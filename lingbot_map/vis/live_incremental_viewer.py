# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Incremental viser scene for real-time streaming (one point cloud + frustum per frame).

Unlike :class:`PointCloudViewer`, this does not require all frames up front; call
:meth:`LiveIncrementalViewer.append_frame` as each frame is reconstructed.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import List, Optional

import cv2
import numpy as np
import viser
import viser.transforms as tf

from lingbot_map.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from lingbot_map.vis.sky_segmentation import apply_sky_segmentation


class LiveIncrementalViewer:
    """
    Minimal viser server that grows the scene frame-by-frame.

    Uses the same ``extrinsic`` / ``intrinsic`` / depth tensors as
    :class:`PointCloudViewer` (i.e. the numpy arrays from ``prepare_for_visualization``
    after ``demo.py`` postprocess).
    """

    def __init__(
        self,
        port: int = 8080,
        conf_threshold: float = 1.5,
        downsample_factor: int = 10,
        point_size: float = 0.00001,
        mask_sky: bool = False,
        sky_mask_dir: Optional[str] = None,
        use_point_map: bool = False,
        max_frustums: int = 30,
        pc_hud_history: int = 60,
        pc_hud_height: int = 220,
    ) -> None:
        self.port = port
        self.conf_threshold = conf_threshold
        self.downsample_factor = max(1, int(downsample_factor))
        self.point_size = point_size
        self.mask_sky = mask_sky
        self.sky_mask_dir = sky_mask_dir
        self.use_point_map = use_point_map
        self.max_frustums = max(1, int(max_frustums))
        self.pc_hud_history = max(1, int(pc_hud_history))
        self.pc_hud_height = max(32, int(pc_hud_height))
        # Rolling CPU cache of (pts_world, colors_u8) per appended frame used to
        # rasterize a first-person view of the accumulated point cloud.
        self._pc_cache: List[tuple[np.ndarray, np.ndarray]] = []

        if self.mask_sky and not self.sky_mask_dir:
            self.sky_mask_dir = os.path.join(
                tempfile.gettempdir(), "lingbot_map_live_sky_masks"
            )
            os.makedirs(self.sky_mask_dir, exist_ok=True)

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(
            titlebar_content=None,
            control_layout="collapsible",
            control_width="large",
        )

        self._frame_count = 0
        self.pc_handles: List = []
        self.cam_handles: List = []
        # Point size and confidence threshold are now fixed at construction time
        # (tunable via CLI flags) — the sliders were removed to save screen space.
        # Kept as mutable attributes so code that reads them still works.
        self._point_size_value = float(point_size)
        self._show_frustum_image = self.server.gui.add_checkbox(
            "Show image on current frustum", initial_value=True
        )
        self._show_hud = self.server.gui.add_checkbox(
            "Show camera HUD", initial_value=True
        )
        self._show_pc_hud = self.server.gui.add_checkbox(
            "Show point cloud HUD", initial_value=True
        )
        self._pos_label = self.server.gui.add_text(
            "Camera XYZ (m)", "+0.000, +0.000, +0.000"
        )
        self._speed_label = self.server.gui.add_text(
            "Walking speed", "0.00 m/s (0.0 km/h)"
        )
        self._reset_btn = self.server.gui.add_button("Reset origin to current camera")
        # Re-center button: also has an "undo" companion to go back to raw world frame.
        self._undo_reset_btn = self.server.gui.add_button("Clear origin reset")
        # Single composite HUD image: camera frame + point-cloud render placed
        # side by side (same height). Sidebar auto-widens to fit the image.
        self._hud_composite = self.server.gui.add_image(
            np.zeros((self.pc_hud_height, self.pc_hud_height * 2, 3), dtype=np.uint8),
            label="Camera HUD  |  Point cloud from camera",
            format="jpeg",
            jpeg_quality=65,
        )

        # The /live frame is the parent for all reconstruction nodes. Shifting
        # its position effectively translates the whole scene, so the "reset
        # origin" button is a one-line transform instead of per-node rewrites.
        self._live_frame = self.server.scene.add_frame("/live", show_axes=False)
        self._world_offset = np.zeros(3, dtype=np.float64)
        self._last_cam_world: Optional[np.ndarray] = None
        # Speed estimation state (EMA-smoothed so 5-Hz sampling isn't jittery).
        self._prev_cam_world: Optional[np.ndarray] = None
        self._prev_cam_ts: Optional[float] = None
        self._speed_ema: Optional[float] = None
        self._speed_ema_alpha: float = 0.35

        @self._reset_btn.on_click
        def _(_) -> None:
            if self._last_cam_world is None:
                return
            self._world_offset = self._last_cam_world.astype(np.float64).copy()
            self._live_frame.position = tuple((-self._world_offset).tolist())

        @self._undo_reset_btn.on_click
        def _(_) -> None:
            self._world_offset = np.zeros(3, dtype=np.float64)
            self._live_frame.position = (0.0, 0.0, 0.0)

    def _render_pc_from_camera(
        self,
        ext: np.ndarray,
        intr: np.ndarray,
        H_in: int,
        W_in: int,
    ) -> np.ndarray:
        """
        Project cached world points into the current camera and rasterize to a
        small uint8 RGB image. Uses the input-frame intrinsics (``ext`` is
        world->camera, matching ``pred_dict['extrinsic']``).
        """
        H = self.pc_hud_height
        W = max(1, int(round(H * (W_in / max(H_in, 1)))))
        img = np.zeros((H, W, 3), dtype=np.uint8)
        if not self._pc_cache:
            return img

        pts_all = np.concatenate([p for p, _ in self._pc_cache], axis=0)
        cols_all = np.concatenate([c for _, c in self._pc_cache], axis=0)
        if pts_all.size == 0:
            return img

        scale = H / max(H_in, 1)
        fx = intr[0, 0] * scale
        fy = intr[1, 1] * scale
        cx = intr[0, 2] * scale
        cy = intr[1, 2] * scale

        R = ext[:3, :3]
        t = ext[:3, 3]
        p_cam = pts_all @ R.T + t  # (N, 3)
        z = p_cam[:, 2]
        mask = z > 0.01
        if not np.any(mask):
            return img
        p_cam = p_cam[mask]
        cols = cols_all[mask]
        z = z[mask]

        u = (fx * p_cam[:, 0] / z + cx).astype(np.int32)
        v = (fy * p_cam[:, 1] / z + cy).astype(np.int32)
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_bounds):
            return img
        u = u[in_bounds]
        v = v[in_bounds]
        cols = cols[in_bounds]
        z = z[in_bounds]

        # Paint far-to-near so near points overwrite far ones (cheap z-buffer).
        order = np.argsort(-z)
        u = u[order]
        v = v[order]
        cols = cols[order]

        # 2x2 splat so sparse points are actually visible at this resolution.
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
            uu = np.minimum(u + du, W - 1)
            vv = np.minimum(v + dv, H - 1)
            img[vv, uu] = cols
        return img

    def _filter_points(
        self,
        world_points: np.ndarray,
        colors_hw3: np.ndarray,
        conf: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flatten, confidence-filter, and spatially subsample one frame."""
        pts = world_points.reshape(-1, 3)
        cols = colors_hw3.reshape(-1, 3)
        if conf is not None:
            c = conf.reshape(-1)
            mask = c > self.conf_threshold
            pts = pts[mask]
            cols = cols[mask]
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        cols = cols[valid]
        if self.downsample_factor > 1 and len(pts) > 0:
            idx = np.arange(0, len(pts), self.downsample_factor)
            pts = pts[idx]
            cols = cols[idx]
        if cols.dtype != np.uint8:
            cols = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8)
        return pts, cols

    def append_frame(
        self,
        frame_idx: int,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        depth_map: np.ndarray,
        depth_conf: Optional[np.ndarray],
        world_points: Optional[np.ndarray],
        world_points_conf: Optional[np.ndarray],
        image_chw_float: np.ndarray,
        timestamp_s: Optional[float] = None,
    ) -> None:
        """
        Add one frame's geometry to the scene.

        Args:
            frame_idx: Index used in node names (monotonic).
            extrinsic: ``(3, 4)`` camera matrix in the same convention as ``demo.py``
                / :class:`PointCloudViewer` (``pred_dict['extrinsic']``).
            intrinsic: ``(3, 3)`` intrinsics for this frame.
            depth_map: ``(H, W, 1)`` or ``(H, W)`` depth (used if not ``use_point_map``).
            depth_conf: ``(H, W)`` optional confidence for depth path.
            world_points: ``(H, W, 3)`` if ``use_point_map``.
            world_points_conf: ``(H, W)`` if ``use_point_map``.
            image_chw_float: ``(3, H, W)`` RGB in ``[0, 1]`` for vertex colors.
        """
        ext = np.asarray(extrinsic, dtype=np.float64)
        intr = np.asarray(intrinsic, dtype=np.float64)
        if ext.shape != (3, 4):
            raise ValueError(f"extrinsic must be (3,4), got {ext.shape}")
        if intr.shape != (3, 3):
            raise ValueError(f"intrinsic must be (3,3), got {intr.shape}")

        if not self.use_point_map:
            d = np.asarray(depth_map, dtype=np.float32)
            if d.ndim == 2:
                d = d[..., np.newaxis]
            wpts = unproject_depth_map_to_point_map(
                d[np.newaxis], ext[np.newaxis], intr[np.newaxis]
            )[0]
            conf = depth_conf
        else:
            wpts = np.asarray(world_points, dtype=np.float32)
            conf = world_points_conf
            if conf is None:
                conf = np.ones(wpts.shape[:2], dtype=np.float32)

        colors = np.asarray(image_chw_float, dtype=np.float32)
        colors = np.clip(colors.transpose(1, 2, 0), 0.0, 1.0)

        if self.mask_sky and conf is not None and conf.size > 0:
            conf_img = np.asarray(conf, dtype=np.float32)
            S = np.stack([image_chw_float], axis=0)
            C = np.stack([conf_img], axis=0)
            C = apply_sky_segmentation(
                C,
                image_folder=None,
                images=S,
                sky_mask_dir=self.sky_mask_dir,
            )
            conf = C[0]

        pts, cols_u8 = self._filter_points(wpts, colors, conf)
        if len(pts) == 0:
            pts = np.zeros((0, 3), dtype=np.float32)
            cols_u8 = np.zeros((0, 3), dtype=np.uint8)

        pc = self.server.scene.add_point_cloud(
            name=f"/live/pc/{frame_idx:06d}",
            points=pts,
            colors=cols_u8,
            point_size=self._point_size_value,
        )
        self.pc_handles.append(pc)

        # Maintain a bounded CPU cache for the "point cloud from camera" HUD.
        self._pc_cache.append((pts.astype(np.float32, copy=False), cols_u8))
        while len(self._pc_cache) > self.pc_hud_history:
            self._pc_cache.pop(0)

        cam_to_world = closed_form_inverse_se3(ext[np.newaxis])[0]
        R = cam_to_world[:3, :3]
        t = cam_to_world[:3, 3]
        q = tf.SO3.from_matrix(R).wxyz

        # Cache the current world-space camera position for the reset button and
        # update the displayed XYZ (relative to the current origin offset).
        self._last_cam_world = np.asarray(t, dtype=np.float64).copy()
        disp = self._last_cam_world - self._world_offset
        self._pos_label.value = f"{disp[0]:+.3f}, {disp[1]:+.3f}, {disp[2]:+.3f}"

        # Walking speed = ||Δposition|| / Δt. Use sender timestamp if available
        # (more accurate than wall-clock at the viewer), else fall back.
        now_ts = float(timestamp_s) if timestamp_s is not None else time.perf_counter()
        if self._prev_cam_world is not None and self._prev_cam_ts is not None:
            dt = now_ts - self._prev_cam_ts
            if dt > 1e-3:
                v = float(np.linalg.norm(self._last_cam_world - self._prev_cam_world) / dt)
                # Clip absurd jumps (e.g. first-frame scale settling) from skewing the EMA.
                if v < 20.0:
                    if self._speed_ema is None:
                        self._speed_ema = v
                    else:
                        a = self._speed_ema_alpha
                        self._speed_ema = a * v + (1.0 - a) * self._speed_ema
                    kmh = self._speed_ema * 3.6
                    mph = self._speed_ema * 2.2369363
                    self._speed_label.value = (
                        f"{self._speed_ema:.2f} m/s ({kmh:.1f} km/h, {mph:.1f} mph)"
                    )
        self._prev_cam_world = self._last_cam_world
        self._prev_cam_ts = now_ts
        focal = float(intr[0, 0])
        pp = (float(intr[0, 2]), float(intr[1, 2]))
        fov = 2 * np.arctan(pp[0] / max(focal, 1e-6))
        aspect = pp[0] / max(pp[1], 1e-6)

        # Revert the previously-latest camera back to the "old" blue color and
        # strip its image (only the current frustum shows the live view).
        if self.cam_handles:
            try:
                self.cam_handles[-1].color = (90, 140, 220)
                self.cam_handles[-1].image = None
            except Exception:
                pass

        # Build a uint8 HxWx3 RGB image for the frustum / HUD panel.
        rgb_u8 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)

        fr = self.server.scene.add_camera_frustum(
            name=f"/live/cam/{frame_idx:06d}",
            fov=float(fov),
            aspect=float(aspect),
            wxyz=q,
            position=t,
            scale=0.06,
            color=(40, 220, 90),
            image=rgb_u8 if self._show_frustum_image.value else None,
            format="jpeg",
            jpeg_quality=70,
        )
        self.cam_handles.append(fr)

        # Build the composite HUD (camera frame + point-cloud render, same
        # height, placed side by side with a thin dark separator).
        show_cam = bool(self._show_hud.value)
        show_pc = bool(self._show_pc_hud.value)
        if show_cam or show_pc:
            try:
                H_in, W_in = rgb_u8.shape[:2]
                H = self.pc_hud_height
                W_side = max(1, int(round(H * (W_in / max(H_in, 1)))))
                panels: List[np.ndarray] = []
                if show_cam:
                    # Downscale camera frame to HUD height (INTER_AREA is
                    # the right sampler for shrinking).
                    cam_small = cv2.resize(
                        rgb_u8, (W_side, H), interpolation=cv2.INTER_AREA
                    )
                    panels.append(cam_small)
                if show_pc:
                    pc_view = self._render_pc_from_camera(ext, intr, H_in, W_in)
                    # _render_pc_from_camera already returns shape (H, W_side, 3).
                    if pc_view.shape[0] != H or pc_view.shape[1] != W_side:
                        pc_view = cv2.resize(
                            pc_view, (W_side, H), interpolation=cv2.INTER_NEAREST
                        )
                    panels.append(pc_view)
                if len(panels) == 2:
                    # 2-pixel dark separator between the two views.
                    sep = np.zeros((H, 2, 3), dtype=np.uint8)
                    composite = np.concatenate([panels[0], sep, panels[1]], axis=1)
                else:
                    composite = panels[0]
                self._hud_composite.image = composite
            except Exception:
                pass

        # Keep only the current frustum plus the most recent (max_frustums - 1)
        # older ones; evict anything older so the scene stays readable.
        while len(self.cam_handles) > self.max_frustums:
            old = self.cam_handles.pop(0)
            try:
                old.remove()
            except Exception:
                pass

        self._frame_count += 1

    def set_status(self, text: str) -> None:
        # GUI status widget was removed to save screen space; retained as a no-op
        # so external callers (demo_live.py) don't need to be aware.
        del text

    def evict_oldest_until(self, max_frames: int) -> None:
        """
        Drop the oldest point clouds / camera frustums from the scene so that
        at most ``max_frames`` remain. Cheap way to cap browser memory and
        per-update WebSocket payload when streaming indefinitely.
        """
        if max_frames <= 0:
            return
        while len(self.pc_handles) > max_frames:
            h = self.pc_handles.pop(0)
            try:
                h.remove()
            except Exception:
                pass
        while len(self.cam_handles) > max_frames:
            h = self.cam_handles.pop(0)
            try:
                h.remove()
            except Exception:
                pass
