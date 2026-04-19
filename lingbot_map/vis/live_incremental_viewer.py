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
from typing import List, Optional

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
    ) -> None:
        self.port = port
        self.conf_threshold = conf_threshold
        self.downsample_factor = max(1, int(downsample_factor))
        self.point_size = point_size
        self.mask_sky = mask_sky
        self.sky_mask_dir = sky_mask_dir
        self.use_point_map = use_point_map

        if self.mask_sky and not self.sky_mask_dir:
            self.sky_mask_dir = os.path.join(
                tempfile.gettempdir(), "lingbot_map_live_sky_masks"
            )
            os.makedirs(self.sky_mask_dir, exist_ok=True)

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        self._frame_count = 0
        self.pc_handles: List = []
        self.cam_handles: List = []
        self._status = self.server.gui.add_text("Live status", "Waiting for frames…")
        self._frame_label = self.server.gui.add_text("Frames appended", "0")
        self._show_cam = self.server.gui.add_checkbox("Show cameras", initial_value=True)
        self._psize = self.server.gui.add_slider(
            "Point size", min=0.00001, max=0.01, step=0.00001, initial_value=point_size
        )
        self._conf = self.server.gui.add_slider(
            "Conf threshold", min=0.5, max=5.0, step=0.05, initial_value=conf_threshold
        )

        @self._psize.on_update
        def _(_) -> None:
            for h in self.pc_handles:
                h.point_size = self._psize.value

        @self._conf.on_update
        def _(_) -> None:
            self.conf_threshold = float(self._conf.value)

        @self._show_cam.on_update
        def _(_) -> None:
            v = self._show_cam.value
            for h in self.cam_handles:
                h.visible = v

        self.server.scene.add_frame("/live", show_axes=False)

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
            point_size=self._psize.value,
        )
        self.pc_handles.append(pc)

        cam_to_world = closed_form_inverse_se3(ext[np.newaxis])[0]
        R = cam_to_world[:3, :3]
        t = cam_to_world[:3, 3]
        q = tf.SO3.from_matrix(R).wxyz
        focal = float(intr[0, 0])
        pp = (float(intr[0, 2]), float(intr[1, 2]))
        fov = 2 * np.arctan(pp[0] / max(focal, 1e-6))
        aspect = pp[0] / max(pp[1], 1e-6)

        fr = self.server.scene.add_camera_frustum(
            name=f"/live/cam/{frame_idx:06d}",
            fov=float(fov),
            aspect=float(aspect),
            wxyz=q,
            position=t,
            scale=0.05,
            color=(90, 140, 220),
        )
        self.cam_handles.append(fr)

        self._frame_count += 1
        self._frame_label.value = str(self._frame_count)
        self._status.value = f"Last frame index {frame_idx}"

    def set_status(self, text: str) -> None:
        self._status.value = text
