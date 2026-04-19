#!/usr/bin/env python3
"""LingBot-Map real-time demo: WebSocket JPEG ingest + incremental viser.

Companion app (e.g. Meta Wearables SDK on a phone) sends binary WebSocket
messages::

    <uint64 little-endian timestamp_us><uint32 little-endian jpeg_len><jpeg bytes>

See ``lingbot_map/live/ingest.py`` and the README "Live streaming" section.

Example::

    pip install -e ".[vis,live]"
    python demo_live.py --model_path /path/to/checkpoint.pt --ingest_port 8765

    # In another terminal (webcam smoke test):
    python -m lingbot_map.live.webcam_client --url ws://127.0.0.1:8765/ws
"""

from __future__ import annotations

import argparse
import contextlib
import os
import queue
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

import demo as batch_demo
from lingbot_map.live.ingest import FrameMessage, IngestStats, run_ingest_server_thread
from lingbot_map.utils.load_fn import preprocess_single_frame
from lingbot_map.vis.live_incremental_viewer import LiveIncrementalViewer


def _maybe_compile_streaming(model: torch.nn.Module, enabled: bool) -> None:
    """Apply ``torch.compile`` to heavy blocks (keeps ``point_head`` for depth/points)."""
    if not enabled:
        return
    try:
        agg = model.aggregator
        for i, b in enumerate(agg.frame_blocks):
            agg.frame_blocks[i] = torch.compile(b, mode="reduce-overhead")
        for i, b in enumerate(agg.patch_embed.blocks):
            agg.patch_embed.blocks[i] = torch.compile(b, mode="reduce-overhead")
        for b in agg.global_blocks:
            if hasattr(b, "attn_pre"):
                b.attn_pre = torch.compile(b.attn_pre, mode="reduce-overhead")
            if hasattr(b, "ffn_residual"):
                b.ffn_residual = torch.compile(b.ffn_residual, mode="reduce-overhead")
            b.attn.proj = torch.compile(b.attn.proj, mode="reduce-overhead")
        print("torch.compile applied to aggregator blocks (point_head kept).")
    except Exception as exc:  # pragma: no cover
        print(f"torch.compile skipped: {exc}")


def _cudagraph_mark() -> None:
    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()


def _append_vis_frame(
    viewer: LiveIncrementalViewer,
    vis: dict,
    vis_index: int,
    scene_frame_id: int,
    *,
    use_point_map: bool,
    timestamp_s: Optional[float] = None,
) -> None:
    depth = vis.get("depth")
    if not use_point_map and depth is None:
        return
    if use_point_map and vis.get("world_points") is None:
        return
    dc = vis.get("depth_conf")
    dc_i = dc[vis_index] if dc is not None else None
    wp = vis.get("world_points") if use_point_map else None
    wpc = vis.get("world_points_conf") if use_point_map else None
    if use_point_map:
        d_i = np.zeros((1, 1, 1), dtype=np.float32)
    else:
        d_i = depth[vis_index]
    viewer.append_frame(
        scene_frame_id,
        vis["extrinsic"][vis_index],
        vis["intrinsic"][vis_index],
        d_i,
        dc_i,
        wp[vis_index] if wp is not None else None,
        wpc[vis_index] if wpc is not None else None,
        vis["images"][vis_index],
        timestamp_s=timestamp_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LingBot-Map: live WebSocket ingest + streaming 3D")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ingest_host", type=str, default="0.0.0.0")
    parser.add_argument("--ingest_port", type=int, default=8765)
    parser.add_argument("--image_size", type=int, default=518,
                        help="Input width for crop preprocess. Try 378 if GPU cannot keep up.")
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--keyframe_interval", type=int, default=4,
                        help="After scale frames, keep every N-th frame in KV cache (1 = all).")
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=1,
                        help="Live default 1 for speed (batch demo defaults to 4).")
    parser.add_argument("--use_sdpa", action="store_true", default=False)
    parser.add_argument("--use_point_map", action="store_true", default=False,
                        help="Visualize model world_points instead of depth unprojection.")
    parser.add_argument("--compile", action="store_true", help="torch.compile aggregator blocks")
    parser.add_argument("--port", type=int, default=8080, help="viser HTTP port")
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=10,
                        help="Spatial subsample of points per frame (higher = fewer points, faster browser)")
    parser.add_argument("--viewer_stride", type=int, default=1,
                        help="Append every N-th frame to viser (model still runs every frame). "
                             "Increase to 4-10 if the browser is laggy.")
    parser.add_argument("--max_scene_frames", type=int, default=0,
                        help="If >0, evict the oldest point cloud / frustum once this many are in the scene "
                             "(rolling window, prevents unbounded growth).")
    parser.add_argument("--max_frustums", type=int, default=30,
                        help="Only keep the current camera frustum plus this many - 1 older ones "
                             "(older frustums are removed from the scene). Point clouds are unaffected.")
    parser.add_argument("--pc_hud_history", type=int, default=60,
                        help="How many past frames of points to include in the 'point cloud HUD' render.")
    parser.add_argument("--pc_hud_height", type=int, default=220,
                        help="Pixel height of each HUD panel; width follows input aspect. "
                             "Composite image is ~2x this wide (camera + point cloud side by side).")
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--mask_sky", action="store_true")
    parser.add_argument("--sky_mask_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available; live inference will be very slow.")

    frame_queue: queue.Queue[FrameMessage] = queue.Queue(maxsize=2)
    ingest_stats = IngestStats()
    run_ingest_server_thread(
        frame_queue,
        host=args.ingest_host,
        port=args.ingest_port,
        max_queue_size=2,
        stats=ingest_stats,
    )
    time.sleep(0.6)
    print(f"Ingest WebSocket: ws://{args.ingest_host}:{args.ingest_port}/ws")
    print("Send JPEG frames as binary: <uint64 ts_us><uint32 len><jpeg>")

    ns = argparse.Namespace(
        mode="streaming",
        image_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        num_scale_frames=args.num_scale_frames,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
        model_path=args.model_path,
    )
    model = batch_demo.load_model(ns, device)
    _maybe_compile_streaming(model, args.compile)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        print(f"Casting aggregator to {dtype} (heads kept in fp32)")
        model.aggregator = model.aggregator.to(dtype=dtype)

    scale_frames = args.num_scale_frames
    print(f"Waiting for {scale_frames} frames to build scale batch…")
    scale_cpu = []
    scale_ts = []
    for _ in range(scale_frames):
        msg = frame_queue.get()
        rgb = cv2.cvtColor(msg.bgr, cv2.COLOR_BGR2RGB)
        scale_cpu.append(
            preprocess_single_frame(
                rgb, mode="crop", image_size=args.image_size, patch_size=args.patch_size
            )
        )
        scale_ts.append(msg.timestamp_us / 1e6)
    scale_batch = torch.cat(scale_cpu, dim=0).unsqueeze(0).to(device)
    del scale_cpu

    viewer = LiveIncrementalViewer(
        port=args.port,
        conf_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        point_size=args.point_size,
        mask_sky=args.mask_sky,
        sky_mask_dir=args.sky_mask_dir,
        use_point_map=args.use_point_map,
        max_frustums=args.max_frustums,
        pc_hud_history=args.pc_hud_history,
        pc_hud_height=args.pc_hud_height,
    )
    print(f"viser: http://localhost:{args.port}")

    model.clean_kv_cache()
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=dtype)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )

    viewer.set_status("Running scale-phase forward…")
    _cudagraph_mark()
    with torch.no_grad(), autocast_ctx:
        out = model.forward(
            scale_batch,
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,
            causal_inference=True,
        )
    pred, images_cpu = batch_demo.postprocess(out, scale_batch[0])
    vis = batch_demo.prepare_for_visualization(pred, images_cpu)
    for i in range(vis["images"].shape[0]):
        ts_i = scale_ts[i] if i < len(scale_ts) else None
        _append_vis_frame(
            viewer, vis, i, i,
            use_point_map=args.use_point_map,
            timestamp_s=ts_i,
        )
    viewer.set_status(f"Scale done ({scale_frames} frames). Streaming…")
    del out, pred, images_cpu, vis, scale_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    frame_idx = scale_frames
    t_report = time.perf_counter()
    proc_since_report = 0
    ingest_recv_at_report = ingest_stats.received
    ingest_drop_at_report = ingest_stats.dropped
    t_prep_ms = t_fwd_ms = t_post_ms = t_view_ms = 0.0

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    while True:
        msg = frame_queue.get()
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(msg.bgr, cv2.COLOR_BGR2RGB)
        x_cpu = preprocess_single_frame(
            rgb, mode="crop", image_size=args.image_size, patch_size=args.patch_size
        )
        # preprocess_single_frame returns (1, 3, H, W); aggregator wants (B, S, 3, H, W).
        x = x_cpu.to(device).unsqueeze(0)  # -> (1, 1, 3, H, W)
        t1 = time.perf_counter()

        is_keyframe = (args.keyframe_interval <= 1) or (
            (frame_idx - scale_frames) % args.keyframe_interval == 0
        )
        if not is_keyframe:
            model._set_skip_append(True)
        _cudagraph_mark()
        with torch.no_grad(), autocast_ctx:
            out = model.forward(
                x,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=1,
                causal_inference=True,
            )
        if not is_keyframe:
            model._set_skip_append(False)
        _sync()
        t2 = time.perf_counter()

        # Skip viewer updates when --viewer_stride > 1 to keep the browser snappy
        # (model still does the per-frame forward to maintain KV cache continuity).
        push_to_viewer = (
            args.viewer_stride <= 1
            or ((frame_idx - scale_frames) % args.viewer_stride == 0)
        )
        if push_to_viewer:
            pred, img_c = batch_demo.postprocess(out, x[0])
            vis = batch_demo.prepare_for_visualization(pred, img_c)
            t3 = time.perf_counter()
            _append_vis_frame(
                viewer, vis, 0, frame_idx,
                use_point_map=args.use_point_map,
                timestamp_s=msg.timestamp_us / 1e6,
            )
            if args.max_scene_frames > 0:
                viewer.evict_oldest_until(args.max_scene_frames)
            del pred, img_c, vis
            t4 = time.perf_counter()
        else:
            t3 = t2
            t4 = t2
        frame_idx += 1
        proc_since_report += 1
        t_prep_ms += (t1 - t0) * 1000.0
        t_fwd_ms += (t2 - t1) * 1000.0
        t_post_ms += (t3 - t2) * 1000.0
        t_view_ms += (t4 - t3) * 1000.0

        now = time.perf_counter()
        dt = now - t_report
        if dt >= 1.0:
            recv_now = ingest_stats.received
            drop_now = ingest_stats.dropped
            input_fps = (recv_now - ingest_recv_at_report) / dt
            drop_rate = (drop_now - ingest_drop_at_report) / dt
            proc_fps = proc_since_report / dt
            qdepth = frame_queue.qsize()
            n = max(proc_since_report, 1)
            line = (
                f"[live] input {input_fps:5.1f} fps | proc {proc_fps:5.1f} fps | "
                f"dropped {drop_rate:4.1f}/s (total {drop_now}) | "
                f"queue {qdepth}/{frame_queue.maxsize} | frame #{frame_idx} | "
                f"ms/frame prep={t_prep_ms/n:5.1f} fwd={t_fwd_ms/n:5.1f} "
                f"post={t_post_ms/n:5.1f} view={t_view_ms/n:5.1f}"
            )
            print(line, flush=True)
            viewer.set_status(f"in {input_fps:.1f} / proc {proc_fps:.1f} fps | drop {drop_rate:.1f}/s")
            t_report = now
            proc_since_report = 0
            ingest_recv_at_report = recv_now
            ingest_drop_at_report = drop_now
            t_prep_ms = t_fwd_ms = t_post_ms = t_view_ms = 0.0

        del out, x


if __name__ == "__main__":
    main()
