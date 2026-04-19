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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

import demo as batch_demo
from lingbot_map.live.ingest import FrameMessage, run_ingest_server_thread
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
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--mask_sky", action="store_true")
    parser.add_argument("--sky_mask_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available; live inference will be very slow.")

    frame_queue: queue.Queue[FrameMessage] = queue.Queue(maxsize=2)
    run_ingest_server_thread(
        frame_queue,
        host=args.ingest_host,
        port=args.ingest_port,
        max_queue_size=2,
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
    for _ in range(scale_frames):
        msg = frame_queue.get()
        rgb = cv2.cvtColor(msg.bgr, cv2.COLOR_BGR2RGB)
        scale_cpu.append(
            preprocess_single_frame(
                rgb, mode="crop", image_size=args.image_size, patch_size=args.patch_size
            )
        )
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
        _append_vis_frame(viewer, vis, i, i, use_point_map=args.use_point_map)
    viewer.set_status(f"Scale done ({scale_frames} frames). Streaming…")
    del out, pred, images_cpu, vis, scale_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    frame_idx = scale_frames
    t_last = time.perf_counter()
    n_frames = 0
    while True:
        msg = frame_queue.get()
        rgb = cv2.cvtColor(msg.bgr, cv2.COLOR_BGR2RGB)
        x_cpu = preprocess_single_frame(
            rgb, mode="crop", image_size=args.image_size, patch_size=args.patch_size
        )
        x = x_cpu.to(device).unsqueeze(0).unsqueeze(0)

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

        pred, img_c = batch_demo.postprocess(out, x[0])
        vis = batch_demo.prepare_for_visualization(pred, img_c)
        _append_vis_frame(viewer, vis, 0, frame_idx, use_point_map=args.use_point_map)
        frame_idx += 1
        n_frames += 1
        if n_frames % 30 == 0:
            elapsed = time.perf_counter() - t_last
            fps = 30.0 / elapsed if elapsed > 0 else 0.0
            t_last = time.perf_counter()
            viewer.set_status(f"Streaming ~{fps:.1f} FPS (last 30 frames)")

        del out, pred, img_c, vis, x


if __name__ == "__main__":
    main()
