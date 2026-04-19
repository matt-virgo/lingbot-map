#!/usr/bin/env python3
"""
Stream a local video file to ``demo_live.py`` ingest WebSocket at wall-clock FPS.

Useful as a **mock Ray-Ban / glasses feed**: paces frames in real time instead of
preloading them, so the receiving server sees them arrive over the wire just as
it would from a phone-relayed live source.

Requires: pip install opencv-python websockets  (or ``pip install 'lingbot-map[live]'``)

Examples::

    # Send the bundled clip at 10 FPS (matches batch demo's --fps default)
    python -m lingbot_map.live.video_client \\
        --video videos/IMG_2559_000_lores.mp4 --fps 10

    # Loop forever (good for long-running stress / FPS measurements)
    python -m lingbot_map.live.video_client \\
        --video videos/IMG_2559_000_lores.mp4 --fps 10 --loop
"""

from __future__ import annotations

import argparse
import asyncio
import os
import struct
import time

import cv2

try:
    import websockets
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Install websockets: pip install websockets  "
        "(or pip install 'lingbot-map[live]')"
    ) from e


async def _stream_video(
    url: str,
    video_path: str,
    target_fps: float,
    jpeg_quality: int,
    loop: bool,
    start_frame: int,
    max_frames: int,
) -> None:
    if not os.path.exists(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    period = 1.0 / max(target_fps, 0.1)
    sent = 0

    async with websockets.connect(url, max_size=None) as ws:
        print(f"Connected to {url}; streaming {video_path} at ~{target_fps} FPS")
        while True:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise SystemExit(f"Could not open video {video_path}")
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
            stride = max(1, round(src_fps / target_fps))
            print(
                f"  source FPS={src_fps:.2f}, total={total}, "
                f"stride={stride} (every {stride}-th frame)"
            )

            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx < start_frame:
                    idx += 1
                    continue
                if (idx - start_frame) % stride != 0:
                    idx += 1
                    continue

                t0 = time.perf_counter()
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
                )
                if ok:
                    jpeg = buf.tobytes()
                    ts_us = time.time_ns() // 1000
                    msg = struct.pack("<QI", ts_us, len(jpeg)) + jpeg
                    await ws.send(msg)
                    sent += 1
                    if sent % 30 == 0:
                        print(f"  sent {sent} frames")

                idx += 1
                if max_frames > 0 and sent >= max_frames:
                    cap.release()
                    print(f"Reached --max-frames={max_frames}; exiting.")
                    return

                elapsed = time.perf_counter() - t0
                await asyncio.sleep(max(0.0, period - elapsed))

            cap.release()
            if not loop:
                print(f"End of video. Sent {sent} frames.")
                return
            print("Looping...")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Mock Ray-Ban: stream a local video file to demo_live ingest."
    )
    p.add_argument("--url", type=str, default="ws://127.0.0.1:8765/ws")
    p.add_argument("--video", type=str, required=True, help="Path to a .mp4/.mov file")
    p.add_argument("--fps", type=float, default=10.0,
                   help="Target send rate in frames/sec (independent of source FPS)")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--loop", action="store_true", help="Restart the video on EOF")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Skip this many source frames before sending")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after sending this many frames (0 = no limit)")
    args = p.parse_args()
    asyncio.run(_stream_video(
        args.url, args.video, args.fps, args.jpeg_quality,
        args.loop, args.start_frame, args.max_frames,
    ))


if __name__ == "__main__":
    main()
