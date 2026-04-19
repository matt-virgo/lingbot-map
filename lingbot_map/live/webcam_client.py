#!/usr/bin/env python3
"""
Send webcam frames to ``demo_live.py`` ingest WebSocket (smoke test).

Requires: pip install opencv-python websockets

Example::

    python -m lingbot_map.live.webcam_client --url ws://127.0.0.1:8765/ws --camera 0
"""

from __future__ import annotations

import argparse
import asyncio
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


async def _run(url: str, camera_id: int, target_fps: float, jpeg_quality: int) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera_id}")
    period = 1.0 / max(target_fps, 0.1)
    async with websockets.connect(url, max_size=None) as ws:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.05)
                continue
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            )
            if not ok:
                continue
            jpeg = buf.tobytes()
            ts_us = time.time_ns() // 1000
            msg = struct.pack("<QI", ts_us, len(jpeg)) + jpeg
            await ws.send(msg)
            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0.0, period - elapsed))


def main() -> None:
    p = argparse.ArgumentParser(description="Webcam -> demo_live JPEG WebSocket client")
    p.add_argument("--url", type=str, default="ws://127.0.0.1:8765/ws")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--jpeg-quality", type=int, default=85)
    args = p.parse_args()
    asyncio.run(_run(args.url, args.camera, args.fps, args.jpeg_quality))


if __name__ == "__main__":
    main()
