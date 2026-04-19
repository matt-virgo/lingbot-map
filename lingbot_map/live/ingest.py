# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
WebSocket ingest for live frames (e.g. Meta Wearables companion app).

Wire format (one WebSocket *binary* message per frame)::

    <uint64 little-endian timestamp_us>
    <uint32 little-endian jpeg_len>
    <jpeg_len bytes JPEG payload>

Decoded frames are BGR ``numpy.uint8`` arrays pushed to a thread-safe queue with
oldest-drop when full (keeps latency bounded).
"""

from __future__ import annotations

import asyncio
import struct
import threading
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import PlainTextResponse
except ImportError as e:  # pragma: no cover
    FastAPI = None  # type: ignore
    WebSocket = None  # type: ignore
    WebSocketDisconnect = Exception  # type: ignore
    PlainTextResponse = None  # type: ignore
    _FASTAPI_IMPORT_ERROR = e
else:
    _FASTAPI_IMPORT_ERROR = None


@dataclass
class FrameMessage:
    """One decoded video frame from the ingest WebSocket."""

    timestamp_us: int
    """Monotonic or wall-clock microsecond stamp from sender."""

    bgr: np.ndarray
    """OpenCV BGR image, shape ``(H, W, 3)``, ``uint8``."""


@dataclass
class IngestStats:
    """Cumulative counters updated by the ingest server (thread-safe enough
    for simple read-only sampling from the main thread; ints are atomic in
    CPython)."""

    received: int = 0
    decoded: int = 0
    dropped: int = 0
    decode_errors: int = 0


def _decode_jpeg_message(data: bytes) -> Tuple[int, np.ndarray]:
    if len(data) < 12:
        raise ValueError(f"message too short: {len(data)} bytes (need >= 12)")
    timestamp_us, jpeg_len = struct.unpack_from("<QI", data, 0)
    if jpeg_len < 1 or jpeg_len > 50 * 1024 * 1024:
        raise ValueError(f"invalid jpeg_len={jpeg_len}")
    if len(data) < 12 + jpeg_len:
        raise ValueError(f"truncated payload: need {12 + jpeg_len}, got {len(data)}")
    jpeg = data[12 : 12 + jpeg_len]
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed (not a valid JPEG?)")
    return int(timestamp_us), bgr


def create_ingest_app(
    frame_queue: "Queue",
    max_queue_size: int = 2,
    on_health: Optional[Callable[[], str]] = None,
    stats: Optional[IngestStats] = None,
) -> "FastAPI":
    """
    Build a FastAPI app that accepts JPEG frames on ``/ws``.

    Args:
        frame_queue: Thread-safe ``queue.Queue`` (maxsize should equal
            ``max_queue_size``). Oldest frame is dropped when full.
        max_queue_size: Expected queue capacity (for docs / assertions only).
        on_health: Optional callback returning a short status string for ``/health``.
    """
    if FastAPI is None:
        raise ImportError(
            "Live ingest requires fastapi and uvicorn. "
            "Install with: pip install 'lingbot-map[live]'"
        ) from _FASTAPI_IMPORT_ERROR

    if frame_queue.maxsize != max_queue_size:
        # Allow mismatch but warn via docstring; caller should set maxsize=2
        pass

    app = FastAPI(title="LingBot-Map live ingest", version="0.1.0")
    _stats = stats if stats is not None else IngestStats()

    def _put_drop_oldest(msg: FrameMessage) -> None:
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
                _stats.dropped += 1
            except Empty:
                pass
        try:
            frame_queue.put_nowait(msg)
        except Full:
            try:
                frame_queue.get_nowait()
                _stats.dropped += 1
            except Empty:
                pass
            try:
                frame_queue.put_nowait(msg)
            except Full:
                _stats.dropped += 1

    @app.get("/health")
    async def health():
        body = "ok"
        if on_health is not None:
            try:
                body = f"ok {on_health()}"
            except Exception as exc:  # pragma: no cover
                body = f"ok (health callback error: {exc})"
        return PlainTextResponse(body)

    @app.websocket("/ws")
    async def websocket_ingest(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_bytes()
                _stats.received += 1

                def _decode_and_enqueue(raw: bytes) -> None:
                    ts_us, bgr = _decode_jpeg_message(raw)
                    _put_drop_oldest(FrameMessage(timestamp_us=ts_us, bgr=bgr))
                    _stats.decoded += 1

                try:
                    await asyncio.to_thread(_decode_and_enqueue, data)
                except ValueError as exc:
                    _stats.decode_errors += 1
                    await websocket.send_text(f"error: {exc}")
                    continue
        except WebSocketDisconnect:
            pass

    return app


def run_ingest_server_thread(
    frame_queue: "Queue",
    host: str = "0.0.0.0",
    port: int = 8765,
    *,
    max_queue_size: int = 2,
    log_level: str = "warning",
    stats: Optional[IngestStats] = None,
) -> threading.Thread:
    """
    Start uvicorn serving ``create_ingest_app`` in a daemon thread.

    Returns the started thread (call ``.join()`` from main if you want to block).
    """
    if FastAPI is None:
        raise ImportError(
            "Live ingest requires fastapi and uvicorn. "
            "Install with: pip install 'lingbot-map[live]'"
        ) from _FASTAPI_IMPORT_ERROR

    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Live ingest requires uvicorn. Install with: pip install 'lingbot-map[live]'"
        ) from e

    app = create_ingest_app(frame_queue, max_queue_size=max_queue_size, stats=stats)

    def _serve() -> None:
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level=log_level,
            ws_ping_interval=None,
            ws_ping_timeout=None,
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve())

    t = threading.Thread(target=_serve, name="lingbot-live-ingest", daemon=True)
    t.start()
    return t
