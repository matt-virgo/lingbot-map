# LingBot-Map Live Streaming: Android Integration Spec

> Target audience: Android developer building a companion app that streams first-person video from **Meta Ray-Ban Meta smart glasses** (via the **Meta Wearables SDK**) to a running `lingbot-map` real-time reconstruction server.
>
> This document is the complete contract: endpoint, wire format, reference code, image guidance, and acceptance criteria. It is written to be drop-in consumable by Claude Code.

---

## 1. What you are building

An Android app that:

1. Authenticates with the user's Meta Ray-Ban Meta glasses via the **Meta Wearables SDK**.
2. Subscribes to the glasses' live forward-facing camera stream.
3. Re-encodes each frame to JPEG.
4. Streams each JPEG over a **binary WebSocket** to the `lingbot-map` ingest server using the exact wire format specified below.
5. Reconnects on network failures and drops oldest-frames under backpressure.

Everything downstream of your WebSocket send (3D reconstruction, pose estimation, visualization) is already implemented and running on the server.

---

## 2. Server endpoint

| Purpose | URL | Method |
|---|---|---|
| Frame ingest | `ws://195.242.10.98:8765/ws` | WebSocket (binary frames) |
| Health check | `http://195.242.10.98:8765/health` | HTTP GET |

- The server is directly reachable over the internet on this IP; no VPN or tunnel needed.
- Because the endpoint is currently plain `ws://`, **you must allow cleartext traffic to this host** in your Android `network_security_config.xml` (see snippet below), or the OS will block the connection silently.
- IP may change if the server VM is restarted. The operator will announce any change in advance; design your app to read the URL from a config (BuildConfig, remote config, or a settings screen) rather than hardcoding.
- `/health` returns `"ok"` if the ingest thread is alive. Use it as a liveness probe before connecting the WebSocket.
- For production deployment (post-demo), the endpoint will move to `wss://<dns-name>/ws` with TLS; keep scheme handling in your config-driven URL so you can flip to it without code changes.

### Required Android `network_security_config.xml` snippet

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="false">195.242.10.98</domain>
    </domain-config>
</network-security-config>
```

Reference it from your manifest:

```xml
<application
    android:networkSecurityConfig="@xml/network_security_config"
    ...>
```

---

## 3. Wire format

Every frame is **one binary WebSocket message** laid out as:

```
Offset  Size  Field            Type                    Notes
──────  ────  ───────────────  ─────────────────────   ────────────────────────────
0       8     timestamp_us     uint64 little-endian    Microsecond capture time
8       4     jpeg_len         uint32 little-endian    Length of JPEG payload (bytes)
12      N     jpeg_bytes       raw JPEG (SOI..EOI)     Standard baseline JPEG, RGB
```

Total message size = `12 + jpeg_len` bytes.

### Field semantics

- **`timestamp_us`** — microsecond timestamp at the point the frame left the glasses' sensor (or as close to that as the SDK provides). Used for relative motion only; absolute epoch alignment is **not** required. Monotonic is preferred (`SystemClock.elapsedRealtimeNanos() / 1000`) because walking speed is derived from timestamp deltas; wall-clock jumps from NTP syncs would poison the estimate.
- **`jpeg_len`** — exact byte length of the JPEG bytes that follow. Max enforced server-side: 50 MB (way more than needed).
- **`jpeg_bytes`** — standard baseline JPEG, YCbCr or RGB, any resolution. The server decodes with OpenCV `cv2.imdecode` and resizes/crops internally.

### Validation

- The server responds with a **text WebSocket frame** (`error: <message>`) if it cannot decode a message. Non-fatal; continue sending subsequent frames. Log the error on the Android side for debugging.
- The server never sends binary frames to you. Any binary data you receive is a protocol violation — treat it as a fatal error and reconnect.

---

## 4. Frame source: Meta Wearables SDK

Use the Meta Wearables SDK's **live camera preview stream** (not the record-then-upload path). High-level flow:

```
Glasses camera  ──BT/WiFi──▶  Wearables service on phone
                                       │
                                       ▼
                             Your WearablesSdkClient
                                       │
                                       ▼  onFrameAvailable(frame)
                                Convert → JPEG
                                       │
                                       ▼
                              WebSocket.send(binary)
```

Exact SDK calls depend on your SDK version and entitlements. Pseudocode assuming a `WearablesCameraSession` / `CameraFrameCallback` abstraction (adapt to the actual class names in your SDK distribution):

```kotlin
val session = WearablesClient.getInstance(context)
    .cameraSession(CameraId.FORWARD)
    .configure(
        resolution = Resolution(720, 540),
        targetFps = 5,
        pixelFormat = PixelFormat.RGB_888,   // or YUV420; see §6
    )

session.setFrameCallback { frame: WearablesCameraFrame ->
    streamer.sendFrame(frame)
}
session.start()
```

**Requirements you must satisfy via the SDK:**

- Request the camera-stream permission/entitlement during onboarding.
- Handle the SDK's "glasses disconnected / paired device not in range" events — pause streaming, surface status to UI, resume cleanly when the glasses reconnect.
- Do **not** hold frames longer than ~100 ms. The SDK will start backpressuring the glasses if you do, leading to frame drops at the source.

---

## 5. Reference Android client implementation

### 5.1 Dependencies (`build.gradle.kts`)

```kotlin
dependencies {
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    // Meta Wearables SDK deps go here (per your SDK docs).
}
```

### 5.2 WebSocket streamer

```kotlin
package com.example.lingbotclient

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import okhttp3.*
import okio.ByteString.Companion.toByteString
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Streams JPEG frames to the lingbot-map ingest server over a binary WebSocket.
 *
 * - Bounded outgoing queue (capacity 2) with oldest-drop backpressure — matches
 *   the server's ingest policy, so no buffering lag accumulates on the phone.
 * - Automatic exponential-backoff reconnect on socket failure.
 * - All socket I/O happens on a single worker thread; frame capture can call
 *   `enqueueFrame` from any thread.
 */
class LingBotStreamer(
    private val serverUrl: String,
    private val jpegQuality: Int = 85,
) {
    private val client = OkHttpClient.Builder()
        .pingInterval(15, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private val worker = Executors.newSingleThreadExecutor()
    private val queue = ConcurrentLinkedDeque<ByteArray>()
    private val queueCap = 2

    private val running = AtomicBoolean(false)
    @Volatile private var ws: WebSocket? = null
    @Volatile private var connected = false
    private var backoffMs = 500L

    fun start() {
        if (running.getAndSet(true)) return
        worker.execute(::runLoop)
    }

    fun stop() {
        running.set(false)
        ws?.close(1000, "client shutdown")
        ws = null
        worker.shutdown()
    }

    /** Call from the Meta Wearables frame callback. Non-blocking. */
    fun enqueueFrame(bitmap: Bitmap, timestampUs: Long) {
        val msg = encodeMessage(bitmap, timestampUs) ?: return
        // Oldest-drop: if queue is full, discard the oldest pending frame.
        while (queue.size >= queueCap) queue.pollFirst()
        queue.addLast(msg)
    }

    private fun encodeMessage(bitmap: Bitmap, timestampUs: Long): ByteArray? {
        val jpeg = ByteArrayOutputStream(64 * 1024).also {
            bitmap.compress(Bitmap.CompressFormat.JPEG, jpegQuality, it)
        }.toByteArray()
        val header = ByteBuffer.allocate(12)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putLong(timestampUs)
            .putInt(jpeg.size)
            .array()
        val out = ByteArray(header.size + jpeg.size)
        System.arraycopy(header, 0, out, 0, header.size)
        System.arraycopy(jpeg, 0, out, header.size, jpeg.size)
        return out
    }

    private fun runLoop() {
        while (running.get()) {
            if (!connected) connect()
            if (!connected) {
                Thread.sleep(backoffMs)
                backoffMs = (backoffMs * 2).coerceAtMost(10_000)
                continue
            }
            backoffMs = 500L
            val msg = queue.pollFirst()
            if (msg == null) {
                Thread.sleep(5)
                continue
            }
            val sent = ws?.send(msg.toByteString()) ?: false
            if (!sent) {
                Log.w(TAG, "send() returned false; treating as disconnected")
                connected = false
            }
        }
    }

    private fun connect() {
        val req = Request.Builder().url(serverUrl).build()
        ws = client.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                connected = true
                Log.i(TAG, "WS connected: $serverUrl")
            }
            override fun onMessage(ws: WebSocket, text: String) {
                Log.w(TAG, "server said: $text")
            }
            override fun onClosing(ws: WebSocket, code: Int, reason: String) {
                connected = false
                ws.close(1000, null)
            }
            override fun onFailure(ws: WebSocket, t: Throwable, resp: Response?) {
                connected = false
                Log.e(TAG, "WS failure", t)
            }
        })
    }

    companion object { private const val TAG = "LingBotStreamer" }
}
```

### 5.3 Wiring into the Wearables frame callback

```kotlin
val streamer = LingBotStreamer(
    serverUrl = BuildConfig.LINGBOT_WS_URL, // from app config, e.g. ws://192.168.1.42:8765/ws
)
streamer.start()

session.setFrameCallback { frame ->
    // Convert whatever the SDK gives you (YUV, RGB planes, ImageProxy, etc.) to Bitmap.
    val bitmap: Bitmap = frame.toBitmap()
    val tsUs: Long = frame.timestampNanos / 1_000L
    streamer.enqueueFrame(bitmap, tsUs)
}
```

---

## 6. Image guidance

| Parameter | Recommended | Why |
|---|---|---|
| Send rate | **5 fps** | Matches server processing rate at `--image_size 518`. Higher = server drops, wasting battery + bandwidth. |
| Resolution | **720×540** or **640×480** | Server resizes to ~518×392 internally. Larger pre-compression is pointless. |
| JPEG quality | **85** | Sweet spot: ~30 KB/frame typical, no visible artifacts at reconstruction scale. <70 hurts quality. |
| Color space | **RGB** (sRGB) | Server assumes RGB. If SDK gives you YUV420/NV21, convert via `RenderScript` or `libyuv` before JPEG encode. |
| Timestamp source | **`SystemClock.elapsedRealtimeNanos()`** | Monotonic; immune to NTP jumps. Epoch alignment not required. |
| Orientation | **Upright, no rotation metadata** | Strip EXIF. Rotate pixels before encode if the sensor is mounted sideways. |

### Bandwidth budget

At 5 fps × 30 KB/frame = **150 KB/s ≈ 1.2 Mbps** uplink. Comfortable on WiFi, workable on 4G, tight on weak cellular.

### Common visual-quality pitfalls

Observed on first-run integrations; if the server's side-by-side HUD shows a squished camera view or a dark reconstruction view, it is almost always one of these:

1. **Squished / stretched camera feed**
    - **Symptom**: faces elongated, straight vertical lines tilted, circle appears as oval.
    - **Cause**: the bitmap you send has non-square pixels, i.e. its `width:height` ratio doesn't match the sensor's native aspect. Happens when you `Bitmap.createScaledBitmap(src, 720, 540, ...)` on a frame whose native aspect is 16:9 or 4:3, without letterbox/padding.
    - **Fix**: pass the frame through **without aspect-ratio distortion**. Either send the native-resolution bitmap directly (preferred — server will resize correctly) or, if you must resize, use a resize that preserves aspect and center-crops/pads to the target. The server accepts any resolution and does its own proportional resize to `518 × H` internally.
    - **Quick check**: print `bitmap.width / bitmap.height.toFloat()` before encoding. Ray-Ban Meta forward camera native aspect is **~4:3 (≈ 1.33)** for photo mode and **~16:9 (≈ 1.78)** for video mode. If you're sending e.g. 720×540 (1.33) but your native source is 16:9, you're stretching it.

2. **Dark or black reconstruction view ("Point cloud from camera")**
    - **Symptom**: the right-hand HUD panel looks almost black even while the left panel shows a well-lit scene.
    - **Cause (partial)**: typically means the scene has depth/texture the model isn't confident about yet, so few points are projected into the frame. With typical downsampling and confidence thresholds, an under-populated view renders almost entirely as background.
    - **Cause (your side)**: sending frames that are **too dark or low-contrast** (e.g. indoor without proper white balance, or applying gamma/tone-mapping before encoding) starves the model of features, so it produces fewer confident depth points → sparser reconstruction → darker HUD.
    - **Fix**:
        - Let the Ray-Ban auto-exposure do its job; **do not** manually lower exposure, apply gamma, or tone-map.
        - Encode in **sRGB color space** (default `Bitmap.compress(JPEG)` output). Do not tag with a non-standard color profile.
        - Do not pre-darken to reduce file size. Use the JPEG quality knob (80–90) for that.
        - Make sure you're not accidentally encoding the alpha channel or a grayscale `Bitmap.Config.ALPHA_8`. Use `Bitmap.Config.ARGB_8888` (or at least RGB_565 if memory is tight) as the source.

3. **Upside-down or sideways reconstruction**
    - **Symptom**: the camera HUD looks fine but the 3D scene's "up" is wrong; walking forward moves the point cloud sideways.
    - **Cause**: JPEG EXIF orientation tag set, or the frame is pre-rotated but the sensor-reported dimensions weren't swapped accordingly.
    - **Fix**: **bake rotation into the pixels** before encoding. Strip EXIF. Our server uses `cv2.imdecode` which ignores EXIF orientation, so any rotation flag you set will be silently discarded — the pixels must already be upright.

4. **Weird hue / washed-out colors**
    - **Cause**: YUV → RGB conversion with wrong color matrix (BT.601 vs BT.709) or incomplete range expansion (limited-range 16-235 treated as full-range 0-255).
    - **Fix**: if you're pulling YUV from the Wearables SDK, convert using an SDK-provided helper if available. Otherwise use `YuvImage.compressToJpeg` (NV21 in, JPEG out) rather than rolling your own matrix.

### How to self-verify image quality from the app

Before blaming the server, periodically save a debug JPEG to the phone's local storage (the exact bytes you're about to send over the socket), then open it in the phone's Gallery. If it looks wrong there, it's an encoding issue. If it looks right there but wrong in the server's HUD, ping the server operator.

---

## 7. Backpressure & reconnection

### On the phone

- **Queue capacity**: **2 frames**. If a new frame arrives when 2 are already queued, **drop the oldest**. This is the same policy the server uses, and together they eliminate latency buildup anywhere in the pipeline.
- **Never** hold all frames to "not lose any" — dropped frames are normal and correct behavior in real-time streaming. Lag is much worse than drop.

### Reconnection

- On `onFailure` / `onClosing`: mark `connected = false`, sleep with **exponential backoff starting at 500 ms, capped at 10 s**, reconnect.
- Do **not** buffer frames during the disconnected interval — those frames are stale by the time you reconnect anyway. Just drop and move forward.
- Do not send application-level pings; OkHttp's `pingInterval(15s)` handles keepalive.

### Detecting a zombie server

If you send frames for > 3 seconds and `onMessage` (text or binary) is never called and the socket doesn't fail, the server might be frozen. Optional: issue a periodic `GET /health` from a separate thread and force-reconnect if it 5xx's or times out.

---

## 8. UI / UX requirements (minimum viable)

Status indicators your app should surface:

| State | Visible to user |
|---|---|
| Searching for glasses | "Looking for Ray-Ban Meta…" |
| Glasses connected, camera not started | "Ready to stream" + Start button |
| Streaming, server reachable | Green "Streaming @ N fps" + last-frame timestamp |
| Streaming, server disconnected | Yellow "Reconnecting…" + retry count |
| Streaming, glasses out of range | Red "Glasses disconnected" |

A tiny FPS counter driven by actual send rate is strongly recommended for debugging.

---

## 9. Local end-to-end test (before real glasses integration)

1. Ask the server operator to start `demo_live.py` on a machine reachable on your WiFi. They will share the URL (`ws://192.168.x.x:8765/ws`) and browser URL (`http://192.168.x.x:8080`).
2. From the phone's browser: load `http://<host>:8765/health` → expect `"ok"`.
3. Temporarily swap the Wearables frame callback for a **phone rear camera** source (CameraX `ImageAnalysis` at 5 fps — see §10 for snippet). This proves the WebSocket and image pipeline in isolation from the SDK.
4. Server operator watches their terminal telemetry:
   ```
   [live] input 5.0 fps | proc 5.0 fps | dropped 0.0/s (total 0) | queue 1/2 ...
   ```
   If `input ≈ send rate` and `dropped = 0`, the phone side is good.
5. Server operator loads `http://<host>:8080` → 3D reconstruction visibly updates as the phone moves.
6. Only after (5) is green: swap the frame source from CameraX to the Meta Wearables SDK session.

## 10. Optional: CameraX fallback for development

Useful during bring-up before the Meta SDK is integrated:

```kotlin
val imageAnalysis = ImageAnalysis.Builder()
    .setTargetResolution(Size(720, 540))
    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
    .build()
    .apply {
        setAnalyzer(ContextCompat.getMainExecutor(context)) { proxy ->
            val now = SystemClock.elapsedRealtimeNanos() / 1000
            if (now - lastSentUs >= 200_000L) {  // ~5 Hz
                val bm = proxy.toBitmap()
                streamer.enqueueFrame(bm, now)
                lastSentUs = now
            }
            proxy.close()
        }
    }
cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, imageAnalysis)
```

---

## 11. Acceptance criteria

The Android integration is **done** when:

- [ ] App authenticates with Meta Wearables SDK and acquires live forward-camera frames.
- [ ] App streams JPEG frames at a steady **5 fps ± 0.5 fps** for at least 10 minutes.
- [ ] Server telemetry (shared by operator): `input 5.0 fps | dropped 0.0/s | queue 0-1/2`.
- [ ] `timestamp_us` deltas across frames are monotonic and match `elapsedRealtimeNanos`-derived deltas within 5 ms.
- [ ] Killing WiFi on the phone for 10 s and restoring it: stream resumes within 15 s, no app crash, no buffered old frames sent.
- [ ] Server-side reconstruction (`http://<host>:8080`) visually tracks actual user motion with < 2 s end-to-end latency (capture → point cloud appearing in browser).
- [ ] App backgrounding / foregrounding: stream pauses/resumes cleanly, no crash.

---

## 12. Debugging quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `input 0.0 fps` on server | App not connected, or on wrong network | Verify `/health` from phone browser; check server IP. |
| `input 5 fps` but `dropped` high | Processing can't keep up | Not your problem — server operator will tune. |
| Server logs `error: cv2.imdecode failed` | Malformed JPEG | Check color space conversion; ensure `Bitmap.compress(JPEG, …)` succeeded. |
| Reconstruction is drifting sideways | Wrong pose, usually from rotation/flip | Ensure frames are upright, no EXIF rotation flag. |
| High latency even though `dropped=0` | Phone queueing socket writes | Confirm queue capacity is 2, oldest-drop is actually evicting. |

---

## 13. Open questions to resolve with the server operator

- Exact server host / URL for the demo window.
- Whether TLS will be terminated in front (`wss://`) or plain (`ws://`).
- Whether to flip to `--fps 8` or stay at 5 — depends on whether the server gets a GPU upgrade before the demo.
- Any authentication / auth token needed on the WebSocket upgrade request (currently none).

---

**Contact**: ping the LingBot-Map server maintainer for any ambiguity in this spec. The server-side code is in [`lingbot-map/lingbot_map/live/ingest.py`](../lingbot-map/lingbot_map/live/ingest.py) — read it if anything here is unclear; the Python implementation is the ground truth.
