"""Serves annotated frames as MJPEG over HTTP."""
from __future__ import annotations

import json
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

from .camera import Camera
from .detector import draw

PAGE = b"""<!doctype html><meta charset=utf-8>
<title>EEE_project - live</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{margin:0;background:#111;color:#eee;font:14px/1.5 system-ui,sans-serif;
      display:flex;flex-direction:column;align-items:center;gap:12px;padding:16px}
 h1{font-size:16px;font-weight:600;margin:0;color:#aaa}
 img{max-width:100%;border-radius:8px;box-shadow:0 2px 16px #0008}
 #stats{font-variant-numeric:tabular-nums;color:#8f8;min-height:1.5em;text-align:center}
</style>
<h1>EEE_project &mdash; live vehicle detection</h1>
<img src="/stream.mjpg" alt="live detection">
<div id=stats>connecting&hellip;</div>
<script>
const q = location.search;
document.querySelector('img').src = '/stream.mjpg' + q;
setInterval(async () => {
  const el = document.getElementById('stats');
  try {
    const d = await (await fetch('/stats.json' + q)).json();
    const counts = Object.entries(d.counts).map(([k, v]) => k + ':' + v).join('  ');
    el.textContent = d.fps.toFixed(1) + ' FPS | ' + d.objects + ' objects | ' + counts;
  } catch (e) { el.textContent = 'stream offline'; }
}, 1000);
</script>
"""


class Pipeline(threading.Thread):
    """Detects on the newest frame and publishes an annotated JPEG.

    One pipeline serves every viewer, so N browser tabs cost the same as one.
    """

    def __init__(self, camera: Camera, detector, classes, quality: int = 80):
        super().__init__(daemon=True)
        self.camera = camera
        self.detector = detector
        self.classes = classes
        self.quality = quality
        self.jpeg: bytes | None = None
        self.condition = threading.Condition()
        self.fps = 0.0
        self.objects = 0
        self.counts: dict[str, int] = {}
        self.running = True

    def run(self) -> None:
        last_seq, previous = -1, time.time()
        while self.running:
            seq, frame = self.camera.read()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq

            detections = self.detector(frame)
            draw(frame, detections, self.classes)

            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-6, now - previous))
            previous = now
            self.objects = len(detections)
            self.counts = dict(Counter(self.classes[d[4]] for d in detections).most_common(8))

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                continue
            with self.condition:
                self.jpeg = buf.tobytes()
                self.condition.notify_all()

    def next_jpeg(self, timeout: float = 5.0) -> bytes | None:
        with self.condition:
            self.condition.wait(timeout)
            return self.jpeg

    def stop(self) -> None:
        self.running = False


class Handler(BaseHTTPRequestHandler):
    pipeline: Pipeline = None
    token: str | None = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _authorized(self) -> bool:
        if not self.token:
            return True
        query = parse_qs(urlparse(self.path).query)
        return query.get("t", [None])[0] == self.token

    def do_GET(self):
        if not self._authorized():
            self.send_error(403, "missing or wrong token")
            return

        path = urlparse(self.path).path
        if path == "/stream.mjpg":
            self._stream()
        elif path == "/stats.json":
            self._json({"fps": self.pipeline.fps,
                        "objects": self.pipeline.objects,
                        "counts": self.pipeline.counts})
        else:
            self._bytes(PAGE, "text/html; charset=utf-8")

    def _bytes(self, body: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict):
        self._bytes(json.dumps(payload).encode(), "application/json")

    def _stream(self):
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        try:
            while True:
                jpeg = self.pipeline.next_jpeg()
                if jpeg is None:
                    continue
                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(pipeline: Pipeline, host: str, port: int, token: str | None = None):
    Handler.pipeline = pipeline
    Handler.token = token
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
