"""Serves annotated frames as MJPEG over HTTP, with a browser-side source picker."""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

from pathlib import Path

from .camera import Camera
from .detector import draw

PAGE = r"""<!doctype html><meta charset=utf-8>
<title>EEE_project - live</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
/* shadcn/ui design tokens - dark (slate) */
:root{
  --background:222.2 84% 4.9%;  --foreground:210 40% 98%;
  --popover:222.2 84% 4.9%;     --popover-foreground:210 40% 98%;
  --primary:210 40% 98%;        --primary-foreground:222.2 47.4% 11.2%;
  --secondary:217.2 32.6% 17.5%;--secondary-foreground:210 40% 98%;
  --muted:217.2 32.6% 17.5%;    --muted-foreground:215 20.2% 65.1%;
  --accent:217.2 32.6% 17.5%;   --accent-foreground:210 40% 98%;
  --destructive:0 62.8% 50%;
  --border:217.2 32.6% 17.5%;   --input:217.2 32.6% 17.5%;
  --ring:212.7 26.8% 83.9%;     --radius:0.5rem;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:#000;color:hsl(var(--foreground));overflow:hidden;
     font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
     -webkit-font-smoothing:antialiased}

#stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center}
#video{width:100%;height:100%;object-fit:contain;display:none}
#idle{text-align:center;padding:24px;color:hsl(var(--muted-foreground))}
#idle h2{font-size:18px;font-weight:600;margin:0 0 6px;color:hsl(var(--foreground));
         letter-spacing:-.01em}

/* bottom bar */
#bar{position:fixed;left:0;right:0;bottom:0;display:flex;gap:12px;align-items:center;
     padding:16px 18px;background:linear-gradient(transparent,rgba(0,0,0,.85));
     transition:opacity .25s}
#bar.hide{opacity:0;pointer-events:none}
#meta{flex:1;display:flex;gap:14px;align-items:baseline;min-width:0}
#stats{font-variant-numeric:tabular-nums;color:hsl(var(--foreground));
       font-size:13px;font-weight:500;white-space:nowrap}
#srcLabel{color:hsl(var(--muted-foreground));white-space:nowrap;overflow:hidden;
          text-overflow:ellipsis;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}

/* shadcn button */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;
     white-space:nowrap;border-radius:calc(var(--radius) - 2px);font-size:14px;
     font-weight:500;cursor:pointer;transition:background-color .15s,color .15s;
     border:1px solid transparent;height:40px;padding:0 16px;font-family:inherit}
.btn:focus-visible{outline:none;box-shadow:0 0 0 2px hsl(var(--background)),
                   0 0 0 4px hsl(var(--ring))}
.btn svg{width:16px;height:16px;flex:none}
.btn-default{background:hsl(var(--primary));color:hsl(var(--primary-foreground))}
.btn-default:hover{background:hsl(var(--primary)/.9)}
.btn-secondary{background:hsl(var(--secondary));color:hsl(var(--secondary-foreground))}
.btn-secondary:hover{background:hsl(var(--secondary)/.8)}
.btn-outline{border-color:hsl(var(--border));background:transparent;
             color:hsl(var(--foreground))}
.btn-outline:hover{background:hsl(var(--accent));color:hsl(var(--accent-foreground))}
.btn-icon{width:40px;height:40px;padding:0}
.btn-ghost{background:rgba(255,255,255,.08);color:hsl(var(--foreground));
           backdrop-filter:blur(8px);border-color:rgba(255,255,255,.12)}
.btn-ghost:hover{background:hsl(var(--accent))}

/* shadcn dialog */
#overlay{position:fixed;inset:0;background:rgb(0 0 0/.8);display:none;
         align-items:center;justify-content:center;padding:16px;z-index:50}
#overlay.show{display:flex;animation:fade .15s ease-out}
@keyframes fade{from{opacity:0}to{opacity:1}}
#dialog{position:relative;width:100%;max-width:512px;display:flex;flex-direction:column;
        gap:16px;background:hsl(var(--popover));color:hsl(var(--popover-foreground));
        border:1px solid hsl(var(--border));border-radius:var(--radius);padding:24px;
        box-shadow:0 10px 15px -3px rgb(0 0 0/.3),0 4px 6px -4px rgb(0 0 0/.3);
        animation:zoom .15s ease-out}
@keyframes zoom{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:scale(1)}}
#dialog h2{margin:0;font-size:18px;font-weight:600;letter-spacing:-.01em}
#dialog p.desc{margin:6px 0 0;font-size:14px;color:hsl(var(--muted-foreground))}
#close{position:absolute;top:16px;right:16px;border-radius:4px;opacity:.7;
       background:none;border:none;color:inherit;cursor:pointer;padding:4px;
       display:inline-flex}
#close:hover{opacity:1}
#close svg{width:16px;height:16px}

/* shadcn input */
.input{display:flex;height:40px;width:100%;border-radius:calc(var(--radius) - 2px);
       border:1px solid hsl(var(--input));background:transparent;padding:8px 12px;
       font:13px ui-monospace,SFMono-Regular,Menlo,monospace;
       color:hsl(var(--foreground))}
.input::placeholder{color:hsl(var(--muted-foreground))}
.input:focus-visible{outline:none;box-shadow:0 0 0 2px hsl(var(--background)),
                     0 0 0 4px hsl(var(--ring))}
.footer{display:flex;justify-content:flex-end;gap:8px}
#msg{font-size:13px;min-height:1.25em;color:hsl(var(--muted-foreground))}
#msg.err{color:hsl(var(--destructive))}
#msg.ok{color:#4ade80}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}
</style>

<div id=stage>
  <img id=video alt="live detection">
  <div id=idle>
    <h2>No source connected</h2>
    <div>Open the camera picker to connect a stream.</div>
  </div>
</div>

<div id=bar>
  <div id=meta>
    <span id=stats>starting&hellip;</span>
    <span id=srcLabel></span>
  </div>
  <button id=srcBtn class="btn btn-ghost btn-icon" title="Source" aria-label="Source">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/>
    </svg>
  </button>
  <button id=fsBtn class="btn btn-ghost btn-icon" title="Fullscreen (f)" aria-label="Fullscreen">
    <svg id=fsIcon viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/>
      <path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>
    </svg>
  </button>
</div>

<div id=overlay role=dialog aria-modal=true aria-labelledby=dlgTitle>
  <div id=dialog>
    <button id=close aria-label="Close">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M18 6 6 18"/><path d="m6 6 12 12"/>
      </svg>
    </button>
    <div>
      <h2 id=dlgTitle>Connect a camera</h2>
      <p class=desc>An RTSP URL, or a webcam index such as <code>0</code>.</p>
    </div>
    <label class=sr for=url>Stream URL</label>
    <input id=url class=input autocomplete=off spellcheck=false
           placeholder="rtsp://user:pass@192.168.1.64:554/stream1">
    <div id=msg></div>
    <div class=footer>
      <button id=cancel class="btn btn-outline">Cancel</button>
      <button id=go class="btn btn-default">Connect</button>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const overlay = $('overlay'), bar = $('bar'), video = $('video'), msg = $('msg');

const MAXIMIZE = '<path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/>' +
                 '<path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>';
const MINIMIZE = '<path d="M8 3v3a2 2 0 0 1-2 2H3"/><path d="M21 8h-3a2 2 0 0 1-2-2V3"/>' +
                 '<path d="M3 16h3a2 2 0 0 1 2 2v3"/><path d="M16 21v-3a2 2 0 0 1 2-2h3"/>';

function url(path, extra) {
  const p = new URLSearchParams(location.search);
  if (extra) for (const k in extra) p.set(k, extra[k]);
  const s = p.toString();
  return s ? path + '?' + s : path;
}

// Re-request the MJPEG stream on each reconnect so a stalled connection from
// the previous source cannot keep showing a frozen frame.
let live = null;
function setLive(on) {
  if (on === live) return;
  live = on;
  if (on) {
    video.src = url('/stream.mjpg', {r: Date.now()});
    video.style.display = 'block'; $('idle').style.display = 'none';
  } else {
    video.removeAttribute('src');
    video.style.display = 'none'; $('idle').style.display = 'block';
  }
}

// A looping file source reports a brief drop at every EOF, and RTSP cameras
// blip when they reconnect. Only fall back to the idle screen once the source
// has been down for a few polls, so the view does not flicker.
let down = 0;
async function poll() {
  try {
    const d = await (await fetch(url('/api/status'))).json();
    down = d.connected ? 0 : down + 1;
    if (d.connected) setLive(true);
    else if (down >= 4) setLive(false);

    if (d.connected) {
      const c = Object.entries(d.counts).map(([k, v]) => k + ' ' + v).join('   ');
      $('stats').textContent = d.fps.toFixed(1) + ' fps · ' + d.objects +
                               ' objects' + (c ? ' · ' + c : '');
      $('srcLabel').textContent = d.source;
    } else if (d.source) {
      $('stats').textContent = d.error ? 'reconnecting' : 'connecting';
      $('srcLabel').textContent = d.source;
    } else {
      $('stats').textContent = 'no source connected';
      $('srcLabel').textContent = '';
    }
  } catch (e) { $('stats').textContent = 'server offline'; }
}
setInterval(poll, 1000); poll();

function openDialog() { overlay.classList.add('show'); $('url').focus(); }
function closeDialog() { overlay.classList.remove('show'); msg.textContent = ''; }

$('go').onclick = async () => {
  const v = $('url').value.trim();
  if (!v) { msg.textContent = 'Enter a URL'; msg.className = 'err'; return; }
  msg.textContent = 'Connecting…'; msg.className = '';
  try {
    const r = await fetch(url('/api/connect'), {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({url: v})});
    const d = await r.json();
    if (!r.ok) { msg.textContent = d.error || ('HTTP ' + r.status); msg.className = 'err'; return; }
    msg.textContent = 'Accepted'; msg.className = 'ok';
    live = null; down = 0;
    setTimeout(closeDialog, 600);
  } catch (e) { msg.textContent = String(e); msg.className = 'err'; }
};

$('srcBtn').onclick = openDialog;
$('cancel').onclick = closeDialog;
$('close').onclick = closeDialog;
overlay.addEventListener('click', e => { if (e.target === overlay) closeDialog(); });
$('url').addEventListener('keydown', e => { if (e.key === 'Enter') $('go').click(); });

$('fsBtn').onclick = () => document.fullscreenElement
  ? document.exitFullscreen() : document.documentElement.requestFullscreen();
document.addEventListener('fullscreenchange', () => {
  const full = !!document.fullscreenElement;
  $('fsIcon').innerHTML = full ? MINIMIZE : MAXIMIZE;
  $('fsBtn').title = full ? 'Exit fullscreen (f)' : 'Fullscreen (f)';
});

addEventListener('keydown', e => {
  if (e.key === 'Escape') closeDialog();
  if (e.key === 'f' && !overlay.classList.contains('show')) $('fsBtn').click();
});

// Hide the control bar during a demo, bring it back on any input.
let timer;
function wake() {
  bar.classList.remove('hide');
  clearTimeout(timer);
  timer = setTimeout(() => {
    if (!overlay.classList.contains('show')) bar.classList.add('hide');
  }, 3000);
}
addEventListener('mousemove', wake); addEventListener('touchstart', wake); wake();
</script>
""".encode("utf-8")


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def redact(source) -> str:
    """Hide the password in an RTSP URL before it reaches a browser or a log."""
    if source is None:
        return ""
    text = str(source)
    if "://" in text and "@" in text:
        scheme, rest = text.split("://", 1)
        credentials, host = rest.rsplit("@", 1)
        user = credentials.split(":")[0]
        return f"{scheme}://{user}:***@{host}"
    return text


def parse_source(raw: str):
    """Validate a browser-supplied source.

    Deliberately narrow: a deployed server that will open any string the
    network hands it can be used to probe hosts behind the firewall or to
    read local files.
    """
    value = (raw or "").strip()
    if not value:
        raise ValueError("empty source")
    if value.isdigit():
        return int(value)
    low = value.lower()
    if low.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return value

    # Local files are allowed only inside the project directory and only with
    # a video extension, so this cannot be used to probe the filesystem.
    candidate = Path(value).expanduser()
    root = Path.cwd().resolve()
    try:
        resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        resolved.relative_to(root)
    except (ValueError, OSError):
        raise ValueError("file sources must live inside the project directory")
    if resolved.suffix.lower() not in VIDEO_SUFFIXES or not resolved.is_file():
        raise ValueError("use an rtsp:// URL, an http(s):// stream, or a webcam index")
    return str(resolved)


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
    allow_connect: bool = False
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _authorized(self) -> bool:
        if not self.token:
            return True
        query = parse_qs(urlparse(self.path).query)
        return query.get("t", [None])[0] == self.token

    # -- routing -----------------------------------------------------------

    def do_GET(self):
        if not self._authorized():
            self.send_error(403, "missing or wrong token")
            return

        path = urlparse(self.path).path
        if path == "/stream.mjpg":
            self._stream()
        elif path == "/stats.json":
            self._json(self._stats())
        elif path == "/api/status":
            self._json(self._status())
        else:
            self._bytes(PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        if not self._authorized():
            self.send_error(403, "missing or wrong token")
            return
        if urlparse(self.path).path != "/api/connect":
            self.send_error(404, "no such endpoint")
            return
        if not self.allow_connect:
            self._json({"error": "source switching is disabled on this server"}, 403)
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 4096:
            self._json({"error": "bad request body"}, 400)
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            source = parse_source(payload.get("url", ""))
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
            return
        except Exception:
            self._json({"error": "bad request body"}, 400)
            return

        self.pipeline.camera.switch(source)
        self._json({"ok": True, "source": redact(source)})

    # -- payload helpers ---------------------------------------------------

    def _stats(self) -> dict:
        return {"fps": self.pipeline.fps,
                "objects": self.pipeline.objects,
                "counts": self.pipeline.counts}

    def _status(self) -> dict:
        camera = self.pipeline.camera
        payload = self._stats()
        payload.update({"connected": bool(camera.connected),
                        "source": redact(camera.source),
                        "error": camera.error,
                        "can_connect": bool(self.allow_connect)})
        return payload

    def _bytes(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200):
        self._bytes(json.dumps(payload).encode(), "application/json", status)

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


def serve(pipeline: Pipeline, host: str, port: int, token: str | None = None,
          allow_connect: bool = False):
    Handler.pipeline = pipeline
    Handler.token = token
    Handler.allow_connect = allow_connect
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
