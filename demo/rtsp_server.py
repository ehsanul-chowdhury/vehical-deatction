#!/usr/bin/env python3
"""Publish a video file as a localhost RTSP stream, looping forever.

Stands in for an IP camera when you don't have one to hand. PyAV bundles the
ffmpeg libraries, so there is nothing to install system-wide - no MediaMTX
and no ffmpeg CLI.

Note the direction. A real camera is the server and the detector dials in.
ffmpeg cannot act as an RTSP server on the sending side, so here the detector
listens and this script dials in to it. The RTSP session on the wire is the
same either way, and detect.py needs no code change - only an env var telling
OpenCV to listen. This script retries forever, so whenever the detector
re-listens the stream comes back on its own.

Start the detector FIRST, in terminal 1:

    OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_flags;listen|rtsp_transport;tcp" \
        python detect.py --source rtsp://127.0.0.1:8554/live

Then this, in terminal 2:

    python demo/rtsp_server.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import av
import numpy as np


def probe(src: Path) -> tuple[int, int, float]:
    container = av.open(str(src))
    stream = container.streams.video[0]
    size = (stream.codec_context.width, stream.codec_context.height)
    fps = float(stream.average_rate or 25)
    container.close()
    return size[0], size[1], fps


def detect_banner(src: Path) -> int:
    """Height of the solid dark strip burned across the top, if any.

    The bundled demo clip is a rendered detect.py output, so it carries an
    annotation banner. Cropping it keeps the fake camera's picture clean.
    """
    container = av.open(str(src))
    try:
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="bgr24")
            for y in range(min(120, img.shape[0])):
                if float((img[y].max(axis=1) < 60).mean()) < 0.80:
                    return y + (y % 2)          # keep the height even for h264
            return 0
    finally:
        container.close()
    return 0


def session(src: Path, url: str, width: int, height: int, fps: float,
            loop: bool, crop_top: int = 0) -> int:
    """One RTSP session. Returns the number of frames sent."""
    # av.open does not dial out: the RTSP ANNOUNCE happens on the first mux,
    # so a failure surfaces there, not here.
    out = av.open(url, "w", format="rtsp", options={"rtsp_transport": "tcp"})
    ostream = out.add_stream("libx264", rate=int(round(fps)))
    ostream.width, ostream.height = width, height
    ostream.pix_fmt = "yuv420p"
    # zerolatency keeps the encoder from buffering frames, which otherwise
    # shows up as the detector running several seconds behind.
    ostream.options = {"preset": "veryfast", "tune": "zerolatency"}

    interval = 1.0 / fps
    sent = 0
    announced = False
    start = time.time()
    try:
        while True:
            inp = av.open(str(src))
            for frame in inp.decode(video=0):
                # Pace against the wall clock so it behaves like a live feed
                # rather than dumping the whole file as fast as it decodes.
                delay = (start + sent * interval) - time.time()
                if delay > 0:
                    time.sleep(delay)
                if crop_top:
                    img = frame.to_ndarray(format="bgr24")[crop_top:]
                    frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                frame.pts = None
                for packet in ostream.encode(frame):
                    out.mux(packet)
                    if not announced:
                        print("connected - streaming (ctrl-c to stop)", flush=True)
                        announced = True
                sent += 1
            inp.close()
            if not loop:
                return sent
    finally:
        try:
            for packet in ostream.encode(None):
                out.mux(packet)
        except Exception:
            pass
        try:
            out.close()
        except Exception:
            pass
    return sent


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve a video file over RTSP on localhost.")
    ap.add_argument("--source", default="demo/dhaka/EEE_project_demo.mp4")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--path", default="live")
    ap.add_argument("--once", action="store_true", help="one pass instead of looping")
    ap.add_argument("--retry", type=float, default=2.0,
                    help="seconds between reconnect attempts (0 to give up)")
    ap.add_argument("--crop-top", type=int, default=-1,
                    help="rows to cut off the top; -1 auto-detects a burned-in banner")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"source not found: {src}")

    width, height, fps = probe(src)
    crop = detect_banner(src) if args.crop_top < 0 else args.crop_top
    if crop:
        height -= crop
        height -= height % 2
        print(f"crop    {crop}px banner off the top")
    url = f"rtsp://{args.host}:{args.port}/{args.path}"
    print(f"source  {src.name}  {width}x{height} @ {fps:g}fps -> h264")
    print(f"publish {url}")
    print("the detector must already be listening", flush=True)

    total, failures = 0, 0
    try:
        while True:
            try:
                total += session(src, url, width, height, fps,
                                 loop=not args.once, crop_top=crop)
                failures = 0
                if args.once:
                    break
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # Nothing listening yet, the detector dropped the session and
                # is listening again, or another publisher holds the slot.
                # Keep trying, but say it once rather than once per attempt.
                failures += 1
                if failures == 1:
                    print(f"  waiting for a detector to listen on {url}", flush=True)
                    print("  start detect.py first; only one publisher at a time",
                          flush=True)
                elif failures % 15 == 0:
                    print(f"  still waiting after {failures} attempts "
                          f"({type(exc).__name__})", flush=True)
            if args.retry <= 0:
                break
            time.sleep(args.retry)
    except KeyboardInterrupt:
        pass
    print(f"stopped after {total} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
