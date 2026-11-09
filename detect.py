#!/usr/bin/env python3
"""Run vehicle detection on a camera, video file or webcam.

    python detect.py                        prompt for a source, view in browser
    python detect.py --setup                build an RTSP URL step by step
    python detect.py --source rtsp://...    view in browser at :8000
    python detect.py --source 0 --window    desktop window instead

Server deployment - start with no camera and pick one from the browser:

    python detect.py --source none --connect --token SECRET
"""
from __future__ import annotations

import argparse
import getpass
import os
import socket
import sys
from pathlib import Path
from urllib.parse import quote

from src.camera import Camera
from src.detector import Detector, load_classes
from src.server import Pipeline, serve
from src import viewer

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "EEE_project_yolov9s.onnx"

# Stream path differs per manufacturer; these cover most cameras sold locally.
RTSP_PATHS = {
    "hikvision": "/Streaming/Channels/101",
    "dahua": "/cam/realmonitor?channel=1&subtype=0",
    "tplink": "/stream1",
    "reolink": "/h264Preview_01_main",
    "other": "/stream1",
}


def lan_ip() -> str:
    """Best-effort local address, for telling the user where to point a phone."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def build_rtsp_url() -> str:
    print("\nRTSP setup. Press enter to accept the value in brackets.\n")
    brands = ", ".join(RTSP_PATHS)
    brand = input(f"Camera brand ({brands}) [other]: ").strip().lower() or "other"
    if brand not in RTSP_PATHS:
        print(f"  unknown brand, using the generic path {RTSP_PATHS['other']}")
        brand = "other"

    ip = input("Camera IP address (e.g. 192.168.1.64): ").strip()
    if not ip:
        sys.exit("no IP given")
    port = input("RTSP port [554]: ").strip() or "554"
    user = input("Username [admin]: ").strip() or "admin"
    # getpass keeps the password out of the terminal and out of shell history.
    password = getpass.getpass("Password (not shown): ")

    path = RTSP_PATHS[brand]
    credentials = f"{quote(user, safe='')}:{quote(password, safe='')}@" if user else ""
    url = f"rtsp://{credentials}{ip}:{port}{path}"
    print(f"\nUsing: rtsp://{user}:***@{ip}:{port}{path}\n")
    return url


def resolve_source(args):
    """Command line, then RTSP_URL, then ask. None means start with no camera."""
    if args.setup:
        return build_rtsp_url()
    if args.source:
        if args.source.lower() in ("none", "idle"):
            return None
        return int(args.source) if args.source.isdigit() else args.source

    from_env = os.environ.get("RTSP_URL")
    if from_env:
        print("Using RTSP_URL from the environment.")
        return from_env

    # With the browser picker enabled there is nothing to prompt for.
    if args.connect:
        return None

    print("\nNo source given. Options:")
    print("  - paste an RTSP URL        rtsp://user:pass@192.168.1.64:554/stream1")
    print("  - a webcam index           0")
    print("  - a path to a video file")
    print("  - type 'setup' to build an RTSP URL step by step\n")
    answer = input("Source [0]: ").strip() or "0"
    if answer.lower() == "setup":
        return build_rtsp_url()
    return int(answer) if answer.isdigit() else answer


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Vehicle detection on RTSP, webcam or video.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--source", help="RTSP URL, webcam index, or video path")
    ap.add_argument("--setup", action="store_true", help="build an RTSP URL interactively")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--names", type=Path, default=None, help="override the class list file")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="must match the export; these models are fixed at 640")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--no-stretch", action="store_true",
                    help="use letterbox preprocessing (only for letterbox-trained models)")
    ap.add_argument("--connect", action="store_true",
                    help="let viewers set the source from the browser; needs --token off localhost")
    ap.add_argument("--window", action="store_true", help="desktop window instead of browser")
    ap.add_argument("--save", type=Path, default=None, help="record annotated mp4 (window mode)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default=os.environ.get("VIEW_TOKEN"),
                    help="require ?t=TOKEN to view; use this when exposing a server")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    args = ap.parse_args()

    if not args.model.exists():
        sys.exit(f"model not found: {args.model}")

    # Anyone who can reach the picker can make this process dial out to any
    # address it can route to, so off localhost it has to be behind a token.
    local_only = args.host in ("127.0.0.1", "localhost", "::1")
    if args.connect and not local_only and not args.token:
        sys.exit("--connect on a public interface needs --token SECRET "
                 "(or bind with --host 127.0.0.1)")

    source = resolve_source(args)
    classes = load_classes(args.model, args.names)
    detector = Detector(args.model, args.imgsz, args.conf, args.iou,
                        classes, stretch=not args.no_stretch)

    print(f"model   {args.model.name}  ({len(classes)} classes)")
    print(f"backend {detector.backend}")
    print(f"input   {args.imgsz}px {'letterbox' if args.no_stretch else 'stretch'}")

    if source is not None:
        shown = source if isinstance(source, int) else str(source).split("@")[-1]
        print(f"source  {shown}")

    camera = Camera(source)
    if source is None:
        print("source  none - waiting for one from the browser")
    elif not camera.wait_until_ready():
        camera.stop()
        if not args.connect:
            sys.exit("no frames after 20s. Check the URL, credentials and network.")
        print("warning no frames yet; pick another source from the browser")

    if args.window:
        if source is None:
            camera.stop()
            sys.exit("--window needs a --source; the picker is browser-only")
        try:
            return viewer.run(camera, detector, classes, args.save)
        finally:
            camera.stop()

    pipeline = Pipeline(camera, detector, classes, args.quality)
    pipeline.start()
    server = serve(pipeline, args.host, args.port, args.token, args.connect)

    suffix = f"?t={args.token}" if args.token else ""
    print(f"\n  local   http://localhost:{args.port}{suffix}")
    if args.host == "0.0.0.0":
        print(f"  network http://{lan_ip()}:{args.port}{suffix}")
        if not args.token:
            print("\n  Reachable by anyone who can route to this machine.")
            print("  On a public server pass --token to require a secret in the URL.")
    if args.connect:
        print("\n  Source picker enabled - press 'Source' in the page to change camera.")
    print("\n  ctrl-c to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        pipeline.stop()
        camera.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
