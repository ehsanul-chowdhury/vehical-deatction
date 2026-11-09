"""Frame capture from RTSP, webcam or video file."""
from __future__ import annotations

import os
import threading
import time

import cv2

# TCP survives congested wifi far better than the UDP default, and without a
# connect timeout OpenCV blocks for ~30s on a wrong address, which makes the
# browser "Connect" button feel broken. Only applied if the operator has not
# set their own options.
RTSP_DEFAULTS = "rtsp_transport;tcp|stimeout;5000000"


class Camera(threading.Thread):
    """Reads continuously and keeps only the newest frame.

    RTSP delivers frames faster than CPU inference consumes them. Without this
    the view falls further behind real time with every frame.

    The source can be changed at runtime with switch(), so a deployed server
    can be pointed at a different camera without a restart.
    """

    def __init__(self, source=None, reconnect: bool = True):
        super().__init__(daemon=True)
        self.source = source
        self.reconnect = reconnect
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.running = True
        self.connected = False
        self.error = ""
        self.cap = None
        self._pending = source
        self._swap = threading.Event()
        # macOS raises the camera permission prompt on the first VideoCapture
        # call, and AVFoundation can only do that from the thread running the
        # main loop. Webcams therefore have to be opened on the caller's
        # thread. Network and file sources stay lazy so a slow or not-yet-
        # listening RTSP endpoint cannot block startup.
        if isinstance(source, int):
            self.cap = self._open()
        self.start()

    # -- source management -------------------------------------------------

    def switch(self, source) -> None:
        """Point the capture at a new source. Takes effect within a second."""
        self._pending = source
        self._swap.set()

    def _open(self):
        if isinstance(self.source, str) and self.source.startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", RTSP_DEFAULTS)
        cap = cv2.VideoCapture(self.source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _pause(self, seconds: float) -> None:
        """Sleep, but wake immediately if someone asks for a new source."""
        self._swap.wait(seconds)

    # -- capture loop ------------------------------------------------------

    def run(self) -> None:
        while self.running:
            if self._swap.is_set():
                self._swap.clear()
                self._close()
                with self.lock:
                    self.source = self._pending
                    self.frame = None
                self.connected = False
                self.error = ""

            if self.source is None:
                self._pause(0.1)
                continue

            if self.cap is None or not self.cap.isOpened():
                self._close()
                self.cap = self._open()
                if not self.cap.isOpened():
                    self.connected = False
                    self.error = "could not open source"
                    if not self.reconnect:
                        break
                    self._pause(2.0)
                    continue

            ok, frame = self.cap.read()
            if not ok:
                self.connected = False
                self.error = "stream dropped"
                self._close()
                if not self.reconnect:
                    self.running = False
                    break
                self._pause(1.0)
                continue

            self.connected = True
            self.error = ""
            with self.lock:
                self.frame = frame
                self.seq += 1

    # -- readers -----------------------------------------------------------

    def read(self):
        """Return (sequence number, frame copy). Frame is None before first read."""
        with self.lock:
            return self.seq, (None if self.frame is None else self.frame.copy())

    def wait_until_ready(self, timeout: float = 20.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.read()[1] is not None:
                return True
            time.sleep(0.1)
        return False

    def stop(self) -> None:
        self.running = False
        self._swap.set()
        self._close()
