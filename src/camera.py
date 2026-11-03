"""Frame capture from RTSP, webcam or video file."""
from __future__ import annotations

import threading
import time

import cv2


class Camera(threading.Thread):
    """Reads continuously and keeps only the newest frame.

    RTSP delivers frames faster than CPU inference consumes them. Without this
    the view falls further behind real time with every frame.
    """

    def __init__(self, source, reconnect: bool = True):
        super().__init__(daemon=True)
        self.source = source
        self.reconnect = reconnect
        self.cap = self._open()
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.running = True
        self.start()

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def run(self) -> None:
        while self.running:
            if not self.cap.isOpened():
                if not self.reconnect:
                    break
                time.sleep(2.0)
                self.cap = self._open()
                continue

            ok, frame = self.cap.read()
            if not ok:
                if not self.reconnect:
                    self.running = False
                    break
                time.sleep(1.0)
                self.cap.release()
                self.cap = self._open()
                continue

            with self.lock:
                self.frame = frame
                self.seq += 1

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
        try:
            self.cap.release()
        except Exception:
            pass
