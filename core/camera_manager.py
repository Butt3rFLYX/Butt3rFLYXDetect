"""Webcam / video source with automatic reconnect."""
from __future__ import annotations

import sys
import time

import numpy as np


class CameraManager:
    def __init__(self, index: int = 0, source: str = "", size: tuple[int, int] = (1280, 720), fps: int = 30) -> None:
        self.index, self.source, self.size, self.fps = index, source, size, fps
        self.cap = None
        self.error: str | None = None
        self.is_file = bool(source)
        self._last_retry = 0.0
        self.actual_size: tuple[int, int] | None = None

    def open(self) -> bool:
        import cv2
        self.release()
        try:
            if self.source:
                self.cap = cv2.VideoCapture(self.source)
            elif sys.platform == "darwin":
                self.cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
            elif sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.index)
        except Exception as e:  # noqa: BLE001
            self.error = f"Camera error: {e}"
            return False
        if self.cap is None or not self.cap.isOpened():
            self.error = (f"Video file could not be opened: {self.source}" if self.source else
                          f"Camera {self.index} not found or no permission (macOS: System Settings > "
                          f"Privacy & Security > Camera).")
            self.cap = None
            return False
        if not self.source:
            w, h = self.size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.actual_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.error = None
        return True

    @property
    def is_open(self) -> bool:
        return self.cap is not None

    def read(self) -> np.ndarray | None:
        """Returns a frame, or None. Tries to reconnect a lost camera every 2 s."""
        if self.cap is None:
            if time.monotonic() - self._last_retry > 2.0:
                self._last_retry = time.monotonic()
                self.open()
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if self.is_file:           # loop video files
                import cv2
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if ok:
                    return frame
            self.error = "Camera signal lost - reconnecting..."
            self.release()
            return None
        return frame

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:  # noqa: BLE001
                pass
        self.cap = None


def probe_cameras(max_index: int = 5) -> list[int]:
    """Indices that can be opened. Opening cameras takes time - only call on user request."""
    import cv2
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" else cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            found.append(i)
        if cap is not None:
            cap.release()
    return found
