"""Optional naming of people via face_recognition (feature of the original app).

DISABLED by default. Only active when `face_identity` is switched on in the settings AND
the optional package `face_recognition` is installed AND a known_faces/<Name>/<images>
folder exists. Face encodings are only held in memory, nothing is written to disk.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .geometry import DEFAULT_MIN_KP_CONF, NOSE

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class FaceIdentifier:
    def __init__(self, known_dir: str, tolerance: float = 0.6) -> None:
        import face_recognition as fr  # optional dependency
        self.fr = fr
        self.tolerance = tolerance
        self.encodings, self.labels = [], []
        for label in sorted(os.listdir(known_dir)):
            sub = os.path.join(known_dir, label)
            if not os.path.isdir(sub):
                continue
            for img in sorted(os.listdir(sub)):
                if not img.lower().endswith(IMAGE_EXT):
                    continue
                try:
                    enc = fr.face_encodings(fr.load_image_file(os.path.join(sub, img)))
                except Exception:  # noqa: BLE001
                    continue
                if enc:
                    self.encodings.append(enc[0])
                    self.labels.append(label)

    @property
    def summary(self) -> str:
        return f"{len(self.encodings)} faces / {len(set(self.labels))} people"

    def identify(self, frame_bgr, scale: float = 0.5):
        """[((cx, cy), name), ...] in original pixel coordinates."""
        import cv2
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = self.fr.face_locations(rgb)
        encs = self.fr.face_encodings(rgb, locs)
        out = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            name = "Unknown"
            if self.encodings:
                d = self.fr.face_distance(self.encodings, enc)
                i = int(np.argmin(d))
                if d[i] <= self.tolerance:
                    name = self.labels[i]
            out.append((((left + right) / 2 / scale, (top + bottom) / 2 / scale), name))
        return out


def assign_faces(faces, persons):
    """persons: {pid: (box, xy, conf)} -> yields (pid, name) for the closest nose."""
    for (cx, cy), name in faces:
        best, best_d = None, float("inf")
        for pid, (box, xy, conf) in persons.items():
            x1, y1, x2, y2 = box
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            ref = xy[NOSE] if conf[NOSE] >= DEFAULT_MIN_KP_CONF else np.array([(x1 + x2) / 2, y1])
            d = math.hypot(cx - ref[0], cy - ref[1])
            if d < best_d:
                best, best_d = pid, d
        if best is not None:
            yield best, name
