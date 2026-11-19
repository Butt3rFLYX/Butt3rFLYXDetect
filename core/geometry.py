"""Geometry helpers and COCO keypoint definitions (YOLO-Pose).

"left"/"right" in keypoint names always means the PERSON's own side,
not the side of the image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# COCO-17 keypoints as produced by YOLO11-pose
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# (a, b, body-part group) - group decides the neon colour of the bone
SKELETON: list[tuple[int, int, str]] = [
    (L_SH, R_SH, "torso"), (L_SH, L_HIP, "torso"), (R_SH, R_HIP, "torso"), (L_HIP, R_HIP, "torso"),
    (L_SH, L_EL, "left_arm"), (L_EL, L_WR, "left_arm"),
    (R_SH, R_EL, "right_arm"), (R_EL, R_WR, "right_arm"),
    (L_HIP, L_KNEE, "left_leg"), (L_KNEE, L_ANK, "left_leg"),
    (R_HIP, R_KNEE, "right_leg"), (R_KNEE, R_ANK, "right_leg"),
    (NOSE, L_EYE, "head"), (NOSE, R_EYE, "head"), (L_EYE, L_EAR, "head"), (R_EYE, R_EAR, "head"),
]

DEFAULT_MIN_KP_CONF = 0.4


@dataclass
class Keypoints:
    """17 keypoints of one person with per-point confidence."""
    xy: np.ndarray            # (17, 2) pixel coordinates
    conf: np.ndarray          # (17,)
    min_conf: float = DEFAULT_MIN_KP_CONF

    def get(self, i: int) -> np.ndarray | None:
        if i >= len(self.conf) or self.conf[i] < self.min_conf:
            return None
        return self.xy[i].astype(float)

    def visible(self, *idx: int) -> bool:
        return all(self.get(i) is not None for i in idx)

    def mean_conf(self, *idx: int) -> float:
        vals = [float(self.conf[i]) for i in idx if i < len(self.conf)]
        return float(np.mean(vals)) if vals else 0.0


def mid(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def dist(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def joint_angle(a, b, c) -> float | None:
    """Angle at b in degrees between b->a and b->c (works in 2D and 3D)."""
    if a is None or b is None or c is None:
        return None
    v1 = np.asarray(a, float) - np.asarray(b, float)
    v2 = np.asarray(c, float) - np.asarray(b, float)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def tilt_from_vertical(lower, upper) -> float | None:
    """0 deg = segment lower->upper points straight up, 90 = horizontal, 180 = down.
    Image y points downwards."""
    if lower is None or upper is None:
        return None
    v = np.asarray(upper, float) - np.asarray(lower, float)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return math.degrees(math.acos(float(np.clip(-v[1] / n, -1.0, 1.0))))


def angle_from_down(origin, point) -> float | None:
    """Angle of origin->point measured from straight down (0) over horizontal (90) to up (180).
    Used for arm elevation: arm hanging = 0, T-pose = 90, arm straight up = 180."""
    t = tilt_from_vertical(origin, point)
    return None if t is None else 180.0 - t


def soft_margin(value: float | None, threshold: float, width: float, above: bool = True) -> float:
    """How clearly `value` is on the expected side of `threshold`, mapped to 0..1.
    0.5 = exactly on the threshold, 1.0 = at least `width` past it."""
    if value is None:
        return 0.0
    d = (value - threshold) if above else (threshold - value)
    return float(np.clip(0.5 + 0.5 * d / max(width, 1e-6), 0.0, 1.0))


def count_swings(values, min_amp: float) -> int:
    """Count direction reversals of a 1-D signal, ignoring jitter below `min_amp`.
    (Taken from the original action_recognition_app.py.)"""
    swings, direction, anchor = 0, 0, None
    for x in values:
        if anchor is None:
            anchor = x
            continue
        d = x - anchor
        if abs(d) >= min_amp:
            new_dir = 1 if d > 0 else -1
            if direction != 0 and new_dir != direction:
                swings += 1
            direction, anchor = new_dir, x
        elif direction != 0 and (x - anchor) * direction > 0:
            anchor = x
    return swings


def linear_velocity(ts, values) -> float | None:
    """Least-squares slope of values over time (robust against single-frame jitter)."""
    if len(ts) < 3:
        return None
    t = np.asarray(ts, float)
    if t[-1] - t[0] < 1e-3:
        return None
    v = np.asarray(values, float)
    t = t - t.mean()
    denom = float(np.dot(t, t))
    if denom < 1e-9:
        return None
    return float(np.dot(t, v - v.mean()) / denom)


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0
