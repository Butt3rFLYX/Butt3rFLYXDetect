"""Synthetic test data: COCO skeletons, MediaPipe hands and faces built from simple geometry.

Coordinates are given in torso units (T) relative to the hip centre; image y points down.
For a person FACING the camera, the person's LEFT side is on the image's RIGHT (+x).
"""
from __future__ import annotations

import math

import numpy as np

from core.geometry import KEYPOINT_NAMES, Keypoints
from core.pose_detector import PoseDetection

T = 100.0
ORIGIN = np.array([640.0, 420.0])

STAND_FRONT = {
    "nose": (0, -1.45), "left_eye": (0.07, -1.52), "right_eye": (-0.07, -1.52),
    "left_ear": (0.16, -1.47), "right_ear": (-0.16, -1.47),
    "left_shoulder": (0.42, -1.0), "right_shoulder": (-0.42, -1.0),
    "left_elbow": (0.48, -0.5), "right_elbow": (-0.48, -0.5),
    "left_wrist": (0.5, -0.02), "right_wrist": (-0.5, -0.02),
    "left_hip": (0.22, 0.0), "right_hip": (-0.22, 0.0),
    "left_knee": (0.22, 0.88), "right_knee": (-0.22, 0.88),
    "left_ankle": (0.22, 1.78), "right_ankle": (-0.22, 1.78),
}


def pose(overrides: dict | None = None, hidden: tuple = (), base: dict | None = None, offset=(0.0, 0.0),
         scale: float = T, conf: float = 0.9):
    """Returns (Keypoints, box). `hidden` keypoints get confidence 0."""
    pts = dict(base or STAND_FRONT)
    pts.update(overrides or {})
    xy = np.zeros((17, 2))
    c = np.full(17, conf)
    for i, name in enumerate(KEYPOINT_NAMES):
        x, y = pts[name]
        xy[i] = ORIGIN + np.array(offset) * T + np.array([x, y]) * scale
        if name in hidden:
            c[i] = 0.0
    vis = xy[c > 0.4]
    pad = 0.1 * scale
    box = (vis[:, 0].min() - pad, vis[:, 1].min() - pad, vis[:, 0].max() + pad, vis[:, 1].max() + pad)
    return Keypoints(xy, c), box


def detection(kps: Keypoints, box, raw_id=1) -> PoseDetection:
    return PoseDetection(raw_id, np.array(box, float), kps.xy.copy(), kps.conf.copy(), 0.9)


def side_view(face_right: bool = True) -> dict:
    """Standing seen from the side, facing image-right (or image-left)."""
    s = 1 if face_right else -1
    return {
        "nose": (0.25 * s, -1.42), "left_eye": (0.2 * s, -1.5), "right_eye": (0.18 * s, -1.5),
        "left_ear": (0.0, -1.47), "right_ear": (0.0, -1.47),
        "left_shoulder": (0.05, -1.0), "right_shoulder": (-0.05, -1.0),
        "left_elbow": (0.05, -0.5), "right_elbow": (-0.05, -0.5),
        "left_wrist": (0.08, -0.02), "right_wrist": (-0.05, -0.02),
        "left_hip": (0.03, 0.0), "right_hip": (-0.03, 0.0),
        "left_knee": (0.03, 0.88), "right_knee": (-0.03, 0.88),
        "left_ankle": (0.03, 1.78), "right_ankle": (-0.03, 1.78),
    }


def back_view() -> dict:
    d = dict(STAND_FRONT)
    for a, b in (("left_shoulder", "right_shoulder"), ("left_elbow", "right_elbow"), ("left_wrist", "right_wrist"),
                 ("left_hip", "right_hip"), ("left_knee", "right_knee"), ("left_ankle", "right_ankle"),
                 ("left_ear", "right_ear"), ("left_eye", "right_eye")):
        d[a], d[b] = STAND_FRONT[b], STAND_FRONT[a]
    return d


# --------------------------------------------------------------------------
# Hands: wrist at origin, fingers pointing up (-y), RIGHT hand, palm towards the camera
# --------------------------------------------------------------------------
FINGER_X = {"index": -0.3, "middle": -0.1, "ring": 0.1, "pinky": 0.3}


def hand(extended: dict | None = None, rotate_deg: float = 0.0, scale: float = 50.0, center=(400.0, 300.0),
         mirror: bool = False, spread: dict | None = None, thumb_dir=None) -> np.ndarray:
    ext = {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}
    ext.update(extended or {})
    spread = spread or {}
    lm = np.zeros((21, 3))
    if thumb_dir is not None:          # explicit direction for an extended thumb (unit vector, image coords)
        d = np.array(thumb_dir, float) / np.linalg.norm(thumb_dir)
        base = np.array([-0.35, -0.3, 0.0])
        lm[1] = base
        lm[2] = base + np.array([d[0], d[1], 0]) * 0.3
        lm[3] = base + np.array([d[0], d[1], 0]) * 0.6
        lm[4] = base + np.array([d[0], d[1], 0]) * 0.9
    elif ext["thumb"]:
        lm[1:5] = [(-0.35, -0.3, 0), (-0.6, -0.5, 0), (-0.8, -0.7, 0), (-0.95, -0.85, 0)]
    else:
        lm[1:5] = [(-0.35, -0.3, 0), (-0.45, -0.55, -0.05), (-0.3, -0.75, -0.1), (-0.12, -0.8, -0.1)]
    for name, base in zip(("index", "middle", "ring", "pinky"), (5, 9, 13, 17)):
        x = FINGER_X[name]
        dx = spread.get(name, 0.0)
        lm[base] = (x, -1.0, 0)
        if ext[name]:
            lm[base + 1] = (x + dx * 0.4, -1.45, 0)
            lm[base + 2] = (x + dx * 0.7, -1.75, 0)
            lm[base + 3] = (x + dx, -2.0, 0)
        else:
            lm[base + 1] = (x, -1.35, -0.2)
            lm[base + 2] = (x, -1.1, -0.35)
            lm[base + 3] = (x, -0.9, -0.2)
    if mirror:          # left hand: mirror image of the right hand
        lm[:, 0] *= -1
    a = math.radians(rotate_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    lm[:, :2] = lm[:, :2] @ rot.T
    lm *= scale
    lm[:, 0] += center[0]
    lm[:, 1] += center[1]
    return lm


# --------------------------------------------------------------------------
# Faces (subset of the 478 face-mesh landmarks)
# --------------------------------------------------------------------------
def face(nose_dx: float = 0.0, nose_y: float = 0.246, roll_deg: float = 0.0, iris_dx: float = 0.0,
         iris_dy: float = 0.0, scale: float = 60.0, eye_open: float = 0.12, center=(500.0, 200.0)) -> np.ndarray:
    lm = np.zeros((478, 3))
    pts = {
        234: (-1.0, 0.0), 454: (1.0, 0.0), 1: (nose_dx, nose_y), 152: (0.0, 1.0), 10: (0.0, -1.0),
        33: (-0.6, -0.3), 133: (-0.2, -0.3), 362: (0.2, -0.3), 263: (0.6, -0.3),
        159: (-0.4, -0.3 - eye_open / 2), 145: (-0.4, -0.3 + eye_open / 2),
        386: (0.4, -0.3 - eye_open / 2), 374: (0.4, -0.3 + eye_open / 2),
        468: (-0.4 + iris_dx, -0.3 + iris_dy), 473: (0.4 + iris_dx, -0.3 + iris_dy),
    }
    a = math.radians(roll_deg)
    for i, (x, y) in pts.items():
        xr, yr = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
        lm[i] = (center[0] + xr * scale, center[1] + yr * scale, 0.0)
    return lm
