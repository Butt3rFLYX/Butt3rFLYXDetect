"""Hand / finger analysis on MediaPipe Hand Landmarker output (21 landmarks per hand).

Pure functions + small temporal trackers, so everything is unit-testable without a camera.
Coordinates: x, y in pixels (y points down), z = MediaPipe relative depth scaled to pixels
(smaller z = closer to the camera).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import count_swings, joint_angle, linear_velocity, soft_margin
from .labels import Label

WRIST = 0
THUMB = (1, 2, 3, 4)          # cmc, mcp, ip, tip
INDEX = (5, 6, 7, 8)          # mcp, pip, dip, tip
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = {"thumb": THUMB, "index": INDEX, "middle": MIDDLE, "ring": RING, "pinky": PINKY}
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class HandObservation:
    landmarks: np.ndarray                 # (21, 3)
    side: str                             # "left" / "right" from the PERSON's point of view
    score: float                          # detector confidence (handedness score)
    fingers: dict[str, bool] = field(default_factory=dict)
    finger_scores: dict[str, float] = field(default_factory=dict)
    count: int = 0
    gestures: list[Label] = field(default_factory=list)
    palm: Label | None = None
    motions: list[Label] = field(default_factory=list)
    palm_size: float = 0.0

    @property
    def center(self) -> np.ndarray:
        return self.landmarks[[0, 5, 9, 13, 17], :2].mean(axis=0)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xy = self.landmarks[:, :2]
        return float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())

    @property
    def primary(self) -> Label | None:
        return self.gestures[0] if self.gestures else None


def palm_size(lm: np.ndarray) -> float:
    """Wrist -> middle MCP in 3D, plus index MCP -> pinky MCP; robust size reference."""
    a = float(np.linalg.norm(lm[9] - lm[0]))
    b = float(np.linalg.norm(lm[5] - lm[17]))
    return max(a, 1.3 * b, 1e-3)


def finger_extension(lm: np.ndarray) -> dict[str, float]:
    """Extension score 0..1 per finger (>= 0.5 means extended)."""
    out: dict[str, float] = {}
    w = lm[WRIST]
    for name in ("index", "middle", "ring", "pinky"):
        mcp, pip, dip, tip = (lm[i] for i in FINGERS[name])
        straight = joint_angle(mcp, pip, tip) or 0.0
        reach = np.linalg.norm(tip - w) / max(np.linalg.norm(pip - w), 1e-6)
        out[name] = 0.5 * soft_margin(straight, 140, 30) + 0.5 * soft_margin(reach, 1.15, 0.25)
    cmc, mcp, ip, tip = (lm[i] for i in THUMB)
    straight = joint_angle(mcp, ip, tip) or 0.0
    away = np.linalg.norm(tip - lm[17]) / max(np.linalg.norm(ip - lm[17]), 1e-6)
    spread = np.linalg.norm(tip - lm[5]) / palm_size(lm)
    out["thumb"] = (0.35 * soft_margin(straight, 145, 25) + 0.35 * soft_margin(away, 1.08, 0.12)
                    + 0.3 * soft_margin(spread, 0.45, 0.2))
    return out


def palm_orientation(lm: np.ndarray, side: str) -> Label | None:
    """Palm or back of the hand towards the camera, from the winding order of
    wrist -> index MCP -> pinky MCP. Unreliable when the hand is seen edge-on."""
    v1 = lm[5, :2] - lm[0, :2]
    v2 = lm[17, :2] - lm[0, :2]
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return None
    cz = float(v1[0] * v2[1] - v1[1] * v2[0]) / n
    if abs(cz) < 0.2:
        return None
    palm_towards = cz > 0 if side == "right" else cz < 0
    return Label("palm_facing_camera" if palm_towards else "back_of_hand", soft_margin(abs(cz), 0.2, 0.4))


def _direction(vec2: np.ndarray, facing_camera: bool) -> str:
    """Pointing direction from the PERSON's view. Up/down are unambiguous;
    left/right: when the person faces the camera, image-left is the person's right."""
    dx, dy = float(vec2[0]), float(vec2[1])
    if abs(dy) >= abs(dx):
        return "pointing_up" if dy < 0 else "pointing_down"
    image_left = dx < 0
    person_right = image_left if facing_camera else not image_left
    return "pointing_right" if person_right else "pointing_left"


def classify_hand(lm: np.ndarray, side: str, score: float, facing_camera: bool = True) -> HandObservation:
    obs = HandObservation(landmarks=lm, side=side, score=float(score))
    ps = palm_size(lm)
    obs.palm_size = ps
    ext = finger_extension(lm)
    obs.finger_scores = ext
    fingers = {k: v >= 0.5 for k, v in ext.items()}
    obs.fingers = fingers
    obs.count = sum(fingers.values())
    obs.palm = palm_orientation(lm, side)

    def certainty(keys_on, keys_off) -> float:
        vals = [ext[k] for k in keys_on] + [1 - ext[k] for k in keys_off]
        return float(score * np.mean([min(1.0, 0.5 + abs(v - 0.5)) for v in vals])) if vals else score

    f = fingers
    four = [f["index"], f["middle"], f["ring"], f["pinky"]]
    g: list[Label] = []
    thumb_vec = lm[4, :2] - lm[2, :2]
    tl = float(np.linalg.norm(thumb_vec)) or 1e-6
    pinch = float(np.linalg.norm(lm[4] - lm[8])) / ps

    if pinch < 0.3 and f["middle"] and f["ring"] and f["pinky"]:
        g.append(Label("ok_sign", certainty(["middle", "ring", "pinky"], []) * soft_margin(pinch, 0.3, 0.2, above=False)))
    elif not any(four) and f["thumb"] and abs(thumb_vec[1]) / tl > 0.6:
        key = "thumbs_up" if thumb_vec[1] < 0 else "thumbs_down"
        g.append(Label(key, certainty(["thumb"], ["index", "middle", "ring", "pinky"])))
    elif not any(four) and not f["thumb"]:
        g.append(Label("fist", certainty([], list(FINGERS))))
    elif all(four) and f["thumb"]:
        g.append(Label("open_hand", certainty(list(FINGERS), [])))
    elif f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
        a = _angle_between(lm[8, :2] - lm[5, :2], lm[12, :2] - lm[9, :2])
        if a > 10 and not f["thumb"]:
            g.append(Label("peace", certainty(["index", "middle"], ["ring", "pinky", "thumb"])))
    elif f["index"] and f["pinky"] and not f["middle"] and not f["ring"]:
        g.append(Label("rock", certainty(["index", "pinky"], ["middle", "ring"])))
    elif f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        vec = lm[8] - lm[5]
        flat_len = float(np.linalg.norm(vec[:2])) / ps
        depth = float(vec[2]) / ps
        c = certainty(["index"], ["middle", "ring", "pinky"])
        if flat_len < 0.45 and depth < -0.35:
            g.append(Label("pointing_at_camera", c * 0.7))
        else:
            g.append(Label(_direction(vec[:2], facing_camera), c))
    if not g:
        if obs.count == 3:
            g.append(Label("three_fingers", score * 0.9))
        elif obs.count == 4:
            g.append(Label("four_fingers", score * 0.9))
    obs.gestures = g
    return obs


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n < 1e-6:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / n, -1, 1))))


def two_hand_gestures(left: HandObservation | None, right: HandObservation | None) -> list[Label]:
    if left is None or right is None:
        return []
    out: list[Label] = []
    ps = (left.palm_size + right.palm_size) / 2
    c = min(left.score, right.score)
    d_centers = float(np.linalg.norm(left.center - right.center)) / ps
    thumbs = float(np.linalg.norm(left.landmarks[4, :2] - right.landmarks[4, :2])) / ps
    idx = float(np.linalg.norm(left.landmarks[8, :2] - right.landmarks[8, :2])) / ps
    idx_above = left.landmarks[8, 1] < left.landmarks[4, 1] and right.landmarks[8, 1] < right.landmarks[4, 1]
    if thumbs < 0.5 and idx < 0.5 and idx_above and d_centers > 0.6:
        out.append(Label("heart_hands", c * 0.75))
    elif d_centers < 1.2:
        out.append(Label("hands_together", c * soft_margin(d_centers, 1.2, 0.6, above=False)))
    lp, rp = left.primary, right.primary
    if lp and rp and lp.key.startswith("pointing") and rp.key.startswith("pointing"):
        out.append(Label("pointing_both_hands", min(lp.confidence, rp.confidence)))
    return out


# --------------------------------------------------------------------------
# Temporal hand analysis
# --------------------------------------------------------------------------
@dataclass
class _HandSample:
    t: float
    wrist: np.ndarray
    ps: float
    gesture: str | None
    fingers: tuple
    roll: float


class HandMotionTracker:
    """Tracks one hand (one side of one person) over time."""
    WINDOW = 2.0

    def __init__(self) -> None:
        self.hist: deque[_HandSample] = deque(maxlen=120)
        self.last_event: dict[str, float] = {}

    def update(self, obs: HandObservation, t: float) -> list[Label]:
        lm = obs.landmarks
        v = lm[17, :2] - lm[5, :2]
        roll = math.degrees(math.atan2(float(v[1]), float(v[0])))
        g = obs.primary.key if obs.primary else None
        self.hist.append(_HandSample(t, lm[0, :2].copy(), obs.palm_size, g,
                                     tuple(obs.fingers.get(k, False) for k in FINGERS), roll))
        while self.hist and t - self.hist[0].t > self.WINDOW:
            self.hist.popleft()
        return self._motions(t)

    def reset(self) -> None:
        self.hist.clear()

    def _motions(self, now: float) -> list[Label]:
        h = list(self.hist)
        out: list[Label] = []
        if len(h) < 5:
            return out
        ps = float(np.median([s.ps for s in h]))
        recent = [s for s in h if now - s.t <= 1.5]
        xs = [float(s.wrist[0]) for s in recent]
        if len(xs) >= 6 and count_swings(xs, 0.4 * ps) >= 3:
            out.append(Label("waving_hand", 0.8))
        else:
            seg = [s for s in h if now - s.t <= 0.4]
            if len(seg) >= 3:
                vx = linear_velocity([s.t for s in seg], [s.wrist[0] for s in seg]) or 0.0
                vy = linear_velocity([s.t for s in seg], [s.wrist[1] for s in seg]) or 0.0
                sp = math.hypot(vx, vy) / ps
                if sp > 2.5:
                    if abs(vx) >= abs(vy):
                        key = "hand_moving_left" if vx < 0 else "hand_moving_right"
                    else:
                        key = "hand_moving_up" if vy < 0 else "hand_moving_down"
                    out.append(Label(key, soft_margin(sp, 2.5, 2.0)))
        # opening / closing within the last 0.8 s
        seg = [s for s in h if now - s.t <= 0.8]
        gs = [s.gesture for s in seg]
        if "fist" in gs and "open_hand" in gs:
            i_f = max(i for i, x in enumerate(gs) if x == "fist")
            i_o = max(i for i, x in enumerate(gs) if x == "open_hand")
            first_f = min(i for i, x in enumerate(gs) if x == "fist")
            first_o = min(i for i, x in enumerate(gs) if x == "open_hand")
            if gs[-1] == "open_hand" and first_f < i_o and i_f < i_o:
                out.append(Label("hand_opening", 0.8))
            elif gs[-1] == "fist" and first_o < i_f and i_o < i_f:
                out.append(Label("hand_closing", 0.8))
        # rotation of the knuckle line
        if len(seg) >= 3:
            rolls = np.unwrap(np.radians([s.roll for s in seg]))
            if abs(math.degrees(rolls[-1] - rolls[0])) > 70:
                out.append(Label("hand_rotating", 0.6))
        # finger wiggle while the hand itself stays roughly in place
        seg = [s for s in h if now - s.t <= 1.2]
        toggles = sum(1 for a, b in zip(seg, seg[1:]) if a.fingers != b.fingers)
        if len(seg) >= 6 and toggles >= 4:
            moved = float(np.linalg.norm(seg[-1].wrist - seg[0].wrist)) / ps
            if moved < 0.6:
                out.append(Label("fingers_moving", 0.6))
        return out


class TwoHandMotionTracker:
    """Distance between both hands over time -> clapping / moving apart / together."""

    def __init__(self) -> None:
        self.hist: deque[tuple[float, float]] = deque(maxlen=120)

    def update(self, left: HandObservation | None, right: HandObservation | None, t: float) -> list[Label]:
        if left is None or right is None:
            self.hist.clear()
            return []
        ps = (left.palm_size + right.palm_size) / 2
        d = float(np.linalg.norm(left.center - right.center)) / ps
        self.hist.append((t, d))
        while self.hist and t - self.hist[0][0] > 2.0:
            self.hist.popleft()
        out: list[Label] = []
        ds = [x for _, x in self.hist]
        if len(ds) >= 8 and count_swings(ds, 0.6) >= 3 and min(ds) < 1.3:
            out.append(Label("clapping", 0.7))
            return out
        seg = [(tt, x) for tt, x in self.hist if t - tt <= 0.5]
        if len(seg) >= 3:
            v = linear_velocity([a for a, _ in seg], [b for _, b in seg]) or 0.0
            if v > 2.0:
                out.append(Label("hands_moving_apart", soft_margin(v, 2.0, 2.0)))
            elif v < -2.0:
                out.append(Label("hands_moving_together", soft_margin(-v, 2.0, 2.0)))
        return out
