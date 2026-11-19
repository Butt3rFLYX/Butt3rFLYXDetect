"""Head orientation and (where possible) eye-gaze analysis.

Two sources:
  * MediaPipe Face Landmarker (478 landmarks incl. iris)  -> head yaw/pitch/roll + gaze
  * YOLO pose keypoints (nose, eyes, ears)                -> coarse head direction fallback

Head direction ("KOPF") and gaze ("BLICK") are reported separately. Gaze is only
reported when the eyes are large enough, open, and both eyes agree - otherwise the
status is "uncertain" and only the head direction is shown.
Angles are from the PERSON's point of view: yaw > 0 = head turned to the person's left,
roll > 0 = head tilted towards the person's left shoulder.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import L_EAR, L_EYE, NOSE, R_EAR, R_EYE, Keypoints, count_swings, linear_velocity, soft_margin
from .labels import Label

# Face-mesh landmark ids
NOSE_TIP, CHIN, FOREHEAD = 1, 152, 10
CHEEK_IMG_L, CHEEK_IMG_R = 234, 454
EYE_A = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}      # eye on the image-left
EYE_B = {"outer": 263, "inner": 362, "top": 386, "bottom": 374}     # eye on the image-right
IRIS_CENTERS = (468, 473)

NOSE_DEPTH_RATIO = 0.5     # nose protrusion / half face width (for the yaw estimate)
DEFAULT_PITCH_RATIO = 0.42 # neutral (nose-eyes)/(chin-eyes) ratio before calibration
YAW_THRESHOLD = 15.0
ROLL_THRESHOLD = 12.0
PITCH_THRESHOLD = 0.07
MIN_EYE_WIDTH_PX = 14.0


@dataclass
class HeadInfo:
    source: str | None = None             # "face_mesh" | "pose"
    yaw: float | None = None
    pitch_ratio: float | None = None
    pitch: float | None = None            # approx degrees, > 0 = looking up
    roll: float | None = None
    direction: Label | None = None
    tilt: Label | None = None
    gaze: Label | None = None
    gaze_status: str = "n/a"             # ok | uncertain | n/a
    gaze_h: float | None = None
    gaze_v: float | None = None
    looking: Label | None = None
    eyes_closed: Label | None = None
    motions: list[Label] = field(default_factory=list)
    face_box: tuple | None = None
    confidence: float = 0.0
    landmarks: np.ndarray | None = None   # (478, 3) when face mesh was used

    def labels(self) -> list[Label]:
        out = [x for x in (self.direction, self.tilt, self.looking, self.eyes_closed) if x is not None]
        return out + list(self.motions)


def _rotate(points: np.ndarray, center: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(-angle_rad), math.sin(-angle_rad)
    p = points - center
    return np.stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c], axis=1) + center


def analyse_face_mesh(lm: np.ndarray, presence: float = 1.0, pitch_neutral: float = DEFAULT_PITCH_RATIO) -> HeadInfo:
    """lm: (N>=468, 3) landmarks in pixels."""
    info = HeadInfo(source="face_mesh", landmarks=lm, confidence=float(presence))
    xy = lm[:, :2].astype(float)
    info.face_box = (float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max()))
    a_out, b_out = xy[EYE_A["outer"]], xy[EYE_B["outer"]]
    roll_img = math.atan2(b_out[1] - a_out[1], b_out[0] - a_out[0])
    info.roll = math.degrees(roll_img)          # image-right eye lower -> tilted to the person's left
    eye_mid = (a_out + b_out) / 2
    r = _rotate(xy, eye_mid, roll_img)          # face-aligned coordinates

    half = (r[CHEEK_IMG_R, 0] - r[CHEEK_IMG_L, 0]) / 2
    if half <= 1:
        return info
    center_x = (r[CHEEK_IMG_R, 0] + r[CHEEK_IMG_L, 0]) / 2
    yr = float(np.clip((r[NOSE_TIP, 0] - center_x) / half, -1.5, 1.5))
    info.yaw = math.degrees(math.atan(yr / NOSE_DEPTH_RATIO))   # nose -> image right = person's left

    eyes_y = (r[EYE_A["outer"], 1] + r[EYE_B["outer"], 1]) / 2
    span = r[CHIN, 1] - eyes_y
    if span > 1:
        info.pitch_ratio = float((r[NOSE_TIP, 1] - eyes_y) / span)
        info.pitch = (pitch_neutral - info.pitch_ratio) * 150.0     # rough degrees, > 0 = up

    conf = float(presence)
    info.direction = _direction_label(info.yaw, info.pitch_ratio, pitch_neutral, conf)
    if abs(info.roll) >= ROLL_THRESHOLD:
        info.tilt = Label("head_tilted_left" if info.roll > 0 else "head_tilted_right",
                          conf * soft_margin(abs(info.roll), ROLL_THRESHOLD, 10), detail=f"~{abs(info.roll):.0f}°")

    _gaze(info, r, conf)
    pitch_ok = info.pitch_ratio is None or abs(info.pitch_ratio - pitch_neutral) < PITCH_THRESHOLD
    if abs(info.yaw) < 12 and pitch_ok:
        gaze_ok = info.gaze_status != "ok" or (info.gaze_h is not None and abs(info.gaze_h) < 0.1)
        if gaze_ok and info.eyes_closed is None:
            info.looking = Label("looking_at_camera", conf * soft_margin(abs(info.yaw), 12, 8, above=False)
                                 * (1.0 if info.gaze_status == "ok" else 0.7))
    return info


def _direction_label(yaw, pitch_ratio, neutral, conf) -> Label:
    cands = []
    if yaw is not None and abs(yaw) >= YAW_THRESHOLD:
        cands.append((soft_margin(abs(yaw), YAW_THRESHOLD, 15), "head_left" if yaw > 0 else "head_right",
                      f"~{abs(yaw):.0f}°"))
    if pitch_ratio is not None and abs(pitch_ratio - neutral) >= PITCH_THRESHOLD:
        d = pitch_ratio - neutral
        cands.append((soft_margin(abs(d), PITCH_THRESHOLD, 0.08), "head_down" if d > 0 else "head_up", ""))
    if not cands:
        m = 1.0 if yaw is None else soft_margin(abs(yaw), YAW_THRESHOLD, 10, above=False)
        return Label("head_forward", conf * m)
    best = max(cands)
    return Label(best[1], conf * best[0], detail=best[2])


def _gaze(info: HeadInfo, r: np.ndarray, conf: float) -> None:
    hs, vs, ears, widths = [], [], [], []
    for eye, iris_id in ((EYE_A, IRIS_CENTERS[0]), (EYE_B, IRIS_CENTERS[1])):
        p_out, p_in = r[eye["outer"]], r[eye["inner"]]
        left_c, right_c = (p_out, p_in) if p_out[0] < p_in[0] else (p_in, p_out)
        w = right_c[0] - left_c[0]
        if w <= 1:
            return
        widths.append(w)
        center = (left_c + right_c) / 2
        if r.shape[0] > max(IRIS_CENTERS):
            # take whichever iris centre lies in this eye (robust to index conventions)
            iris = min((r[i] for i in IRIS_CENTERS), key=lambda p: float(np.linalg.norm(p - center)))
            hs.append(float((iris[0] - left_c[0]) / w - 0.5))
            vs.append(float((iris[1] - center[1]) / w))
        ears.append(float(abs(r[eye["bottom"], 1] - r[eye["top"], 1]) / w))
    if ears and max(ears) < 0.12:
        info.eyes_closed = Label("eyes_closed", conf * soft_margin(max(ears), 0.12, 0.06, above=False))
    if len(hs) < 2:
        info.gaze_status = "n/a"
        return
    info.gaze_h, info.gaze_v = float(np.mean(hs)), float(np.mean(vs))
    reliable = (min(widths) >= MIN_EYE_WIDTH_PX and info.eyes_closed is None
                and info.yaw is not None and abs(info.yaw) < 35 and abs(hs[0] - hs[1]) < 0.15)
    if not reliable:
        info.gaze_status = "uncertain"
        return
    info.gaze_status = "ok"
    h, v = info.gaze_h, info.gaze_v
    gc = conf * soft_margin(min(widths), MIN_EYE_WIDTH_PX, 10) * (1 - min(1.0, abs(hs[0] - hs[1]) / 0.15) * 0.5)
    if abs(h) < 0.07 and abs(v) < 0.06:
        info.gaze = Label("gaze_center", gc)
    elif abs(h) / 0.07 >= abs(v) / 0.06:
        # iris towards the image-right = person looks to their own left
        info.gaze = Label("gaze_left" if h > 0 else "gaze_right", gc * soft_margin(abs(h), 0.07, 0.08))
    else:
        info.gaze = Label("gaze_down" if v > 0 else "gaze_up", gc * soft_margin(abs(v), 0.06, 0.06))


def analyse_head_from_pose(kps: Keypoints) -> HeadInfo | None:
    """Coarse head direction from the 5 YOLO face keypoints."""
    nose, le, re_ = kps.get(NOSE), kps.get(L_EAR), kps.get(R_EAR)
    ley, rey = kps.get(L_EYE), kps.get(R_EYE)
    if all(p is None for p in (nose, le, re_, ley, rey)):
        return None
    info = HeadInfo(source="pose")
    c = kps.mean_conf(NOSE, L_EYE, R_EYE, L_EAR, R_EAR) * 0.75
    info.confidence = c
    if nose is None and ley is None and rey is None:
        info.direction = Label("head_away", c)
        return info
    if ley is not None and rey is not None:
        # eye line: the person's left eye is on the image-right when facing the camera
        a, b = (rey, ley) if rey[0] < ley[0] else (ley, rey)
        info.roll = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        if abs(info.roll) >= ROLL_THRESHOLD + 5:
            info.tilt = Label("head_tilted_left" if info.roll > 0 else "head_tilted_right", c * 0.8,
                              detail=f"~{abs(info.roll):.0f}°")
    if nose is not None and le is not None and re_ is not None:
        half = abs(le[0] - re_[0]) / 2
        if half > 1:
            r = (nose[0] - (le[0] + re_[0]) / 2) / half
            info.yaw = math.degrees(math.atan(float(np.clip(r, -1.5, 1.5)) / 0.6))
    elif nose is not None and (le is None) != (re_ is None):
        # only one ear visible -> head strongly turned. Seeing the LEFT ear = head turned to the person's RIGHT
        info.yaw = -60.0 if le is not None else 60.0
    if info.yaw is not None:
        info.direction = _direction_label(info.yaw, None, DEFAULT_PITCH_RATIO, c)
    elif nose is not None:
        info.direction = Label("head_forward", c * 0.5)
    return info


class HeadMotionTracker:
    """Nodding / shaking / turning + adaptive neutral pitch per person."""

    def __init__(self) -> None:
        self.hist: deque[tuple[float, float | None, float | None]] = deque(maxlen=200)
        self.pitch_samples: deque[float] = deque(maxlen=300)

    @property
    def pitch_neutral(self) -> float:
        n = len(self.pitch_samples)
        if n < 20:
            return DEFAULT_PITCH_RATIO
        med = float(np.median(self.pitch_samples))
        w = min(1.0, n / 100)
        return (1 - w) * DEFAULT_PITCH_RATIO + w * med

    def update(self, info: HeadInfo, t: float) -> list[Label]:
        if info.source == "face_mesh" and info.pitch_ratio is not None and info.yaw is not None and abs(info.yaw) < 15:
            self.pitch_samples.append(info.pitch_ratio)
        self.hist.append((t, info.yaw, info.pitch_ratio))
        while self.hist and t - self.hist[0][0] > 2.0:
            self.hist.popleft()
        out: list[Label] = []
        seg = [(tt, y, p) for tt, y, p in self.hist if t - tt <= 1.5]
        yaws = [y for _, y, _ in seg if y is not None]
        pitches = [p for _, _, p in seg if p is not None]
        if len(yaws) >= 6 and count_swings(yaws, 8.0) >= 2:
            out.append(Label("shaking_head", 0.7))
        elif len(pitches) >= 6 and count_swings(pitches, 0.04) >= 2:
            out.append(Label("nodding", 0.65))
        else:
            ys = [(tt, y) for tt, y, _ in seg if y is not None and t - tt <= 0.4]
            if len(ys) >= 3:
                v = linear_velocity([a for a, _ in ys], [b for _, b in ys]) or 0.0
                if abs(v) > 60:
                    out.append(Label("head_turning", soft_margin(abs(v), 60, 60), detail="(L)" if v > 0 else "(R)"))
        return out
