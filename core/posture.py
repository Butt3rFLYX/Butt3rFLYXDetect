"""Single-frame body-pose analysis on YOLO11-pose keypoints.

All lengths are normalised by the torso length T (shoulder-mid to hip-mid),
so the rules are independent of the distance to the camera.
Every label carries a confidence computed from
  (mean keypoint confidence of the joints used) x (how clearly the rule matched).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import (L_ANK, L_EAR, L_EL, L_EYE, L_HIP, L_KNEE, L_SH, L_WR, NOSE, R_ANK, R_EAR,
                       R_EL, R_EYE, R_HIP, R_KNEE, R_SH, R_WR, Keypoints, angle_from_down, dist,
                       joint_angle, mid, soft_margin, tilt_from_vertical)
from .labels import Label

# ---- thresholds (tunable, all relative to torso length T) ----
LYING_TORSO_ANGLE = 60
UPRIGHT_TORSO_ANGLE = 40
STAND_LEG_VERT = 1.35        # hip->ankle vertical distance for standing
SQUAT_LEG_VERT = 0.95
DEEP_SQUAT_LEG_VERT = 0.5
STAND_KNEE_ANGLE = 145
SIT_KNEE_ANGLE = 140
HAND_RAISE_MARGIN = 0.1
STRAIGHT_ELBOW = 145
SIDE_ON_RATIO = 0.30         # shoulder width / T below this -> body seen from the side
FRONT_RATIO = 0.42


@dataclass
class PoseFrame:
    """Everything derived from one person in one frame."""
    t: float
    box: tuple[float, float, float, float]
    kps: Keypoints
    hip: np.ndarray | None = None
    shoulder: np.ndarray | None = None
    torso_len: float | None = None
    torso_angle: float | None = None
    leg_vert: float | None = None
    ankle_y: float | None = None          # mean of both ankles (only if both visible)
    l_ank: np.ndarray | None = None
    r_ank: np.ndarray | None = None
    l_knee: np.ndarray | None = None
    r_knee: np.ndarray | None = None
    l_wr: np.ndarray | None = None
    r_wr: np.ndarray | None = None
    l_sh: np.ndarray | None = None
    r_sh: np.ndarray | None = None
    nose: np.ndarray | None = None
    shoulder_ratio: float | None = None   # signed (left.x - right.x) / T
    hip_ratio: float | None = None
    facing_side: str | None = None        # front | back | side_l | side_r  (side_l: nose -> image left)
    raised_l_wr_x: float | None = None
    raised_r_wr_x: float | None = None
    arm_elev: dict = field(default_factory=dict)
    joint_angles: dict = field(default_factory=dict)
    posture: Label | None = None
    modifiers: list[Label] = field(default_factory=list)
    facing: Label | None = None
    truncated: bool = False               # box touches the image border -> person only partly visible

    @property
    def box_h(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def center(self) -> np.ndarray:
        if self.hip is not None:
            return self.hip
        x1, y1, x2, y2 = self.box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], float)

    def has(self, key: str) -> bool:
        return (self.posture is not None and self.posture.key == key) or any(m.key == key for m in self.modifiers)


def _c(kps: Keypoints, *idx: int) -> float:
    return kps.mean_conf(*idx)


def analyse_pose(kps: Keypoints, box, t: float) -> PoseFrame:
    f = PoseFrame(t=t, box=tuple(float(v) for v in box), kps=kps)
    g = kps.get
    l_sh, r_sh, l_hip, r_hip = g(L_SH), g(R_SH), g(L_HIP), g(R_HIP)
    l_el, r_el, l_wr, r_wr = g(L_EL), g(R_EL), g(L_WR), g(R_WR)
    l_kn, r_kn, l_an, r_an = g(L_KNEE), g(R_KNEE), g(L_ANK), g(R_ANK)
    nose = g(NOSE)
    f.l_sh, f.r_sh, f.l_wr, f.r_wr, f.nose = l_sh, r_sh, l_wr, r_wr, nose
    f.l_knee, f.r_knee, f.l_ank, f.r_ank = l_kn, r_kn, l_an, r_an

    sh, hip = mid(l_sh, r_sh), mid(l_hip, r_hip)
    f.shoulder, f.hip = sh, hip
    if sh is None:
        return f
    if hip is not None and float(np.linalg.norm(sh - hip)) >= 5:
        T = float(np.linalg.norm(sh - hip))
        f.torso_angle = tilt_from_vertical(hip, sh)
    elif l_sh is not None and r_sh is not None:
        # only shoulders visible (e.g. webcam at a desk): estimate T from shoulder width
        T = max(abs(float(l_sh[0] - r_sh[0])) / 0.9, float(np.linalg.norm(l_sh - r_sh)) / 0.9)
        if T < 5:
            return f
        f.torso_angle = 0.0 if hip is None else tilt_from_vertical(hip, sh)
    else:
        return f
    f.torso_len = T
    tang = f.torso_angle if f.torso_angle is not None else 0.0

    if l_sh is not None and r_sh is not None:
        f.shoulder_ratio = float(l_sh[0] - r_sh[0]) / T
    if l_hip is not None and r_hip is not None:
        f.hip_ratio = float(l_hip[0] - r_hip[0]) / T

    # ---- joint angles ----
    ja = {
        "l_elbow": joint_angle(l_sh, l_el, l_wr), "r_elbow": joint_angle(r_sh, r_el, r_wr),
        "l_knee": joint_angle(l_hip, l_kn, l_an), "r_knee": joint_angle(r_hip, r_kn, r_an),
        "l_hip": joint_angle(l_sh, l_hip, l_kn), "r_hip": joint_angle(r_sh, r_hip, r_kn),
        "l_shoulder": joint_angle(l_el, l_sh, l_hip), "r_shoulder": joint_angle(r_el, r_sh, r_hip),
    }
    f.joint_angles = {k: v for k, v in ja.items() if v is not None}
    f.arm_elev = {"l": angle_from_down(l_sh, l_wr), "r": angle_from_down(r_sh, r_wr)}

    if l_an is not None and r_an is not None:
        f.ankle_y = float((l_an[1] + r_an[1]) / 2)
    ank_y = f.ankle_y if f.ankle_y is not None else (
        float(l_an[1]) if l_an is not None else (float(r_an[1]) if r_an is not None else None))
    if hip is not None and ank_y is not None:
        f.leg_vert = (ank_y - float(hip[1])) / T

    knees = [a for a in (ja["l_knee"], ja["r_knee"]) if a is not None]
    knee = float(np.mean(knees)) if knees else None

    _facing(f, kps)
    _base_posture(f, kps, T, tang, knee, l_kn, r_kn, l_an, r_an, l_wr, r_wr, l_sh, r_sh, l_el, r_el)
    _arm_modifiers(f, kps, T, tang, l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hip, r_hip, nose)
    _leg_modifiers(f, kps, T, l_kn, r_kn, l_an, r_an, hip)
    _lean_modifiers(f, kps, T, tang, sh, hip, nose)
    return f


# --------------------------------------------------------------------------
def _facing(f: PoseFrame, kps: Keypoints) -> None:
    if f.shoulder_ratio is None:
        return
    r = f.shoulder_ratio
    face_vis = kps.get(NOSE) is not None and (kps.get(L_EYE) is not None or kps.get(R_EYE) is not None)
    face_conf = _c(kps, NOSE, L_EYE, R_EYE)
    sh_conf = _c(kps, L_SH, R_SH)
    if abs(r) < SIDE_ON_RATIO:
        nose, sh = kps.get(NOSE), f.shoulder
        ref = nose
        if ref is None:   # use the visible ear as a hint where the face points
            le, re_ = kps.get(L_EAR), kps.get(R_EAR)
            ref = le if le is not None and re_ is None else (re_ if re_ is not None and le is None else None)
            if ref is not None:  # ear sits behind the face -> face points the other way
                ref = 2 * sh - ref if sh is not None else None
        if ref is not None and sh is not None:
            f.facing_side = "side_l" if ref[0] < sh[0] else "side_r"
        conf = sh_conf * soft_margin(abs(r), SIDE_ON_RATIO, 0.15, above=False)
        f.facing = Label("side_on", conf)
    elif r > 0 and face_vis:
        f.facing_side = "front"
        f.facing = Label("facing_camera", min(sh_conf, 0.5 + 0.5 * face_conf) * soft_margin(r, SIDE_ON_RATIO, 0.25))
    elif not face_vis:
        f.facing_side = "back"
        f.facing = Label("back_to_camera", sh_conf * (1.0 - 0.5 * face_conf) * soft_margin(abs(r), SIDE_ON_RATIO, 0.25))
    else:
        # face visible but shoulders crossed: ambiguous (YOLO left/right swap) -> trust the face
        f.facing_side = "front"
        f.facing = Label("facing_camera", 0.5 * face_conf)


def _base_posture(f, kps, T, tang, knee, l_kn, r_kn, l_an, r_an, l_wr, r_wr, l_sh, r_sh, l_el, r_el):
    x1, y1, x2, y2 = f.box
    w, h = x2 - x1, y2 - y1
    hip = f.hip
    torso_c = _c(kps, L_SH, R_SH, L_HIP, R_HIP)
    leg_c = _c(kps, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK)
    lv = f.leg_vert

    if hip is None:
        if tang < UPRIGHT_TORSO_ANGLE:
            f.posture = Label("upright_upper_body", _c(kps, L_SH, R_SH) * 0.8)
        return

    # ---------- torso far from vertical: bending / plank / lying ----------
    if tang > LYING_TORSO_ANGLE or (w > 1.3 * h and tang > 45):
        if lv is not None and lv > 1.1 and tang < 130:
            f.posture = Label("bending_forward", min(torso_c, leg_c) * soft_margin(lv, 1.1, 0.4))
            return
        # plank / push-up: straight arms below the shoulders, legs roughly in line with the torso
        arms_down = []
        for sh_, el_, wr_ in ((l_sh, l_el, l_wr), (r_sh, r_el, r_wr)):
            ea = joint_angle(sh_, el_, wr_)
            if sh_ is not None and wr_ is not None and ea is not None:
                arms_down.append(ea > STRAIGHT_ELBOW and (wr_[1] - sh_[1]) / T > 0.6)
        if arms_down and all(arms_down) and 60 <= tang <= 120 and lv is not None and abs(lv) < 0.7:
            f.posture = Label("plank_position", torso_c * 0.8)
            return
        conf = torso_c * soft_margin(tang, LYING_TORSO_ANGLE, 25)
        if w > 1.3 * h:
            conf = max(conf, torso_c * 0.75)
        sub = _lying_subtype(f, kps, T)
        f.posture = Label(sub or "lying", conf if sub is None else conf * 0.7)
        return

    # ---------- kneeling ----------
    kneel = {}
    for side, kn, an in (("l", l_kn, l_an), ("r", r_kn, r_an)):
        if kn is not None and an is not None:
            kneel[side] = ((an[1] - kn[1]) / T < 0.3, (kn[1] - hip[1]) / T)
    if len(kneel) == 2 and lv is not None:
        down = [s for s, (low, hk) in kneel.items() if low and hk > 0.45]
        if len(down) == 2:
            f.posture = Label("kneeling", leg_c * 0.85)
            return
        if len(down) == 1:
            other = "r" if down[0] == "l" else "l"
            oka = f.joint_angles.get(f"{other}_knee")
            if oka is not None and 55 < oka < 125 and not kneel[other][0]:
                f.posture = Label("kneeling_one_knee", leg_c * 0.75)
                return

    # ---------- lunge: feet far apart, front knee bent, back leg extended, torso upright ----------
    if lv is not None and tang < 30 and _is_lunge(f, T):
        f.posture = Label("lunge", leg_c * 0.7)
        return

    # ---------- legs fully visible: use vertical hip->ankle distance ----------
    if lv is not None:
        if lv >= STAND_LEG_VERT:
            f.posture = Label("standing", leg_c * soft_margin(lv, STAND_LEG_VERT, 0.3))
            return
        if lv < DEEP_SQUAT_LEG_VERT:
            key = "crouching" if tang > 35 else "deep_squat"
            f.posture = Label(key, leg_c * soft_margin(lv, DEEP_SQUAT_LEG_VERT, 0.3, above=False))
            return
        if lv < SQUAT_LEG_VERT and (tang > 12 or (knee is not None and knee < 100)):
            key = "crouching" if tang > 35 else "squatting"
            f.posture = Label(key, leg_c * soft_margin(lv, SQUAT_LEG_VERT, 0.3, above=False))
            return
        if knee is None or knee < SIT_KNEE_ANGLE or lv < 1.15:
            f.posture = Label("sitting", leg_c * soft_margin(lv, STAND_LEG_VERT, 0.35, above=False))
            return
        f.posture = Label("standing", leg_c * 0.55)
        return

    # ---------- knees visible, ankles not (e.g. sitting at a desk) ----------
    thighs = [angle_from_down(hip, k) for k in (l_kn, r_kn) if k is not None]
    if thighs:
        thigh = float(np.mean(thighs))       # 0 = thigh hangs down, 90 = horizontal
        if thigh > 55:
            f.posture = Label("sitting", _c(kps, L_HIP, R_HIP, L_KNEE, R_KNEE) * soft_margin(thigh, 55, 25))
            return
        if thigh < 30 and tang < UPRIGHT_TORSO_ANGLE:
            f.posture = Label("standing", _c(kps, L_HIP, R_HIP, L_KNEE, R_KNEE) * 0.6)
            return
    if tang < UPRIGHT_TORSO_ANGLE:
        f.posture = Label("upright_upper_body", torso_c * soft_margin(tang, UPRIGHT_TORSO_ANGLE, 20, above=False))


def _is_lunge(f: PoseFrame, T: float) -> bool:
    if f.l_ank is None or f.r_ank is None:
        return False
    spread = abs(float(f.l_ank[0] - f.r_ank[0])) / T
    ka, kb = f.joint_angles.get("l_knee"), f.joint_angles.get("r_knee")
    if ka is None or kb is None:
        return False
    return spread > 0.8 and min(ka, kb) < 120 and max(ka, kb) > 130


def _lying_subtype(f: PoseFrame, kps: Keypoints, T: float) -> str | None:
    """Very rough 2D heuristic - marked experimental."""
    l_sh, r_sh = kps.get(L_SH), kps.get(R_SH)
    if l_sh is None or r_sh is None:
        return None
    if float(np.linalg.norm(l_sh - r_sh)) / T < 0.35:
        return "lying_side"
    eyes = kps.get(L_EYE) is not None and kps.get(R_EYE) is not None
    if kps.get(NOSE) is not None and eyes:
        return "lying_back"
    if kps.get(NOSE) is None and kps.get(L_EYE) is None and kps.get(R_EYE) is None:
        return "lying_front"
    return None


def _arm_modifiers(f, kps, T, tang, l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hip, r_hip, nose):
    mods = f.modifiers
    if tang >= UPRIGHT_TORSO_ANGLE:
        return
    arm_c = {"l": _c(kps, L_SH, L_EL, L_WR), "r": _c(kps, R_SH, R_EL, R_WR)}
    el_ang = {"l": f.joint_angles.get("l_elbow"), "r": f.joint_angles.get("r_elbow")}
    elev = f.arm_elev
    straight = {s: el_ang[s] is not None and el_ang[s] > STRAIGHT_ELBOW for s in "lr"}
    both_c = min(arm_c["l"], arm_c["r"])

    def both(lo, hi):
        return all(straight[s] and elev[s] is not None and lo <= elev[s] <= hi for s in "lr")

    named = None
    sw = float(np.linalg.norm(l_sh - r_sh)) if l_sh is not None and r_sh is not None else 0.8 * T
    wr_d = dist(l_wr, r_wr)
    if both(65, 115):
        named = "t_pose"
    elif both(150, 180) and wr_d is not None and wr_d < 1.3 * sw:
        named = "reaching_up"
    elif both(118, 172):
        named = "y_pose"
    elif both(18, 55):
        named = "a_pose"
    if named:
        mods.append(Label(named, both_c * 0.95))
    else:
        for s, key in (("l", "left_arm_extended"), ("r", "right_arm_extended")):
            if straight[s] and elev[s] is not None and 60 <= elev[s] <= 120:
                mods.append(Label(key, arm_c[s] * 0.9))

    # raised hands (from the original app) / hands above head
    margin = HAND_RAISE_MARGIN * T
    left_up = l_wr is not None and l_sh is not None and l_wr[1] < l_sh[1] - margin
    right_up = r_wr is not None and r_sh is not None and r_wr[1] < r_sh[1] - margin
    if left_up:
        f.raised_l_wr_x = float(l_wr[0])
    if right_up:
        f.raised_r_wr_x = float(r_wr[0])
    head_y = nose[1] - 0.05 * T if nose is not None else (min(l_sh[1], r_sh[1]) - 0.6 * T
                                                         if l_sh is not None and r_sh is not None else None)
    if named not in ("t_pose", "y_pose", "reaching_up"):
        if left_up and right_up and head_y is not None and l_wr[1] < head_y and r_wr[1] < head_y:
            if not _hands_behind_head(f, kps, T, l_el, r_el, l_wr, r_wr, l_sh, r_sh, nose):
                mods.append(Label("hands_above_head", both_c * soft_margin(head_y - max(l_wr[1], r_wr[1]), 0, 0.3 * T)))
        elif left_up and right_up:
            mods.append(Label("raising_both_hands", both_c * 0.9))
        elif left_up:
            mods.append(Label("raising_left_hand", arm_c["l"] * soft_margin(l_sh[1] - l_wr[1], margin, 0.3 * T)))
        elif right_up:
            mods.append(Label("raising_right_hand", arm_c["r"] * soft_margin(r_sh[1] - r_wr[1], margin, 0.3 * T)))
    elif named == "y_pose":
        pass

    if _hands_behind_head(f, kps, T, l_el, r_el, l_wr, r_wr, l_sh, r_sh, nose):
        mods.append(Label("hands_behind_head", _c(kps, L_EL, R_EL, L_SH, R_SH) * 0.75))

    if f.hip is None or l_wr is None or r_wr is None or l_sh is None or r_sh is None:
        return
    hip = f.hip
    in_torso_band = all(sh_y - 0.15 * T <= w[1] <= hip[1] + 0.15 * T
                        for w, sh_y in ((l_wr, l_sh[1]), (r_wr, r_sh[1])))
    # arms crossed: the wrists are in the opposite order compared to the shoulders
    if in_torso_band and abs(float(l_sh[0] - r_sh[0])) / T > 0.35:
        crossed = (l_wr[0] - r_wr[0]) * (l_sh[0] - r_sh[0]) < 0
        near = True
        if l_el is not None and r_el is not None:
            near = (dist(l_wr, r_el) or 9e9) < 0.6 * T and (dist(r_wr, l_el) or 9e9) < 0.6 * T
        if crossed and near:
            mods.append(Label("arms_crossed", both_c * 0.8))
    # hands on hips
    if l_hip is not None and r_hip is not None and l_el is not None and r_el is not None:
        cx = hip[0]
        on = (dist(l_wr, l_hip) < 0.4 * T and dist(r_wr, r_hip) < 0.4 * T
              and abs(l_el[0] - cx) > abs(l_wr[0] - cx) + 0.05 * T
              and abs(r_el[0] - cx) > abs(r_wr[0] - cx) + 0.05 * T
              and (el_ang["l"] or 180) < 140 and (el_ang["r"] or 180) < 140)
        if on:
            mods.append(Label("hands_on_hips", both_c * 0.8))
    # hands together in front of the body (pose-only estimate)
    if wr_d is not None and wr_d < 0.3 * T and not any(m.key == "arms_crossed" for m in mods):
        if l_wr[1] > l_sh[1] - 0.1 * T:
            mods.append(Label("hands_together_pose", both_c * soft_margin(wr_d, 0.3 * T, 0.2 * T, above=False)))


def _hands_behind_head(f, kps, T, l_el, r_el, l_wr, r_wr, l_sh, r_sh, nose) -> bool:
    if l_el is None or r_el is None or l_sh is None or r_sh is None:
        return False
    if not (l_el[1] < l_sh[1] + 0.05 * T and r_el[1] < r_sh[1] + 0.05 * T):
        return False
    head = nose if nose is not None else mid(kps.get(L_EAR), kps.get(R_EAR))
    if head is None:
        return False
    for el, wr, sh in ((l_el, l_wr, l_sh), (r_el, r_wr, r_sh)):
        if wr is not None:
            if dist(wr, head) > 0.65 * T or wr[1] > sh[1]:
                return False
            ang = joint_angle(sh, el, wr)
            if ang is not None and ang > 100:
                return False
    # elbows must stick out sideways further than the head width
    return abs(float(l_el[0] - r_el[0])) > 1.1 * abs(float(l_sh[0] - r_sh[0]))


def _leg_modifiers(f, kps, T, l_kn, r_kn, l_an, r_an, hip):
    if f.posture is None or f.posture.key not in ("standing", "lunge") or hip is None:
        return
    if l_an is not None and r_an is not None:
        d = float(l_an[1] - r_an[1]) / T
        if abs(d) > 0.35:
            f.modifiers.append(Label("standing_one_leg", _c(kps, L_ANK, R_ANK, L_KNEE, R_KNEE)
                                     * soft_margin(abs(d), 0.35, 0.3)))
    for kn, key, idx in ((l_kn, "knee_raised_left", L_KNEE), (r_kn, "knee_raised_right", R_KNEE)):
        if kn is not None and (kn[1] - hip[1]) / T < 0.35:
            f.modifiers.append(Label(key, _c(kps, idx, L_HIP, R_HIP) * soft_margin(0.35 - (kn[1] - hip[1]) / T, 0, 0.3)))


def _lean_modifiers(f, kps, T, tang, sh, hip, nose):
    if hip is None or f.posture is None or f.posture.key not in ("standing", "sitting", "upright_upper_body", "lunge"):
        return
    c = _c(kps, L_SH, R_SH, L_HIP, R_HIP)
    dx = float(sh[0] - hip[0])
    if f.facing_side in ("front", "back") and 10 <= tang <= 45:
        to_image_right = dx > 0
        # facing the camera, the person's left side is on the image's right
        person_left = to_image_right if f.facing_side == "front" else not to_image_right
        f.modifiers.append(Label("leaning_left" if person_left else "leaning_right", c * soft_margin(tang, 10, 15)))
    elif f.facing_side in ("side_l", "side_r") and 12 <= tang <= 50:
        face_dir = -1 if f.facing_side == "side_l" else 1
        key = "leaning_forward" if dx * face_dir > 0 else "leaning_back"
        f.modifiers.append(Label(key, c * 0.8 * soft_margin(tang, 12, 15)))
