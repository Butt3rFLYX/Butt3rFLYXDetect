"""Geometry + single-frame posture recognition on synthetic skeletons."""
import math

import numpy as np
import pytest

from core.geometry import (Keypoints, angle_from_down, count_swings, iou, joint_angle, linear_velocity, soft_margin,
                           tilt_from_vertical)
from core.posture import analyse_pose
from tests.synthetic import back_view, pose, side_view


def keys(f):
    return ({f.posture.key} if f.posture else set()) | {m.key for m in f.modifiers}


# ---------------- geometry ----------------
def test_joint_angle_basic():
    assert joint_angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1])) == pytest.approx(90)
    assert joint_angle(np.array([-1, 0]), np.array([0, 0]), np.array([1, 0])) == pytest.approx(180)
    assert joint_angle(None, np.array([0, 0]), np.array([1, 0])) is None
    assert joint_angle(np.array([0, 0]), np.array([0, 0]), np.array([1, 0])) is None   # degenerate


def test_joint_angle_3d():
    assert joint_angle(np.array([1, 0, 0]), np.zeros(3), np.array([0, 0, 1])) == pytest.approx(90)


def test_tilt_and_elevation():
    assert tilt_from_vertical(np.array([0, 10]), np.array([0, 0])) == pytest.approx(0)      # straight up
    assert tilt_from_vertical(np.array([0, 0]), np.array([10, 0])) == pytest.approx(90)
    assert angle_from_down(np.array([0, 0]), np.array([0, 10])) == pytest.approx(0)          # arm hanging
    assert angle_from_down(np.array([0, 0]), np.array([10, 0])) == pytest.approx(90)         # T-pose
    assert angle_from_down(np.array([0, 0]), np.array([0, -10])) == pytest.approx(180)       # arm up


def test_helpers():
    assert count_swings([0, 1, 0, 1, 0, 1], 0.5) == 4
    assert count_swings([0, 0.1, 0.0, 0.1], 0.5) == 0
    assert linear_velocity([0, 1, 2, 3], [0, 2, 4, 6]) == pytest.approx(2)
    assert linear_velocity([0, 1], [0, 1]) is None
    assert soft_margin(10, 10, 5) == pytest.approx(0.5)
    assert soft_margin(20, 10, 5) == 1.0
    assert soft_margin(None, 10, 5) == 0.0
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0


# ---------------- postures ----------------
def test_standing_front():
    f = analyse_pose(*pose(), t=0)
    assert f.posture.key == "standing"
    assert f.facing.key == "facing_camera"
    assert 0 < f.posture.confidence <= 1
    assert f.torso_len == pytest.approx(100, rel=0.01)


def test_sitting_side_view():
    o = {"left_hip": (0.03, 0), "right_hip": (-0.03, 0), "left_knee": (0.85, 0.05), "right_knee": (0.8, 0.05),
         "left_ankle": (0.85, 0.95), "right_ankle": (0.8, 0.95)}
    f = analyse_pose(*pose(o, base=side_view()), t=0)
    assert f.posture.key == "sitting"


def test_sitting_at_desk_without_ankles():
    o = {"left_knee": (0.22, 0.05), "right_knee": (-0.22, 0.05)}
    f = analyse_pose(*pose(o, hidden=("left_ankle", "right_ankle"), base=side_view()), t=0)
    # thighs horizontal -> sitting
    o2 = {"left_knee": (0.9, 0.05), "right_knee": (0.85, 0.05)}
    f2 = analyse_pose(*pose(o2, hidden=("left_ankle", "right_ankle"), base=side_view()), t=0)
    assert f2.posture.key == "sitting"
    assert f.posture is None or f.posture.key != "lying"


def test_squat_and_deep_squat():
    base = side_view()
    sq = {"left_shoulder": (0.4, -0.95), "right_shoulder": (0.3, -0.95), "nose": (0.65, -1.35),
          "left_knee": (0.6, -0.1), "right_knee": (0.55, -0.1), "left_ankle": (0.3, 0.6), "right_ankle": (0.25, 0.6)}
    f = analyse_pose(*pose(sq, base=base), t=0)
    assert f.posture.key == "squatting"
    deep = dict(sq, left_ankle=(0.3, 0.35), right_ankle=(0.25, 0.35), left_knee=(0.6, -0.2), right_knee=(0.55, -0.2))
    f = analyse_pose(*pose(deep, base=base), t=0)
    assert f.posture.key == "deep_squat"


def test_crouching():
    base = side_view()
    c = {"left_shoulder": (0.75, -0.7), "right_shoulder": (0.7, -0.7), "nose": (1.0, -0.9),
         "left_knee": (0.6, -0.1), "right_knee": (0.55, -0.1), "left_ankle": (0.3, 0.45), "right_ankle": (0.25, 0.45)}
    assert analyse_pose(*pose(c, base=base), t=0).posture.key == "crouching"


def test_kneeling():
    o = {"left_knee": (0.22, 0.85), "right_knee": (-0.22, 0.85), "left_ankle": (0.22, 0.95), "right_ankle": (-0.22, 0.95)}
    assert analyse_pose(*pose(o), t=0).posture.key == "kneeling"


def test_lying_and_bending():
    lying = {"nose": (-1.5, -0.05), "left_eye": (-1.55, -0.1), "right_eye": (-1.55, 0.0), "left_ear": (-1.45, -0.15),
             "right_ear": (-1.45, 0.05), "left_shoulder": (-1.0, -0.25), "right_shoulder": (-1.0, 0.25),
             "left_elbow": (-0.6, -0.3), "right_elbow": (-0.6, 0.3), "left_wrist": (-0.2, -0.3), "right_wrist": (-0.2, 0.3),
             "left_hip": (0.0, -0.2), "right_hip": (0.0, 0.2), "left_knee": (0.9, -0.2), "right_knee": (0.9, 0.2),
             "left_ankle": (1.8, -0.2), "right_ankle": (1.8, 0.2)}
    f = analyse_pose(*pose(lying), t=0)
    assert f.posture.key.startswith("lying")
    bend = {"left_shoulder": (0.92, -0.32), "right_shoulder": (0.85, -0.25), "nose": (1.3, -0.1),
            "left_elbow": (0.95, 0.2), "right_elbow": (0.9, 0.2), "left_wrist": (1.0, 0.6), "right_wrist": (0.95, 0.6)}
    f = analyse_pose(*pose(bend, base=side_view()), t=0)
    assert f.posture.key == "bending_forward"


def test_plank():
    plank = {"nose": (-1.5, -0.2), "left_eye": (-1.55, -0.25), "right_eye": (-1.55, -0.2),
             "left_shoulder": (-1.0, -0.05), "right_shoulder": (-1.0, 0.05), "left_elbow": (-1.0, 0.4),
             "right_elbow": (-1.02, 0.4), "left_wrist": (-1.0, 0.8), "right_wrist": (-1.02, 0.8),
             "left_hip": (0.0, 0.0), "right_hip": (0.0, 0.05), "left_knee": (0.9, 0.1), "right_knee": (0.9, 0.15),
             "left_ankle": (1.8, 0.2), "right_ankle": (1.8, 0.25)}
    assert analyse_pose(*pose(plank), t=0).posture.key == "plank_position"


def test_arm_poses():
    t_pose = {"left_elbow": (0.92, -1.0), "right_elbow": (-0.92, -1.0), "left_wrist": (1.42, -1.0), "right_wrist": (-1.42, -1.0)}
    assert "t_pose" in keys(analyse_pose(*pose(t_pose), t=0))
    d = (math.sin(math.radians(45)), -math.cos(math.radians(45)))
    y_pose = {"left_elbow": (0.42 + 0.5 * d[0], -1 + 0.5 * d[1]), "left_wrist": (0.42 + d[0], -1 + d[1]),
              "right_elbow": (-0.42 - 0.5 * d[0], -1 + 0.5 * d[1]), "right_wrist": (-0.42 - d[0], -1 + d[1])}
    assert "y_pose" in keys(analyse_pose(*pose(y_pose), t=0))
    a = (math.sin(math.radians(35)), math.cos(math.radians(35)))
    a_pose = {"left_elbow": (0.42 + 0.5 * a[0], -1 + 0.5 * a[1]), "left_wrist": (0.42 + a[0], -1 + a[1]),
              "right_elbow": (-0.42 - 0.5 * a[0], -1 + 0.5 * a[1]), "right_wrist": (-0.42 - a[0], -1 + a[1])}
    assert "a_pose" in keys(analyse_pose(*pose(a_pose), t=0))
    up = {"left_elbow": (0.4, -1.5), "left_wrist": (0.38, -2.0), "right_elbow": (-0.4, -1.5), "right_wrist": (-0.38, -2.0)}
    assert "reaching_up" in keys(analyse_pose(*pose(up), t=0))


def test_raised_hands():
    above = {"left_elbow": (0.45, -1.4), "left_wrist": (0.2, -1.8), "right_elbow": (-0.45, -1.4), "right_wrist": (-0.2, -1.8)}
    k = keys(analyse_pose(*pose(above), t=0))
    assert "hands_above_head" in k and "hands_behind_head" not in k
    right = {"right_elbow": (-0.6, -1.15), "right_wrist": (-0.65, -1.3)}
    k = keys(analyse_pose(*pose(right), t=0))
    assert "raising_right_hand" in k and "raising_left_hand" not in k
    left = {"left_elbow": (0.6, -1.15), "left_wrist": (0.65, -1.3)}
    assert "raising_left_hand" in keys(analyse_pose(*pose(left), t=0))


def test_hands_behind_head():
    o = {"left_elbow": (0.75, -1.35), "right_elbow": (-0.75, -1.35), "left_wrist": (0.15, -1.55), "right_wrist": (-0.15, -1.55)}
    assert "hands_behind_head" in keys(analyse_pose(*pose(o), t=0))


def test_arms_crossed_and_hands_on_hips():
    crossed = {"left_wrist": (-0.25, -0.55), "right_wrist": (0.25, -0.55), "left_elbow": (0.45, -0.5), "right_elbow": (-0.45, -0.5)}
    assert "arms_crossed" in keys(analyse_pose(*pose(crossed), t=0))
    hips = {"left_wrist": (0.25, -0.05), "right_wrist": (-0.25, -0.05), "left_elbow": (0.65, -0.45), "right_elbow": (-0.65, -0.45)}
    assert "hands_on_hips" in keys(analyse_pose(*pose(hips), t=0))


def test_one_leg_and_knee_raise():
    o = {"left_knee": (0.3, 0.25), "left_ankle": (0.3, 1.0)}
    k = keys(analyse_pose(*pose(o), t=0))
    assert "standing_one_leg" in k and "knee_raised_left" in k and "knee_raised_right" not in k


def test_leaning_left_right_person_perspective():
    from tests.synthetic import STAND_FRONT
    shift = 0.35
    upper = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist")
    lean_img_right = {n: (STAND_FRONT[n][0] + shift, STAND_FRONT[n][1]) for n in upper}
    # facing the camera: image right = person's LEFT
    assert "leaning_left" in keys(analyse_pose(*pose(lean_img_right), t=0))
    lean_img_left = {n: (STAND_FRONT[n][0] - shift, STAND_FRONT[n][1]) for n in upper}
    assert "leaning_right" in keys(analyse_pose(*pose(lean_img_left), t=0))


def test_facing_back_and_side():
    f = analyse_pose(*pose(base=back_view(), hidden=("nose", "left_eye", "right_eye")), t=0)
    assert f.facing.key == "back_to_camera" and f.facing_side == "back"
    f = analyse_pose(*pose(base=side_view(face_right=True)), t=0)
    assert f.facing.key == "side_on" and f.facing_side == "side_r"
    f = analyse_pose(*pose(base=side_view(face_right=False)), t=0)
    assert f.facing_side == "side_l"


def test_missing_keypoints_do_not_crash():
    kps = Keypoints(np.zeros((17, 2)), np.zeros(17))
    f = analyse_pose(kps, (0, 0, 10, 10), t=0)
    assert f.posture is None and f.modifiers == []
    only_upper = pose(hidden=("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"))
    f = analyse_pose(*only_upper, t=0)
    assert f.posture.key == "upright_upper_body"


def test_low_confidence_is_reflected():
    hi = analyse_pose(*pose(conf=0.95), t=0).posture.confidence
    lo = analyse_pose(*pose(conf=0.5), t=0).posture.confidence
    assert lo < hi
