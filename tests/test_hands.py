"""Hand / finger gestures and head pose on synthetic landmarks."""
import numpy as np
import pytest

from core.geometry import Keypoints
from core.hand_gestures import (HandMotionTracker, TwoHandMotionTracker, classify_hand, finger_extension,
                                two_hand_gestures)
from core.head_pose import HeadMotionTracker, analyse_face_mesh, analyse_head_from_pose
from tests.synthetic import face, hand

FOLD4 = {"index": False, "middle": False, "ring": False, "pinky": False}


def g(lm, side="right", facing=True):
    obs = classify_hand(lm, side, 0.95, facing)
    return obs, (obs.primary.key if obs.primary else None)


def test_finger_extension_scores():
    ext = finger_extension(hand())
    assert all(v >= 0.5 for v in ext.values())
    ext = finger_extension(hand({"thumb": False, **FOLD4}))
    assert all(v < 0.5 for v in ext.values())


def test_open_hand_and_fist():
    obs, key = g(hand())
    assert key == "open_hand" and obs.count == 5
    obs, key = g(hand({"thumb": False, **FOLD4}))
    assert key == "fist" and obs.count == 0


def test_thumbs_up_and_down():
    _, key = g(hand(FOLD4, thumb_dir=(-0.2, -1)))
    assert key == "thumbs_up"
    _, key = g(hand(FOLD4, thumb_dir=(-0.2, 1)))
    assert key == "thumbs_down"


def test_peace_rock_ok_counts():
    _, key = g(hand({"thumb": False, "ring": False, "pinky": False}, spread={"index": -0.4, "middle": 0.1}))
    assert key == "peace"
    _, key = g(hand({"middle": False, "ring": False}))
    assert key == "rock"
    lm = hand()
    lm[4] = lm[8] + np.array([3.0, 3.0, 0])      # thumb tip touches index tip
    lm[3] = lm[8] + np.array([20.0, 25.0, 0])
    _, key = g(lm)
    assert key == "ok_sign"
    obs, key = g(hand({"thumb": False, "pinky": False}))
    assert key == "three_fingers" and obs.count == 3
    obs, key = g(hand({"thumb": False}))
    assert key == "four_fingers" and obs.count == 4


def test_pointing_directions_person_perspective():
    only_index = {"thumb": False, "middle": False, "ring": False, "pinky": False}
    assert g(hand(only_index))[1] == "pointing_up"
    assert g(hand(only_index, rotate_deg=180))[1] == "pointing_down"
    # index towards the image-LEFT: for a person facing the camera that is the person's RIGHT
    assert g(hand(only_index, rotate_deg=-90))[1] == "pointing_right"
    assert g(hand(only_index, rotate_deg=90))[1] == "pointing_left"
    # person with the back to the camera: image-left is the person's left
    assert g(hand(only_index, rotate_deg=-90), facing=False)[1] == "pointing_left"


def test_pointing_at_camera():
    lm = hand({"thumb": False, "middle": False, "ring": False, "pinky": False})
    # foreshorten the index finger and move it towards the camera (negative z)
    for i, f in ((6, 0.3), (7, 0.2), (8, 0.15)):
        lm[i, :2] = lm[5, :2] + (lm[i, :2] - lm[5, :2]) * f
        lm[i, 2] = -40.0 * (i - 5)
    assert g(lm)[1] == "pointing_at_camera"


def test_palm_orientation_uses_handedness():
    obs, _ = g(hand(), side="right")
    assert obs.palm.key == "palm_facing_camera"
    obs, _ = g(hand(mirror=True), side="left")
    assert obs.palm.key == "palm_facing_camera"
    obs, _ = g(hand(mirror=True), side="right")        # right hand seen from the back
    assert obs.palm.key == "back_of_hand"


def test_two_hand_gestures():
    left = classify_hand(hand(mirror=True, center=(380, 300)), "left", 0.9)
    right = classify_hand(hand(center=(420, 300)), "right", 0.9)
    assert "hands_together" in {x.key for x in two_hand_gestures(left, right)}
    assert two_hand_gestures(None, right) == []
    # heart: thumb tips touch below, index tips touch above, hands apart
    ll = hand(mirror=True, center=(340, 300))
    rr = hand(center=(460, 300))
    mid_top = (ll[8] + rr[8]) / 2
    mid_bot = (ll[4] + rr[4]) / 2 + np.array([0, 60, 0])
    ll[8], rr[8] = mid_top, mid_top + 1
    ll[4], rr[4] = mid_bot, mid_bot + 1
    keys = {x.key for x in two_hand_gestures(classify_hand(ll, "left", 0.9), classify_hand(rr, "right", 0.9))}
    assert "heart_hands" in keys


def test_hand_motion_waving_and_open_close():
    tr = HandMotionTracker()
    out = []
    for i in range(45):
        t = i / 30
        x = 400 + 40 * np.sin(2 * np.pi * 2.5 * t)
        out = tr.update(classify_hand(hand(center=(x, 300)), "right", 0.9), t)
    assert "waving_hand" in {m.key for m in out}
    tr = HandMotionTracker()
    for i in range(10):
        tr.update(classify_hand(hand({"thumb": False, **FOLD4}), "right", 0.9), i / 30)
    for i in range(10, 16):
        out = tr.update(classify_hand(hand(), "right", 0.9), i / 30)
    assert "hand_opening" in {m.key for m in out}


def test_hand_motion_direction():
    tr = HandMotionTracker()
    out = []
    for i in range(15):
        out = tr.update(classify_hand(hand(center=(400 - 12 * i, 300)), "right", 0.9), i / 30)
    assert "hand_moving_left" in {m.key for m in out}


def test_clapping():
    tr = TwoHandMotionTracker()
    out = []
    for i in range(45):
        t = i / 30
        d = 60 + 55 * np.sin(2 * np.pi * 2.5 * t)
        out = tr.update(classify_hand(hand(mirror=True, center=(400 - d, 300)), "left", 0.9),
                        classify_hand(hand(center=(400 + d, 300)), "right", 0.9), t)
    assert "clapping" in {m.key for m in out}


# ---------------- head ----------------
def test_head_forward_and_looking_at_camera():
    h = analyse_face_mesh(face())
    assert h.direction.key == "head_forward"
    assert abs(h.yaw) < 1 and abs(h.roll) < 1
    assert h.gaze_status == "ok" and h.gaze.key == "gaze_center"
    assert h.looking is not None


def test_head_yaw_sign_person_perspective():
    # nose towards the image-right = the person turned the head to their LEFT
    h = analyse_face_mesh(face(nose_dx=0.5))
    assert h.direction.key == "head_left" and h.yaw == pytest.approx(45, abs=1)
    assert analyse_face_mesh(face(nose_dx=-0.5)).direction.key == "head_right"


def test_head_pitch_and_roll():
    assert analyse_face_mesh(face(nose_y=0.0)).direction.key == "head_up"
    assert analyse_face_mesh(face(nose_y=0.5)).direction.key == "head_down"
    h = analyse_face_mesh(face(roll_deg=20))
    assert h.tilt.key == "head_tilted_left" and h.roll == pytest.approx(20, abs=1)
    assert h.direction.key == "head_forward"         # roll is removed before yaw/pitch


def test_gaze_and_reliability():
    h = analyse_face_mesh(face(iris_dx=0.08))
    assert h.gaze_status == "ok" and h.gaze.key == "gaze_left"
    small = analyse_face_mesh(face(scale=20))         # eyes too small -> only head direction
    assert small.gaze_status == "uncertain" and small.gaze is None
    closed = analyse_face_mesh(face(eye_open=0.02))
    assert closed.eyes_closed is not None and closed.gaze is None


def test_head_from_pose_fallback():
    xy = np.zeros((17, 2))
    conf = np.zeros(17)
    xy[0], xy[3], xy[4] = (110, 100), (130, 100), (70, 100)     # nose, left ear (img right), right ear
    xy[1], xy[2] = (118, 95), (98, 95)
    conf[[0, 1, 2, 3, 4]] = 0.9
    h = analyse_head_from_pose(Keypoints(xy, conf))
    assert h.source == "pose" and h.direction.key == "head_left"
    conf2 = conf.copy()
    conf2[4] = 0
    h = analyse_head_from_pose(Keypoints(xy, conf2))           # only the LEFT ear visible -> turned right
    assert h.direction.key == "head_right"
    conf3 = np.zeros(17)
    conf3[[3, 4]] = 0.9
    assert analyse_head_from_pose(Keypoints(xy, conf3)).direction.key == "head_away"
    assert analyse_head_from_pose(Keypoints(xy, np.zeros(17))) is None


def test_head_shake_and_nod():
    tr = HeadMotionTracker()
    out = []
    for i in range(40):
        t = i / 30
        out = tr.update(analyse_face_mesh(face(nose_dx=0.3 * np.sin(2 * np.pi * 2 * t))), t)
    assert "shaking_head" in {m.key for m in out}
    tr = HeadMotionTracker()
    for i in range(40):
        t = i / 30
        out = tr.update(analyse_face_mesh(face(nose_y=0.246 + 0.12 * np.sin(2 * np.pi * 2 * t))), t)
    assert "nodding" in {m.key for m in out}
