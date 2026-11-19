"""Second batch: covers the remaining postures, movements, transitions, exercises, head and hand labels."""
import math


from core.action_engine import ActionEngine
from core.hand_gestures import HandMotionTracker, TwoHandMotionTracker, classify_hand, two_hand_gestures
from core.head_pose import HeadMotionTracker, analyse_face_mesh
from core.mediapipe_models import RawHand
from core.motion_analyzer import MotionAnalyzer
from core.posture import analyse_pose
from tests.synthetic import STAND_FRONT, detection, face, hand, pose, side_view

FPS = 30
SQUAT = {"left_shoulder": (0.4, -0.95), "right_shoulder": (0.3, -0.95), "nose": (0.65, -1.35),
         "left_knee": (0.6, -0.1), "right_knee": (0.55, -0.1), "left_ankle": (0.3, 0.6), "right_ankle": (0.25, 0.6)}
BEND = {"left_shoulder": (0.92, -0.32), "right_shoulder": (0.85, -0.25), "nose": (1.3, -0.1),
        "left_elbow": (0.95, 0.2), "right_elbow": (0.9, 0.2), "left_wrist": (1.0, 0.6), "right_wrist": (0.95, 0.6)}
LYING = {"nose": (-1.5, 1.6), "left_eye": (-1.55, 1.55), "right_eye": (-1.55, 1.65), "left_ear": (-1.45, 1.5),
         "right_ear": (-1.45, 1.7), "left_shoulder": (-1.0, 1.45), "right_shoulder": (-1.0, 1.75),
         "left_elbow": (-0.6, 1.45), "right_elbow": (-0.6, 1.75), "left_wrist": (-0.2, 1.45), "right_wrist": (-0.2, 1.75),
         "left_hip": (0.0, 1.45), "right_hip": (0.0, 1.75), "left_knee": (0.9, 1.45), "right_knee": (0.9, 1.75),
         "left_ankle": (1.8, 1.45), "right_ankle": (1.8, 1.75)}


def keys(f):
    return ({f.posture.key} if f.posture else set()) | {m.key for m in f.modifiers}


def run(frames):
    ma = MotionAnalyzer()
    return [ma.update(analyse_pose(k, b, t)) for t, k, b in frames]


def seq(*parts):
    """parts: (seconds, overrides, base) -> frames with continuous time."""
    out, t = [], 0.0
    for secs, o, base, *rest in parts:
        off = rest[0] if rest else (0, 0)
        for _ in range(int(secs * FPS)):
            out.append((t, *pose(o, base=base, offset=off)))
            t += 1 / FPS
    return out


def all_instant(res):
    return [lab.key for r in res for lab in r.instant_events]


def all_ex(res):
    return {e.key for r in res for e in r.exercises}


# ---------------- postures ----------------
def test_lunge_and_one_knee():
    lunge = {"left_hip": (0.1, 0.0), "right_hip": (-0.1, 0.0), "left_knee": (0.75, 0.15), "left_ankle": (0.75, 1.05),
             "right_knee": (-0.6, 0.75), "right_ankle": (-1.1, 1.05)}
    f = analyse_pose(*pose(lunge, base=side_view()), t=0)
    assert f.posture.key == "lunge"
    one = {"left_knee": (0.85, 0.05), "left_ankle": (0.85, 0.95), "right_knee": (-0.2, 0.85), "right_ankle": (-0.7, 0.95)}
    assert analyse_pose(*pose(one, base=side_view()), t=0).posture.key == "kneeling_one_knee"


def test_lying_subtypes():
    side = dict(LYING, left_shoulder=(-1.0, 1.58), right_shoulder=(-1.0, 1.62))
    assert analyse_pose(*pose(side), t=0).posture.key == "lying_side"
    wide = dict(LYING, left_shoulder=(-1.0, 1.35), right_shoulder=(-1.0, 1.85))   # shoulders seen from above
    assert analyse_pose(*pose(wide), t=0).posture.key == "lying_back"
    assert analyse_pose(*pose(wide, hidden=("nose", "left_eye", "right_eye")), t=0).posture.key == "lying_front"


def test_leaning_forward_back_side_view():
    base = side_view(face_right=True)
    upper = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist")
    fwd = {n: (base[n][0] + 0.4, base[n][1]) for n in upper}
    assert "leaning_forward" in keys(analyse_pose(*pose(fwd, base=base), t=0))
    back = {n: (base[n][0] - 0.4, base[n][1]) for n in upper}
    back["nose"] = (back["nose"][0] + 0.0, back["nose"][1])
    assert "leaning_back" in keys(analyse_pose(*pose(back, base=base), t=0))


def test_both_hands_and_single_arm_extended():
    both = {"left_elbow": (0.6, -1.15), "left_wrist": (0.62, -1.3), "right_elbow": (-0.6, -1.15), "right_wrist": (-0.62, -1.3)}
    assert "raising_both_hands" in keys(analyse_pose(*pose(both), t=0))
    arm = {"left_elbow": (0.92, -1.0), "left_wrist": (1.42, -1.0)}
    k = keys(analyse_pose(*pose(arm), t=0))
    assert "left_arm_extended" in k and "t_pose" not in k
    arm_r = {"right_elbow": (-0.92, -1.0), "right_wrist": (-1.42, -1.0)}
    assert "right_arm_extended" in keys(analyse_pose(*pose(arm_r), t=0))


def test_hands_together_pose():
    o = {"left_wrist": (0.08, -0.5), "right_wrist": (-0.08, -0.5), "left_elbow": (0.4, -0.45), "right_elbow": (-0.4, -0.45)}
    assert "hands_together_pose" in keys(analyse_pose(*pose(o), t=0))


def test_torso_twist_against_baseline():
    ma = MotionAnalyzer()
    for i in range(40):
        ma.update(analyse_pose(*pose(), t=i / FPS))
    tw = {"left_shoulder": (0.18, -1.0), "right_shoulder": (-0.18, -1.0)}
    r = ma.update(analyse_pose(*pose(tw), t=41 / FPS))
    for i in range(42, 52):
        r = ma.update(analyse_pose(*pose(tw), t=i / FPS))
    assert "torso_twisted" in {m.key for m in r.modifiers}


# ---------------- movement ----------------
def test_idle_while_sitting_and_slow_walk_left():
    sit = {"left_hip": (0.03, 0), "right_hip": (-0.03, 0), "left_knee": (0.85, 0.05), "right_knee": (0.8, 0.05),
           "left_ankle": (0.85, 0.95), "right_ankle": (0.8, 0.95)}
    assert run(seq((1.5, sit, side_view())))[-1].movement.key == "idle"
    frames = [(i / FPS, *pose(base=side_view(False), offset=(-1.0 * i / FPS, 0))) for i in range(40)]
    r = run(frames)[-1]
    assert r.movement.key == "walking_slow" and r.direction.key == "moving_screen_left"


def test_moving_away():
    frames = []
    for i in range(45):
        t = i / FPS
        sw = 0.15 * math.sin(2 * math.pi * 1.8 * t)
        frames.append((t, *pose({"left_ankle": (0.22, 1.78 + sw), "right_ankle": (-0.22, 1.78 - sw)},
                                scale=150 * math.exp(-0.45 * t))))
    assert run(frames)[-1].movement.key == "moving_away"


def test_walking_and_running_in_place():
    def frames(freq, amp):
        out = []
        for i in range(60):
            t = i / FPS
            s = amp * math.sin(2 * math.pi * freq * t)
            out.append((t, *pose({"left_ankle": (0.22, 1.78 - max(s, 0)), "right_ankle": (-0.22, 1.78 - max(-s, 0)),
                                  "left_knee": (0.22, 0.88 - max(s, 0)), "right_knee": (-0.22, 0.88 - max(-s, 0))})))
        return out
    assert run(frames(1.0, 0.3))[-1].movement.key == "walking_in_place"
    assert run(frames(2.0, 0.6))[-1].movement.key == "running_in_place"


def test_repeated_jumps_and_turning_label():
    parts = [(1.0, {}, STAND_FRONT)]
    for _ in range(3):
        parts += [(0.15, {}, STAND_FRONT, (0, -0.5)), (0.6, {}, STAND_FRONT)]
    res = run(seq(*parts))
    assert any(r.movement is not None and r.movement.key == "jumping_repeatedly" for r in res)
    res = run(seq((0.5, {}, STAND_FRONT), (0.5, {}, side_view(False))))
    assert any(r.movement is not None and r.movement.key == "turning" for r in res)


def test_stumble():
    frames, t = [], 0.0
    for _ in range(30):
        frames.append((t, *pose()))
        t += 1 / FPS
    upper = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist")
    for i, (lean, dx) in enumerate([(0.2, 0.1), (0.45, 0.25), (0.6, 0.4), (0.45, 0.5), (0.2, 0.55), (0.0, 0.55)]):
        o = {n: (STAND_FRONT[n][0] + lean, STAND_FRONT[n][1]) for n in upper}
        for _ in range(2):
            frames.append((t, *pose(o, offset=(dx, 0))))
            t += 1 / FPS
    for _ in range(8):
        frames.append((t, *pose(offset=(0.55, 0))))
        t += 1 / FPS
    assert "stumbling" in all_instant(run(frames))


# ---------------- transitions ----------------
def test_squat_bend_and_lie_transitions():
    k = all_instant(run(seq((1, {}, STAND_FRONT), (1, SQUAT, side_view()), (1, {}, STAND_FRONT))))
    assert "squatting_down" in k and "rising_from_squat" in k
    k = all_instant(run(seq((1, {}, STAND_FRONT), (1, BEND, side_view()), (1, {}, STAND_FRONT))))
    assert "bending_down" in k and "straightening_up" in k
    sit = {"left_hip": (0.03, 0), "right_hip": (-0.03, 0), "left_knee": (0.85, 0.05), "right_knee": (0.8, 0.05),
           "left_ankle": (0.85, 0.95), "right_ankle": (0.8, 0.95)}
    # slow lying down from sitting (no fall), then getting up
    k = all_instant(run(seq((1, sit, side_view()), (1, LYING, None, (0, -1.4)), (1, sit, side_view()))))
    assert "lying_down" in k and "getting_up" in k and "falling" not in k


# ---------------- exercises ----------------
def test_left_wave_arm_repeats_leg_raises_balance_arm_circles():
    frames = [(i / FPS, *pose({"left_elbow": (0.6, -1.1), "left_wrist": (0.6 + 0.3 * math.sin(2 * math.pi * 2.5 * i / FPS), -1.5)}))
              for i in range(45)]
    assert "waving_left_hand" in all_ex(run(frames))
    frames = [(i / FPS, *pose({"right_elbow": (-0.6, -0.5 - 0.3 * math.sin(2 * math.pi * 2 * i / FPS)),
                               "right_wrist": (-0.6, -0.5 - 0.9 * math.sin(2 * math.pi * 2 * i / FPS))})) for i in range(75)]
    assert "repetitive_arm_motion" in all_ex(run(frames))
    raise_ = {"left_knee": (0.3, 0.25), "left_ankle": (0.3, 1.0)}
    parts = []
    for _ in range(3):
        parts += [(0.4, raise_, STAND_FRONT), (0.4, {}, STAND_FRONT)]
    res = run(seq((0.6, {}, STAND_FRONT), *parts, (0.6, {}, STAND_FRONT)))
    assert "leg_raises" in all_ex(res)
    assert "balance_one_leg" in all_ex(run(seq((3.6, raise_, STAND_FRONT))))
    frames = []
    for i in range(80):
        a = 2 * math.pi * 1.2 * i / FPS
        frames.append((i / FPS, *pose({"left_elbow": (0.42 + 0.45 * math.sin(a), -1 + 0.45 * math.cos(a)),
                                       "left_wrist": (0.42 + 0.9 * math.sin(a), -1 + 0.9 * math.cos(a))})))
    assert "arm_circles" in all_ex(run(frames))


def test_repetitive_leg_motion():
    frames = [(i / FPS, *pose({"left_knee": (0.3, 0.88 - 0.3 * max(0, math.sin(2 * math.pi * 1.5 * i / FPS))),
                               "left_ankle": (0.3, 1.78 - 0.5 * max(0, math.sin(2 * math.pi * 1.5 * i / FPS)))}))
              for i in range(80)]
    assert "repetitive_leg_motion" in all_ex(run(frames))


# ---------------- head ----------------
def test_head_right_tilt_gaze_directions_eyes_closed():
    assert analyse_face_mesh(face(roll_deg=-20)).tilt.key == "head_tilted_right"
    assert analyse_face_mesh(face()).looking.key == "looking_at_camera"
    assert analyse_face_mesh(face(iris_dx=-0.08)).gaze.key == "gaze_right"
    assert analyse_face_mesh(face(iris_dy=-0.06)).gaze.key == "gaze_up"
    assert analyse_face_mesh(face(iris_dy=0.06)).gaze.key == "gaze_down"
    assert analyse_face_mesh(face(eye_open=0.02)).eyes_closed.key == "eyes_closed"


def test_head_turning_motion():
    tr = HeadMotionTracker()
    out = []
    for i in range(10):
        out = tr.update(analyse_face_mesh(face(nose_dx=-0.6 + 0.12 * i)), i / FPS)
    assert "head_turning" in {m.key for m in out}


# ---------------- hands ----------------
FOLD = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}


def test_hand_closing_rotation_moves_and_fingers():
    tr = HandMotionTracker()
    for i in range(10):
        tr.update(classify_hand(hand(), "right", 0.9), i / FPS)
    out = []
    for i in range(10, 16):
        out = tr.update(classify_hand(hand(FOLD), "right", 0.9), i / FPS)
    assert "hand_closing" in {m.key for m in out}
    for direction, vec in (("hand_moving_right", (12, 0)), ("hand_moving_up", (0, -12)), ("hand_moving_down", (0, 12))):
        tr = HandMotionTracker()
        for i in range(15):
            out = tr.update(classify_hand(hand(center=(400 + vec[0] * i, 300 + vec[1] * i)), "right", 0.9), i / FPS)
        assert direction in {m.key for m in out}
    tr = HandMotionTracker()
    for i in range(20):
        out = tr.update(classify_hand(hand(rotate_deg=6 * i), "right", 0.9), i / FPS)
    assert "hand_rotating" in {m.key for m in out}
    tr = HandMotionTracker()
    for i in range(30):
        out = tr.update(classify_hand(hand({"index": i % 4 < 2, "middle": i % 4 >= 2}), "right", 0.9), i / FPS)
    assert "fingers_moving" in {m.key for m in out}


def test_two_hands_pointing_and_moving_apart_together():
    point = {"thumb": False, "middle": False, "ring": False, "pinky": False}
    l = classify_hand(hand(point, mirror=True, center=(200, 300)), "left", 0.9)
    r = classify_hand(hand(point, center=(600, 300)), "right", 0.9)
    assert "pointing_both_hands" in {g.key for g in two_hand_gestures(l, r)}
    for key, sign in (("hands_moving_apart", 1), ("hands_moving_together", -1)):
        tr = TwoHandMotionTracker()
        out = []
        for i in range(15):
            d = 150 + sign * 10 * i
            out = tr.update(classify_hand(hand(mirror=True, center=(400 - d, 300)), "left", 0.9),
                            classify_hand(hand(center=(400 + d, 300)), "right", 0.9), i / FPS)
        assert key in {m.key for m in out}


def test_engine_assigns_hands_to_person_side_and_detects_hand_over_face():
    eng = ActionEngine(min_event_duration=0.0)
    k, b = pose()
    # right wrist of a person facing the camera is on the image LEFT
    rw = k.xy[10]
    lm = hand(center=(rw[0], rw[1] - 60))
    lm[:, :2] += rw - lm[0, :2]
    raw = RawHand(lm, "Left", 0.95)
    snaps, _ = eng.process(0.0, [detection(k, b)], [1], [(1, "entered")], [raw], [])
    assert snaps[0].hands["right"] is not None and snaps[0].hands["left"] is None
    # hand directly in front of the face
    nose = k.xy[0]
    lm2 = hand(center=(nose[0], nose[1] + 50), scale=40)
    raw2 = RawHand(lm2, "Left", 0.95, owner=(1, "right"))
    snaps, _ = eng.process(0.1, [detection(k, b)], [1], [], [raw2], [])
    assert "hand_over_face" in {g.key for g in snaps[0].hands["right"].gestures}


def test_hopping_on_one_leg():
    one = {"left_knee": (0.3, 0.25), "left_ankle": (0.3, 1.0)}
    res = run(seq((1.2, one, STAND_FRONT), (0.15, one, STAND_FRONT, (0, -0.5)), (0.3, one, STAND_FRONT)))
    assert any(r.movement is not None and r.movement.key == "hopping_one_leg" for r in res)
