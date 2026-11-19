"""Temporal motion analysis + action engine events on synthetic sequences."""
import math

import numpy as np

from core.action_engine import ActionEngine
from core.motion_analyzer import MotionAnalyzer
from core.posture import analyse_pose
from tests.synthetic import STAND_FRONT, back_view, detection, pose, side_view

FPS = 30
SIT = {"left_hip": (0.03, 0), "right_hip": (-0.03, 0), "left_knee": (0.85, 0.05), "right_knee": (0.8, 0.05),
       "left_ankle": (0.85, 0.95), "right_ankle": (0.8, 0.95)}
SQUAT = {"left_shoulder": (0.4, -0.95), "right_shoulder": (0.3, -0.95), "nose": (0.65, -1.35),
         "left_knee": (0.6, -0.1), "right_knee": (0.55, -0.1), "left_ankle": (0.3, 0.6), "right_ankle": (0.25, 0.6)}


def run(frames, ma=None):
    """frames: iterable of (t, kps, box). Returns (analyzer, list of results)."""
    ma = ma or MotionAnalyzer()
    out = [ma.update(analyse_pose(k, b, t)) for t, k, b in frames]
    return ma, out


def still(seconds, t0=0.0, **kw):
    n = int(seconds * FPS)
    return [(t0 + i / FPS, *pose(**kw)) for i in range(n)]


def instant_keys(results):
    return [lab.key for r in results for lab in r.instant_events]


def test_standing_still():
    _, res = run(still(1.5))
    assert res[-1].movement.key == "standing_still"
    assert res[-1].posture.key == "standing"


def test_walking_sideways_with_steps():
    frames = []
    for i in range(int(1.6 * FPS)):
        t = i / FPS
        dx = 2.0 * t                      # 2 torso lengths per second to the image right
        sw = 0.25 * math.sin(2 * math.pi * 1.8 * t)
        o = {"left_ankle": (0.03 + sw, 1.78), "right_ankle": (-0.03 - sw, 1.78)}
        frames.append((t, *pose(o, base=side_view(face_right=True), offset=(dx, 0))))
    _, res = run(frames)
    r = res[-1]
    assert r.movement.key in ("walking", "walking_fast")
    assert r.direction.key == "moving_screen_right"
    assert 1.5 < r.speed < 2.5


def test_running_speed_class():
    frames = [(i / FPS, *pose(base=side_view(), offset=(5.0 * i / FPS, 0))) for i in range(int(1.2 * FPS))]
    _, res = run(frames)
    assert res[-1].movement.key in ("running", "sprinting")


def test_walking_backward_is_detected_from_facing():
    # facing image-right but moving image-left
    frames = [(i / FPS, *pose(base=side_view(face_right=True), offset=(-1.5 * i / FPS, 0))) for i in range(int(1.2 * FPS))]
    _, res = run(frames)
    assert res[-1].movement.key == "walking_backward"


def test_sidestepping_when_facing_camera():
    frames = [(i / FPS, *pose(offset=(1.2 * i / FPS, 0))) for i in range(int(1.2 * FPS))]
    _, res = run(frames)
    assert res[-1].movement.key == "sidestepping"


def test_approaching_camera():
    frames = []
    for i in range(int(1.5 * FPS)):
        t = i / FPS
        s = 100 * math.exp(0.45 * t)
        sw = 0.15 * math.sin(2 * math.pi * 1.8 * t)
        frames.append((t, *pose({"left_ankle": (0.22, 1.78 + sw), "right_ankle": (-0.22, 1.78 - sw)}, scale=s)))
    _, res = run(frames)
    assert res[-1].movement.key == "approaching_camera"


def test_sit_down_transition_and_stand_up():
    frames = still(1.0) + [(1.0 + i / FPS, *pose(SIT, base=side_view())) for i in range(30)]
    frames += [(2.0 + i / FPS, *pose()) for i in range(30)]
    _, res = run(frames)
    k = instant_keys(res)
    assert "sitting_down" in k and "standing_up" in k
    assert k.index("sitting_down") < k.index("standing_up")


def test_squat_repetitions_counted():
    frames, t = [], 0.0
    for _ in range(3):
        for _ in range(15):
            frames.append((t, *pose()))
            t += 1 / FPS
        for _ in range(15):
            frames.append((t, *pose(SQUAT, base=side_view())))
            t += 1 / FPS
    for _ in range(10):
        frames.append((t, *pose()))
        t += 1 / FPS
    _, res = run(frames)
    reps = [e for e in res[-1].exercises if e.key == "squat_reps"]
    assert reps and reps[0].detail == "x3"


def test_jump_detected():
    frames = still(1.2)
    for i in range(5):
        frames.append((1.2 + i / FPS, *pose(offset=(0, -0.5))))
    _, res = run(frames)
    assert "jumping" in instant_keys(res)


def test_fall_detected():
    lying = {"nose": (-1.5, 1.6), "left_eye": (-1.55, 1.55), "right_eye": (-1.55, 1.65), "left_ear": (-1.45, 1.5),
             "right_ear": (-1.45, 1.7), "left_shoulder": (-1.0, 1.5), "right_shoulder": (-1.0, 1.7),
             "left_elbow": (-0.6, 1.5), "right_elbow": (-0.6, 1.7), "left_wrist": (-0.2, 1.5), "right_wrist": (-0.2, 1.7),
             "left_hip": (0.0, 1.5), "right_hip": (0.0, 1.7), "left_knee": (0.9, 1.5), "right_knee": (0.9, 1.7),
             "left_ankle": (1.8, 1.5), "right_ankle": (1.8, 1.7)}
    # realistic fall: the body goes from standing to lying within 0.4 s
    frames = still(1.0)
    for i in range(1, 13):
        a = i / 12
        mix = {k: tuple((1 - a) * np.array(STAND_FRONT[k]) + a * np.array(lying[k])) for k in STAND_FRONT}
        frames.append((1.0 + i / FPS, *pose(mix)))
    frames += [(1.4 + i / FPS, *pose(lying)) for i in range(15)]
    _, res = run(frames)
    assert "falling" in instant_keys(res)
    assert res[-1].movement.key == "falling"
    assert "lying_down" not in instant_keys(res)       # a fall is not reported as "lying down"


def test_turning_180_and_360():
    seq = [("front", {}, STAND_FRONT, ()), ("side_l", {}, side_view(False), ()),
           ("back", {}, back_view(), ("nose", "left_eye", "right_eye")), ("side_r", {}, side_view(True), ()),
           ("front", {}, STAND_FRONT, ())]
    frames, t = [], 0.0
    for _, o, base, hidden in seq:
        for _ in range(12):
            frames.append((t, *pose(o, base=base, hidden=hidden)))
            t += 1 / FPS
    _, res = run(frames)
    k = instant_keys(res)
    assert "turned_90" in k and "turned_180" in k and "turned_360" in k
    ev = [lab for r in res for lab in r.instant_events if lab.key == "turned_360"][0]
    assert ev.detail == "(R)"           # front -> nose to image-left = person turns to their right


def test_waving_right_hand():
    frames = []
    for i in range(int(1.5 * FPS)):
        t = i / FPS
        x = -0.6 + 0.3 * math.sin(2 * math.pi * 2.5 * t)
        frames.append((t, *pose({"right_elbow": (-0.6, -1.1), "right_wrist": (x, -1.5)})))
    _, res = run(frames)
    assert any(e.key == "waving_right_hand" for e in res[-1].exercises)


def test_jumping_jacks():
    open_ = {"left_wrist": (0.7, -1.6), "right_wrist": (-0.7, -1.6), "left_elbow": (0.65, -1.3), "right_elbow": (-0.65, -1.3),
             "left_ankle": (0.7, 1.75), "right_ankle": (-0.7, 1.75)}
    frames, t = [], 0.0
    for _ in range(3):
        for o in (open_, {}):
            for _ in range(8):
                frames.append((t, *pose(o)))
                t += 1 / FPS
    _, res = run(frames)
    assert any(e.key == "jumping_jacks" for e in res[-1].exercises)


# ---------------- action engine / events ----------------
def test_engine_events_and_debounce():
    eng = ActionEngine(min_event_duration=0.3)
    events = []
    for i in range(30):
        k, b = pose()
        snaps, ev = eng.process(i / FPS, [detection(k, b)], [1], [(1, "entered")] if i == 0 else [], [], [])
        events += ev
    types = [e.event_type for e in events]
    assert types[0] == "entered"
    st = [e for e in events if e.event_type == "state" and e.label == "standing"]
    assert len(st) == 1                       # emitted once, not every frame
    assert snaps[0].name == "PERSON 001"
    assert snaps[0].motion.posture.key == "standing"
    assert 0 < snaps[0].confidence <= 1


def test_engine_flicker_is_not_reported():
    eng = ActionEngine(min_event_duration=0.5)
    events = []
    for i in range(60):
        o = {"right_elbow": (-0.6, -1.15), "right_wrist": (-0.65, -1.3)} if i % 6 == 0 else {}
        k, b = pose(o)
        _, ev = eng.process(i / FPS, [detection(k, b)], [1], [], [], [])
        events += ev
    assert not any(e.label == "raising_right_hand" for e in events)


def test_engine_left_event_has_duration():
    eng = ActionEngine()
    k, b = pose()
    eng.process(0.0, [detection(k, b)], [1], [(1, "entered")], [], [])
    _, ev = eng.process(5.0, [], [], [(1, "left")], [], [])
    assert ev[0].event_type == "left" and ev[0].duration_s == 5.0
    assert 1 not in eng.persons


def test_teleport_resets_history_instead_of_fake_events():
    # an ID swap moves the "same" person 5 torso lengths within one frame -> history reset, no jump/turn
    frames = still(1.2) + [(1.2 + i / FPS, *pose(base=back_view(), hidden=("nose", "left_eye", "right_eye"),
                                                    offset=(5, -0.6))) for i in range(20)]
    _, res = run(frames)
    keys = instant_keys(res)
    assert "jumping" not in keys and not any(k.startswith("turned") for k in keys)
