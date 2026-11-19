"""Temporal motion analysis for one tracked person.

Works on a time window of PoseFrames and combines:
  * hip velocity (least squares)          -> walking / running / direction
  * change of body scale                   -> approaching / moving away
  * alternating ankle / knee motion        -> steps, in-place walking / running
  * posture state changes                  -> sitting down, standing up, ...
  * facing-state sequence                  -> turning 90 / 180 / 360 deg
  * hip drop + torso tilt (original logic) -> possible fall
  * foot lift (original logic)             -> jump
  * oscillations                           -> waving, repeated arm / leg movement, exercises
Speeds are expressed in torso lengths per second (T/s), 1 T is roughly 0.5 m.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import L_ANK, L_SH, R_ANK, R_SH, count_swings, linear_velocity, soft_margin
from .labels import Label
from .posture import PoseFrame

HISTORY_S = 4.0
UPRIGHT = 40
FALL_WINDOW_S, FALL_DROP, FALL_END_ANGLE, FALL_HOLD_S = 1.0, 1.0, 55, 3.0
JUMP_RISE, JUMP_HOLD_S = 0.2, 0.6
WAVE_MIN_SWINGS, WAVE_WINDOW_S, WAVE_AMPLITUDE = 3, 1.5, 0.15
TRANSITION_HOLD_S = 1.5
# lateral speed classes in T/s
SPEED_CLASSES = [(0.6, "walking_slow"), (1.6, "walking"), (3.0, "walking_fast"), (4.0, "running"), (7.0, "sprinting")]
APPROACH_RATE = 0.25        # relative scale change per second

GROUP = {
    "standing": "STAND", "lunge": "STAND", "sitting": "SIT", "squatting": "SQUAT", "deep_squat": "SQUAT",
    "crouching": "SQUAT", "bending_forward": "BEND", "lying": "LYING", "lying_back": "LYING",
    "lying_front": "LYING", "lying_side": "LYING", "plank_position": "LYING", "kneeling": "KNEEL",
    "kneeling_one_knee": "KNEEL", "upright_upper_body": "UPPER",
}
TRANSITIONS = {
    ("STAND", "SIT"): "sitting_down", ("SIT", "STAND"): "standing_up",
    ("STAND", "SQUAT"): "squatting_down", ("SQUAT", "STAND"): "rising_from_squat",
    ("STAND", "BEND"): "bending_down", ("BEND", "STAND"): "straightening_up",
    ("STAND", "LYING"): "lying_down", ("SIT", "LYING"): "lying_down", ("KNEEL", "LYING"): "lying_down",
    ("SQUAT", "LYING"): "lying_down",
    ("LYING", "SIT"): "getting_up", ("LYING", "KNEEL"): "getting_up", ("LYING", "STAND"): "getting_up",
    ("LYING", "SQUAT"): "getting_up", ("KNEEL", "STAND"): "standing_up", ("SQUAT", "SIT"): "sitting_down",
}
FACING_CYCLE = ["front", "side_l", "back", "side_r"]   # +1 step = person turns to their right


@dataclass
class MotionResult:
    posture: Label | None = None
    modifiers: list[Label] = field(default_factory=list)
    facing: Label | None = None
    movement: Label | None = None
    direction: Label | None = None
    transitions: list[Label] = field(default_factory=list)
    exercises: list[Label] = field(default_factory=list)
    instant_events: list[Label] = field(default_factory=list)   # one-shot events (jump, fall, turn, rep)
    speed: float = 0.0                       # T/s (lateral)
    velocity: np.ndarray | None = None       # px/s of the hip centre
    scale_rate: float = 0.0
    step_freq: float = 0.0
    trajectory: list = field(default_factory=list)
    body_turn_steps: int = 0


class MotionAnalyzer:
    def __init__(self, smooth_frames: int = 7) -> None:
        self.smooth_frames = smooth_frames
        self.reset()

    def reset(self) -> None:
        self.hist: deque[PoseFrame] = deque(maxlen=240)
        self.fall_until = 0.0
        self.jump_until = 0.0
        self.jump_times: deque[float] = deque(maxlen=10)
        self.jump_one_leg = False
        self.stumble_until = 0.0
        self.stable_group: str | None = None
        self.stable_since = 0.0
        self.active_transitions: dict[str, float] = {}
        self.squat_reps: deque[float] = deque(maxlen=200)
        self.leg_raises: deque[float] = deque(maxlen=200)
        self.prev_knee_raised = False
        self.one_leg_since: float | None = None
        self.facing_state: str | None = None
        self.facing_candidate: tuple[str | None, int] = (None, 0)
        self.turn_steps = 0
        self.turn_unknown = 0
        self.turn_last_change = 0.0
        self.turn_emitted: set[str] = set()
        self.jj_phase: str | None = None
        self.jj_cycles: deque[float] = deque(maxlen=50)
        self.twist_ratios: deque[float] = deque(maxlen=300)

    # ------------------------------------------------------------------
    def update(self, f: PoseFrame) -> MotionResult:
        now = f.t
        if self._teleported(f):
            self.reset()          # ID swap / tracking glitch: never mix two motion histories
        self.hist.append(f)
        while self.hist and now - self.hist[0].t > HISTORY_S:
            self.hist.popleft()
        res = MotionResult()
        hist = [h for h in self.hist if h.hip is not None and h.torso_len]
        T = float(np.median([h.torso_len for h in hist[-15:]])) if hist else (f.torso_len or 0.0)

        self._smooth(res, now)
        res.trajectory = [tuple(map(float, h.hip)) for h in hist[-40:]]

        if f.hip is not None and f.torso_len:
            self._fall(f, hist, res, now)
            self._jump(f, hist, res, now)
        self._transitions(res, now)
        self._stumble(f, res, now)
        self._turning(res, now)
        if T > 0:
            self._locomotion(f, hist, res, now, T)
            self._exercises(f, res, now, T)
            self._twist(f, res)
        return res

    def _teleported(self, f: PoseFrame) -> bool:
        """True if the body centre moved faster than humanly possible since the last frame."""
        last = next((h for h in reversed(self.hist) if h.hip is not None and h.torso_len), None)
        if last is None or f.hip is None or not f.torso_len:
            return False
        dt = f.t - last.t
        if dt <= 0 or dt > 0.5:
            return False
        d = float(np.linalg.norm(f.hip - last.hip)) / max(f.torso_len, last.torso_len)
        return d > 0.8 and d / dt > 15.0

    # ------------------------------------------------------------------
    def _recent(self, now: float, seconds: float) -> list[PoseFrame]:
        return [h for h in self.hist if now - h.t <= seconds]

    def _smooth(self, res: MotionResult, now: float) -> None:
        recent = list(self.hist)[-self.smooth_frames:]
        n = len(recent)
        keys = [h.posture.key if h.posture else None for h in recent]
        key, votes = Counter(keys).most_common(1)[0]
        if key is not None:
            confs = [h.posture.confidence for h in recent if h.posture and h.posture.key == key]
            res.posture = Label(key, float(np.mean(confs)) * votes / n)
        mod_counts: Counter = Counter()
        mod_conf: dict[str, list[float]] = {}
        for h in recent:
            for m in h.modifiers:
                mod_counts[m.key] += 1
                mod_conf.setdefault(m.key, []).append(m.confidence)
        for k, c in mod_counts.items():
            if c * 2 > n:
                res.modifiers.append(Label(k, float(np.mean(mod_conf[k])) * c / n))
        fkeys = [h.facing.key if h.facing else None for h in recent]
        fk, fv = Counter(fkeys).most_common(1)[0]
        if fk is not None:
            fc = [h.facing.confidence for h in recent if h.facing and h.facing.key == fk]
            res.facing = Label(fk, float(np.mean(fc)) * fv / n)
        # facing side (for turning)
        # only confident, fully visible frames may change the facing state (used for turn detection)
        sides = [h.facing_side for h in recent
                 if h.facing_side and not h.truncated and h.facing is not None and h.facing.confidence >= 0.5
                 and min(h.kps.conf[L_SH], h.kps.conf[R_SH]) >= 0.6]
        if sides:
            side, sv = Counter(sides).most_common(1)[0]
            self._facing_observe(side if sv * 2 > len(sides) else None, now, res)

    # ---------------- fall / jump (original algorithms) ----------------
    def _fall(self, f, hist, res, now):
        if now >= self.fall_until:
            window = [h for h in hist if now - h.t <= FALL_WINDOW_S and h.torso_angle is not None
                      and h.torso_angle < UPRIGHT]
            if window and f.torso_angle is not None and f.torso_angle > FALL_END_ANGLE:
                top = min(window, key=lambda h: h.hip[1])
                drop = (f.hip[1] - top.hip[1]) / top.torso_len
                if drop > FALL_DROP:
                    self.fall_until = now + FALL_HOLD_S
                    conf = soft_margin(drop, FALL_DROP, 0.8) * soft_margin(f.torso_angle, FALL_END_ANGLE, 25)
                    res.instant_events.append(Label("falling", conf))
                    self.stable_group = "LYING"     # don't report "lying down" for a fall
        if now < self.fall_until:
            res.movement = Label("falling", 0.9)

    def _jump(self, f, hist, res, now):
        # both feet must leave the ground individually, with reliable ankle keypoints
        # (prevents false jumps from keypoint jitter on partly visible people)
        reliable = (not f.truncated and f.l_ank is not None and f.r_ank is not None
                    and min(f.kps.conf[L_ANK], f.kps.conf[R_ANK]) >= 0.5)
        if reliable and now >= self.jump_until:
            base = [h for h in hist if 0.3 <= now - h.t <= 1.5 and h.l_ank is not None and h.r_ank is not None
                    and h.posture is not None and h.posture.key in ("standing", "lunge")]
            if len(base) >= 5:
                ground_l = float(np.median([h.l_ank[1] for h in base]))
                ground_r = float(np.median([h.r_ank[1] for h in base]))
                base_hip = float(np.median([h.hip[1] for h in base]))
                rise = JUMP_RISE * f.torso_len
                if (f.l_ank[1] < ground_l - rise and f.r_ank[1] < ground_r - rise
                        and f.hip[1] < base_hip - 0.75 * rise):
                    self.jump_until = now + JUMP_HOLD_S
                    self.jump_times.append(now)
                    self.jump_one_leg = sum(1 for h in base if h.has("standing_one_leg")) > len(base) / 2
                    lift = (min(ground_l - f.l_ank[1], ground_r - f.r_ank[1])) / f.torso_len
                    res.instant_events.append(Label("jumping", soft_margin(lift, JUMP_RISE, 0.3)))
        if res.movement is None and now < self.jump_until + 0.8 and self.jump_times:
            recent = [t for t in self.jump_times if now - t <= 3.0]
            if now < self.jump_until or len(recent) >= 2:
                if self.jump_one_leg:
                    res.movement = Label("hopping_one_leg", 0.6)
                elif len(recent) >= 2:
                    res.movement = Label("jumping_repeatedly", 0.8, detail=f"x{len(recent)}")
                elif now < self.jump_until:
                    res.movement = Label("jumping", 0.8)

    # ---------------- stumble: short torso pitch + jerk, then recovered upright ----------------
    def _stumble(self, f, res, now):
        if now < self.fall_until or f.torso_angle is None or f.torso_angle > 20 or now < self.stumble_until:
            return
        seg = [h for h in self.hist if now - h.t <= 1.2 and h.torso_angle is not None and h.hip is not None and h.torso_len]
        if len(seg) < 8:
            return
        i_peak = max(range(len(seg)), key=lambda i: seg[i].torso_angle)
        peak = seg[i_peak]
        before = seg[:i_peak]
        if peak.torso_angle < 28 or not before or i_peak >= len(seg) - 2:
            return
        if peak.torso_angle - min(h.torso_angle for h in before) < 20 or peak.t - before[0].t > 0.9:
            return
        T = float(np.median([h.torso_len for h in seg]))
        v = [abs(b.hip[0] - a.hip[0]) / max(b.t - a.t, 1e-3) / T for a, b in zip(seg, seg[1:])]
        if max(v) < 1.5 or peak.posture is None or peak.posture.key not in ("standing", "lunge", "bending_forward"):
            return
        self.stumble_until = now + 2.0
        res.instant_events.append(Label("stumbling", 0.5))
        if res.movement is None:
            res.movement = Label("stumbling", 0.5)

    # ---------------- posture transitions ----------------
    def _transitions(self, res, now):
        group = GROUP.get(res.posture.key) if res.posture else None
        if group and group != "UPPER" and group != self.stable_group:
            prev = self.stable_group
            self.stable_group, self.stable_since = group, now
            key = TRANSITIONS.get((prev, group)) if prev else None
            if key and now >= self.fall_until:
                self.active_transitions[key] = now + TRANSITION_HOLD_S
                res.instant_events.append(Label(key, res.posture.confidence))
                if key == "rising_from_squat":
                    self.squat_reps.append(now)
        for key, until in list(self.active_transitions.items()):
            if now < until:
                res.transitions.append(Label(key, 0.8))
            else:
                del self.active_transitions[key]

    # ---------------- turning via facing-state sequence ----------------
    def _facing_observe(self, side: str | None, now: float, res: MotionResult) -> None:
        if side is None:
            return
        cand, n = self.facing_candidate
        self.facing_candidate = (side, n + 1) if side == cand else (side, 1)
        if self.facing_candidate[1] < 3 or side == self.facing_state:
            return
        prev = self.facing_state
        self.facing_state = side
        if prev is None:
            return
        if now - self.turn_last_change > 2.5:
            self.turn_steps, self.turn_unknown, self.turn_emitted = 0, 0, set()
        self.turn_last_change = now
        d = (FACING_CYCLE.index(side) - FACING_CYCLE.index(prev)) % 4
        if d == 1:
            self.turn_steps += 1
        elif d == 3:
            self.turn_steps -= 1
        elif d == 2:
            # skipped a state (e.g. side-left -> side-right): a real turn passes through front/back,
            # so this is keypoint flicker - do not count it
            self.turn_steps, self.turn_unknown, self.turn_emitted = 0, 0, set()
            return
        total = abs(self.turn_steps) + self.turn_unknown
        direction = "" if self.turn_steps == 0 else ("(R)" if self.turn_steps > 0 else "(L)")
        for steps, key in ((4, "turned_360"), (2, "turned_180"), (1, "turned_90")):
            if total >= steps:
                if key not in self.turn_emitted:
                    self.turn_emitted.update(k for s, k in ((4, "turned_360"), (2, "turned_180"), (1, "turned_90")) if s <= steps)
                    res.instant_events.append(Label(key, 0.6, detail=direction))
                break

    def _turning(self, res, now):
        if self.facing_state is not None and now - self.turn_last_change < 0.8 and self.turn_last_change > 0:
            res.body_turn_steps = self.turn_steps
            if res.movement is None:
                d = "(R)" if self.turn_steps > 0 else ("(L)" if self.turn_steps < 0 else "")
                res.movement = Label("turning", 0.6, detail=d)

    # ---------------- walking / running / direction ----------------
    def _locomotion(self, f, hist, res, now, T):
        seg = [h for h in hist if now - h.t <= 0.8]
        ts = [h.t for h in seg]
        vx = linear_velocity(ts, [h.hip[0] for h in seg]) if len(seg) >= 4 else None
        vy = linear_velocity(ts, [h.hip[1] for h in seg]) if len(seg) >= 4 else None
        if vx is not None:
            res.velocity = np.array([vx, vy or 0.0])
            res.speed = abs(vx) / T
        seg1 = [h for h in hist if now - h.t <= 1.0]
        if len(seg1) >= 5:
            sr = linear_velocity([h.t for h in seg1], [math.log(h.torso_len) for h in seg1])
            res.scale_rate = sr or 0.0
        # alternating feet / knees within 2 s
        seg2 = [h for h in self.hist if now - h.t <= 2.0]
        dur = (seg2[-1].t - seg2[0].t) if len(seg2) > 1 else 0.0
        ank_x = [(h.l_ank[0] - h.r_ank[0]) / T for h in seg2 if h.l_ank is not None and h.r_ank is not None]
        ank_y = [(h.l_ank[1] - h.r_ank[1]) / T for h in seg2 if h.l_ank is not None and h.r_ank is not None]
        kn_y = [(h.l_knee[1] - h.r_knee[1]) / T for h in seg2 if h.l_knee is not None and h.r_knee is not None]
        steps = max(count_swings(ank_x, 0.12), count_swings(ank_y, 0.12), count_swings(kn_y, 0.12))
        res.step_freq = steps / dur if dur > 0.5 else 0.0
        knee_amp = (max(kn_y) - min(kn_y)) if len(kn_y) >= 4 else 0.0

        if res.movement is not None:
            return
        pk = res.posture.key if res.posture else None
        upright = pk in ("standing", "lunge", None) and (f.torso_angle is None or f.torso_angle < UPRIGHT)
        if not upright:
            if self._still(now, T):
                res.movement = Label("idle", 0.8)
            return
        s, zs = res.speed, res.scale_rate
        if s > SPEED_CLASSES[0][0] and s > 2.5 * abs(zs):
            key = SPEED_CLASSES[0][1]
            for thr, k in SPEED_CLASSES:
                if s >= thr:
                    key = k
            if key in ("walking", "walking_fast") and res.step_freq > 2.6 and s > 2.0:
                key = "running"
            conf = soft_margin(s, SPEED_CLASSES[0][0], 0.6)
            res.direction = Label("moving_screen_left" if vx < 0 else "moving_screen_right", conf)
            facing = Counter(h.facing_side for h in seg if h.facing_side).most_common(1)
            fs = facing[0][0] if facing else None
            if fs in ("front", "back") and s < 3.0:
                key = "sidestepping"
            elif fs in ("side_l", "side_r") and s < 3.0:
                nose_dir = -1 if fs == "side_l" else 1
                if nose_dir * vx < 0:
                    key = "walking_backward"
            res.movement = Label(key, conf, detail=f"{s:.1f} T/s")
            return
        if abs(zs) > APPROACH_RATE and steps >= 2:
            fs = f.facing_side
            if zs > 0 and fs == "back":
                res.movement = Label("walking_backward", 0.5)
            else:
                res.movement = Label("approaching_camera" if zs > 0 else "moving_away", soft_margin(abs(zs), APPROACH_RATE, 0.3))
            return
        if steps >= 3 and s < 0.4:
            key = "running_in_place" if (res.step_freq > 2.2 or knee_amp > 0.9) else "walking_in_place"
            res.movement = Label(key, 0.6)
            return
        if self._still(now, T):
            res.movement = Label("standing_still" if pk == "standing" else "idle", 0.85)

    def _still(self, now: float, T: float) -> bool:
        seg = self._recent(now, 1.0)
        if len(seg) < 5:
            return False
        speeds = []
        for a, b in zip(seg, seg[1:]):
            dt = b.t - a.t
            if dt <= 0:
                continue
            m = (a.kps.conf >= a.kps.min_conf) & (b.kps.conf >= b.kps.min_conf)
            if m.sum() < 4:
                continue
            speeds.append(float(np.mean(np.linalg.norm(b.kps.xy[m] - a.kps.xy[m], axis=1))) / dt / T)
        return bool(speeds) and float(np.median(speeds)) < 0.3

    # ---------------- exercises / repetitive motion ----------------
    def _exercises(self, f, res, now, T):
        ex = res.exercises
        # waving (original logic, per side)
        seg = self._recent(now, WAVE_WINDOW_S)
        waving = set()
        for attr, key in (("raised_l_wr_x", "waving_left_hand"), ("raised_r_wr_x", "waving_right_hand")):
            xs = [getattr(h, attr) for h in seg if getattr(h, attr) is not None]
            if len(xs) >= 6 and count_swings(xs, WAVE_AMPLITUDE * T) >= WAVE_MIN_SWINGS:
                ex.append(Label(key, 0.85))
                waving.add(key)
        # repeated arm movement
        seg = self._recent(now, 2.5)
        if not waving:
            for side in ("l", "r"):
                ys = [((h.l_wr if side == "l" else h.r_wr)[1] - h.shoulder[1]) / T for h in seg
                      if (h.l_wr if side == "l" else h.r_wr) is not None and h.shoulder is not None]
                if len(ys) >= 8 and count_swings(ys, 0.35) >= 4:
                    ex.append(Label("repetitive_arm_motion", 0.7))
                    break
        # repeated leg movement (not while walking)
        if res.movement is None or res.movement.key in ("standing_still", "idle", "walking_in_place", "running_in_place"):
            for attr in ("l_ank", "r_ank"):
                ys = [(getattr(h, attr)[1] - h.hip[1]) / T for h in seg if getattr(h, attr) is not None and h.hip is not None]
                if len(ys) >= 8 and count_swings(ys, 0.25) >= 4:
                    ex.append(Label("repetitive_leg_motion", 0.6))
                    break
        # squat repetitions
        reps = [t for t in self.squat_reps if now - t <= 30]
        if reps and now - reps[-1] <= 10:
            ex.append(Label("squat_reps", 0.8, detail=f"x{len(reps)}"))
        # leg raises (knee raised onsets)
        raised = f.has("knee_raised_left") or f.has("knee_raised_right")
        if raised and not self.prev_knee_raised:
            self.leg_raises.append(now)
        self.prev_knee_raised = raised
        lr = [t for t in self.leg_raises if now - t <= 10]
        if len(lr) >= 2 and res.movement is not None and res.movement.key not in ("running_in_place", "walking_in_place"):
            ex.append(Label("leg_raises", 0.7, detail=f"x{len(lr)}"))
        # balance on one leg
        if any(m.key == "standing_one_leg" for m in res.modifiers):
            if self.one_leg_since is None:
                self.one_leg_since = now
            held = now - self.one_leg_since
            if held >= 3.0:
                ex.append(Label("balance_one_leg", 0.8, detail=f"{held:.0f}s"))
        else:
            self.one_leg_since = None
        # jumping jacks: alternate "arms up + legs apart" / "arms down + legs together"
        phase = self._jj_phase(f, T)
        if phase and phase != self.jj_phase:
            if self.jj_phase == "open" and phase == "closed":
                self.jj_cycles.append(now)
            self.jj_phase = phase
        cyc = [t for t in self.jj_cycles if now - t <= 4.0]
        if len(cyc) >= 2:
            ex.append(Label("jumping_jacks", 0.65, detail=f"x{len(cyc)}"))
        # arm circles: wrist angle around the shoulder keeps rotating
        seg3 = self._recent(now, 3.0)
        for side in ("l", "r"):
            angs = []
            for h in seg3:
                wr, sh = (h.l_wr, h.l_sh) if side == "l" else (h.r_wr, h.r_sh)
                if wr is not None and sh is not None:
                    angs.append(math.atan2(wr[1] - sh[1], wr[0] - sh[0]))
            if len(angs) >= 10:
                total = abs(float(np.unwrap(angs)[-1] - np.unwrap(angs)[0]))
                if math.degrees(total) > 540:
                    ex.append(Label("arm_circles", 0.6))
                    break

    def _jj_phase(self, f: PoseFrame, T: float) -> str | None:
        if any(p is None for p in (f.l_wr, f.r_wr, f.l_sh, f.r_sh, f.l_ank, f.r_ank)):
            return None
        sw = abs(float(f.l_sh[0] - f.r_sh[0])) or 0.8 * T
        spread = abs(float(f.l_ank[0] - f.r_ank[0]))
        arms_up = f.l_wr[1] < f.l_sh[1] and f.r_wr[1] < f.r_sh[1]
        arms_down = f.l_wr[1] > f.l_sh[1] + 0.5 * T and f.r_wr[1] > f.r_sh[1] + 0.5 * T
        if arms_up and spread > 1.3 * sw:
            return "open"
        if arms_down and spread < 0.9 * sw:
            return "closed"
        return None

    def _twist(self, f: PoseFrame, res: MotionResult) -> None:
        if f.facing_side != "front" or f.shoulder_ratio is None or not f.hip_ratio or f.hip_ratio < 0.2:
            return
        ratio = f.shoulder_ratio / f.hip_ratio
        if len(self.twist_ratios) >= 30:
            base = float(np.median(self.twist_ratios))
            if ratio < 0.65 * base:
                res.modifiers.append(Label("torso_twisted", 0.55))
                return
        self.twist_ratios.append(ratio)
