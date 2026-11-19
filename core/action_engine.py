"""Central action-recognition engine.

Fuses per person: body pose (YOLO) + motion history + hands (MediaPipe) + head/face
(MediaPipe or YOLO fallback) into one PersonSnapshot, and turns state changes into
de-duplicated, debounced events for the timeline and the log files.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .geometry import L_EAR, L_EL, L_EYE, L_WR, NOSE, R_EAR, R_EL, R_EYE, R_WR, Keypoints
from .hand_gestures import HandMotionTracker, HandObservation, TwoHandMotionTracker, classify_hand, two_hand_gestures
from .head_pose import HeadInfo, HeadMotionTracker, analyse_face_mesh, analyse_head_from_pose
from .labels import Label, label_text
from .mediapipe_models import RawFace, RawHand
from .motion_analyzer import MotionAnalyzer, MotionResult
from .pose_detector import PoseDetection
from .posture import PoseFrame, analyse_pose

CACHE_HANDS_S = 0.3        # re-use hand/face results while those models are skipped for a frame
CACHE_FACE_S = 0.4


@dataclass
class EventRecord:
    timestamp: str                     # ISO time
    person_id: int | None
    event_type: str                    # entered | left | reacquired | state | ended | instant | command | system
    label: str                         # label key
    text: str                          # human readable
    posture: str | None = None
    movement: str | None = None
    head_direction: str | None = None
    left_hand: str | None = None
    right_hand: str | None = None
    confidence: float | None = None
    duration_s: float | None = None
    t: float = 0.0

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "t"}
        if d["confidence"] is not None:
            d["confidence"] = round(float(d["confidence"]), 3)
        if d["duration_s"] is not None:
            d["duration_s"] = round(float(d["duration_s"]), 2)
        return d


@dataclass
class PersonSnapshot:
    pid: int
    name: str
    box: np.ndarray
    pose: PoseFrame
    motion: MotionResult
    head: HeadInfo | None = None
    hands: dict = field(default_factory=lambda: {"left": None, "right": None})
    two_hand: list[Label] = field(default_factory=list)
    confidence: float = 0.0
    color_state: str = "normal"        # normal | gesture | success | uncertain | alarm
    tracked_for: float = 0.0
    recent_events: list[str] = field(default_factory=list)

    @property
    def kps(self) -> Keypoints:
        return self.pose.kps

    def all_labels(self) -> list[Label]:
        m = self.motion
        out = [x for x in (m.posture, m.facing, m.movement, m.direction) if x is not None]
        out += m.modifiers + m.transitions + m.exercises + self.two_hand
        if self.head is not None:
            out += self.head.labels()
            if self.head.gaze is not None:
                out.append(self.head.gaze)
        for h in self.hands.values():
            if h is not None:
                out += h.gestures + h.motions
        return out

    def label_keys(self) -> set[str]:
        return {lab.key for lab in self.all_labels()}

    def hand_key(self, side: str) -> str | None:
        h = self.hands.get(side)
        if h is None:
            return None
        return h.primary.key if h.primary else f"fingers_{h.count}"


class _Channel:
    """Single-valued state that must persist `min_dur` before it counts as changed."""

    def __init__(self) -> None:
        self.value: str | None = None
        self.since = 0.0
        self.cand: str | None = None
        self.cand_since = 0.0

    def update(self, value: str | None, t: float, min_dur: float):
        if value == self.value:
            self.cand = None
            return None
        if value != self.cand:
            self.cand, self.cand_since = value, t
        if t - self.cand_since >= min_dur:
            prev, prev_dur = self.value, t - self.since
            self.value, self.since, self.cand = value, self.cand_since, None
            return prev, prev_dur
        return None


@dataclass
class _PersonState:
    pid: int
    first_seen: float
    motion: MotionAnalyzer = field(default_factory=MotionAnalyzer)
    hand_motion: dict = field(default_factory=lambda: {"left": HandMotionTracker(), "right": HandMotionTracker()})
    two_hand_motion: TwoHandMotionTracker = field(default_factory=TwoHandMotionTracker)
    head_motion: HeadMotionTracker = field(default_factory=HeadMotionTracker)
    channels: dict = field(default_factory=dict)
    set_labels: dict = field(default_factory=dict)      # key -> [cand_since, last_seen, started]
    last_instant: dict = field(default_factory=dict)
    hand_cache: dict = field(default_factory=dict)      # side -> (t, HandObservation)
    face_cache: tuple | None = None
    name_votes: Counter = field(default_factory=Counter)
    events: deque = field(default_factory=lambda: deque(maxlen=8))


SINGLE_CHANNELS = ("posture", "movement", "head", "gaze", "left_hand", "right_hand")


class ActionEngine:
    def __init__(self, keypoint_conf: float = 0.4, min_event_duration: float = 0.5, lang: str = "en") -> None:
        self.keypoint_conf = keypoint_conf
        self.min_event_duration = min_event_duration
        self.lang = lang
        self.persons: dict[int, _PersonState] = {}

    def reset(self) -> None:
        self.persons.clear()

    def person_name(self, pid: int) -> str:
        st = self.persons.get(pid)
        if st is not None:
            known = [(n, c) for n, c in st.name_votes.most_common() if n != "Unknown"]
            if known and known[0][1] >= 2:
                return known[0][0]
        return f"PERSON {pid:03d}"

    def add_name_vote(self, pid: int, name: str) -> None:
        if pid in self.persons:
            self.persons[pid].name_votes[name] += 1

    # ------------------------------------------------------------------
    def process(self, t: float, detections: list[PoseDetection], pids: list[int], tracker_events,
                hands: list[RawHand] | None, faces: list[RawFace] | None,
                frame_size: tuple[int, int] | None = None) -> tuple[list[PersonSnapshot], list[EventRecord]]:
        """hands / faces = None means "model not run this frame" -> cached results are reused."""
        events: list[EventRecord] = []
        snaps: list[PersonSnapshot] = []
        for pid, kind in tracker_events:
            if kind == "left":
                name = self.person_name(pid)
                st = self.persons.pop(pid, None)
                dur = t - st.first_seen if st else None
                events.append(self._event(None, pid, "left", "left", f"{name} LEFT FRAME", t, duration=dur))
            elif kind == "rematched":
                st = self.persons.get(pid)
                if st is not None:          # possibly a different person now: start a fresh history
                    first, votes = st.first_seen, st.name_votes
                    self.persons[pid] = _PersonState(pid, first, name_votes=votes)
            elif kind in ("entered", "reacquired"):
                if pid not in self.persons or kind == "reacquired":
                    old = self.persons.get(pid)
                    self.persons[pid] = _PersonState(pid, old.first_seen if old else t,
                                                     name_votes=old.name_votes if old else Counter())
                txt = "ENTERED FRAME" if kind == "entered" else "RE-ACQUIRED"
                events.append(self._event(None, pid, kind, kind, f"{self.person_name(pid)} {txt}", t))

        frames: dict[int, PoseFrame] = {}
        for det, pid in zip(detections, pids):
            st = self.persons.setdefault(pid, _PersonState(pid, t))
            kps = Keypoints(det.xy, det.conf, self.keypoint_conf)
            pf = analyse_pose(kps, det.box, t)
            if frame_size is not None:
                W, H = frame_size
                x1, y1, x2, y2 = det.box
                pf.truncated = x1 <= 3 or x2 >= W - 3 or y2 >= H - 3
            frames[pid] = pf

        hand_assign = self._assign_hands(hands, frames) if hands is not None else None
        face_assign = self._assign_faces(faces, frames) if faces is not None else None

        for det, pid in zip(detections, pids):
            st = self.persons[pid]
            pf = frames[pid]
            facing_cam = pf.facing_side != "back"
            snap = PersonSnapshot(pid=pid, name=self.person_name(pid), box=det.box, pose=pf,
                                  motion=st.motion.update(pf), tracked_for=t - st.first_seen)
            # ---- head ----
            head = None
            if face_assign is not None:
                raw = face_assign.get(pid)
                st.face_cache = (t, raw) if raw is not None else None
            if st.face_cache is not None and t - st.face_cache[0] <= CACHE_FACE_S:
                raw = st.face_cache[1]
                head = analyse_face_mesh(raw.landmarks, raw.score, st.head_motion.pitch_neutral)
            if head is None:
                head = analyse_head_from_pose(pf.kps)
            if head is not None:
                head.motions = st.head_motion.update(head, t)
            snap.head = head
            # ---- hands ----
            for side in ("left", "right"):
                obs = None
                if hand_assign is not None:
                    raw = hand_assign.get((pid, side))
                    if raw is not None:
                        obs = classify_hand(raw.landmarks, side, raw.score, facing_cam)
                        st.hand_cache[side] = (t, obs)
                    else:
                        st.hand_cache.pop(side, None)
                if obs is None and side in st.hand_cache and t - st.hand_cache[side][0] <= CACHE_HANDS_S:
                    obs = st.hand_cache[side][1]
                if obs is not None:
                    if self._hand_over_face(obs, head, pf.kps):
                        if not any(g.key == "hand_over_face" for g in obs.gestures):
                            obs.gestures.append(Label("hand_over_face", obs.score * 0.85))
                    obs.motions = st.hand_motion[side].update(obs, t)
                else:
                    st.hand_motion[side].reset()
                snap.hands[side] = obs
            snap.two_hand = two_hand_gestures(snap.hands["left"], snap.hands["right"])
            snap.two_hand += st.two_hand_motion.update(snap.hands["left"], snap.hands["right"], t)
            self._finalise(snap)
            events += self._events_for(st, snap, t)
            snap.recent_events = list(st.events)
            snaps.append(snap)
        return snaps, events

    # ------------------------------------------------------------------
    def _finalise(self, snap: PersonSnapshot) -> None:
        m = snap.motion
        confs = [x.confidence for x in (m.posture, m.movement) if x is not None]
        if snap.head is not None and snap.head.direction is not None:
            confs.append(snap.head.direction.confidence)
        for h in snap.hands.values():
            if h is not None and h.primary is not None:
                confs.append(h.primary.confidence)
        snap.confidence = float(np.mean(confs)) if confs else 0.0
        keys = snap.label_keys()
        if "falling" in keys:
            snap.color_state = "alarm"
        elif snap.confidence < 0.45:
            snap.color_state = "uncertain"
        elif m.transitions or m.exercises:
            snap.color_state = "success"
        elif any(h is not None and h.gestures for h in snap.hands.values()) or snap.two_hand:
            snap.color_state = "gesture"
        else:
            snap.color_state = "normal"

    def _ctx(self, snap: PersonSnapshot) -> dict:
        m = snap.motion
        return dict(posture=m.posture.key if m.posture else None,
                    movement=m.movement.key if m.movement else None,
                    head_direction=snap.head.direction.key if snap.head and snap.head.direction else None,
                    left_hand=snap.hand_key("left"), right_hand=snap.hand_key("right"))

    def _event(self, snap, pid, etype, key, text, t, conf=None, duration=None) -> EventRecord:
        ctx = self._ctx(snap) if snap is not None else {}
        ev = EventRecord(timestamp=datetime.now().isoformat(timespec="milliseconds"), person_id=pid,
                         event_type=etype, label=key, text=text, confidence=conf, duration_s=duration, t=t, **ctx)
        st = self.persons.get(pid) if pid is not None else None
        if st is not None:
            st.events.appendleft(text)
        return ev

    def _events_for(self, st: _PersonState, snap: PersonSnapshot, t: float) -> list[EventRecord]:
        out: list[EventRecord] = []
        name = snap.name
        m = snap.motion
        values = {
            "posture": m.posture, "movement": m.movement,
            "head": snap.head.direction if snap.head else None,
            "gaze": snap.head.gaze if snap.head else None,
            "left_hand": snap.hands["left"].primary if snap.hands["left"] else None,
            "right_hand": snap.hands["right"].primary if snap.hands["right"] else None,
        }
        prefix = {"posture": "IS", "movement": "STARTED", "head": "HEAD:", "gaze": "GAZE:",
                  "left_hand": "LEFT HAND:", "right_hand": "RIGHT HAND:"}
        for ch in SINGLE_CHANNELS:
            lab = values[ch]
            chan = st.channels.setdefault(ch, _Channel())
            changed = chan.update(lab.key if lab else None, t, self.min_event_duration)
            if changed is None:
                continue
            prev, prev_dur = changed
            if lab is None:
                continue
            txt = f"{name} {prefix[ch]} {lab.text(self.lang)}"
            out.append(self._event(snap, snap.pid, "state", lab.key, txt, t, lab.confidence,
                                   prev_dur if prev is not None else None))
        # multi-valued labels (pose modifiers, gestures of both hands, exercises, ...)
        current = {lab.key: lab for lab in m.modifiers + m.exercises + snap.two_hand}
        for h in snap.hands.values():
            if h is not None:
                for g in h.motions:
                    current[g.key] = g
        for key, lab in current.items():
            rec = st.set_labels.get(key)
            if rec is None:
                st.set_labels[key] = [t, t, False]
                continue
            rec[1] = t
            if not rec[2] and t - rec[0] >= self.min_event_duration:
                rec[2] = True
                out.append(self._event(snap, snap.pid, "state", key, f"{name}: {lab.text(self.lang)}", t, lab.confidence))
        for key, rec in list(st.set_labels.items()):
            if key not in current and t - rec[1] > 0.5:
                if rec[2]:
                    out.append(self._event(snap, snap.pid, "ended", key,
                                           f"{name}: {label_text(key, self.lang)} ENDED", t, None, rec[1] - rec[0]))
                    st.events.popleft()   # keep "ended" out of the on-screen mini history
                del st.set_labels[key]
        # one-shot events (falls, jumps, transitions, turns)
        for lab in m.instant_events:
            last = st.last_instant.get(lab.key, -1e9)
            if t - last < 1.0:
                continue
            st.last_instant[lab.key] = t
            out.append(self._event(snap, snap.pid, "instant", lab.key, f"{name} {lab.text(self.lang)}", t, lab.confidence))
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _wrist_estimate(kps: Keypoints, wr: int, el: int, T: float | None):
        w = kps.get(wr)
        if w is not None:
            return w, 1.0
        e = kps.get(el)
        return (e, 0.6) if e is not None else (None, 0.0)

    def _assign_hands(self, hands: list[RawHand], frames: dict[int, PoseFrame]) -> dict:
        """Match each MediaPipe hand to the closest YOLO wrist -> person + anatomical side."""
        out, used_h, used_s = {}, set(), set()
        for hi, hand in enumerate(hands):          # hands found in a wrist crop already know their owner
            if hand.owner is not None and hand.owner[0] in frames and hand.owner not in used_s:
                out[hand.owner] = hand
                used_h.add(hi)
                used_s.add(hand.owner)
        pairs = []
        for hi, hand in enumerate(hands):
            if hi in used_h:
                continue
            wrist = hand.landmarks[0, :2]
            for pid, pf in frames.items():
                T = pf.torso_len or max(pf.box_h / 3, 1.0)
                for side, wr, el in (("left", L_WR, L_EL), ("right", R_WR, R_EL)):
                    pos, q = self._wrist_estimate(pf.kps, wr, el, T)
                    if pos is None:
                        continue
                    d = float(np.linalg.norm(wrist - pos)) / T
                    limit = 0.6 if q == 1.0 else 1.0
                    if d < limit:
                        pairs.append((d / q, hi, pid, side))
        for _, hi, pid, side in sorted(pairs):
            if hi in used_h or (pid, side) in used_s:
                continue
            out[(pid, side)] = hands[hi]
            used_h.add(hi)
            used_s.add((pid, side))
        # hands without a wrist match: inside a person box -> MediaPipe handedness.
        # MediaPipe assumes a mirrored (selfie) image; we analyse the raw image, so swap.
        for hi, hand in enumerate(hands):
            if hi in used_h:
                continue
            c = hand.landmarks[:, :2].mean(axis=0)
            for pid, pf in frames.items():
                x1, y1, x2, y2 = pf.box
                if x1 <= c[0] <= x2 and y1 <= c[1] <= y2:
                    side = "right" if hand.mp_label == "Left" else "left"
                    if (pid, side) not in used_s:
                        out[(pid, side)] = hand
                        used_s.add((pid, side))
                        used_h.add(hi)
                    break
        return out

    def _assign_faces(self, faces: list[RawFace], frames: dict[int, PoseFrame]) -> dict:
        out, used_f = {}, set()
        for fi, face in enumerate(faces):
            if face.owner is not None and face.owner in frames and face.owner not in out:
                out[face.owner] = face
                used_f.add(fi)
        pairs = []
        for fi, face in enumerate(faces):
            if fi in used_f:
                continue
            nose = face.landmarks[1, :2]
            for pid, pf in frames.items():
                ref = pf.kps.get(NOSE)
                T = pf.torso_len or max(pf.box_h / 3, 1.0)
                if ref is None:
                    x1, y1, x2, y2 = pf.box
                    if not (x1 <= nose[0] <= x2 and y1 <= nose[1] <= y1 + 0.4 * (y2 - y1)):
                        continue
                    ref = np.array([(x1 + x2) / 2, y1 + 0.12 * (y2 - y1)])
                d = float(np.linalg.norm(nose - ref)) / T
                if d < 0.6:
                    pairs.append((d, fi, pid))
        for _, fi, pid in sorted(pairs):
            if fi in used_f or pid in out:
                continue
            out[pid] = faces[fi]
            used_f.add(fi)
        return out

    @staticmethod
    def _hand_over_face(obs: HandObservation, head: HeadInfo | None, kps: Keypoints) -> bool:
        if head is not None and head.face_box is not None:
            fb = head.face_box
        else:
            pts = [kps.get(i) for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)]
            pts = [p for p in pts if p is not None]
            if len(pts) < 2:
                return False
            arr = np.array(pts)
            pad = max(float(np.ptp(arr[:, 0])), 10.0) * 0.4
            fb = (arr[:, 0].min() - pad, arr[:, 1].min() - 2 * pad, arr[:, 0].max() + pad, arr[:, 1].max() + 3 * pad)
        hb = obs.bbox
        ix = max(0.0, min(fb[2], hb[2]) - max(fb[0], hb[0]))
        iy = max(0.0, min(fb[3], hb[3]) - max(fb[1], hb[1]))
        face_area = max((fb[2] - fb[0]) * (fb[3] - fb[1]), 1.0)
        return ix * iy / face_area > 0.3
