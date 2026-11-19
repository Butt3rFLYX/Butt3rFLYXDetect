"""Vision pipeline: one call per camera frame.

camera frame -> YOLO pose + ByteTrack -> stable IDs -> (MediaPipe hands / face, every n-th
frame depending on the performance mode) -> action engine -> FrameAnalysis
Every optional model can fail to load without stopping the others.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .action_engine import ActionEngine, EventRecord, PersonSnapshot
from .config import AppConfig
from .mediapipe_models import FaceLandmarkerModel, HandLandmarkerModel, OptionalModelError, RawHand  # noqa: F401
from .person_tracker import PersonTracker
from .pose_detector import ModelLoadError, PoseDetector


@dataclass
class FrameAnalysis:
    persons: list[PersonSnapshot] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    stray_hands: list[RawHand] = field(default_factory=list)    # hands not assigned to any person
    timings_ms: dict = field(default_factory=dict)
    frame_size: tuple[int, int] = (0, 0)
    paused: bool = False


def dedupe_hands(hands: list[RawHand]) -> list[RawHand]:
    """The same physical hand can show up in two wrist crops (e.g. a hand in front of another
    person). Keep only the assignment whose YOLO wrist matched best."""
    keep: list[RawHand] = []
    for h in sorted(hands, key=lambda x: x.match):
        size = float(np.linalg.norm(h.landmarks[9, :2] - h.landmarks[0, :2])) or 1.0
        dup = any(float(np.mean(np.linalg.norm(h.landmarks[:, :2] - k.landmarks[:, :2], axis=1))) < 0.3 * size
                  for k in keep)
        if not dup:
            keep.append(h)
    return keep


class FpsMeter:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha, self.fps, self.last = alpha, 0.0, None

    def tick(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        if self.last is not None:
            dt = now - self.last
            if dt > 0:
                self.fps = 1 / dt if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha / dt
        self.last = now
        return self.fps


class VisionPipeline:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.pose: PoseDetector | None = None
        self.hands: HandLandmarkerModel | None = None
        self.face: FaceLandmarkerModel | None = None
        self.face_id = None
        self.tracker = PersonTracker()
        self.tracker.configure(cfg.tracking_sensitivity)
        self.engine = ActionEngine(cfg.keypoint_confidence, cfg.min_event_duration, cfg.language)
        self.status: dict[str, str] = {"pose": "offline", "hands": "offline", "face": "offline",
                                       "identity": "disabled", "device": "-"}
        self.frame_no = 0
        self._prev: list = []
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mediapipe")

    # ------------------------------------------------------------------
    def load_models(self, progress=None) -> None:
        cfg, prof = self.cfg, self.cfg.profile
        say = progress or (lambda *_: None)
        if cfg.enable_pose:
            say("pose", "loading")
            try:
                self.pose = PoseDetector(cfg.resolve(cfg.model_path), cfg.device, prof["imgsz"],
                                         cfg.pose_confidence, cfg.allow_model_download)
                dev = self.pose.load()
                self.status["pose"], self.status["device"] = "online", dev.upper()
            except ModelLoadError as e:
                self.pose = None
                self.status["pose"] = f"error: {e}"
        else:
            self.status["pose"] = "disabled"
        say("pose", self.status["pose"])
        self._load_hands()
        say("hands", self.status["hands"])
        self._load_face()
        say("face", self.status["face"])
        if cfg.face_identity:
            try:
                from .face_identity import FaceIdentifier
                d = cfg.resolve(cfg.known_faces_dir)
                if not d.is_dir():
                    self.status["identity"] = f"error: folder {d.name} missing"
                else:
                    self.face_id = FaceIdentifier(str(d))
                    self.status["identity"] = f"online ({self.face_id.summary})"
            except ImportError:
                self.status["identity"] = "error: face_recognition not installed"
        say("identity", self.status["identity"])

    def _load_hands(self) -> None:
        cfg = self.cfg
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        if not cfg.enable_hands:
            self.status["hands"] = "disabled"
            return
        try:
            self.hands = HandLandmarkerModel(cfg.resolve(cfg.hand_model_path), cfg.allow_model_download)
            self.status["hands"] = "online"
        except (OptionalModelError, RuntimeError, ValueError) as e:
            self.status["hands"] = f"error: {e}"

    def _load_face(self) -> None:
        cfg = self.cfg
        if self.face is not None:
            self.face.close()
            self.face = None
        if not cfg.enable_head:
            self.status["face"] = "disabled"
            return
        try:
            self.face = FaceLandmarkerModel(cfg.resolve(cfg.face_model_path), cfg.allow_model_download)
            self.status["face"] = "online"
        except (OptionalModelError, RuntimeError, ValueError) as e:
            self.status["face"] = f"error: {e}"

    def apply_config(self, new: AppConfig) -> None:
        """Apply changed settings without restarting the app."""
        old = self.cfg
        self.cfg = new
        self.tracker.configure(new.tracking_sensitivity)
        self.engine.keypoint_conf = new.keypoint_confidence
        self.engine.min_event_duration = new.min_event_duration
        self.engine.lang = new.language
        reload_pose = (old.model_path != new.model_path or old.device != new.device
                       or old.enable_pose != new.enable_pose)
        if self.pose is not None and not reload_pose:
            self.pose.imgsz = new.profile["imgsz"]
            self.pose.conf = new.pose_confidence
        if reload_pose:
            self.pose = None
            self.tracker.reset()
            self.engine.reset()
            self._prev = []
            if new.enable_pose:
                try:
                    self.pose = PoseDetector(new.resolve(new.model_path), new.device, new.profile["imgsz"],
                                             new.pose_confidence, new.allow_model_download)
                    self.status["device"] = self.pose.load().upper()
                    self.status["pose"] = "online"
                except ModelLoadError as e:
                    self.pose = None
                    self.status["pose"] = f"error: {e}"
            else:
                self.status["pose"] = "disabled"
        if (old.enable_hands != new.enable_hands or old.performance_mode != new.performance_mode
                or (new.enable_hands and self.hands is None)):
            self._load_hands()
        if (old.enable_head != new.enable_head or old.performance_mode != new.performance_mode
                or (new.enable_head and self.face is None)):
            self._load_face()

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, t: float | None = None) -> FrameAnalysis:
        t = time.monotonic() if t is None else t
        self.frame_no += 1
        prof = self.cfg.profile
        h, w = frame.shape[:2]
        fa = FrameAnalysis(frame_size=(w, h))
        if self.pose is None:
            return fa
        # MediaPipe (CPU) runs on crops around each person's head / wrists. The crops come from the
        # previous frame's skeletons so MediaPipe can run in parallel with YOLO (GPU) - one frame lag.
        do_hands = self.hands is not None and self._prev and self.frame_no % prof["hand_every"] == 0
        do_face = self.face is not None and self._prev and self.frame_no % prof["face_every"] == 0
        fut = self._pool.submit(self._mediapipe_job, frame, self._prev, do_hands, do_face) if (do_hands or do_face) else None
        t0 = time.perf_counter()
        try:
            dets = self.pose.detect(frame)
        except Exception as e:  # noqa: BLE001 - keep running, report in status
            self.status["pose"] = f"error: {e}"
            dets = []
        fa.timings_ms["pose"] = (time.perf_counter() - t0) * 1000
        pids, tr_events = self.tracker.update([(d.raw_id, d.box) for d in dets], t)
        hands = faces = None
        if fut is not None:
            hands, faces, timings = fut.result()
            fa.timings_ms.update(timings)
        self._prev = list(zip(dets, pids))

        if self.face_id is not None and dets and self.frame_no % 10 == 0:
            persons = {pid: (d.box, d.xy, d.conf) for d, pid in zip(dets, pids)}
            from .face_identity import assign_faces
            try:
                for pid, name in assign_faces(self.face_id.identify(frame), persons):
                    self.engine.add_name_vote(pid, name)
            except Exception:  # noqa: BLE001
                pass

        t2 = time.perf_counter()
        fa.persons, fa.events = self.engine.process(t, dets, pids, tr_events, hands, faces, (w, h))
        fa.timings_ms["analysis"] = (time.perf_counter() - t2) * 1000
        return fa

    # ------------------------------------------------------------------
    def _mediapipe_job(self, frame, prev, do_hands: bool, do_face: bool):
        prof = self.cfg.profile
        order = sorted(prev, key=lambda dp: -(dp[0].box[3] - dp[0].box[1]))   # largest people first
        timings: dict = {}
        hands = faces = None
        if do_hands:
            t1 = time.perf_counter()
            hands = []
            try:
                for d, pid in order:
                    hands += self._hands_for(frame, d, pid)
                    if len(hands) >= prof["max_hands"]:
                        break
                hands = dedupe_hands(hands)
            except Exception as e:  # noqa: BLE001
                self.status["hands"] = f"error: {e}"
                self.hands, hands = None, None
            timings["hands"] = (time.perf_counter() - t1) * 1000
        if do_face:
            t1 = time.perf_counter()
            faces = []
            try:
                for d, pid in order[:prof["max_faces"]]:
                    f = self._face_for(frame, d, pid)
                    if f is not None:
                        faces.append(f)
            except Exception as e:  # noqa: BLE001
                self.status["face"] = f"error: {e}"
                self.face, faces = None, None
            timings["face"] = (time.perf_counter() - t1) * 1000
        return hands, faces, timings

    def _scale(self, d) -> float:
        kc = self.cfg.keypoint_confidence
        sh = [d.xy[i] for i in (5, 6) if d.conf[i] >= kc]
        hp = [d.xy[i] for i in (11, 12) if d.conf[i] >= kc]
        if sh and hp:
            return float(np.linalg.norm(np.mean(sh, 0) - np.mean(hp, 0)))
        if len(sh) == 2:
            return float(np.linalg.norm(sh[0] - sh[1])) / 0.9
        return float(d.box[3] - d.box[1]) / 3.0

    def _hands_for(self, frame, d, pid) -> list[RawHand]:
        kc = self.cfg.keypoint_confidence
        T = self._scale(d)
        out: list[RawHand] = []
        for side, wr, el in (("left", 9, 7), ("right", 10, 8)):
            if d.conf[wr] < kc:
                continue
            w = d.xy[wr]
            c = w + 0.35 * (w - d.xy[el]) if d.conf[el] >= kc else w
            side_px = max(1.0 * T, 48.0)
            found = self.hands.detect_crop(frame, c[0], c[1], side_px)
            if not found:
                continue
            best = min(found, key=lambda h: float(np.linalg.norm(h.landmarks[0, :2] - w)))
            dw = float(np.linalg.norm(best.landmarks[0, :2] - w))
            if dw <= 0.6 * T:
                best.owner = (pid, side)
                best.match = dw / T
                out.append(best)
        return out

    def _face_for(self, frame, d, pid):
        kc = self.cfg.keypoint_confidence
        pts = [d.xy[i] for i in range(5) if d.conf[i] >= kc]
        if len(pts) < 2 or d.conf[0] < kc:
            return None          # no visible face
        pts = np.array(pts)
        T = self._scale(d)
        span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])))
        side = max(2.4 * span, 0.75 * T, 40.0)
        c = pts.mean(axis=0)
        f = self.face.detect_crop(frame, c[0], c[1] + 0.05 * side, side)
        if f is not None:
            f.owner = pid
        return f

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for m in (self.hands, self.face):
            if m is not None:
                m.close()
