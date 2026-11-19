"""Background thread: camera -> AI pipeline -> HUD rendering -> logging.

The GUI thread only receives finished QImages and plain display data, so the interface
stays responsive no matter how heavy the models are.
"""
from __future__ import annotations

import copy
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime

from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal
from PySide6.QtGui import QImage

from core.camera_manager import CameraManager
from core.config import AppConfig
from core.gesture_control import GestureController
from core.pipeline import FpsMeter, VisionPipeline
from logging_system.event_logger import EventLogger

from .hud_renderer import HudRenderer, qimage_to_bgr
from .view_model import PersonView, build_view


@dataclass
class FramePacket:
    image: QImage
    persons: list[PersonView] = field(default_factory=list)
    fps: float = 0.0
    timings: dict = field(default_factory=dict)
    camera_ok: bool = True
    camera_error: str | None = None
    paused: bool = False
    recording: bool = False
    status: dict = field(default_factory=dict)
    log_count: int = 0


class VisionWorker(QThread):
    frame_ready = Signal(object)          # FramePacket
    events_ready = Signal(list)           # [EventRecord]
    load_progress = Signal(str, str)      # module, status
    models_ready = Signal(dict)
    camera_state = Signal(bool, str)
    command = Signal(str)                 # gesture-control commands
    error = Signal(str)

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self._mutex = QMutex()
        self._pending_cfg: AppConfig | None = None
        self._running = True
        self.paused = False
        self._rec_request: bool | None = None
        self.recording = False
        self._writer = None
        self.rec_path = None
        self.logger: EventLogger | None = None
        self.ready = False
        self.status: dict = {}

    # ---- called from the GUI thread ----
    def update_config(self, cfg: AppConfig) -> None:
        with QMutexLocker(self._mutex):
            self._pending_cfg = copy.deepcopy(cfg)

    def set_paused(self, paused: bool) -> None:
        self.paused = paused

    def set_recording(self, on: bool) -> None:
        self._rec_request = on

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._run()
        except Exception:  # noqa: BLE001
            self.error.emit("Worker crashed:\n" + traceback.format_exc())

    def _run(self) -> None:
        cfg = self.cfg
        cam = CameraManager(cfg.camera_index, cfg.camera_source, cfg.frame_size, cfg.target_fps)
        self.load_progress.emit("camera", "opening")
        cam_ok = cam.open()
        self.load_progress.emit("camera", "online" if cam_ok else f"error: {cam.error}")
        self.camera_state.emit(cam_ok, cam.error or "")
        pipe = VisionPipeline(cfg)
        pipe.load_models(lambda mod, st: self.load_progress.emit(mod, st))
        self.status = dict(pipe.status)
        self.ready = True
        self.models_ready.emit(dict(pipe.status))
        renderer = HudRenderer(cfg)
        fps = FpsMeter()
        gestures = GestureController(cfg.gesture_hold_seconds)
        self._set_logging(cfg)
        t_start = time.monotonic()
        last_cam_ok = cam_ok
        try:
            while self._running:
                loop_t0 = time.perf_counter()
                with QMutexLocker(self._mutex):
                    new_cfg, self._pending_cfg = self._pending_cfg, None
                if new_cfg is not None:
                    cam = self._apply_config(cfg, new_cfg, cam, pipe, renderer, gestures)
                    cfg = self.cfg = new_cfg
                if self._rec_request is not None:
                    self._toggle_recording(self._rec_request, cfg)
                    self._rec_request = None

                frame = cam.read()
                ok = frame is not None
                if ok != last_cam_ok:
                    self.camera_state.emit(ok, cam.error or "")
                    last_cam_ok = ok
                if frame is None:
                    self.msleep(50)
                    continue
                now = time.monotonic()
                fa = None
                if not self.paused:
                    fa = pipe.process(frame, now)
                    if fa.events:
                        self.events_ready.emit(fa.events)
                        if self.logger is not None:
                            self.logger.log(fa.events)
                progress = None
                if cfg.gesture_control and fa is not None or (cfg.gesture_control and self.paused):
                    keys = set()
                    if fa is not None and fa.persons:
                        keys = fa.persons[0].label_keys()
                    elif self.paused:
                        # while paused, a quick pose-only check so "fist" can resume
                        fa_tmp = pipe.process(frame, now)
                        keys = fa_tmp.persons[0].label_keys() if fa_tmp.persons else set()
                    cmd = gestures.update(keys, now)
                    if gestures.current:
                        from core.gesture_control import COMMANDS
                        progress = (COMMANDS[gestures.current], gestures.progress(now))
                    if cmd:
                        self.command.emit(cmd)
                img = renderer.render(frame, fa, now - t_start, self.paused, progress)
                if self.recording and self._writer is not None:
                    self._writer.write(qimage_to_bgr(img))
                self.status = dict(pipe.status)
                packet = FramePacket(
                    image=img, fps=fps.tick(),
                    persons=[build_view(s, cfg.language) for s in fa.persons] if fa is not None else [],
                    timings=dict(fa.timings_ms) if fa is not None else {}, camera_ok=True,
                    paused=self.paused, recording=self.recording, status=self.status,
                    log_count=self.logger.count if self.logger else 0)
                self.frame_ready.emit(packet)
                # frame pacing (target FPS); camera read normally limits this already
                budget = 1.0 / max(cfg.target_fps, 1)
                spent = time.perf_counter() - loop_t0
                if spent < budget and cam.is_file:
                    self.msleep(int((budget - spent) * 1000))
        finally:
            self._toggle_recording(False, cfg)
            if self.logger is not None:
                self.logger.close()
            cam.release()
            pipe.close()

    def _set_logging(self, cfg: AppConfig) -> None:
        if self.logger is not None:
            self.logger.close()
            self.logger = None
        if cfg.logging_enabled and cfg.log_formats:
            try:
                self.logger = EventLogger(cfg.resolve(cfg.log_dir), cfg.log_formats)
            except OSError as e:
                self.error.emit(f"Logging could not be started: {e}")

    def _apply_config(self, old: AppConfig, new: AppConfig, cam, pipe, renderer, gestures):
        renderer.cfg = new
        gestures.hold_seconds = new.gesture_hold_seconds
        if (old.camera_index, old.camera_source, old.resolution, old.target_fps) != \
                (new.camera_index, new.camera_source, new.resolution, new.target_fps):
            cam.release()
            cam = CameraManager(new.camera_index, new.camera_source, new.frame_size, new.target_fps)
            ok = cam.open()
            self.camera_state.emit(ok, cam.error or "")
            pipe.tracker.reset()
            pipe.engine.reset()
            if pipe.pose is not None:
                pipe.pose.reset_tracking()
        pipe.apply_config(new)
        if (old.logging_enabled, old.log_formats, old.log_dir) != (new.logging_enabled, new.log_formats, new.log_dir):
            self._set_logging(new)
        self.models_ready.emit(dict(pipe.status))
        return cam

    def _toggle_recording(self, on: bool, cfg: AppConfig) -> None:
        import cv2
        if on and not self.recording:
            d = cfg.resolve(cfg.recording_dir)
            d.mkdir(parents=True, exist_ok=True)
            self.rec_path = d / f"butt3rflyx_{datetime.now():%Y%m%d_%H%M%S}.mp4"
            self._pending_writer = True
            self.recording = True
            self._writer = None
            self._rec_fps = max(cfg.target_fps, 10)
        elif not on and self.recording:
            self.recording = False
            if self._writer is not None:
                self._writer.release()
            self._writer = None
        if self.recording and self._writer is None:
            # writer is created lazily with the size of the first rendered frame
            class _Lazy:
                def __init__(s, path, fps):
                    s.path, s.fps, s.w = path, fps, None

                def write(s, frame):
                    if s.w is None:
                        h, w = frame.shape[:2]
                        s.w = cv2.VideoWriter(str(s.path), cv2.VideoWriter_fourcc(*"mp4v"), s.fps, (w, h))
                    s.w.write(frame)

                def release(s):
                    if s.w is not None:
                        s.w.release()
            self._writer = _Lazy(self.rec_path, self._rec_fps)
