"""Main dashboard window."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QPolygonF, QShortcut
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSplitter,
                               QVBoxLayout, QWidget)

from core.config import AppConfig, save_config
from logging_system.export_manager import export_events

from .cyber_intro import butterfly_curve
from .dashboard import ModuleStatus, PersonPanel, Timeline, TopBar, VideoView, section_title
from .i18n import tr
from .settings_window import SettingsDialog
from .theme import CYAN, DIM, GREEN, RED, VIOLET, YELLOW
from .worker import FramePacket, VisionWorker

HELP = """KEYBOARD
  SPACE  pause / resume tracking        S  settings
  R      start / stop video recording   E  export events
  L      logging on / off               G  gesture control on / off
  K      skeleton    H  HUD    F  hand landmarks    A  joint angles
  V      motion vectors    T  trajectories    M  mirror display
  F11    fullscreen    F1  this help    Q / ESC  quit"""


def make_icon(size: int = 256) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(4, 5, 12))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pts = butterfly_curve(1500) * size * 0.4 + size / 2
    from PySide6.QtCore import QPointF
    poly = QPolygonF([QPointF(float(x), float(y)) for x, y in pts])
    p.setPen(QPen(QColor(VIOLET), size / 40))
    p.drawPolyline(poly)
    p.setPen(QPen(QColor(CYAN), size / 110))
    p.drawPolyline(poly)
    p.end()
    return QIcon(pm)


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig, config_path: Path, worker: VisionWorker) -> None:
        super().__init__()
        self.cfg = cfg
        self.config_path = config_path
        self.worker = worker
        self.history: list = []
        self.setWindowTitle("BUTT3RFLYX // ADVANCED AI MOTION TRACKING SYSTEM")
        self.setWindowIcon(make_icon())
        self.resize(1500, 900)
        self.setMinimumSize(1120, 700)

        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.top = TopBar()
        v.addWidget(self.top)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(10, 8, 10, 8)
        bl.setSpacing(6)
        v.addWidget(body, 1)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(4)
        bl.addWidget(split, 1)
        self.video = VideoView()
        split.addWidget(self.video)

        right = QWidget()
        right.setMinimumWidth(360)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        self.modules = ModuleStatus()
        rl.addWidget(self.modules)
        self.analysis_title = section_title(tr("live_analysis", cfg.language))
        rl.addWidget(self.analysis_title)
        self.persons = PersonPanel()
        rl.addWidget(self.persons, 1)
        self.exp_note = QLabel(tr("exp_note", cfg.language))
        self.exp_note.setStyleSheet(f"color:{YELLOW}; font-size:10px;")
        rl.addWidget(self.exp_note)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([1080, 420])

        bottom = QHBoxLayout()
        tl_box = QVBoxLayout()
        self.events_title = section_title(tr("events", cfg.language), VIOLET)
        tl_box.addWidget(self.events_title)
        self.timeline = Timeline()
        self.timeline.setFixedHeight(150)
        tl_box.addWidget(self.timeline)
        bottom.addLayout(tl_box, 1)
        btns = QVBoxLayout()
        btns.addSpacing(22)
        self.b_settings = QPushButton("⚙  " + tr("settings", cfg.language))
        self.b_pause = QPushButton("❚❚  " + tr("pause", cfg.language))
        self.b_pause.setCheckable(True)
        self.b_rec = QPushButton("●  " + tr("record", cfg.language))
        self.b_rec.setCheckable(True)
        self.b_log = QPushButton("≣  " + tr("logging", cfg.language))
        self.b_log.setCheckable(True)
        self.b_log.setChecked(cfg.logging_enabled)
        self.b_export = QPushButton("⇩  " + tr("export", cfg.language))
        for b in (self.b_settings, self.b_pause, self.b_rec, self.b_log, self.b_export):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setMinimumWidth(150)
            btns.addWidget(b)
        btns.addStretch(1)
        bottom.addLayout(btns)
        bl.addLayout(bottom)

        self.b_settings.clicked.connect(self.open_settings)
        self.b_pause.toggled.connect(self.set_paused)
        self.b_rec.toggled.connect(self.set_recording)
        self.b_log.toggled.connect(self.set_logging)
        self.b_export.clicked.connect(self.export)
        self._shortcuts()

        worker.frame_ready.connect(self.on_frame)
        worker.events_ready.connect(self.on_events)
        worker.models_ready.connect(self.on_models)
        worker.load_progress.connect(self.on_progress)
        worker.camera_state.connect(self.on_camera)
        worker.command.connect(self.on_command)
        worker.error.connect(self.on_error)
        self._apply_ui_cfg()

    # ------------------------------------------------------------------
    def _shortcuts(self) -> None:
        def sc(key, fn):
            s = QShortcut(QKeySequence(key), self)
            s.activated.connect(fn)

        sc("Space", lambda: self.b_pause.toggle())
        sc("S", self.open_settings)
        sc("R", lambda: self.b_rec.toggle())
        sc("L", lambda: self.b_log.toggle())
        sc("E", self.export)
        sc("G", lambda: self._toggle("gesture_control"))
        sc("K", lambda: self._toggle("show_skeleton"))
        sc("H", lambda: self._toggle("show_hud"))
        sc("F", lambda: self._toggle("show_fingers"))
        sc("A", lambda: self._toggle("show_angles"))
        sc("V", lambda: self._toggle("show_vectors"))
        sc("T", lambda: self._toggle("show_trajectories"))
        sc("M", lambda: self._toggle("mirror"))
        sc("F11", self.toggle_fullscreen)
        sc("F1", lambda: QMessageBox.information(self, "BUTT3RFLYX // HELP", HELP))
        sc("Q", self.close)
        sc("Escape", lambda: self.showNormal() if self.isFullScreen() else self.close())

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def _toggle(self, key: str) -> None:
        setattr(self.cfg, key, not getattr(self.cfg, key))
        self.timeline.add_system(f"{key.replace('_', ' ').upper()}: {'ON' if getattr(self.cfg, key) else 'OFF'}", CYAN)
        self._push_cfg()

    def _push_cfg(self) -> None:
        self.worker.update_config(self.cfg)
        self._apply_ui_cfg()
        try:
            save_config(self.cfg, self.config_path)
        except OSError as e:
            self.timeline.add_system(f"CONFIG NOT SAVED: {e}", RED)

    def _apply_ui_cfg(self) -> None:
        lang = self.cfg.language
        self.top.lang = self.video.lang = self.modules.lang = lang
        self.top.mode = self.cfg.performance_mode.upper()
        self.top.logging = self.cfg.logging_enabled
        self.top.gesture_ctrl = self.cfg.gesture_control
        self.video.reduce = self.cfg.reduce_effects
        self.analysis_title.setText(tr("live_analysis", lang))
        self.events_title.setText(tr("events", lang))
        self.exp_note.setText(tr("exp_note", lang))

    # ------------------------------------------------------------------
    def on_frame(self, pk: FramePacket) -> None:
        self.video.message = ""
        self.video.set_image(pk.image)
        self.top.fps = pk.fps
        self.top.persons = len(pk.persons)
        self.top.paused = pk.paused
        self.top.recording = pk.recording
        self.top.camera_ok = True
        self.top.device = pk.status.get("device", "-")
        self.persons.update_views(pk.persons, self.cfg.language)
        self.modules.set_data(pk.status, pk.timings, pk.log_count)

    def on_events(self, events: list) -> None:
        self.history.extend(events)
        if len(self.history) > 50000:
            del self.history[:10000]
        self.timeline.add_events(events)

    def on_models(self, status: dict) -> None:
        self.top.ready = status.get("pose", "").startswith("online")
        self.top.device = status.get("device", "-")
        self.modules.set_data(status, {}, self.modules.log_count)
        for k in ("pose", "hands", "face", "identity"):
            st = status.get(k, "")
            if st.startswith("error"):
                self.timeline.add_system(f"{k.upper()} MODULE: {st}", RED)
        if not self.top.ready:
            self.video.message = ("Pose model unavailable - see AI MODULES. " + status.get("pose", ""))

    def on_progress(self, module: str, status: str) -> None:
        if status.startswith("error"):
            self.timeline.add_system(f"{module.upper()}: {status}", RED)

    def on_camera(self, ok: bool, msg: str) -> None:
        self.top.camera_ok = ok
        if not ok:
            self.video.image = None
            self.video.message = msg
            self.timeline.add_system(msg or "CAMERA OFFLINE", RED)
        else:
            self.timeline.add_system("CAMERA ONLINE", GREEN)

    def on_command(self, cmd: str) -> None:
        self.timeline.add_system(f"GESTURE COMMAND: {cmd.replace('_', ' ').upper()}", VIOLET)
        if cmd == "pause_tracking" and not self.b_pause.isChecked():
            self.b_pause.setChecked(True)
        elif cmd == "resume_tracking" and self.b_pause.isChecked():
            self.b_pause.setChecked(False)
        elif cmd == "toggle_skeleton":
            self._toggle("show_skeleton")
        elif cmd == "open_settings":
            self.open_settings()

    def on_error(self, msg: str) -> None:
        self.timeline.add_system(msg.splitlines()[0][:160], RED)
        print(msg)

    # ------------------------------------------------------------------
    def set_paused(self, on: bool) -> None:
        self.worker.set_paused(on)
        self.b_pause.setText(("▶  RESUME" if on else "❚❚  " + tr("pause", self.cfg.language)))

    def set_recording(self, on: bool) -> None:
        self.worker.set_recording(on)
        if on:
            self.timeline.add_system("VIDEO RECORDING STARTED (explicit user action)", RED)
        else:
            path = getattr(self.worker, "rec_path", None)
            self.timeline.add_system(f"RECORDING STOPPED{(' -> ' + str(path)) if path else ''}", YELLOW)

    def set_logging(self, on: bool) -> None:
        if self.cfg.logging_enabled != on:
            self.cfg.logging_enabled = on
            self.timeline.add_system(f"LOGGING {'ON -> ' + self.cfg.log_dir + '/' if on else 'OFF'}", GREEN if on else DIM)
            self._push_cfg()

    def export(self) -> None:
        if not self.history:
            QMessageBox.information(self, "EXPORT", "No events recorded in this session yet.")
            return
        fmt = self.cfg.export_format
        ext = {"sqlite": "db"}.get(fmt, fmt)
        default = str(self.cfg.resolve(self.cfg.log_dir) / f"export_{datetime.now():%Y%m%d_%H%M%S}.{ext}")
        path, _ = QFileDialog.getSaveFileName(self, "Export events", default,
                                              "JSON (*.json);;CSV (*.csv);;Text (*.txt);;SQLite (*.db)")
        if not path:
            return
        try:
            out = export_events(self.history, path)
            self.timeline.add_system(f"EXPORTED {len(self.history)} EVENTS -> {out}", GREEN)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "EXPORT FAILED", str(e))

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            self.cfg = dlg.result_config()
            self.b_log.blockSignals(True)
            self.b_log.setChecked(self.cfg.logging_enabled)
            self.b_log.blockSignals(False)
            self.timeline.add_system("SETTINGS APPLIED & SAVED", GREEN)
            self._push_cfg()

    def closeEvent(self, e) -> None:  # noqa: N802
        self.worker.stop()
        self.worker.wait(4000)
        try:
            save_config(self.cfg, self.config_path)
        except OSError:
            pass
        super().closeEvent(e)
