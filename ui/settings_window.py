"""Settings dialog. Changes are applied live and saved to config.json."""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider,
                               QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from core.config import LOG_FORMATS, PERFORMANCE_PROFILES, RESOLUTIONS, AppConfig

from .theme import DIM, cyber_font


def _check(text: str, value: bool) -> QCheckBox:
    c = QCheckBox(text)
    c.setChecked(value)
    return c


class SettingsDialog(QDialog):
    def __init__(self, cfg: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BUTT3RFLYX // SETTINGS")
        self.setMinimumSize(640, 560)
        self.cfg = copy.deepcopy(cfg)
        lay = QVBoxLayout(self)
        title = QLabel("SYSTEM CONFIGURATION")
        title.setFont(cyber_font(14, bold=True, spacing=5))
        lay.addWidget(title)
        tabs = QTabWidget()
        lay.addWidget(tabs)
        tabs.addTab(self._camera_tab(), "CAMERA")
        tabs.addTab(self._ai_tab(), "AI / MODELS")
        tabs.addTab(self._display_tab(), "DISPLAY")
        tabs.addTab(self._log_tab(), "LOGGING")
        tabs.addTab(self._control_tab(), "CONTROL / PRIVACY")
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    # ------------------------------------------------------------------
    def _camera_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        c = self.cfg
        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 20)
        self.cam_index.setValue(c.camera_index)
        probe = QPushButton("SCAN CAMERAS")
        self.cam_found = QLabel("")
        self.cam_found.setStyleSheet(f"color:{DIM}")
        probe.clicked.connect(self._probe)
        row = QHBoxLayout()
        row.addWidget(self.cam_index)
        row.addWidget(probe)
        row.addWidget(self.cam_found, 1)
        f.addRow("Camera index", row)
        self.cam_source = QLineEdit(c.camera_source)
        self.cam_source.setPlaceholderText("optional: video file instead of webcam")
        b = QPushButton("…")
        b.clicked.connect(lambda: self._pick_file(self.cam_source, "Video (*.mp4 *.mov *.avi *.mkv)"))
        row = QHBoxLayout()
        row.addWidget(self.cam_source)
        row.addWidget(b)
        f.addRow("Video source", row)
        self.resolution = QComboBox()
        self.resolution.addItems(RESOLUTIONS)
        if c.resolution not in RESOLUTIONS:
            self.resolution.addItem(c.resolution)
        self.resolution.setCurrentText(c.resolution)
        f.addRow("Resolution", self.resolution)
        self.fps = QSpinBox()
        self.fps.setRange(5, 120)
        self.fps.setValue(c.target_fps)
        f.addRow("Target FPS", self.fps)
        self.mirror = _check("Mirror display (analysis always on the original image)", c.mirror)
        f.addRow("", self.mirror)
        return w

    def _ai_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        c = self.cfg
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItems(["models/yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt"])
        self.model.setCurrentText(c.model_path)
        b = QPushButton("…")
        b.clicked.connect(lambda: self._pick_file(self.model, "YOLO model (*.pt *.onnx)"))
        row = QHBoxLayout()
        row.addWidget(self.model, 1)
        row.addWidget(b)
        f.addRow("Pose model", row)
        note = QLabel("n = fastest · s/m = more accurate (downloaded automatically on first use)")
        note.setStyleSheet(f"color:{DIM}")
        f.addRow("", note)
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "mps", "cuda"])
        self.device.setCurrentText(c.device)
        f.addRow("Device", self.device)
        self.perf = QComboBox()
        self.perf.addItems(list(PERFORMANCE_PROFILES))
        self.perf.setCurrentText(c.performance_mode)
        f.addRow("Performance mode", self.perf)
        self.pose_conf = QDoubleSpinBox()
        self.pose_conf.setRange(0.1, 0.9)
        self.pose_conf.setSingleStep(0.05)
        self.pose_conf.setValue(c.pose_confidence)
        f.addRow("Person confidence threshold", self.pose_conf)
        self.kp_conf = QDoubleSpinBox()
        self.kp_conf.setRange(0.1, 0.9)
        self.kp_conf.setSingleStep(0.05)
        self.kp_conf.setValue(c.keypoint_confidence)
        f.addRow("Keypoint confidence threshold", self.kp_conf)
        self.sens = QSlider(Qt.Orientation.Horizontal)
        self.sens.setRange(0, 100)
        self.sens.setValue(int(c.tracking_sensitivity * 100))
        f.addRow("Tracking sensitivity (ID keeping)", self.sens)
        self.min_dur = QDoubleSpinBox()
        self.min_dur.setRange(0.0, 5.0)
        self.min_dur.setSingleStep(0.1)
        self.min_dur.setValue(c.min_event_duration)
        f.addRow("Min. event duration (s)", self.min_dur)
        g = QGroupBox("MODULES")
        gl = QVBoxLayout(g)
        self.en_pose = _check("Body pose (YOLO11-pose)", c.enable_pose)
        self.en_hands = _check("Hand & finger tracking (MediaPipe Hand Landmarker)", c.enable_hands)
        self.en_head = _check("Head pose & gaze (MediaPipe Face Landmarker)", c.enable_head)
        for x in (self.en_pose, self.en_hands, self.en_head):
            gl.addWidget(x)
        f.addRow(g)
        return w

    def _display_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        c = self.cfg
        self.disp = {}
        for key, text in (("show_hud", "HUD overlay"), ("show_skeleton", "Skeleton"), ("show_boxes", "Tracking frames"),
                          ("show_fingers", "Hand landmarks"), ("show_finger_labels", "Finger labels (THUMB, INDEX, ...)"),
                          ("show_angles", "Joint angles"), ("show_vectors", "Motion vectors"),
                          ("show_trajectories", "Trajectories"), ("show_head_direction", "Head direction arrow"),
                          ("show_face_mesh", "Face landmarks"), ("intro_enabled", "Intro animation on start"),
                          ("intro_sound", "Synthetic intro sound"), ("reduce_effects", "Reduce animation effects")):
            cb = _check(text, getattr(c, key))
            self.disp[key] = cb
            lay.addWidget(cb)
        row = QHBoxLayout()
        row.addWidget(QLabel("Language / Sprache"))
        self.lang = QComboBox()
        self.lang.addItems(["en", "de"])
        self.lang.setCurrentText(c.language)
        row.addWidget(self.lang)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)
        return w

    def _log_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        c = self.cfg
        self.log_on = _check("Write recognised events to files", c.logging_enabled)
        f.addRow(self.log_on)
        self.log_fmt = {}
        row = QHBoxLayout()
        for fmt in LOG_FORMATS:
            cb = _check(fmt.upper(), fmt in c.log_formats)
            self.log_fmt[fmt] = cb
            row.addWidget(cb)
        f.addRow("Formats", row)
        self.log_dir = QLineEdit(c.log_dir)
        f.addRow("Log folder", self.log_dir)
        self.export_fmt = QComboBox()
        self.export_fmt.addItems(LOG_FORMATS)
        self.export_fmt.setCurrentText(c.export_format)
        f.addRow("Export format", self.export_fmt)
        self.rec_dir = QLineEdit(c.recording_dir)
        f.addRow("Recording folder", self.rec_dir)
        n = QLabel("Video is NEVER recorded automatically. Recording only runs while you\n"
                   "explicitly start it (R key / REC button) and a red REC indicator is shown.")
        n.setStyleSheet(f"color:{DIM}")
        f.addRow(n)
        return w

    def _control_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        c = self.cfg
        self.gesture = _check("Enable gesture control", c.gesture_control)
        f.addRow(self.gesture)
        self.hold = QDoubleSpinBox()
        self.hold.setRange(0.3, 5.0)
        self.hold.setSingleStep(0.1)
        self.hold.setValue(c.gesture_hold_seconds)
        f.addRow("Hold time (s)", self.hold)
        n = QLabel("THUMBS UP = confirm · OPEN HAND = pause · FIST = resume\n"
                   "PEACE = toggle skeleton · HANDS ABOVE HEAD = open settings")
        n.setStyleSheet(f"color:{DIM}")
        f.addRow(n)
        g = QGroupBox("PRIVACY")
        gl = QVBoxLayout(g)
        self.face_id = _check("Face identity (names from known_faces/, needs 'face_recognition')", c.face_identity)
        gl.addWidget(self.face_id)
        self.known = QLineEdit(c.known_faces_dir)
        gl.addWidget(self.known)
        n2 = QLabel("Off by default. Encodings are kept in memory only - nothing biometric is saved.")
        n2.setStyleSheet(f"color:{DIM}")
        gl.addWidget(n2)
        f.addRow(g)
        return w

    # ------------------------------------------------------------------
    def _probe(self) -> None:
        from core.camera_manager import probe_cameras
        self.cam_found.setText("scanning…")
        self.repaint()
        found = probe_cameras(6)
        self.cam_found.setText("found: " + (", ".join(map(str, found)) if found else "none"))

    def _pick_file(self, target, flt: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", flt)
        if path:
            if isinstance(target, QComboBox):
                target.setCurrentText(path)
            else:
                target.setText(path)

    def result_config(self) -> AppConfig:
        c = self.cfg
        c.camera_index = self.cam_index.value()
        c.camera_source = self.cam_source.text().strip()
        c.resolution = self.resolution.currentText()
        c.target_fps = self.fps.value()
        c.mirror = self.mirror.isChecked()
        c.model_path = self.model.currentText().strip() or c.model_path
        c.device = self.device.currentText()
        c.performance_mode = self.perf.currentText()
        c.pose_confidence = round(self.pose_conf.value(), 2)
        c.keypoint_confidence = round(self.kp_conf.value(), 2)
        c.tracking_sensitivity = self.sens.value() / 100
        c.min_event_duration = round(self.min_dur.value(), 2)
        c.enable_pose = self.en_pose.isChecked()
        c.enable_hands = self.en_hands.isChecked()
        c.enable_head = self.en_head.isChecked()
        for key, cb in self.disp.items():
            setattr(c, key, cb.isChecked())
        c.language = self.lang.currentText()
        c.logging_enabled = self.log_on.isChecked()
        c.log_formats = [k for k, cb in self.log_fmt.items() if cb.isChecked()]
        c.log_dir = self.log_dir.text().strip() or "logs"
        c.export_format = self.export_fmt.currentText()
        c.recording_dir = self.rec_dir.text().strip() or "recordings"
        c.gesture_control = self.gesture.isChecked()
        c.gesture_hold_seconds = round(self.hold.value(), 2)
        c.face_identity = self.face_id.isChecked()
        c.known_faces_dir = self.known.text().strip() or "known_faces"
        c.validate()
        return c
