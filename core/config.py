"""Persistent application settings (config.json).

A broken or partially invalid config file never crashes the program: invalid values fall
back to their defaults, an unreadable file is backed up to config.json.bak.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.json"

PERFORMANCE_PROFILES = {
    #            YOLO input   hands every n   faces every n   max hands   max faces
    "eco":      dict(imgsz=320, hand_every=3, face_every=4, max_hands=2, max_faces=1),
    "balanced": dict(imgsz=480, hand_every=2, face_every=2, max_hands=4, max_faces=2),
    "ultra":    dict(imgsz=640, hand_every=1, face_every=1, max_hands=6, max_faces=4),
}
RESOLUTIONS = ["640x480", "960x540", "1280x720", "1920x1080"]
LOG_FORMATS = ["csv", "json", "txt", "sqlite"]


@dataclass
class AppConfig:
    # camera
    camera_index: int = 0
    camera_source: str = ""              # optional video file instead of a webcam
    resolution: str = "1280x720"
    target_fps: int = 30
    mirror: bool = True                  # mirror only the DISPLAY; analysis runs on the original image
    # models
    model_path: str = "models/yolo11n-pose.pt"
    hand_model_path: str = "models/hand_landmarker.task"
    face_model_path: str = "models/face_landmarker.task"
    allow_model_download: bool = True
    device: str = "auto"                 # auto | cpu | mps | cuda
    pose_confidence: float = 0.5
    keypoint_confidence: float = 0.4
    tracking_sensitivity: float = 0.5
    performance_mode: str = "balanced"   # eco | balanced | ultra
    # modules
    enable_pose: bool = True
    enable_hands: bool = True
    enable_head: bool = True
    face_identity: bool = False          # original optional face_recognition naming - OFF by default
    known_faces_dir: str = "known_faces"
    # visualisation
    show_hud: bool = True
    show_skeleton: bool = True
    show_boxes: bool = True
    show_fingers: bool = True
    show_finger_labels: bool = True
    show_angles: bool = False
    show_vectors: bool = True
    show_trajectories: bool = True
    show_head_direction: bool = True
    show_face_mesh: bool = False
    intro_enabled: bool = True
    intro_sound: bool = False
    reduce_effects: bool = False
    language: str = "en"                 # en | de
    # logging
    logging_enabled: bool = False
    log_formats: list = field(default_factory=lambda: ["csv", "json"])
    log_dir: str = "logs"
    min_event_duration: float = 0.5
    export_format: str = "json"
    # control
    gesture_control: bool = False        # OFF by default, must be enabled explicitly
    gesture_hold_seconds: float = 1.0
    # recording (never automatic)
    recording_dir: str = "recordings"

    # ------------------------------------------------------------------
    @property
    def profile(self) -> dict:
        return PERFORMANCE_PROFILES.get(self.performance_mode, PERFORMANCE_PROFILES["balanced"])

    @property
    def frame_size(self) -> tuple[int, int]:
        try:
            w, h = (int(x) for x in self.resolution.lower().split("x"))
            return w, h
        except ValueError:
            return 1280, 720

    def resolve(self, rel: str) -> Path:
        p = Path(os.path.expanduser(rel))
        return p if p.is_absolute() else PROJECT_DIR / p

    def validate(self) -> list[str]:
        """Clamp / repair values in place, return a list of warnings."""
        warn = []
        d = AppConfig()

        def fix(name, ok):
            if not ok:
                warn.append(f"Invalid value for '{name}': {getattr(self, name)!r} -> default {getattr(d, name)!r}")
                setattr(self, name, getattr(d, name))

        fix("performance_mode", self.performance_mode in PERFORMANCE_PROFILES)
        fix("language", self.language in ("en", "de"))
        fix("device", self.device in ("auto", "cpu", "mps", "cuda"))
        fix("export_format", self.export_format in LOG_FORMATS)
        fix("resolution", "x" in str(self.resolution) and all(p.isdigit() for p in str(self.resolution).split("x")))
        fix("target_fps", 1 <= self.target_fps <= 240)
        fix("pose_confidence", 0.05 <= self.pose_confidence <= 0.95)
        fix("keypoint_confidence", 0.05 <= self.keypoint_confidence <= 0.95)
        fix("tracking_sensitivity", 0.0 <= self.tracking_sensitivity <= 1.0)
        fix("min_event_duration", 0.0 <= self.min_event_duration <= 10.0)
        fix("gesture_hold_seconds", 0.2 <= self.gesture_hold_seconds <= 5.0)
        fix("camera_index", 0 <= self.camera_index <= 20)
        self.log_formats = [f for f in self.log_formats if f in LOG_FORMATS] if isinstance(self.log_formats, list) else ["csv"]
        return warn


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> tuple[AppConfig, list[str]]:
    path = Path(path)
    cfg = AppConfig()
    warnings: list[str] = []
    if not path.exists():
        return cfg, warnings
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("top level is not an object")
    except (OSError, ValueError) as e:
        backup = path.with_suffix(".json.bak")
        try:
            shutil.copy(path, backup)
        except OSError:
            pass
        warnings.append(f"Config file unreadable ({e}); using defaults. Backup: {backup.name}")
        return cfg, warnings
    defaults = AppConfig()
    for f in fields(AppConfig):
        if f.name not in data:
            continue
        val, default = data[f.name], getattr(defaults, f.name)
        if isinstance(default, bool):
            ok = isinstance(val, bool)
        elif isinstance(default, int):
            ok = isinstance(val, int) and not isinstance(val, bool)
        elif isinstance(default, float):
            ok = isinstance(val, (int, float)) and not isinstance(val, bool)
            val = float(val) if ok else val
        elif isinstance(default, list):
            ok = isinstance(val, list)
        else:
            ok = isinstance(val, str)
        if ok:
            setattr(cfg, f.name, val)
        else:
            warnings.append(f"Wrong type for '{f.name}' in config, using default.")
    warnings += cfg.validate()
    return cfg, warnings


def save_config(cfg: AppConfig, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
