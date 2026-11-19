"""Tracking IDs, logging/export, configuration, gesture control and error handling."""
import csv
import json
import sqlite3

import numpy as np
import pytest

from core.action_engine import EventRecord
from core.config import AppConfig, load_config, save_config
from core.gesture_control import GestureController
from core.person_tracker import PersonTracker
from logging_system.event_logger import EventLogger
from logging_system.export_manager import export_events

BOX = np.array([100, 100, 200, 400], float)


# ---------------- tracking ----------------
def test_stable_id_for_same_raw_id():
    tr = PersonTracker()
    pids, ev = tr.update([(7, BOX)], 0.0)
    assert pids == [1] and ev == [(1, "entered")]
    pids, ev = tr.update([(7, BOX + 5)], 0.1)
    assert pids == [1] and ev == []


def test_raw_id_switch_keeps_display_id():
    tr = PersonTracker()
    tr.update([(7, BOX)], 0.0)
    pids, ev = tr.update([(8, BOX + 3)], 0.033)          # ByteTrack re-assigned a new raw id
    assert pids == [1] and ev == [(1, "rematched")]      # same display ID, analysis history is reset


def test_reacquire_after_short_loss():
    tr = PersonTracker(reid_timeout=3.0)
    tr.update([(7, BOX)], 0.0)
    tr.update([], 0.5)
    pids, ev = tr.update([(9, BOX + 10)], 1.5)
    assert pids == [1] and (1, "reacquired") in ev


def test_far_detection_gets_new_id_and_timeout_leaves():
    tr = PersonTracker(reid_timeout=1.0)
    tr.update([(7, BOX)], 0.0)
    pids, ev = tr.update([(7, BOX), (8, BOX + np.array([600, 0, 600, 0]))], 0.1)
    assert pids == [1, 2] and (2, "entered") in ev
    tr.update([(8, BOX + np.array([600, 0, 600, 0]))], 0.5)
    _, ev = tr.update([(8, BOX + np.array([600, 0, 600, 0]))], 1.3)
    assert (1, "left") in ev


def test_two_people_never_share_an_id():
    tr = PersonTracker()
    pids, _ = tr.update([(None, BOX), (None, BOX + 2)], 0.0)
    assert len(set(pids)) == 2
    pids, _ = tr.update([(None, BOX), (None, BOX + 2)], 0.03)
    assert len(set(pids)) == 2


def test_sensitivity_configuration():
    tr = PersonTracker()
    tr.configure(0.0)
    low = tr.reid_timeout
    tr.configure(1.0)
    assert tr.reid_timeout > low


# ---------------- logging / export ----------------
def _ev(t=1.0, label="standing", pid=1, conf=0.9):
    return EventRecord(timestamp="2026-11-19T17:30:05.000", person_id=pid, event_type="state", label=label,
                       text=f"PERSON {pid:03d} IS {label.upper()}", posture="standing", movement="walking",
                       head_direction="head_left", right_hand="pointing_right", confidence=conf, duration_s=1.5, t=t)


def test_event_logger_all_formats(tmp_path):
    lg = EventLogger(tmp_path, ["csv", "json", "txt", "sqlite"])
    n = lg.log([_ev(), _ev(), _ev(label="walking")])          # exact duplicate in the same frame is dropped
    lg.close()
    assert n == 2
    rows = list(csv.DictReader(open(next(tmp_path.glob("*.csv")))))
    assert rows[0]["label"] == "standing" and float(rows[0]["confidence"]) == 0.9
    lines = [json.loads(x) for x in open(next(tmp_path.glob("*.jsonl")))]
    assert lines[1]["label"] == "walking" and lines[0]["person_id"] == 1
    assert "PERSON 001" in next(tmp_path.glob("*.txt")).read_text()
    con = sqlite3.connect(tmp_path / "butt3rflyx.db")
    assert con.execute("select count(*) from events").fetchone()[0] == 2


def test_export_roundtrip(tmp_path):
    evs = [_ev(t=i, pid=i % 2 + 1) for i in range(5)]
    data = json.loads(export_events(evs, tmp_path / "x.json").read_text())
    assert len(data) == 5 and data[0]["head_direction"] == "head_left"
    assert len(list(csv.DictReader(open(export_events(evs, tmp_path / "x.csv"))))) == 5
    assert export_events(evs, tmp_path / "x.txt").read_text().count("\n") == 5
    con = sqlite3.connect(export_events(evs, tmp_path / "x.db"))
    assert con.execute("select count(*) from events").fetchone()[0] == 5
    with pytest.raises(ValueError):
        export_events(evs, tmp_path / "x.xyz")


# ---------------- configuration ----------------
def test_config_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    cfg = AppConfig(performance_mode="eco", language="de", log_formats=["txt"])
    save_config(cfg, p)
    cfg2, warn = load_config(p)
    assert warn == [] and cfg2.performance_mode == "eco" and cfg2.language == "de" and cfg2.log_formats == ["txt"]


def test_config_missing_file_gives_defaults(tmp_path):
    cfg, warn = load_config(tmp_path / "nope.json")
    assert cfg == AppConfig() and warn == []


def test_config_corrupt_file_is_backed_up(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{ this is not json")
    cfg, warn = load_config(p)
    assert cfg == AppConfig() and warn and (tmp_path / "c.json.bak").exists()


def test_config_invalid_values_are_repaired(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"performance_mode": "turbo", "pose_confidence": 5, "mirror": "yes",
                             "target_fps": 30, "log_formats": ["csv", "pdf"], "unknown_key": 1}))
    cfg, warn = load_config(p)
    assert cfg.performance_mode == "balanced" and cfg.pose_confidence == 0.5
    assert cfg.mirror is True and cfg.log_formats == ["csv"]
    assert len(warn) >= 3


def test_privacy_defaults():
    c = AppConfig()
    assert c.face_identity is False and c.gesture_control is False and c.logging_enabled is False


# ---------------- gesture control ----------------
def test_gesture_needs_hold_time_and_cooldown():
    gc = GestureController(hold_seconds=1.0, cooldown=2.0)
    assert gc.update({"thumbs_up"}, 0.0) is None
    assert gc.update({"thumbs_up"}, 0.5) is None
    assert gc.update({"thumbs_up"}, 1.05) == "confirm"
    assert gc.update({"thumbs_up"}, 1.5) is None              # fires only once per hold
    gc.update(set(), 1.6)
    gc.update({"fist"}, 1.7)
    assert gc.update({"fist"}, 2.8) is None                    # still in cooldown
    assert gc.update({"fist"}, 3.2) == "resume_tracking"


def test_gesture_switch_resets_timer():
    gc = GestureController(hold_seconds=1.0)
    gc.update({"open_hand"}, 0.0)
    gc.update({"peace"}, 0.9)
    assert gc.update({"peace"}, 1.5) is None
    assert gc.update({"peace"}, 2.0) == "toggle_skeleton"


# ---------------- error handling ----------------
def test_missing_pose_model_raises_clean_error(tmp_path):
    from core.pose_detector import ModelLoadError, PoseDetector
    det = PoseDetector(tmp_path / "missing.pt", allow_download=False)
    with pytest.raises(ModelLoadError):
        det.load()
    assert det.detect(np.zeros((10, 10, 3), np.uint8)) == []


def test_missing_mediapipe_model_is_optional(tmp_path):
    from core.mediapipe_models import ensure_model, OptionalModelError
    with pytest.raises(OptionalModelError):
        ensure_model(tmp_path / "hand_landmarker.task", allow_download=False)


def test_pipeline_survives_missing_models(tmp_path):
    from core.pipeline import VisionPipeline
    cfg = AppConfig(model_path=str(tmp_path / "x.pt"), hand_model_path=str(tmp_path / "h.task"),
                    face_model_path=str(tmp_path / "f.task"), allow_model_download=False)
    p = VisionPipeline(cfg)
    p.load_models()
    assert p.status["pose"].startswith("error") and p.status["hands"].startswith("error")
    fa = p.process(np.zeros((480, 640, 3), np.uint8))
    assert fa.persons == []
    p.close()


def test_camera_missing_video_file():
    from core.camera_manager import CameraManager
    cam = CameraManager(source="/nonexistent/video.mp4")
    assert cam.open() is False and "could not be opened" in cam.error
    assert cam.read() is None
