"""Real models on the sample photos shipped with ultralytics, plus offscreen UI rendering.

Skipped automatically when the models or sample images are not available.
These still are NOT a substitute for a live webcam test (see docs/limitations.md).
"""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _asset(name):
    try:
        import ultralytics
    except ImportError:
        return None
    p = Path(ultralytics.__file__).parent / "assets" / name
    return p if p.exists() else None


@pytest.fixture(scope="module")
def pipeline():
    if not (ROOT / "models" / "yolo11n-pose.pt").exists():
        pytest.skip("pose model missing")
    from core.config import AppConfig
    from core.pipeline import VisionPipeline
    cfg = AppConfig(performance_mode="ultra", allow_model_download=False)
    p = VisionPipeline(cfg)
    p.load_models()
    yield p
    p.close()


def _run(p, name, n=6):
    import cv2
    path = _asset(name)
    if path is None:
        pytest.skip("sample image missing")
    img = cv2.imread(str(path))
    p.tracker.reset()
    p.engine.reset()
    p.pose.reset_tracking()
    p._prev = []
    fa = None
    for i in range(n):
        fa = p.process(img, 100.0 + i / 30)
    return img, fa


def test_bus_people_detected_standing(pipeline):
    _, fa = _run(pipeline, "bus.jpg")
    assert len(fa.persons) >= 3
    assert sum(1 for s in fa.persons if s.motion.posture and s.motion.posture.key == "standing") >= 3
    assert len({s.pid for s in fa.persons}) == len(fa.persons)


def test_zidane_head_and_pointing_hand(pipeline):
    _, fa = _run(pipeline, "zidane.jpg")
    assert len(fa.persons) == 2
    # both men turn their head to their own left (towards the image right)
    for s in fa.persons:
        assert s.head is not None and s.head.yaw is not None and s.head.yaw > 10
    hands = [h for s in fa.persons for h in s.hands.values() if h is not None]
    if pipeline.hands is not None:
        assert any(h.primary is not None and h.primary.key.startswith("pointing") for h in hands)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_intro_renders_all_phases(qapp):
    from ui.cyber_intro import CyberIntro
    st = {"pose": "online", "hands": "online", "face": "disabled", "camera": "error: none"}
    w = CyberIntro(st, reduce_effects=False)
    w.timer.stop()
    w.resize(800, 450)
    for t in (0.5, 1.5, 3.0, 4.0, 4.9, 5.5, 6.2, 7.0, 7.7):
        w.t = t
        img = w.grab().toImage()
        assert not img.isNull() and img.width() == 800
    assert w.ready()
    w.skip()


def test_intro_sound_synthesis(tmp_path):
    from ui.cyber_intro import write_intro_sound
    p = tmp_path / "s.wav"
    write_intro_sound(p, sr=8000)
    assert p.stat().st_size > 1000


def test_hud_and_dashboard_render(qapp, pipeline, tmp_path):
    from core.config import AppConfig
    from ui.hud_renderer import HudRenderer, qimage_to_bgr
    from ui.main_window import MainWindow
    from ui.view_model import build_view
    from ui.worker import FramePacket, VisionWorker
    img, fa = _run(pipeline, "zidane.jpg")
    cfg = AppConfig(show_angles=True, show_face_mesh=True)
    q = HudRenderer(cfg).render(img, fa, 1.0, paused=True, gesture_progress=("confirm", 0.5))
    assert (q.width(), q.height()) == (img.shape[1], img.shape[0])
    assert qimage_to_bgr(q).shape == img.shape
    for lang in ("en", "de"):
        views = [build_view(s, lang) for s in fa.persons]
        assert views and all(len(v.rows) == 7 for v in views)
    w = MainWindow(cfg, tmp_path / "cfg.json", VisionWorker(cfg))
    w.on_frame(FramePacket(image=q, persons=[build_view(s, "en") for s in fa.persons], fps=30, status={"pose": "online"}))
    w.on_events(fa.events)
    assert w.top.persons == 2
    w.close()
