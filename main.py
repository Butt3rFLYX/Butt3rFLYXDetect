#!/usr/bin/env python3
"""BUTT3RFLYX - SEE EVERY MOTION. TRACK THE MOVEMENT. UNDERSTAND THE ACTION.

Start:   python main.py
Options: python main.py --help
The original OpenCV app is still available:  python main.py --legacy  (or action_recognition_app.py)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Butt3rFLYX - AI motion tracking dashboard")
    ap.add_argument("--no-intro", action="store_true", help="skip the intro animation")
    ap.add_argument("--camera", type=int, help="camera index")
    ap.add_argument("--source", help="video file instead of a webcam")
    ap.add_argument("--mode", choices=["eco", "balanced", "ultra"], help="performance mode")
    ap.add_argument("--lang", choices=["en", "de"], help="interface language")
    ap.add_argument("--config", default=str(PROJECT / "config.json"), help="config file path")
    ap.add_argument("--reset-config", action="store_true", help="start with default settings")
    ap.add_argument("--legacy", action="store_true", help="run the original OpenCV action_recognition_app.py")
    ap.add_argument("--quit-after", type=float, default=0, help=argparse.SUPPRESS)   # automated smoke tests
    ap.add_argument("--smoke-shot", default="", help=argparse.SUPPRESS)
    return ap.parse_known_args(argv)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--legacy" in argv:             # original OpenCV app, all other arguments go to it unchanged
        import action_recognition_app
        rest = [a for a in argv if a != "--legacy"]
        if "--model" not in rest:
            rest += ["--model", str(PROJECT / "models" / "yolo11n-pose.pt")]
        sys.argv = [sys.argv[0]] + rest
        action_recognition_app.main()
        return 0
    args, _ = parse_args(argv)

    from PySide6.QtCore import QPropertyAnimation, QTimer
    from PySide6.QtWidgets import QApplication

    from core.config import AppConfig, load_config, save_config
    from ui.cyber_intro import CyberIntro
    from ui.main_window import MainWindow, make_icon
    from ui.theme import stylesheet
    from ui.worker import VisionWorker

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Butt3rFLYX")
    app.setWindowIcon(make_icon())
    app.setStyleSheet(stylesheet())

    cfg_path = Path(args.config)
    cfg, warnings = (AppConfig(), []) if args.reset_config else load_config(cfg_path)
    if args.camera is not None:
        cfg.camera_index = args.camera
    if args.source:
        cfg.camera_source = args.source
    if args.mode:
        cfg.performance_mode = args.mode
    if args.lang:
        cfg.language = args.lang
    if not cfg_path.exists():
        try:
            save_config(cfg, cfg_path)
        except OSError:
            pass

    if sys.platform == "darwin" and not cfg.camera_source:
        # macOS shows the camera permission dialog only when the camera is first opened on the
        # MAIN thread - the worker thread could not request it. Opens briefly, then releases.
        import cv2
        probe = cv2.VideoCapture(cfg.camera_index, cv2.CAP_AVFOUNDATION)
        probe.release()

    worker = VisionWorker(cfg)
    window = MainWindow(cfg, cfg_path, worker)
    for w in warnings:
        window.timeline.add_system("CONFIG: " + w)

    boot_status: dict = {}
    worker.load_progress.connect(lambda mod, st: boot_status.__setitem__(mod, st))
    worker.start()          # models load in the background while the intro plays

    def show_main():
        window.showMaximized()
        window.raise_()
        window.activateWindow()

    if cfg.intro_enabled and not args.no_intro:
        intro = CyberIntro(boot_status, cfg.reduce_effects, cfg.intro_sound, cfg.language)
        screen = app.primaryScreen().geometry()
        intro.setGeometry(screen)
        intro.show()
        intro.activateWindow()
        intro.setFocus()

        def end_intro():
            show_main()
            anim = QPropertyAnimation(intro, b"windowOpacity", intro)
            anim.setDuration(350)
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.finished.connect(intro.close)
            anim.start()
            intro._fade = anim

        intro.finished.connect(end_intro)
    else:
        show_main()

    if args.quit_after > 0:
        stats = {"frames": 0, "events": 0, "max_persons": 0}

        def _count(pk):
            stats["frames"] += 1
            stats["max_persons"] = max(stats["max_persons"], len(pk.persons))
        worker.frame_ready.connect(_count)
        worker.events_ready.connect(lambda ev: stats.__setitem__("events", stats["events"] + len(ev)))

        def _finish():
            if args.smoke_shot:
                window.grab().save(args.smoke_shot)
            print(f"SMOKE frames={stats['frames']} events={stats['events']} max_persons={stats['max_persons']} "
                  f"fps={window.top.fps:.1f} status={worker.status}")
            window.close()
        QTimer.singleShot(int(args.quit_after * 1000), _finish)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
