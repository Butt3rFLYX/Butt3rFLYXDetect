"""YOLO11-pose wrapper with ByteTrack tracking and automatic device selection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ModelLoadError(RuntimeError):
    pass


@dataclass
class PoseDetection:
    raw_id: int | None
    box: np.ndarray        # xyxy
    xy: np.ndarray         # (17, 2)
    conf: np.ndarray       # (17,)
    score: float           # person box confidence


def select_device(preference: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "mps" or preference == "auto":
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except AttributeError:
            pass
        return "cpu"
    return "cpu"


class PoseDetector:
    def __init__(self, model_path: str | Path, device: str = "auto", imgsz: int = 480,
                 conf: float = 0.5, allow_download: bool = True) -> None:
        self.model_path = Path(model_path)
        self.device_pref = device
        self.device = "cpu"
        self.imgsz = imgsz
        self.conf = conf
        self.allow_download = allow_download
        self.model = None

    def load(self) -> str:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ModelLoadError(f"ultralytics not installed: {e}") from e
        src = str(self.model_path)
        if not self.model_path.exists():
            if not self.allow_download:
                raise ModelLoadError(f"Model file not found: {self.model_path}")
            # ultralytics downloads official model names (e.g. yolo11s-pose.pt) automatically
            src = self.model_path.name
        try:
            self.model = YOLO(src)
        except Exception as e:  # noqa: BLE001 - surface any loader error to the UI
            raise ModelLoadError(f"Could not load pose model '{src}': {e}") from e
        if getattr(self.model, "task", "pose") != "pose":
            raise ModelLoadError(f"'{src}' is not a pose model (task={self.model.task})")
        self.device = select_device(self.device_pref)
        # warm-up (also verifies the device works; falls back to CPU otherwise)
        dummy = np.zeros((self.imgsz, self.imgsz, 3), np.uint8)
        try:
            self.model.predict(dummy, imgsz=self.imgsz, device=self.device, verbose=False)
        except Exception:  # noqa: BLE001
            self.device = "cpu"
            self.model.predict(dummy, imgsz=self.imgsz, device=self.device, verbose=False)
        return self.device

    def reset_tracking(self) -> None:
        try:
            pred = self.model.predictor if self.model is not None else None
            if pred is not None and getattr(pred, "trackers", None):
                for tr in pred.trackers:
                    tr.reset()
        except Exception:  # noqa: BLE001
            pass

    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        if self.model is None:
            return []
        r = self.model.track(frame, persist=True, verbose=False, conf=self.conf, imgsz=self.imgsz,
                             device=self.device, tracker="bytetrack.yaml")[0]
        out: list[PoseDetection] = []
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return out
        xys = r.keypoints.xy.cpu().numpy()
        confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xys.shape[:2])
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(boxes)
        for rid, box, xy, conf, sc in zip(ids, boxes, xys, confs, scores):
            out.append(PoseDetection(rid, box.astype(float), xy.astype(float), conf.astype(float), float(sc)))
        return out
