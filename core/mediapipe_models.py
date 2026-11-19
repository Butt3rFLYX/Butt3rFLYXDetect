"""MediaPipe Tasks wrappers: Hand Landmarker (21 points / hand) and Face Landmarker
(478 points incl. iris). Both are optional - if MediaPipe or a model file is missing the
rest of Butt3rFLYX keeps working."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


class OptionalModelError(RuntimeError):
    pass


def ensure_model(path: Path, allow_download: bool) -> Path:
    if path.exists() and path.stat().st_size > 1000:
        return path
    url = MODEL_URLS.get(path.name)
    if not allow_download or url is None:
        raise OptionalModelError(f"Model file missing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise OptionalModelError(f"Download of {path.name} failed: {e}") from e
    return path


@dataclass
class RawHand:
    landmarks: np.ndarray     # (21, 3) pixels in the full frame; z scaled like x
    mp_label: str             # MediaPipe handedness label ("Left"/"Right", assumes a mirrored image!)
    score: float
    owner: tuple | None = None  # (person id, "left"/"right") when detected in a wrist crop
    match: float = 9.9          # wrist distance to the YOLO wrist / torso length


@dataclass
class RawFace:
    landmarks: np.ndarray     # (478, 3) pixels in the full frame
    score: float              # lower bound = configured presence threshold (Tasks API has no per-face score)
    owner: int | None = None  # person id when detected in a head crop


def crop_square(frame: np.ndarray, cx: float, cy: float, side: float, out_size: int):
    """Square crop (zero padded at the borders) resized to out_size. Returns (rgb, x0, y0, scale)
    with full-frame coordinate = crop coordinate / scale + (x0, y0)."""
    import cv2
    h, w = frame.shape[:2]
    side = max(int(side), 8)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    if xb <= xa or yb <= ya:
        return None
    patch = np.zeros((side, side, 3), np.uint8)
    patch[ya - y0:yb - y0, xa - x0:xb - x0] = frame[ya:yb, xa:xb]
    scale = out_size / side
    small = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB), x0, y0, scale


class _ImageTask:
    def _run(self, rgb: np.ndarray):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return self.landmarker.detect(img)

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:  # noqa: BLE001
            pass


class HandLandmarkerModel(_ImageTask):
    """Runs on wrist crops (IMAGE mode) - finds small/distant hands that a full-frame pass misses."""
    CROP = 224

    def __init__(self, model_path: Path, allow_download: bool = True, min_conf: float = 0.5) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:
            raise OptionalModelError(f"mediapipe not installed: {e}") from e
        path = ensure_model(Path(model_path), allow_download)
        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)), running_mode=vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

    def detect_crop(self, frame: np.ndarray, cx: float, cy: float, side: float) -> list[RawHand]:
        c = crop_square(frame, cx, cy, side, self.CROP)
        if c is None:
            return []
        rgb, x0, y0, s = c
        res = self._run(rgb)
        out = []
        for lms, hd in zip(res.hand_landmarks or [], res.handedness or []):
            arr = np.array([[p.x * self.CROP / s + x0, p.y * self.CROP / s + y0, p.z * self.CROP / s] for p in lms], float)
            cat = hd[0] if hd else None
            out.append(RawHand(arr, cat.category_name if cat else "", float(cat.score) if cat else 0.5))
        return out


class FaceLandmarkerModel(_ImageTask):
    """Runs on head crops (IMAGE mode), one face per crop."""
    CROP = 256

    def __init__(self, model_path: Path, allow_download: bool = True, min_conf: float = 0.5) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:
            raise OptionalModelError(f"mediapipe not installed: {e}") from e
        path = ensure_model(Path(model_path), allow_download)
        self._mp = mp
        self.min_conf = min_conf
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)), running_mode=vision.RunningMode.IMAGE,
            num_faces=1, min_face_detection_confidence=min_conf, min_face_presence_confidence=min_conf)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

    def detect_crop(self, frame: np.ndarray, cx: float, cy: float, side: float) -> RawFace | None:
        c = crop_square(frame, cx, cy, side, self.CROP)
        if c is None:
            return None
        rgb, x0, y0, s = c
        res = self._run(rgb)
        if not res.face_landmarks:
            return None
        lms = res.face_landmarks[0]
        arr = np.array([[p.x * self.CROP / s + x0, p.y * self.CROP / s + y0, p.z * self.CROP / s] for p in lms], float)
        return RawFace(arr, self.min_conf)
