# 🦋 BUTT3RFLYX – Advanced AI Motion Tracking System

> **SEE EVERY MOTION. TRACK THE MOVEMENT. UNDERSTAND THE ACTION.**

Butt3rFLYX is a desktop application for **local, transparent** motion analysis via webcam.
It combines **YOLO11-Pose** (full-body skeleton + tracking), **MediaPipe Hand Landmarker**
(21 points per hand) and **MediaPipe Face Landmarker** (478 points incl. iris) into a central
action recognition engine – with a cyberpunk intro, live dashboard, event timeline and logging.

Evolved from `action_recognition_app.py` (still included and runnable, see [Legacy mode](#legacy-mode)).

---

## Contents
1. [Quick start](#quick-start) · 2. [Requirements](#requirements) · 3. [Usage & keys](#usage--keyboard-shortcuts)
4. [Settings](#settings) · 5. [What is detected?](#what-is-detected) · 6. [Models & cameras](#supported-models--cameras)
7. [Performance](#performance) · 8. [Logging & export](#logging--export) · 9. [Privacy](#privacy)
10. [Limitations](#detection-limitations) · 11. [Troubleshooting](#troubleshooting) · 12. [Tests](#tests) · 13. [Architecture](#architecture)

---

## Quick start

```bash
cd Butt3rFLYX
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

On first launch, macOS asks for **camera permission** for the Terminal or Python – please allow it.
The models are already in `models/`. If a file is missing, it is downloaded automatically from the official
source (Ultralytics or Google MediaPipe); this can be disabled with `"allow_model_download": false` in `config.json`.

Useful launch options:

| Option | Effect |
|---|---|
| `--no-intro` | skip the intro |
| `--camera 1` | use a different camera |
| `--source video.mp4` | video file instead of webcam (loops) |
| `--mode eco\|balanced\|ultra` | performance mode |
| `--lang de` | German interface |
| `--reset-config` | start with default settings |
| `--legacy` | start the original OpenCV program |

## Requirements

- **Python 3.10 – 3.13** (developed and tested with **Python 3.13.12**, macOS 27, Apple Silicon)
- Webcam (any UVC/built-in camera OpenCV can open) or a video file
- Libraries (`requirements.txt`): `ultralytics` (incl. PyTorch), `opencv-python`, `numpy`, `lap`,
  `PySide6`, `mediapipe`, `pytest` for tests. Optional: `face_recognition`.
- GPU optional: **CUDA** (NVIDIA) or **MPS** (Apple Silicon) are detected automatically, otherwise CPU.

## Usage & keyboard shortcuts

Startup sequence: black screen → particles + boot terminal → digital neon butterfly
(flaps its wings, dissolves into particles) → glitch logo **B U T T 3 R F L Y X** → **real**
system status (models/camera, loaded in the background meanwhile) → scan transition → dashboard.
The intro can be skipped with **ESC / Space / click** and disabled in the settings.
If the models are still loading at the end of the intro, it waits with "SYNCHRONIZING NEURAL CORES" (max. 25 s).

| Key | Function | Key | Function |
|---|---|---|---|
| `Space` | pause / resume tracking | `S` | settings |
| `R` | start/stop video recording | `E` | export events |
| `L` | logging on/off | `G` | gesture control on/off |
| `K` | skeleton | `H` | entire HUD |
| `F` | hand landmarks | `A` | joint angles |
| `V` | motion vectors | `T` | trajectories |
| `M` | mirror display | `F11` | fullscreen |
| `F1` | help | `Q` / `ESC` | quit |

**Dashboard:** status bar at the top (system, camera, **measured** FPS, person count, mode, device, LOG/REC),
camera image with HUD on the left, AI module status (with runtime per model) and one card per person on the right
(posture, movement, body orientation, head, gaze, left/right hand with finger status `T I M R P`,
active actions, confidence), event timeline at the bottom.

**Skeleton colors:** cyan = normal · violet = gesture detected · green = action/transition detected ·
yellow = uncertain detection (dashed bones = keypoint confidence < 0.55) · red = possible fall.

## Settings

All settings (`S`) are saved to `config.json` and loaded on the next start.
A corrupted file is backed up to `config.json.bak`, invalid values are reset to defaults
and reported in the timeline.

| Section | Settings |
|---|---|
| Camera | index (with camera scan), video file, resolution, target FPS, mirror |
| AI / models | pose model (n/s/m), device (auto/cpu/mps/cuda), performance mode, person and keypoint confidence threshold, tracking sensitivity, minimum event duration, pose/hands/head modules on/off |
| Display | HUD, skeleton, boxes, hand landmarks, finger labels, joint angles, vectors, trajectories, head arrow, face points, intro, intro sound, reduced effects, language DE/EN |
| Logging | on/off, formats CSV/JSON/TXT/SQLite, folder, export format, recording folder |
| Control / privacy | gesture control + hold time, face identity (names) |

## What is detected?

**The complete, auto-generated table of all 131 detections is in [`docs/features.md`](docs/features.md)**
(status, required model, test coverage). Overview:

| Area | Detections | Model |
|---|---|---|
| Posture | standing, sitting, squat, deep squat, crouching, kneeling, kneeling on one knee, lying (+ back/stomach/side exp.), bending forward, push-up position, lunge, upright upper body | YOLO Pose |
| Pose details | leaning left/right/forward/back, standing on one leg, knee raised, arms crossed, hands on hips, raising hand/hands, hands above/behind head, T/Y/A pose, arm extended, stretching, twisted torso | YOLO Pose |
| Orientation | facing camera, sideways, back to camera | YOLO Pose |
| Movement | standing still, calm, slow/normal/fast walking, jogging, running, jumping, repeated jumping, hopping on one leg, sideways, backwards, towards / away from camera, walking/running in place, direction of movement in the image, possible fall, stumbling, turning | YOLO Pose |
| Transitions | sitting down, standing up, squatting down, rising from squat, bending down, straightening up, lying down, getting up from the floor, turn 90°/180°/360° | YOLO Pose |
| Exercise | squats (counter), jumping jacks (counter), arm circles, leg raises (counter), one-leg balance (duration), waving L/R, repeated arm/leg movements | YOLO Pose |
| Head | forward, left, right, up, down, tilted L/R (with angle), away from camera, turning, nodding, shaking, looking roughly at camera | Face Landmarker (fallback: rough from YOLO) |
| Gaze | left, right, up, down, center, eyes closed – **only when reliable**, otherwise "UNCERTAIN → HEAD DIRECTION ONLY" | Face Landmarker (iris) |
| Hand / fingers | hand visible (L/R), status per finger, finger count, palm/back of hand, open hand, fist, thumbs up/down, peace, OK, rock, three/four fingers, pointing up/down/left/right/at camera, hand in front of face | Hand Landmarker |
| Hand movement | waving, hand left/right/up/down, opening/closing hand, rotating hand, moving fingers | Hand Landmarker |
| Two hands | hands together, clapping, heart, pointing with both hands, hands apart/towards each other | Hand Landmarker |

**Left/right:** body sides (hand, arm, knee, head turn, tilt, pointing direction) are always given **from the
person's perspective**. Directions of movement across the image are explicitly labeled "IMAGE LEFT/RIGHT".

**Experimental** detections are marked yellow or with `(EXP)` in the interface.

### Gesture control (OFF by default)

| Gesture (hold 1 s) | Command |
|---|---|
| Thumbs up | confirm |
| Open hand | pause tracking |
| Fist | resume tracking |
| Peace | skeleton on/off |
| Both hands above head | open settings |

Hold time is configurable, followed by a 2 s cooldown; a progress ring shows the hold duration.

## Supported models & cameras

- **Pose:** `models/yolo11n-pose.pt` (default, fast). More accurate: `yolo11s-pose.pt` / `yolo11m-pose.pt`
  (downloaded automatically when selected). Other Ultralytics pose models (`.pt`, `.onnx`) can be selected via "…".
- **Hands:** `models/hand_landmarker.task` (MediaPipe, float16).
- **Face/head/iris:** `models/face_landmarker.task` (MediaPipe, float16).
- **Cameras:** anything OpenCV can open (macOS: AVFoundation, Windows: DirectShow, Linux: V4L2), plus video files.
  On signal loss the camera reconnects automatically every 2 s.

## Performance

| Mode | YOLO input | Hands | Faces | measured* |
|---|---|---|---|---|
| ECO | 320 px | every 3rd frame, max. 2 | every 4th frame, max. 1 | ≈ 55–60 FPS |
| BALANCED | 480 px | every 2nd frame, max. 4 | every 2nd frame, max. 2 | ≈ 37 FPS |
| ULTRA | 640 px | every frame, max. 6 | every frame, max. 4 | ≈ 25 FPS |

\* Pure pipeline processing rate with 3 people and all modules, Apple Silicon, no camera limit.
With a webcam, the camera is usually the bottleneck (typically 30 FPS). Values depend heavily on hardware,
resolution and number of people.

Optimizations: MediaPipe runs **in parallel** to YOLO in its own thread and only on **crops** around the head
and wrists of each person (also finds small/distant hands); results are reused across skipped
frames; bounded history buffers (4 s); camera, AI and HUD rendering run in a
background thread, so the interface stays smooth.

Tips: ECO mode, 960×540 resolution, "reduce animation effects", disable unneeded modules,
good lighting, camera at chest to head height, whole person in frame.

## Logging & export

- Logging is **off by default** (`L` or settings). Files go to `logs/`:
  `butt3rflyx_<date>_<time>.csv`, `.jsonl` (JSON Lines, crash-safe), `.txt`, and `butt3rflyx.db` (SQLite).
- Only **actually detected** events on state changes are stored (debounced via the
  minimum duration), no duplicate entries per frame:
  `timestamp, person_id, event_type, label, text, posture, movement, head_direction, left_hand, right_hand, confidence, duration_s`.
  `duration_s` = duration of the previous state or of the finished action.
- `E` exports all events of the current session as JSON, CSV, TXT or SQLite.

Example (JSON Lines):
```json
{"timestamp": "2026-11-19T17:30:05.120", "person_id": 1, "event_type": "state", "label": "walking",
 "text": "PERSON 001 STARTED WALKING 2.1 T/s", "posture": "standing", "movement": "walking",
 "head_direction": "head_left", "left_hand": null, "right_hand": "pointing_right", "confidence": 0.81, "duration_s": 4.2}
```

## Privacy

- All processing happens **locally**; no images or data are sent anywhere (network only for the
  one-time, optional model download).
- **No video recording without explicit action:** recording only via `R`/REC button, visible as a red
  `● REC` in the status bar; saved to `recordings/`.
- The logger stores **no images**, only text events – and only when logging is enabled.
- Tracking IDs are **temporary** and purely position-based (no biometrics).
- The optional **face identity** (names from `known_faces/<name>/<images>`, a feature of the original program)
  is **off by default**, requires `pip install face_recognition` and keeps face encodings in memory only.
- Please only capture people who have been informed about the analysis.

## Detection limitations

Details in [`docs/limitations.md`](docs/limitations.md). Key points:

- A single 2D camera cannot measure depth: sitting vs. squatting, lying on back vs. stomach, bending forward vs.
  lying down, and direction of rotation are partly ambiguous → such detections are marked experimental.
- **Gaze direction** is only reliable with a close, frontal, well-lit face; otherwise the system shows
  only the **head direction** and "UNCERTAIN".
- Hands are only found if YOLO sees the person's wrist. Hands without a visible body are not evaluated.
- All thresholds are rule-based and were checked with synthetic data, two sample photos and an
  artificial test video – **not** systematically with real webcam footage. Please calibrate live.

## Troubleshooting

| Problem | Solution |
|---|---|
| "NO SIGNAL" / camera not found | macOS: System Settings → Privacy & Security → Camera → allow Terminal/Python. Choose another index in the settings ("SCAN CAMERAS"). Don't keep the camera open in another app. |
| Pose model doesn't load | The AI MODULES panel/timeline shows the error. Check `models/yolo11n-pose.pt` or allow the download. Without a pose model the dashboard shows only the camera image. |
| Hands/head "ERROR" | `pip install mediapipe` or check the `.task` files in `models/`. The other modules keep running. |
| Low FPS | ECO mode, smaller resolution, disable modules, "reduce effects". |
| Wrong side left/right | Left/right is from the person's perspective. Mirroring (`M`) only changes the display, never the analysis. |
| Broken configuration | Backed up and replaced automatically; alternatively `python main.py --reset-config`. |
| GPU not used | Settings → device; on errors the system falls back to CPU automatically. |

## Tests

```bash
python -m pytest -q
```

99 tests: geometry, all postures, movements and transitions (synthetic sequences), hand/finger gestures
and hand movements, head/gaze, tracking IDs, event engine (debouncing, duration), logging/export in all
formats, configuration (incl. corrupted file), gesture control, error handling (missing models/camera),
real models on sample photos, and rendering of intro, HUD and dashboard (offscreen).
**Only testable with a real webcam:** live accuracy, camera reconnection, gesture control in everyday use,
actual FPS on your hardware.

Regenerate the feature table: `python tools/gen_features.py`.

## Architecture

```
Butt3rFLYX/
├── main.py                    entry point: intro + dashboard, arguments, --legacy
├── action_recognition_app.py  original program (still runnable unchanged)
├── config.json                saved settings (generated)
├── models/                    yolo11n-pose.pt, hand_landmarker.task, face_landmarker.task
├── core/                      AI logic, no GUI dependency
│   ├── camera_manager.py      camera/video + reconnect
│   ├── pose_detector.py       YOLO11-Pose + ByteTrack, device selection
│   ├── mediapipe_models.py    hand/face landmarkers on crops
│   ├── person_tracker.py      stable IDs, re-identification after short loss
│   ├── posture.py             single-frame posture analysis
│   ├── motion_analyzer.py     temporal analysis (movement, transitions, exercise, falls)
│   ├── hand_gestures.py       fingers, gestures, hand movements
│   ├── head_pose.py           head direction, gaze, nodding/shaking
│   ├── action_engine.py       fusion + events
│   ├── pipeline.py            per-frame flow, parallel execution
│   ├── labels.py              registry of all detections (status, model, DE/EN)
│   ├── gesture_control.py, face_identity.py, config.py, geometry.py
├── ui/                        PySide6 interface
│   ├── cyber_intro.py, main_window.py, dashboard.py, hud_renderer.py,
│   ├── settings_window.py, worker.py, view_model.py, theme.py, i18n.py
├── logging_system/            event_logger.py, export_manager.py
├── tests/                     99 automated tests
├── tools/gen_features.py      generates docs/features.md
├── docs/                      features.md, architecture.md, limitations.md, README_original.md
└── assets/                    logo/, sounds/ (intro sound is generated synthetically), animations/
```

Details: [`docs/architecture.md`](docs/architecture.md).

## Legacy mode

The original OpenCV program is included unchanged:

```bash
python main.py --legacy --mirror --log actions.csv
```

(all options of `action_recognition_app.py` are passed through; original README: `docs/README_original.md`).
