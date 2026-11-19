#!/usr/bin/env python3
"""
Aktionserkennung per Webcam: erkennt WER (optional per Gesichtserkennung) WAS macht.

Erkannte Zustände / Aktionen:
  Haltung:   steht, sitzt / hockt, liegt, aufrecht (nur Oberkörper sichtbar)
  Arme:      hebt linke Hand, hebt rechte Hand, hebt beide Hände, winkt
  Bewegung:  geht, springt, STURZ

Pose-Erkennung: YOLO-Pose (ultralytics), mehrere Personen gleichzeitig, mit Tracking-IDs.
Namen:       face_recognition (optional) mit Ordner known_faces/<Name>/<Bilder>.

Beenden mit "q".
"""
import argparse
import csv
import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# --------------------------------------------------------------------------
# COCO-Keypoints (YOLO-Pose). "links/rechts" = aus Sicht der Person!
# --------------------------------------------------------------------------
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

# --------------------------------------------------------------------------
# Schwellwerte. Längen sind relativ zur Rumpflänge (Schulter-Mitte bis Hüft-Mitte),
# damit es egal ist, wie weit die Person von der Kamera weg ist.
# --------------------------------------------------------------------------
MIN_KP_CONF = 0.4          # Keypoints darunter gelten als "nicht sichtbar"
LYING_TORSO_ANGLE = 60     # Rumpf so viele Grad gegen die Senkrechte geneigt -> liegt
UPRIGHT_TORSO_ANGLE = 40   # darunter gilt der Rumpf als aufrecht
SIT_KNEE_ANGLE = 120       # Kniewinkel darunter -> sitzt / hockt
STAND_KNEE_ANGLE = 150     # Kniewinkel darüber -> steht
HAND_RAISE_MARGIN = 0.1    # Handgelenk muss so weit über der Schulter sein
FALL_WINDOW_S = 1.0        # Sturz: Zeitfenster
FALL_DROP = 1.0            # Sturz: Hüfte fällt um >= 1 Rumpflänge ...
FALL_END_ANGLE = 55        # ... und der Rumpf ist danach so stark geneigt
JUMP_RISE = 0.2            # Sprung: Füße heben sich um >= 0.2 Rumpflängen
WALK_SPEED = 0.8           # Gehen: Hüfte bewegt sich >= 0.8 Rumpflängen / Sekunde seitlich
WAVE_MIN_SWINGS = 3        # Winken: so viele Richtungswechsel der Hand ...
WAVE_WINDOW_S = 1.5        # ... innerhalb dieses Zeitfensters
WAVE_AMPLITUDE = 0.15      # minimale Ausschlag-Größe pro Schwung
SMOOTH_FRAMES = 7          # Mehrheitsentscheid über so viele Frames gegen Flackern
FALL_HOLD_S = 3.0          # Wie lange "STURZ" angezeigt bleibt
JUMP_HOLD_S = 0.6          # Wie lange "springt" angezeigt bleibt


# --------------------------------------------------------------------------
# Geometrie-Helfer
# --------------------------------------------------------------------------
def kp(xy, conf, i):
    """Keypoint i als np.array oder None, wenn nicht sicher erkannt."""
    if conf[i] < MIN_KP_CONF:
        return None
    return xy[i]


def mid(a, b):
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def joint_angle(a, b, c):
    """Winkel bei b (in Grad) zwischen den Strecken b->a und b->c."""
    if a is None or b is None or c is None:
        return None
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def tilt_from_vertical(lower, upper):
    """0 Grad = Rumpf senkrecht, 90 Grad = waagerecht. (Bild-y zeigt nach unten.)"""
    v = upper - lower
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return math.degrees(math.acos(float(np.clip(-v[1] / n, -1.0, 1.0))))


# --------------------------------------------------------------------------
# Einzelbild-Analyse
# --------------------------------------------------------------------------
@dataclass
class Frame:
    t: float
    hip: np.ndarray | None = None
    ankle_y: float | None = None        # nur gesetzt, wenn beide Füße sichtbar
    torso_len: float | None = None
    torso_angle: float | None = None
    posture: str | None = None
    arms: str | None = None
    l_wrist_x: float | None = None      # nur gesetzt, wenn die Hand gehoben ist
    r_wrist_x: float | None = None


def analyse_frame(xy, conf, box, t):
    """Wertet die Keypoints einer Person in einem Bild aus."""
    f = Frame(t=t)
    l_sh, r_sh = kp(xy, conf, L_SH), kp(xy, conf, R_SH)
    l_hip, r_hip = kp(xy, conf, L_HIP), kp(xy, conf, R_HIP)
    sh, hip = mid(l_sh, r_sh), mid(l_hip, r_hip)
    if sh is None or hip is None:
        return f
    torso_len = float(np.linalg.norm(sh - hip))
    if torso_len < 5:
        return f
    f.hip, f.torso_len = hip, torso_len
    f.torso_angle = tilt_from_vertical(hip, sh)

    l_ank, r_ank = kp(xy, conf, L_ANK), kp(xy, conf, R_ANK)
    if l_ank is not None and r_ank is not None:
        f.ankle_y = float((l_ank[1] + r_ank[1]) / 2)

    # ---- Haltung ----
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    knee_angles = [a for a in (
        joint_angle(l_hip, kp(xy, conf, L_KNEE), l_ank),
        joint_angle(r_hip, kp(xy, conf, R_KNEE), r_ank)) if a is not None]
    knee = float(np.mean(knee_angles)) if knee_angles else None

    if f.torso_angle > LYING_TORSO_ANGLE or (w > 1.3 * h and f.torso_angle > 45):
        f.posture = "liegt"
    elif knee is not None and knee < SIT_KNEE_ANGLE:
        f.posture = "sitzt / hockt"
    elif knee is not None and knee > STAND_KNEE_ANGLE:
        f.posture = "steht"
    elif knee is None and f.torso_angle < UPRIGHT_TORSO_ANGLE:
        f.posture = "aufrecht"   # z. B. am Schreibtisch, Beine nicht im Bild
    elif knee is not None:
        f.posture = "steht (Knie gebeugt)"

    # ---- Arme (nur sinnvoll, wenn der Oberkörper aufrecht ist) ----
    if f.torso_angle < UPRIGHT_TORSO_ANGLE:
        margin = HAND_RAISE_MARGIN * torso_len
        l_wr, r_wr = kp(xy, conf, L_WR), kp(xy, conf, R_WR)
        left_up = l_wr is not None and l_sh is not None and l_wr[1] < l_sh[1] - margin
        right_up = r_wr is not None and r_sh is not None and r_wr[1] < r_sh[1] - margin
        if left_up and right_up:
            f.arms = "hebt beide Hände"
        elif left_up:
            f.arms = "hebt linke Hand"
        elif right_up:
            f.arms = "hebt rechte Hand"
        if left_up:
            f.l_wrist_x = float(l_wr[0])
        if right_up:
            f.r_wrist_x = float(r_wr[0])
    return f


# --------------------------------------------------------------------------
# Zeitliche Analyse pro Person (Sturz, Sprung, Gehen, Winken, Glättung)
# --------------------------------------------------------------------------
def count_swings(xs, min_amp):
    """Zählt Richtungswechsel einer Bewegung, kleine Zitterer werden ignoriert."""
    swings, direction, anchor = 0, 0, None
    for x in xs:
        if anchor is None:
            anchor = x
            continue
        d = x - anchor
        if abs(d) >= min_amp:
            new_dir = 1 if d > 0 else -1
            if direction != 0 and new_dir != direction:
                swings += 1
            direction, anchor = new_dir, x
        elif direction != 0 and (x - anchor) * direction > 0:
            anchor = x   # weiter in dieselbe Richtung -> Umkehrpunkt nachziehen
    return swings


@dataclass
class Person:
    track_id: int
    history: deque = field(default_factory=lambda: deque(maxlen=90))
    name_votes: Counter = field(default_factory=Counter)
    fall_until: float = 0.0
    jump_until: float = 0.0
    last_seen: float = 0.0
    last_logged: str = ""

    @property
    def name(self):
        known = [(n, c) for n, c in self.name_votes.most_common() if n != "Unknown"]
        if known and known[0][1] >= 2:
            return known[0][0]
        return f"Person {self.track_id}"

    def update(self, f: Frame):
        """Neues Frame übernehmen; gibt (Aktionen, neue_Ereignisse) zurück."""
        self.history.append(f)
        self.last_seen = f.t
        events = []
        now = f.t
        hist = [h for h in self.history if h.hip is not None]

        if f.hip is not None:
            # --- Sturz: Hüfte sackt schnell ab und Rumpf kippt in die Waagerechte ---
            if now >= self.fall_until:
                window = [h for h in hist if now - h.t <= FALL_WINDOW_S
                          and h.torso_angle is not None and h.torso_angle < UPRIGHT_TORSO_ANGLE]
                if window and f.torso_angle is not None and f.torso_angle > FALL_END_ANGLE:
                    top = min(window, key=lambda h: h.hip[1])
                    if f.hip[1] - top.hip[1] > FALL_DROP * top.torso_len:
                        self.fall_until = now + FALL_HOLD_S
                        events.append("STURZ erkannt!")

            # --- Sprung: beide Füße heben sich deutlich über den Boden von vorher ---
            if f.ankle_y is not None and now >= self.jump_until:
                base = [h for h in hist if 0.3 <= now - h.t <= 1.5
                        and h.ankle_y is not None and h.posture and h.posture.startswith("steht")]
                if len(base) >= 5:
                    ground = float(np.median([h.ankle_y for h in base]))
                    base_hip = float(np.median([h.hip[1] for h in base]))
                    rise = JUMP_RISE * f.torso_len
                    if f.ankle_y < ground - rise and f.hip[1] < base_hip - 0.75 * rise:
                        self.jump_until = now + JUMP_HOLD_S
                        events.append("springt")

        # --- Glättung: Mehrheitsentscheid über die letzten Frames ---
        recent = list(self.history)[-SMOOTH_FRAMES:]
        posture = Counter(h.posture for h in recent).most_common(1)[0][0]
        arms = Counter(h.arms for h in recent).most_common(1)[0][0]

        actions = []
        if now < self.fall_until:
            actions.append("STURZ!")
        if posture:
            actions.append(posture)
        if now < self.jump_until:
            actions.append("springt")

        # --- Gehen: Hüfte bewegt sich seitlich durchs Bild ---
        if posture and posture.startswith("steht") and now >= self.jump_until:
            walk = [h for h in hist if now - h.t <= 1.0]
            if len(walk) >= 5 and walk[-1].t - walk[0].t > 0.5:
                dx = abs(walk[-1].hip[0] - walk[0].hip[0])
                speed = dx / (walk[-1].t - walk[0].t) / np.median([h.torso_len for h in walk])
                if speed > WALK_SPEED:
                    actions.append("geht")

        # --- Winken: gehobene Hand pendelt seitlich hin und her ---
        waving = False
        for attr in ("l_wrist_x", "r_wrist_x"):
            seg = [h for h in self.history if now - h.t <= WAVE_WINDOW_S]
            xs = [getattr(h, attr) for h in seg if getattr(h, attr) is not None]
            if len(xs) >= 6 and f.torso_len:
                if count_swings(xs, WAVE_AMPLITUDE * f.torso_len) >= WAVE_MIN_SWINGS:
                    waving = True
        if waving:
            actions.append("winkt")
        elif arms:
            actions.append(arms)

        return actions, events


# --------------------------------------------------------------------------
# Gesichtserkennung (optional)
# --------------------------------------------------------------------------
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class FaceIdentifier:
    def __init__(self, known_dir, tolerance=0.6):
        import face_recognition as fr  # nur laden, wenn gebraucht
        self.fr = fr
        self.tolerance = tolerance
        self.encodings, self.labels = [], []
        for label in sorted(os.listdir(known_dir)):
            sub = os.path.join(known_dir, label)
            if not os.path.isdir(sub):
                continue
            for img in sorted(os.listdir(sub)):
                if not img.lower().endswith(IMAGE_EXT):
                    continue
                try:
                    enc = fr.face_encodings(fr.load_image_file(os.path.join(sub, img)))
                except Exception as e:
                    print(f"  Übersprungen ({e}): {sub}/{img}")
                    continue
                if enc:
                    self.encodings.append(enc[0])
                    self.labels.append(label)
                else:
                    print(f"  Kein Gesicht gefunden in: {sub}/{img}")
        print(f"{len(self.encodings)} bekannte Gesichter geladen "
              f"({len(set(self.labels))} Personen).")

    def identify(self, frame_bgr, scale=0.5):
        """Liefert [((cx, cy), name), ...] in Original-Pixelkoordinaten."""
        import cv2
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = self.fr.face_locations(rgb)
        encs = self.fr.face_encodings(rgb, locs)
        out = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            name = "Unknown"
            if self.encodings:
                d = self.fr.face_distance(self.encodings, enc)
                i = int(np.argmin(d))
                if d[i] <= self.tolerance:
                    name = self.labels[i]
            cx, cy = (left + right) / 2 / scale, (top + bottom) / 2 / scale
            out.append(((cx, cy), name))
        return out


def assign_faces(faces, tracks):
    """Ordnet jedes Gesicht der Person zu, deren Nase am nächsten am Gesicht liegt."""
    for (cx, cy), name in faces:
        best, best_d = None, float("inf")
        for pid, (box, xy, conf) in tracks.items():
            x1, y1, x2, y2 = box
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            ref = xy[NOSE] if conf[NOSE] >= MIN_KP_CONF else np.array([(x1 + x2) / 2, y1])
            d = math.hypot(cx - ref[0], cy - ref[1])
            if d < best_d:
                best, best_d = pid, d
        if best is not None:
            yield best, name


# --------------------------------------------------------------------------
# Anzeige
# --------------------------------------------------------------------------
def ascii_text(s):
    """cv2.putText kann keine Umlaute."""
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("Ä", "Ae"), ("Ö", "Oe"),
                 ("Ü", "Ue"), ("ß", "ss")):
        s = s.replace(a, b)
    return s


def draw_person(cv2, frame, box, xy, conf, name, actions):
    alarm = "STURZ!" in actions
    color = (0, 0, 255) if alarm else (0, 200, 0)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    for a, b in SKELETON:
        if conf[a] >= MIN_KP_CONF and conf[b] >= MIN_KP_CONF:
            cv2.line(frame, tuple(map(int, xy[a])), tuple(map(int, xy[b])), (255, 200, 0), 2)
    for i in range(len(xy)):
        if conf[i] >= MIN_KP_CONF:
            cv2.circle(frame, tuple(map(int, xy[i])), 3, (0, 255, 255), -1)
    lines = [name] + actions
    y = max(y1 - 8 - 22 * (len(lines) - 1), 18)
    for i, text in enumerate(lines):
        text = ascii_text(text)
        scale, thick = (0.65, 2) if i == 0 else (0.55, 1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(frame, (x1, y - th - 4), (x1 + tw + 6, y + 4), (0, 0, 0), -1)
        cv2.putText(frame, text, (x1 + 3, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 255) if text == "STURZ!" else (255, 255, 255), thick)
        y += 22


# --------------------------------------------------------------------------
# Hauptprogramm
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Erkennt per Webcam, wer was macht.")
    ap.add_argument("--camera", default="0", help="Kamera-Index oder Videodatei (Standard: 0)")
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="YOLO-Pose-Modell (n=schnell, s/m=genauer). Wird automatisch geladen.")
    ap.add_argument("--known-faces", default="known_faces",
                    help="Ordner mit bekannten Gesichtern: known_faces/<Name>/<Bilder>")
    ap.add_argument("--no-faces", action="store_true", help="Gesichtserkennung ausschalten")
    ap.add_argument("--face-every", type=int, default=10,
                    help="Gesichtserkennung nur jedes n-te Bild (spart Rechenzeit)")
    ap.add_argument("--log", help="Ereignisse zusätzlich in diese CSV-Datei schreiben")
    ap.add_argument("--mirror", action="store_true", help="Bild spiegeln wie ein Spiegel")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    faces = None
    if not args.no_faces:
        if not os.path.isdir(args.known_faces):
            print(f"Ordner '{args.known_faces}' fehlt - Personen werden nur nummeriert.")
        else:
            try:
                faces = FaceIdentifier(args.known_faces)
            except ImportError:
                print("Paket 'face_recognition' nicht installiert - Personen werden nur nummeriert.")

    model = YOLO(args.model)
    source = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Kamera / Video konnte nicht geöffnet werden.")
        return

    log_file = writer = None
    if args.log:
        new = not os.path.exists(args.log)
        log_file = open(args.log, "a", newline="", encoding="utf-8")
        writer = csv.writer(log_file)
        if new:
            writer.writerow(["zeit", "person", "aktionen"])

    def log(person_name, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] {person_name}: {text}")
        if writer:
            writer.writerow([datetime.now().isoformat(timespec="seconds"), person_name, text])
            log_file.flush()

    people: dict[int, Person] = {}
    frame_no, fps, last_t = 0, 0.0, time.time()
    print("Läuft - Beenden mit 'q' im Videofenster.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            frame_no += 1

            r = model.track(frame, persist=True, verbose=False, conf=0.5)[0]
            tracks = {}
            if r.keypoints is not None and len(r.boxes) > 0:
                xys = r.keypoints.xy.cpu().numpy()
                confs = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
                         else np.ones(xys.shape[:2]))
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = (r.boxes.id.int().cpu().tolist() if r.boxes.id is not None
                       else list(range(1, len(boxes) + 1)))
                for pid, box, xy, conf in zip(ids, boxes, xys, confs):
                    tracks[pid] = (box, xy, conf)

            # Namen zuordnen (nicht in jedem Bild, das wäre zu langsam)
            if faces and tracks and frame_no % args.face_every == 0:
                for pid, name in assign_faces(faces.identify(frame), tracks):
                    people.setdefault(pid, Person(pid)).name_votes[name] += 1

            # Analyse immer auf dem Originalbild (sonst wären links/rechts vertauscht),
            # gespiegelt wird nur die Anzeige.
            width = frame.shape[1]
            display = cv2.flip(frame, 1) if args.mirror else frame
            for pid, (box, xy, conf) in tracks.items():
                p = people.setdefault(pid, Person(pid))
                actions, events = p.update(analyse_frame(xy, conf, box, now))
                for e in events:
                    log(p.name, e)
                summary = ", ".join(a for a in actions if a != "STURZ!")
                if summary and summary != p.last_logged:
                    log(p.name, summary)
                    p.last_logged = summary
                if args.mirror:
                    xy = xy.copy()
                    xy[:, 0] = width - xy[:, 0]
                    box = (width - box[2], box[1], width - box[0], box[3])
                draw_person(cv2, display, box, xy, conf, p.name, actions)

            # Personen vergessen, die länger nicht mehr im Bild waren
            for pid in [k for k, v in people.items() if now - v.last_seen > 5 and v.last_seen]:
                del people[pid]

            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt
            cv2.putText(display, f"{fps:.1f} FPS", (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Aktionserkennung", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
