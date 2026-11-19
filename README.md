# 🦋 BUTT3RFLYX – Advanced AI Motion Tracking System

> **SEE EVERY MOTION. TRACK THE MOVEMENT. UNDERSTAND THE ACTION.**

Butt3rFLYX ist eine Desktop-Anwendung zur **lokalen, transparenten** Bewegungsanalyse per Webcam.
Sie kombiniert **YOLO11-Pose** (Ganzkörper-Skelett + Tracking), **MediaPipe Hand Landmarker**
(21 Punkte pro Hand) und **MediaPipe Face Landmarker** (478 Punkte inkl. Iris) zu einer zentralen
Action-Recognition-Engine – mit Cyberpunk-Intro, Live-Dashboard, Ereignis-Timeline und Protokollierung.

Hervorgegangen aus `action_recognition_app.py` (weiterhin enthalten und lauffähig, siehe [Legacy-Modus](#legacy-modus)).

---

## Inhalt
1. [Schnellstart](#schnellstart) · 2. [Voraussetzungen](#voraussetzungen) · 3. [Bedienung & Tasten](#bedienung--tastenkombinationen)
4. [Einstellungen](#einstellungen) · 5. [Was wird erkannt?](#was-wird-erkannt) · 6. [Modelle & Kameras](#unterstützte-modelle--kameras)
7. [Performance](#performance) · 8. [Logging & Export](#logging--export) · 9. [Datenschutz](#datenschutz)
10. [Grenzen](#erkennungsgrenzen) · 11. [Fehlerbehebung](#fehlerbehebung) · 12. [Tests](#tests) · 13. [Architektur](#architektur)

---

## Schnellstart

```bash
cd Butt3rFLYX
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Beim ersten Start fragt macOS nach der **Kamera-Berechtigung** für das Terminal bzw. Python – bitte erlauben.
Die Modelle liegen bereits in `models/`. Fehlt eine Datei, wird sie automatisch aus der offiziellen Quelle
geladen (Ultralytics bzw. Google MediaPipe); abschaltbar mit `"allow_model_download": false` in `config.json`.

Nützliche Startoptionen:

| Option | Wirkung |
|---|---|
| `--no-intro` | Intro überspringen |
| `--camera 1` | andere Kamera |
| `--source video.mp4` | Videodatei statt Webcam (läuft in Schleife) |
| `--mode eco\|balanced\|ultra` | Performance-Modus |
| `--lang de` | deutsche Oberfläche |
| `--reset-config` | mit Standardeinstellungen starten |
| `--legacy` | ursprüngliches OpenCV-Programm starten |

## Voraussetzungen

- **Python 3.10 – 3.13** (entwickelt und getestet mit **Python 3.13.12**, macOS 27, Apple Silicon)
- Webcam (beliebige UVC-/eingebaute Kamera, die OpenCV öffnen kann) oder eine Videodatei
- Bibliotheken (`requirements.txt`): `ultralytics` (inkl. PyTorch), `opencv-python`, `numpy`, `lap`,
  `PySide6`, `mediapipe`, für Tests `pytest`. Optional: `face_recognition`.
- GPU optional: **CUDA** (NVIDIA) oder **MPS** (Apple Silicon) werden automatisch erkannt, sonst CPU.

## Bedienung & Tastenkombinationen

Ablauf beim Start: schwarzer Bildschirm → Partikel + Boot-Terminal → digitaler Neon-Schmetterling
(schlägt mit den Flügeln, zerfällt in Partikel) → Glitch-Logo **B U T T 3 R F L Y X** → **echter**
Systemstatus (Modelle/Kamera, die währenddessen im Hintergrund geladen werden) → Scan-Übergang → Dashboard.
Das Intro ist mit **ESC / Leertaste / Klick** überspringbar und in den Einstellungen abschaltbar.
Falls die Modelle am Ende des Intros noch laden, wartet es mit „SYNCHRONIZING NEURAL CORES“ (max. 25 s).

| Taste | Funktion | Taste | Funktion |
|---|---|---|---|
| `Leertaste` | Tracking pausieren / fortsetzen | `S` | Einstellungen |
| `R` | Videoaufnahme starten/stoppen | `E` | Ereignisse exportieren |
| `L` | Logging an/aus | `G` | Gestensteuerung an/aus |
| `K` | Skelett | `H` | gesamtes HUD |
| `F` | Hand-Landmarks | `A` | Gelenkwinkel |
| `V` | Bewegungsvektoren | `T` | Trajektorien |
| `M` | Anzeige spiegeln | `F11` | Vollbild |
| `F1` | Hilfe | `Q` / `ESC` | Beenden |

**Dashboard:** oben Statusleiste (System, Kamera, **gemessene** FPS, Personenzahl, Modus, Gerät, LOG/REC),
links das Kamerabild mit HUD, rechts KI-Modul-Status (mit Laufzeit pro Modell) und eine Karte pro Person
(Haltung, Bewegung, Körperausrichtung, Kopf, Blick, linke/rechte Hand mit Fingerstatus `T I M R P`,
aktive Aktionen, Konfidenz), unten die Ereignis-Timeline.

**Skelett-Farben:** Cyan = normal · Violett = Geste erkannt · Grün = Aktion/Übergang erkannt ·
Gelb = unsichere Erkennung (gestrichelte Knochen = Keypoint-Konfidenz < 0.55) · Rot = möglicher Sturz.

## Einstellungen

Alle Einstellungen (`S`) werden in `config.json` gespeichert und beim nächsten Start geladen.
Eine fehlerhafte Datei wird nach `config.json.bak` gesichert, ungültige Werte werden auf Standard zurückgesetzt
und in der Timeline gemeldet.

| Bereich | Einstellungen |
|---|---|
| Kamera | Index (mit Kamerasuche), Videodatei, Auflösung, FPS-Ziel, Spiegeln |
| KI / Modelle | Pose-Modell (n/s/m), Gerät (auto/cpu/mps/cuda), Performance-Modus, Personen- und Keypoint-Konfidenzschwelle, Tracking-Empfindlichkeit, Mindestdauer für Ereignisse, Module Pose/Hände/Kopf an/aus |
| Anzeige | HUD, Skelett, Rahmen, Hand-Landmarks, Fingerbeschriftung, Gelenkwinkel, Vektoren, Trajektorien, Kopfpfeil, Gesichtspunkte, Intro, Intro-Sound, reduzierte Effekte, Sprache DE/EN |
| Logging | an/aus, Formate CSV/JSON/TXT/SQLite, Ordner, Exportformat, Aufnahmeordner |
| Steuerung / Datenschutz | Gestensteuerung + Haltezeit, Gesichtsidentität (Namen) |

## Was wird erkannt?

**Die vollständige, automatisch erzeugte Tabelle aller 131 Erkennungen steht in [`docs/features.md`](docs/features.md)**
(Status, benötigtes Modell, Testabdeckung). Kurzüberblick:

| Bereich | Erkennungen | Modell |
|---|---|---|
| Körperhaltung | Stehen, Sitzen, Hocke, tiefe Hocke, Ducken, Knien, Kniestand auf einem Knie, Liegen (+ Rücken/Bauch/Seite exp.), Vorbeugen, Liegestütz-Position, Ausfallschritt, Oberkörper aufrecht | YOLO Pose |
| Pose-Details | Lehnen links/rechts/vorne/hinten, Stand auf einem Bein, Knie gehoben, Arme verschränkt, Hände in den Hüften, Hand/Hände heben, Hände über/hinter dem Kopf, T-/Y-/A-Pose, Arm gestreckt, Strecken, Oberkörper verdreht | YOLO Pose |
| Ausrichtung | zur Kamera, seitlich, Rücken zur Kamera | YOLO Pose |
| Bewegung | Stillstehen, ruhig, langsam/normal/schnell gehen, laufen, rennen, springen, wiederholt springen, hüpfen auf einem Bein, seitwärts, rückwärts, auf die Kamera zu / weg, gehen/laufen auf der Stelle, Bewegungsrichtung im Bild, möglicher Sturz, Stolpern, Drehen | YOLO Pose |
| Übergänge | hinsetzen, aufstehen, in die Hocke, aus der Hocke, bücken, aufrichten, hinlegen, vom Boden aufstehen, Drehung 90°/180°/360° | YOLO Pose |
| Sport | Kniebeugen (Zähler), Hampelmann (Zähler), Armkreisen, Beinheben (Zähler), Balance auf einem Bein (Dauer), Winken L/R, wiederholte Arm-/Beinbewegungen | YOLO Pose |
| Kopf | geradeaus, links, rechts, oben, unten, geneigt L/R (mit Winkel), von der Kamera weg, dreht sich, nickt, schüttelt, schaut ca. zur Kamera | Face Landmarker (Fallback: grob aus YOLO) |
| Blick | links, rechts, oben, unten, Mitte, Augen geschlossen – **nur wenn zuverlässig**, sonst „UNSICHER → NUR KOPFRICHTUNG“ | Face Landmarker (Iris) |
| Hand / Finger | Hand sichtbar (L/R), Fingerstatus je Finger, Fingeranzahl, Handfläche/Handrücken, offene Hand, Faust, Daumen hoch/runter, Peace, OK, Rock, drei/vier Finger, Zeigen oben/unten/links/rechts/zur Kamera, Hand vor dem Gesicht | Hand Landmarker |
| Handbewegung | Winken, Hand nach links/rechts/oben/unten, Hand öffnen/schließen, Hand drehen, Finger bewegen | Hand Landmarker |
| Zwei Hände | Hände zusammen, Klatschen, Herz, mit beiden Händen zeigen, Hände auseinander/zueinander | Hand Landmarker |

**Links/Rechts:** Körperseiten (Hand, Arm, Knie, Kopfdrehung, Neigung, Zeigerichtung) sind immer **aus Sicht
der Person** angegeben. Bewegungsrichtungen durch das Bild sind ausdrücklich als „BILD-LINKS/-RECHTS“ beschriftet.

**Experimentelle** Erkennungen werden im Interface gelb bzw. mit `(EXP)` markiert.

### Gestensteuerung (standardmäßig AUS)

| Geste (1 s halten) | Befehl |
|---|---|
| Daumen hoch | Bestätigen |
| Offene Hand | Tracking pausieren |
| Faust | Tracking fortsetzen |
| Peace | Skelett ein/aus |
| Beide Hände über dem Kopf | Einstellungen öffnen |

Haltezeit einstellbar, danach 2 s Sperrzeit; ein Fortschrittsring zeigt die Haltedauer an.

## Unterstützte Modelle & Kameras

- **Pose:** `models/yolo11n-pose.pt` (Standard, schnell). Genauer: `yolo11s-pose.pt` / `yolo11m-pose.pt`
  (werden bei Auswahl automatisch geladen). Andere Ultralytics-Pose-Modelle (`.pt`, `.onnx`) über „…“ wählbar.
- **Hände:** `models/hand_landmarker.task` (MediaPipe, float16).
- **Gesicht/Kopf/Iris:** `models/face_landmarker.task` (MediaPipe, float16).
- **Kameras:** alles, was OpenCV öffnen kann (macOS: AVFoundation, Windows: DirectShow, Linux: V4L2), plus Videodateien.
  Die Kamera wird bei Signalverlust alle 2 s automatisch neu verbunden.

## Performance

| Modus | YOLO-Eingabe | Hände | Gesichter | gemessen* |
|---|---|---|---|---|
| ECO | 320 px | jedes 3. Bild, max. 2 | jedes 4. Bild, max. 1 | ≈ 55–60 FPS |
| BALANCED | 480 px | jedes 2. Bild, max. 4 | jedes 2. Bild, max. 2 | ≈ 37 FPS |
| ULTRA | 640 px | jedes Bild, max. 6 | jedes Bild, max. 4 | ≈ 25 FPS |

\* Reine Verarbeitungsrate der Pipeline mit 3 Personen und allen Modulen, Apple Silicon, ohne Kamera-Limit.
Mit Webcam begrenzt meist die Kamera (typisch 30 FPS). Die Werte hängen stark von Hardware, Auflösung und
Personenzahl ab.

Optimierungen: MediaPipe läuft **parallel** zu YOLO in einem eigenen Thread und nur auf **Ausschnitten** um Kopf
und Handgelenke jeder Person (findet auch kleine/entfernte Hände); Ergebnisse werden zwischen übersprungenen
Frames weiterverwendet; begrenzte Verlaufspuffer (4 s); Kamera, KI und HUD-Rendering laufen in einem
Hintergrund-Thread, die Oberfläche bleibt flüssig.

Tipps: ECO-Modus, Auflösung 960×540, „Animationseffekte reduzieren“, nicht benötigte Module abschalten,
gute Beleuchtung, Kamera auf Brust- bis Kopfhöhe, ganze Person im Bild.

## Logging & Export

- Logging ist **standardmäßig aus** (`L` oder Einstellungen). Dateien landen in `logs/`:
  `butt3rflyx_<Datum>_<Zeit>.csv`, `.jsonl` (JSON Lines, absturzsicher), `.txt`, sowie `butt3rflyx.db` (SQLite).
- Gespeichert werden nur **tatsächlich erkannte** Ereignisse bei Zustandsänderungen (entprellt über die
  Mindestdauer), keine doppelten Einträge pro Frame:
  `timestamp, person_id, event_type, label, text, posture, movement, head_direction, left_hand, right_hand, confidence, duration_s`.
  `duration_s` = Dauer des vorherigen Zustands bzw. der beendeten Aktion.
- `E` exportiert alle Ereignisse der laufenden Sitzung als JSON, CSV, TXT oder SQLite.

Beispiel (JSON Lines):
```json
{"timestamp": "2026-11-19T17:30:05.120", "person_id": 1, "event_type": "state", "label": "walking",
 "text": "PERSON 001 STARTED WALKING 2.1 T/s", "posture": "standing", "movement": "walking",
 "head_direction": "head_left", "left_hand": null, "right_hand": "pointing_right", "confidence": 0.81, "duration_s": 4.2}
```

## Datenschutz

- Alle Verarbeitung erfolgt **lokal**; es werden keine Bilder oder Daten versendet (Netzwerk nur für den
  einmaligen, abschaltbaren Modell-Download).
- **Keine Videoaufnahme ohne ausdrückliche Aktion:** Aufnahme nur per `R`/REC-Knopf, sichtbar durch rotes
  `● REC` in der Statusleiste; Speicherort `recordings/`.
- Der Logger speichert **keine Bilder**, nur Textereignisse – und nur wenn Logging eingeschaltet ist.
- Tracking-IDs sind **temporär** und rein positionsbasiert (keine Biometrie).
- Die optionale **Gesichtsidentität** (Namen aus `known_faces/<Name>/<Bilder>`, Funktion des Originalprogramms)
  ist **standardmäßig aus**, benötigt `pip install face_recognition` und hält Gesichts-Encodings nur im Arbeitsspeicher.
- Bitte nur Personen erfassen, die über die Analyse informiert sind.

## Erkennungsgrenzen

Ausführlich in [`docs/limitations.md`](docs/limitations.md). Das Wichtigste:

- Eine einzelne 2D-Kamera kann Tiefe nicht messen: Sitzen vs. Hocke, Rücken- vs. Bauchlage, Vorbeugen vs.
  Hinlegen und Drehrichtung sind teils mehrdeutig → solche Erkennungen sind als experimentell markiert.
- **Blickrichtung** ist nur bei nahem, frontalem, gut beleuchtetem Gesicht zuverlässig; sonst zeigt das System
  nur die **Kopfrichtung** und „UNSICHER“.
- Hände werden nur gefunden, wenn YOLO das Handgelenk der Person sieht. Hände ohne sichtbaren Körper werden nicht ausgewertet.
- Alle Schwellwerte sind regelbasiert und wurden mit synthetischen Daten, zwei Beispielfotos und einem
  künstlichen Testvideo geprüft – **nicht** systematisch mit echten Webcam-Aufnahmen. Bitte live kalibrieren.

## Fehlerbehebung

| Problem | Lösung |
|---|---|
| „NO SIGNAL“ / Kamera nicht gefunden | macOS: Systemeinstellungen → Datenschutz & Sicherheit → Kamera → Terminal/Python erlauben. Anderen Index in den Einstellungen („SCAN CAMERAS“) wählen. Kamera nicht in anderer App geöffnet lassen. |
| Pose-Modell lädt nicht | AI-MODULES-Panel/Timeline zeigt den Fehler. Datei `models/yolo11n-pose.pt` prüfen oder Download erlauben. Ohne Pose-Modell zeigt das Dashboard nur das Kamerabild. |
| Hände/Kopf „ERROR“ | `pip install mediapipe` bzw. `.task`-Dateien in `models/` prüfen. Die übrigen Module laufen weiter. |
| Niedrige FPS | ECO-Modus, kleinere Auflösung, Module abschalten, „Effekte reduzieren“. |
| Falsche Seite links/rechts | Links/rechts ist aus Sicht der Person. Spiegeln (`M`) ändert nur die Anzeige, nie die Analyse. |
| Konfiguration kaputt | Wird automatisch gesichert und ersetzt; alternativ `python main.py --reset-config`. |
| GPU wird nicht genutzt | Einstellungen → Gerät; bei Fehlern fällt das System automatisch auf CPU zurück. |

## Tests

```bash
python -m pytest -q
```

99 Tests: Geometrie, alle Körperhaltungen, Bewegungen und Übergänge (synthetische Sequenzen), Hand-/Fingergesten
und Handbewegungen, Kopf/Blick, Tracking-IDs, Ereignis-Engine (Entprellung, Dauer), Logging/Export in allen
Formaten, Konfiguration (inkl. defekter Datei), Gestensteuerung, Fehlerbehandlung (fehlende Modelle/Kamera),
echte Modelle auf Beispielfotos sowie das Rendern von Intro, HUD und Dashboard (offscreen).
**Nur mit echter Webcam prüfbar:** Live-Genauigkeit, Kamera-Wiederverbindung, Gestensteuerung im Alltag,
tatsächliche FPS deiner Hardware.

Die Feature-Tabelle neu erzeugen: `python tools/gen_features.py`.

## Architektur

```
Butt3rFLYX/
├── main.py                    Einstieg: Intro + Dashboard, Argumente, --legacy
├── action_recognition_app.py  Originalprogramm (unverändert lauffähig)
├── config.json                gespeicherte Einstellungen (wird erzeugt)
├── models/                    yolo11n-pose.pt, hand_landmarker.task, face_landmarker.task
├── core/                      KI-Logik, ohne GUI-Abhängigkeit
│   ├── camera_manager.py      Kamera/Video + Reconnect
│   ├── pose_detector.py       YOLO11-Pose + ByteTrack, Geräteauswahl
│   ├── mediapipe_models.py    Hand-/Face-Landmarker auf Ausschnitten
│   ├── person_tracker.py      stabile IDs, Wiedererkennung nach kurzem Verlust
│   ├── posture.py             Einzelbild-Haltungsanalyse
│   ├── motion_analyzer.py     zeitliche Analyse (Bewegung, Übergänge, Sport, Sturz)
│   ├── hand_gestures.py       Finger, Gesten, Handbewegungen
│   ├── head_pose.py           Kopfrichtung, Blick, Nicken/Schütteln
│   ├── action_engine.py       Fusion + Ereignisse
│   ├── pipeline.py            Ablauf pro Frame, parallele Ausführung
│   ├── labels.py              Register aller Erkennungen (Status, Modell, DE/EN)
│   ├── gesture_control.py, face_identity.py, config.py, geometry.py
├── ui/                        PySide6-Oberfläche
│   ├── cyber_intro.py, main_window.py, dashboard.py, hud_renderer.py,
│   ├── settings_window.py, worker.py, view_model.py, theme.py, i18n.py
├── logging_system/            event_logger.py, export_manager.py
├── tests/                     99 automatische Tests
├── tools/gen_features.py      erzeugt docs/features.md
├── docs/                      features.md, architecture.md, limitations.md, README_original.md
└── assets/                    logo/, sounds/ (Intro-Sound wird synthetisch erzeugt), animations/
```

Details: [`docs/architecture.md`](docs/architecture.md).

## Legacy-Modus

Das ursprüngliche OpenCV-Programm ist unverändert enthalten:

```bash
python main.py --legacy --mirror --log aktionen.csv
```

(alle Optionen von `action_recognition_app.py` werden durchgereicht; Original-README: `docs/README_original.md`).
