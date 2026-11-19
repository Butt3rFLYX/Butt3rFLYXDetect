# Architektur

## Datenfluss pro Frame

```
CameraManager.read()                                     (Worker-Thread, ui/worker.py)
      │ BGR-Frame (Original, nie gespiegelt analysiert)
      ▼
VisionPipeline.process()                                  core/pipeline.py
  ├─► [Thread "mediapipe"]  Hand-/Face-Landmarker auf Ausschnitten,
  │       berechnet aus den Skeletten des VORHERIGEN Frames (1 Frame Versatz)
  ├─► PoseDetector.detect()  YOLO11-pose + ByteTrack  (GPU/MPS/CPU, parallel dazu)
  ├─► PersonTracker.update() roh-IDs → stabile IDs (PERSON 001…), Wiedererkennung,
  │       Ereignisse entered / reacquired / rematched / left
  └─► ActionEngine.process()
          ├─ analyse_pose()          Einzelbild: Haltung, Pose-Details, Ausrichtung, Gelenkwinkel
          ├─ MotionAnalyzer.update() Zeitverlauf: Geschwindigkeit, Schritte, Übergänge, Sport,
          │                          Sprung/Sturz/Stolpern, Drehung; Plausibilitäts-Reset
          ├─ Hände zuordnen          Handgelenk-Ausschnitt → (Person, Seite); Duplikate entfernt
          │   classify_hand() + HandMotionTracker + Zwei-Hand-Analyse
          ├─ Kopf                    analyse_face_mesh() oder Fallback analyse_head_from_pose()
          │                          + HeadMotionTracker (Nicken, Schütteln, Drehen, Pitch-Kalibrierung)
          └─ Ereignisse              Entprellung (Mindestdauer), Dauer des Vorzustands,
                                     Einmal-Ereignisse (Sprung, Sturz, Drehung) mit 1 s Sperre
      ▼
HudRenderer.render()  QPainter auf QImage (Worker-Thread)   ui/hud_renderer.py
EventLogger.log()     CSV / JSONL / TXT / SQLite            logging_system/
build_view()          threadsichere Anzeige-Daten           ui/view_model.py
      ▼  Qt-Signale
MainWindow (GUI-Thread): VideoView, TopBar, PersonPanel, ModuleStatus, Timeline
```

## Designentscheidungen

- **core/ ist GUI-frei** und vollständig ohne Kamera testbar (synthetische Keypoints in `tests/synthetic.py`).
- **Label-Register** (`core/labels.py`): jede Erkennung hat Kategorie, Status (stable/experimental),
  Modell und DE/EN-Text. UI, Logs und `docs/features.md` greifen darauf zu – dadurch keine Doppelpflege und
  keine „stillen“ Erkennungen ohne Kennzeichnung.
- **Konfidenzen werden berechnet**, nie geschätzt: mittlere Keypoint-/Landmark-Konfidenz der beteiligten
  Punkte × „wie deutlich die Regel erfüllt ist“ (`soft_margin`), bei geglätteten Labels zusätzlich × Stimmenanteil.
  Ausnahme: MediaPipe Face Landmarker liefert im Tasks-API keinen Score pro Gesicht → es wird die konfigurierte
  Mindestschwelle als Untergrenze verwendet.
- **Normierung auf Rumpflänge T** (Schulter-Mitte bis Hüft-Mitte) macht alle Regeln entfernungsunabhängig.
  Geschwindigkeiten in T/s (1 T ≈ 0,5 m).
- **Glättung:** Mehrheitsentscheid über 7 Frames (aus dem Originalprogramm), Ereignisse erst nach Mindestdauer.
- **Robustheit gegen ID-Wechsel:** Wird eine ID neu zugeordnet („rematched“) oder bewegt sich ein Körper
  schneller als menschenmöglich (> 15 T/s), wird der Bewegungsverlauf dieser Person zurückgesetzt statt zwei
  Personen zu vermischen. Teilweise sichtbare Personen (Rahmen am Bildrand) lösen keine Dreh-/Sprungereignisse aus.
- **Spiegeln** betrifft nur das Zeichnen; die Analyse läuft immer auf dem Originalbild, damit links/rechts stimmt.
- **Asynchronität:** Kamera + KI + Rendering im QThread; MediaPipe zusätzlich parallel zu YOLO im ThreadPool.
- **Fehlertoleranz:** jedes Modell kann einzeln fehlen/scheitern; Status steht im AI-MODULES-Panel und in der
  Timeline. Kamera-Reconnect alle 2 s. Defekte Konfiguration wird gesichert und repariert.

## Erweiterung um eine neue Erkennung

1. Label in `core/labels.py` eintragen (Kategorie, Status, Modell, EN/DE).
2. Regel in `posture.py` (Einzelbild), `motion_analyzer.py` (zeitlich), `hand_gestures.py` oder `head_pose.py`.
3. Test in `tests/` mit synthetischen Daten schreiben.
4. `python tools/gen_features.py` ausführen – die Tabelle zeigt den Test automatisch an.
