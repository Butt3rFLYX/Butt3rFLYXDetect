# Aktionserkennung per Webcam

Erkennt live, **wer** im Bild ist und **was** die Person macht. Mehrere Personen gleichzeitig.

| Kategorie | Erkannt |
|---|---|
| Haltung | steht, sitzt / hockt, liegt, aufrecht (nur Oberkörper im Bild) |
| Arme | hebt linke Hand, hebt rechte Hand, hebt beide Hände, winkt |
| Bewegung | geht, springt, **STURZ** (roter Rahmen) |

"Links/rechts" ist immer aus Sicht der Person gemeint.

## Installation

```bash
pip install -r requirements.txt
pip install face_recognition   # optional, für Namen
```

Das Pose-Modell (`yolo11n-pose.pt`, ca. 6 MB) wird beim ersten Start automatisch heruntergeladen.

## Namen (optional)

Wie im FaceRecognition-Projekt:

```
known_faces/
  Anna/  foto1.jpg  foto2.jpg
  Ben/   foto1.jpg
```

Ohne diesen Ordner heißen die Personen "Person 1", "Person 2" usw.

## Starten

```bash
python action_recognition_app.py
```

Beenden mit `q`. Jede Änderung wird in der Konsole ausgegeben, z. B.:

```
[14:02:11] Anna: steht, hebt linke Hand
[14:02:15] Ben: STURZ erkannt!
```

Nützliche Optionen:

| Option | Wirkung |
|---|---|
| `--log aktionen.csv` | Ausgaben zusätzlich in eine CSV-Datei schreiben |
| `--mirror` | Anzeige spiegeln (wie ein Spiegel) |
| `--camera 1` / `--camera video.mp4` | andere Kamera oder Videodatei |
| `--model yolo11s-pose.pt` | genaueres, aber langsameres Modell |
| `--no-faces` | Gesichtserkennung aus |

## Grenzen

- Sitzen wird am Kniewinkel erkannt. Das klappt am besten, wenn die Kamera die Person schräg oder von der Seite sieht. Von genau vorne sieht ein gebeugtes Knie fast gerade aus.
- Gehen wird nur bei seitlicher Bewegung durchs Bild erkannt, nicht beim Gehen direkt auf die Kamera zu.
- Die Schwellwerte stehen oben in der Datei und lassen sich anpassen.
- Namen gibt es nur, wenn das Gesicht zur Kamera zeigt. Danach bleibt der Name an der Person hängen, auch wenn sie sich wegdreht.
