# Grenzen & Teststand

## Was wurde wie getestet?

| Prüfung | Umfang | Ergebnis |
|---|---|---|
| Unit-Tests mit synthetischen Daten | alle 131 Erkennungen, Tracking, Ereignisse, Logging, Konfiguration, Fehlerfälle | 99/99 bestanden |
| Echte Modelle auf Beispielfotos | `bus.jpg` (4 Personen), `zidane.jpg` (2 Personen, Zeigegeste) aus Ultralytics | Personen, Haltung, Kopfrichtung und Zeigegeste plausibel |
| Komplette App mit Testvideo | künstliches Video (Foto, das seitlich durchs Bild gleitet), Intro → Dashboard → Logging | läuft stabil mit ca. 30 FPS; erkannte „Seitwärtsgehen ≈ 1 T/s“ korrekt |
| **Live-Webcam** | — | **nicht durchgeführt.** Alle Schwellwerte sollten mit echter Kamera überprüft/kalibriert werden. |

Nur mit echter Webcam überprüfbar: Live-Genauigkeit aller Erkennungen, Verhalten bei Licht/Abstand/Winkel,
Kamera-Trennen/Wiederverbinden, Gestensteuerung im Alltag, reale FPS auf deiner Hardware, Intro-Sound.

## Technische Grenzen

### 2D-Kamera
- Keine echte Tiefe. Sitzen vs. Hocke wird über vertikalen Hüft-Knöchel-Abstand, Kniewinkel und
  Oberkörperneigung unterschieden – frontal mit verdeckten Beinen mehrdeutig.
- **Rücken- vs. Bauch- vs. Seitenlage**: nur grobe Heuristik (Schulterbreite, sichtbare Augen) → experimentell.
- **Nach vorne / hinten lehnen** nur in Seitenansicht; **links/rechts lehnen** nur frontal oder von hinten.
- **Drehungen 90/180/360°** werden über die Folge vorne → seitlich → hinten → seitlich erkannt. Schnelle Drehungen
  (< ~0,3 s pro Viertel) oder schlecht sichtbare Schultern werden bewusst nicht gezählt.
- **Rücken zur Kamera**: YOLO vertauscht dabei manchmal linke/rechte Keypoints.
- Gehen direkt auf die Kamera zu/weg wird über Größenänderung + Schrittwechsel erkannt; Geschwindigkeitsklasse
  (langsam/schnell) nur bei Querbewegung.
- **Sturz** = schneller Hüftabfall um ≥ 1 Rumpflänge + Rumpf danach ≥ 55° geneigt. Ein absichtliches schnelles
  Hinlegen kann ebenfalls als möglicher Sturz gemeldet werden. **Kein Ersatz für ein Notfallsystem.**

### Hände (MediaPipe Hand Landmarker)
- Hände werden in Ausschnitten um die YOLO-Handgelenke gesucht. Ist das Handgelenk nicht sichtbar
  (z. B. nur eine Hand vor der Kamera ohne Körper), wird die Hand nicht ausgewertet.
- Überlappen sich Personen, kann eine Hand der falschen Person zugeordnet werden, wenn YOLO das Handgelenk
  selbst falsch platziert (beobachtet auf `zidane.jpg`). Doppelte Zuordnungen derselben Hand werden entfernt.
- Handfläche/Handrücken: aus der Umlaufrichtung der Knöchel – unzuverlässig, wenn die Hand seitlich steht → experimentell.
- „Zur Kamera zeigen“ nutzt die relative MediaPipe-Tiefe → experimentell.
- Zeigerichtung links/rechts aus Sicht der Person: setzt voraus, dass die Körperausrichtung (vorne/hinten) stimmt.

### Kopf & Blick (MediaPipe Face Landmarker)
- **Kopfrichtung** (Yaw/Roll) aus der Lage der Nasenspitze zwischen den Wangen bzw. der Augenlinie – Winkel sind
  Näherungen („~“). **Nicken/Pitch** wird pro Person adaptiv kalibriert und ist experimentell.
- **Blickrichtung** nur, wenn Augen ≥ 14 px breit, geöffnet, Kopf < 35° gedreht und beide Augen übereinstimmen.
  Sonst: „UNSICHER → NUR KOPFRICHTUNG“. Es gibt **kein** echtes 3D-Eye-Tracking (dafür bräuchte es eine
  kalibrierte IR-Eye-Tracking-Kamera).
- Ohne Face-Mesh-Treffer (Profil, weit entfernt) gibt es eine grobe Kopfrichtung aus den 5 YOLO-Gesichtspunkten
  (als experimentell markiert).

### Tracking
- Stabile IDs über ByteTrack + eigene Wiedererkennung nach Position/Bewegung/Größe innerhalb von 1–6 s
  (Tracking-Empfindlichkeit). Kein Aussehen-Merkmal → bei Kreuzungen sehr ähnlicher Personen sind ID-Wechsel möglich.
  Danach wird die Analysehistorie zurückgesetzt, statt Personen zu vermischen.
- Personen, die länger als die Wiedererkennungszeit verschwinden, erhalten eine neue ID.

### Performance
- MediaPipe-Ausschnitte kosten pro Person ~5–8 ms (CPU). Bei vielen Personen begrenzen `max_hands`/`max_faces`
  im Performance-Modus die Last (größte Personen zuerst).

## Nicht implementiert

- Echtes 3D-Eye-Tracking / Blickziel auf dem Bildschirm.
- Zuverlässige Unterscheidung Rücken-/Bauchlage aus einer einzelnen 2D-Ansicht.
- Langfristige Wiedererkennung nach Verlassen des Bildes (bewusst ohne Biometrie).
- Trainierte Klassifikatoren (z. B. LSTM auf Keypoint-Sequenzen): Die Engine ist so aufgebaut, dass ein Klassifikator
  später als weitere Quelle in `ActionEngine` ergänzt werden kann; mangels gelabelter Trainingsdaten wurde bewusst
  kein untrainiertes Modell eingebaut.
