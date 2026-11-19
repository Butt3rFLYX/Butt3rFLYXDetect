# Butt3rFLYX – Übersicht aller Erkennungen

_Automatisch erzeugt von `tools/gen_features.py` aus `core/labels.py` und `tests/`._

**Status-Legende**

- **Implementiert** – regelbasiert umgesetzt, im Interface ohne Markierung.
- **Experimentell** – umgesetzt, aber wegen 2D-Mehrdeutigkeit unzuverlässig; im Interface mit `(EXP)` bzw. gelb markiert.
- **Test** – `✔ synthetisch` = durch automatische Tests mit synthetischen Keypoints/Landmarks abgedeckt; `—` = implementiert, aber ohne eigenen Unit-Test.
- **Keine** Erkennung wurde bisher mit einer echten Webcam systematisch validiert (siehe `limitations.md`). Zusätzlich wurden die echten Modelle auf zwei Beispielfotos und einem künstlichen Testvideo geprüft.

## Körperhaltung

| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |
|---|---|---|---|---|---|
| Stehen | STANDING | Implementiert | YOLO Pose | ✔ synthetisch | `standing` |
| Sitzen | SITTING | Implementiert | YOLO Pose | ✔ synthetisch | `sitting` |
| Hocke | SQUATTING | Implementiert | YOLO Pose | ✔ synthetisch | `squatting` |
| Tiefe Hocke | DEEP SQUAT | Experimentell | YOLO Pose | ✔ synthetisch | `deep_squat` |
| Knien | KNEELING | Experimentell | YOLO Pose | ✔ synthetisch | `kneeling` |
| Auf Einem Knie | KNEELING ON ONE KNEE | Experimentell | YOLO Pose | ✔ synthetisch | `kneeling_one_knee` |
| Liegen | LYING | Implementiert | YOLO Pose | ✔ synthetisch | `lying` |
| Rückenlage | LYING ON BACK | Experimentell | YOLO Pose | ✔ synthetisch | `lying_back` |
| Bauchlage | LYING ON STOMACH | Experimentell | YOLO Pose | ✔ synthetisch | `lying_front` |
| Seitenlage | LYING ON SIDE | Experimentell | YOLO Pose | ✔ synthetisch | `lying_side` |
| Vorbeugen | BENDING FORWARD | Implementiert | YOLO Pose | ✔ synthetisch | `bending_forward` |
| Ducken | CROUCHING | Experimentell | YOLO Pose | ✔ synthetisch | `crouching` |
| Aufrecht (Oberkörper) | UPRIGHT (UPPER BODY) | Implementiert | YOLO Pose | ✔ synthetisch | `upright_upper_body` |
| Liegestütz-Position | PUSH-UP POSITION | Experimentell | YOLO Pose | ✔ synthetisch | `plank_position` |
| Ausfallschritt | LUNGE | Experimentell | YOLO Pose | ✔ synthetisch | `lunge` |

## Pose-Details

| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |
|---|---|---|---|---|---|
| Nach Vorne Gelehnt | LEANING FORWARD | Experimentell | YOLO Pose | ✔ synthetisch | `leaning_forward` |
| Nach Hinten Gelehnt | LEANING BACK | Experimentell | YOLO Pose | ✔ synthetisch | `leaning_back` |
| Nach Links Gelehnt | LEANING LEFT | Implementiert | YOLO Pose | ✔ synthetisch | `leaning_left` |
| Nach Rechts Gelehnt | LEANING RIGHT | Implementiert | YOLO Pose | ✔ synthetisch | `leaning_right` |
| Auf Einem Bein | STANDING ON ONE LEG | Implementiert | YOLO Pose | ✔ synthetisch | `standing_one_leg` |
| Arme Verschränkt | ARMS CROSSED | Experimentell | YOLO Pose | ✔ synthetisch | `arms_crossed` |
| Hände In Den Hüften | HANDS ON HIPS | Experimentell | YOLO Pose | ✔ synthetisch | `hands_on_hips` |
| Hände Über Dem Kopf | HANDS ABOVE HEAD | Implementiert | YOLO Pose | ✔ synthetisch | `hands_above_head` |
| Hände Hinter Dem Kopf | HANDS BEHIND HEAD | Experimentell | YOLO Pose | ✔ synthetisch | `hands_behind_head` |
| Hebt Linke Hand | RAISING LEFT HAND | Implementiert | YOLO Pose | ✔ synthetisch | `raising_left_hand` |
| Hebt Rechte Hand | RAISING RIGHT HAND | Implementiert | YOLO Pose | ✔ synthetisch | `raising_right_hand` |
| Hebt Beide Hände | RAISING BOTH HANDS | Implementiert | YOLO Pose | ✔ synthetisch | `raising_both_hands` |
| T-Pose | T-POSE | Implementiert | YOLO Pose | ✔ synthetisch | `t_pose` |
| Y-Pose | Y-POSE | Implementiert | YOLO Pose | ✔ synthetisch | `y_pose` |
| A-Pose | A-POSE | Experimentell | YOLO Pose | ✔ synthetisch | `a_pose` |
| Linker Arm Gestreckt | LEFT ARM EXTENDED | Implementiert | YOLO Pose | ✔ synthetisch | `left_arm_extended` |
| Rechter Arm Gestreckt | RIGHT ARM EXTENDED | Implementiert | YOLO Pose | ✔ synthetisch | `right_arm_extended` |
| Streckt Sich | STRETCHING UP | Implementiert | YOLO Pose | ✔ synthetisch | `reaching_up` |
| Hände Zusammen | HANDS TOGETHER | Experimentell | YOLO Pose | ✔ synthetisch | `hands_together_pose` |
| Linkes Knie Gehoben | LEFT KNEE RAISED | Implementiert | YOLO Pose | ✔ synthetisch | `knee_raised_left` |
| Rechtes Knie Gehoben | RIGHT KNEE RAISED | Implementiert | YOLO Pose | ✔ synthetisch | `knee_raised_right` |
| Oberkörper Gedreht | TORSO TWISTED | Experimentell | YOLO Pose | ✔ synthetisch | `torso_twisted` |

## Körperausrichtung

| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |
|---|---|---|---|---|---|
| Zur Kamera | FACING CAMERA | Implementiert | YOLO Pose | ✔ synthetisch | `facing_camera` |
| Rücken Zur Kamera | BACK TO CAMERA | Experimentell | YOLO Pose | ✔ synthetisch | `back_to_camera` |
| Seitlich | SIDEWAYS | Implementiert | YOLO Pose | ✔ synthetisch | `side_on` |

## Bewegung

| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |
|---|---|---|---|---|---|
| Ruhig | IDLE | Implementiert | YOLO Pose | ✔ synthetisch | `idle` |
| Stillstehen | STANDING STILL | Implementiert | YOLO Pose | ✔ synthetisch | `standing_still` |
| Langsames Gehen | WALKING SLOWLY | Implementiert | YOLO Pose | ✔ synthetisch | `walking_slow` |
| Gehen | WALKING | Implementiert | YOLO Pose | ✔ synthetisch | `walking` |
| Schnelles Gehen | WALKING FAST | Implementiert | YOLO Pose | ✔ synthetisch | `walking_fast` |
| Laufen | RUNNING | Implementiert | YOLO Pose | ✔ synthetisch | `running` |
| Rennen | SPRINTING | Experimentell | YOLO Pose | ✔ synthetisch | `sprinting` |
| Springen | JUMPING | Implementiert | YOLO Pose | ✔ synthetisch | `jumping` |
| Wiederholtes Springen | JUMPING REPEATEDLY | Implementiert | YOLO Pose | ✔ synthetisch | `jumping_repeatedly` |
| Auf Einem Bein Hüpfen | HOPPING ON ONE LEG | Experimentell | YOLO Pose | ✔ synthetisch | `hopping_one_leg` |
| Seitwärtsgehen | SIDESTEPPING | Experimentell | YOLO Pose | ✔ synthetisch | `sidestepping` |
| Rückwärtsgehen | WALKING BACKWARD | Experimentell | YOLO Pose | ✔ synthetisch | `walking_backward` |
| Auf Kamera Zu | APPROACHING CAMERA | Implementiert | YOLO Pose | ✔ synthetisch | `approaching_camera` |
| Von Kamera Weg | MOVING AWAY | Implementiert | YOLO Pose | ✔ synthetisch | `moving_away` |
| Gehen Auf Der Stelle | WALKING IN PLACE | Experimentell | YOLO Pose | ✔ synthetisch | `walking_in_place` |
| Laufen Auf Der Stelle | RUNNING IN PLACE | Experimentell | YOLO Pose | ✔ synthetisch | `running_in_place` |
| Möglicher Sturz | POSSIBLE FALL | Implementiert | YOLO Pose | ✔ synthetisch | `falling` |
| Stolpern | STUMBLE-LIKE MOTION | Experimentell | YOLO Pose | ✔ synthetisch | `stumbling` |
| Dreht Sich | TURNING | Experimentell | YOLO Pose | ✔ synthetisch | `turning` |

## Bewegungsrichtung

| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |
|---|---|---|---|---|---|
| Bewegt Sich Nach Bild-Links | MOVING SCREEN-LEFT | Implementiert | YOLO Pose | ✔ synthetisch | `moving_screen_left` |
| Bewegt Sich Nach Bild-Rechts | MOVING SCREEN-RIGHT | Implementiert | YOLO Pose | ✔ synthetisch | `moving_screen_right` |

## Zustandswechsel

| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |
|---|---|---|---|---|---|
| Setzt Sich | SITTING DOWN | Implementiert | YOLO Pose | ✔ synthetisch | `sitting_down` |
| Steht Auf | STANDING UP | Implementiert | YOLO Pose | ✔ synthetisch | `standing_up` |
| Geht In Die Hocke | GOING INTO SQUAT | Implementiert | YOLO Pose | ✔ synthetisch | `squatting_down` |
| Aus Der Hocke Auf | RISING FROM SQUAT | Implementiert | YOLO Pose | ✔ synthetisch | `rising_from_squat` |
| Bückt Sich | BENDING DOWN | Implementiert | YOLO Pose | ✔ synthetisch | `bending_down` |
| Richtet Sich Auf | STRAIGHTENING UP | Implementiert | YOLO Pose | ✔ synthetisch | `straightening_up` |
| Legt Sich Hin | LYING DOWN | Implementiert | YOLO Pose | ✔ synthetisch | `lying_down` |
| Steht Vom Boden Auf | GETTING UP FROM FLOOR | Implementiert | YOLO Pose | ✔ synthetisch | `getting_up` |
| 90° Gedreht | TURNED 90° | Experimentell | YOLO Pose | ✔ synthetisch | `turned_90` |
| 180° Gedreht | TURNED 180° | Experimentell | YOLO Pose | ✔ synthetisch | `turned_180` |
| Volle Drehung 360° | FULL TURN 360° | Experimentell | YOLO Pose | ✔ synthetisch | `turned_360` |

## Sport / Wiederholungen

| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |
|---|---|---|---|---|---|
| Kniebeugen | SQUATS | Implementiert | YOLO Pose | ✔ synthetisch | `squat_reps` |
| Hampelmann | JUMPING JACKS | Experimentell | YOLO Pose | ✔ synthetisch | `jumping_jacks` |
| Armkreisen | ARM CIRCLES | Experimentell | YOLO Pose | ✔ synthetisch | `arm_circles` |
| Wiederholte Armbewegung | REPEATED ARM MOVEMENT | Implementiert | YOLO Pose | ✔ synthetisch | `repetitive_arm_motion` |
| Wiederholte Beinbewegung | REPEATED LEG MOVEMENT | Experimentell | YOLO Pose | ✔ synthetisch | `repetitive_leg_motion` |
| Winkt (Links) | WAVING (LEFT HAND) | Implementiert | YOLO Pose | ✔ synthetisch | `waving_left_hand` |
| Winkt (Rechts) | WAVING (RIGHT HAND) | Implementiert | YOLO Pose | ✔ synthetisch | `waving_right_hand` |
| Beinheben | LEG RAISES | Implementiert | YOLO Pose | ✔ synthetisch | `leg_raises` |
| Balance Auf Einem Bein | BALANCING ON ONE LEG | Implementiert | YOLO Pose | ✔ synthetisch | `balance_one_leg` |

## Kopf

| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |
|---|---|---|---|---|---|
| Geradeaus | FORWARD | Implementiert | MediaPipe Face Landmarker | ✔ synthetisch | `head_forward` |
| Nach Links Gedreht | TURNED LEFT | Implementiert | MediaPipe Face Landmarker | ✔ synthetisch | `head_left` |
| Nach Rechts Gedreht | TURNED RIGHT | Implementiert | MediaPipe Face Landmarker | ✔ synthetisch | `head_right` |
| Nach Oben | UP | Experimentell | MediaPipe Face Landmarker | ✔ synthetisch | `head_up` |
| Nach Unten | DOWN | Experimentell | MediaPipe Face Landmarker | ✔ synthetisch | `head_down` |
| Nach Links Geneigt | TILTED LEFT | Implementiert | MediaPipe Face Landmarker | ✔ synthetisch | `head_tilted_left` |
| Nach Rechts Geneigt | TILTED RIGHT | Implementiert | MediaPipe Face Landmarker | ✔ synthetisch | `head_tilted_right` |
| Von Kamera Weg | AWAY FROM CAMERA | Experimentell | YOLO Pose | ✔ synthetisch | `head_away` |
| Kopf Dreht Sich | HEAD TURNING | Experimentell | MediaPipe Face Landmarker | ✔ synthetisch | `head_turning` |
| Schaut Zur Kamera (Ca.) | LOOKING AT CAMERA (APPROX.) | Experimentell | MediaPipe Face Landmarker | ✔ synthetisch | `looking_at_camera` |
| Nickt | NODDING | Experimentell | MediaPipe Face Landmarker | ✔ synthetisch | `nodding` |
| Schüttelt Kopf | SHAKING HEAD | Experimentell | MediaPipe Face Landmarker | ✔ synthetisch | `shaking_head` |

## Blick (Augen)

| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |
|---|---|---|---|---|---|
| Links | LEFT | Experimentell | MediaPipe Face Landmarker | ✔ synthetisch | `gaze_left` |
| Rechts | RIGHT | Experimentell | MediaPipe Face Landmarker | ✔ synthetisch | `gaze_right` |
| Oben | UP | Experimentell | MediaPipe Face Landmarker | ✔ synthetisch | `gaze_up` |
| Unten | DOWN | Experimentell | MediaPipe Face Landmarker | ✔ synthetisch | `gaze_down` |
| Mitte | CENTER | Experimentell | MediaPipe Face Landmarker | ✔ synthetisch | `gaze_center` |
| Augen Geschlossen | EYES CLOSED | Experimentell | MediaPipe Face Landmarker | ✔ synthetisch | `eyes_closed` |

## Handgeste

| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |
|---|---|---|---|---|---|
| Offene Hand | OPEN HAND | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `open_hand` |
| Faust | FIST | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `fist` |
| Daumen Hoch | THUMBS UP | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `thumbs_up` |
| Daumen Runter | THUMBS DOWN | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `thumbs_down` |
| Peace-Zeichen | PEACE | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `peace` |
| Ok-Zeichen | OK SIGN | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `ok_sign` |
| Zeigt Nach Oben | POINTING UP | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `pointing_up` |
| Zeigt Nach Unten | POINTING DOWN | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `pointing_down` |
| Zeigt Nach Links | POINTING LEFT | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `pointing_left` |
| Zeigt Nach Rechts | POINTING RIGHT | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `pointing_right` |
| Zeigt Zur Kamera | POINTING AT CAMERA | Experimentell | MediaPipe Hand Landmarker | ✔ synthetisch | `pointing_at_camera` |
| Rock-Zeichen | ROCK SIGN | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `rock` |
| Drei Finger | THREE FINGERS | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `three_fingers` |
| Vier Finger | FOUR FINGERS | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `four_fingers` |
| Handfläche Zur Kamera | PALM TO CAMERA | Experimentell | MediaPipe Hand Landmarker | ✔ synthetisch | `palm_facing_camera` |
| Handrücken Zur Kamera | BACK OF HAND | Experimentell | MediaPipe Hand Landmarker | ✔ synthetisch | `back_of_hand` |
| Hand Vor Dem Gesicht | HAND IN FRONT OF FACE | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `hand_over_face` |

## Handbewegung

| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |
|---|---|---|---|---|---|
| Winken | WAVING | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `waving_hand` |
| Hand Nach Bild-Links | HAND MOVING SCREEN-LEFT | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `hand_moving_left` |
| Hand Nach Bild-Rechts | HAND MOVING SCREEN-RIGHT | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `hand_moving_right` |
| Hand Nach Oben | HAND MOVING UP | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `hand_moving_up` |
| Hand Nach Unten | HAND MOVING DOWN | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `hand_moving_down` |
| Hand Öffnet Sich | HAND OPENING | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `hand_opening` |
| Hand Schliesst Sich | HAND CLOSING | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `hand_closing` |
| Hand Dreht Sich | HAND ROTATING | Experimentell | MediaPipe Hand Landmarker | ✔ synthetisch | `hand_rotating` |
| Finger Bewegen Sich | FINGERS MOVING | Experimentell | MediaPipe Hand Landmarker | ✔ synthetisch | `fingers_moving` |

## Zwei Hände

| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |
|---|---|---|---|---|---|
| Hände Zusammen | HANDS TOGETHER | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `hands_together` |
| Klatschen | CLAPPING | Experimentell | MediaPipe Hand Landmarker | ✔ synthetisch | `clapping` |
| Herz-Form | HEART SHAPE | Experimentell | MediaPipe Hand Landmarker | ✔ synthetisch | `heart_hands` |
| Zeigt Mit Beiden Händen | POINTING WITH BOTH HANDS | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `pointing_both_hands` |
| Hände Auseinander | HANDS MOVING APART | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `hands_moving_apart` |
| Hände Zueinander | HANDS MOVING TOGETHER | Implementiert | MediaPipe Hand Landmarker | ✔ synthetisch | `hands_moving_together` |

## Nicht implementiert

| Funktion | Grund |
|---|---|
| `true_3d_eye_tracking` | Exact 3D gaze target (needs calibrated eye tracker / IR camera). |
| `lying_subtype_reliable` | Reliable back/stomach distinction from a single 2D view. |
| `full_identity_tracking` | Long-term identity after leaving the frame (only short re-acquisition, no biometrics). |

**Summe:** 131 Erkennungen, davon 52 experimentell, 131 mit eigenem synthetischem Test.
