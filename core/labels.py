"""Central registry of every recognition label Butt3rFLYX can output.

Each label has a category, a status and the model it needs.
  status "stable"       -> rule-based, covered by synthetic unit tests
  status "experimental" -> works in principle, but 2D-camera ambiguity makes it
                           unreliable; marked with "(EXP)" in the interface.
Nothing here is shown unless it was actually computed from the camera frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

POSE, HANDS, FACE = "YOLO Pose", "MediaPipe Hand Landmarker", "MediaPipe Face Landmarker"
S, E = "stable", "experimental"

# key: (category, status, model, english, german)
LABELS: dict[str, tuple[str, str, str, str, str]] = {
    # ---------------- base postures ----------------
    "standing": ("posture", S, POSE, "STANDING", "STEHEN"),
    "sitting": ("posture", S, POSE, "SITTING", "SITZEN"),
    "squatting": ("posture", S, POSE, "SQUATTING", "HOCKE"),
    "deep_squat": ("posture", E, POSE, "DEEP SQUAT", "TIEFE HOCKE"),
    "kneeling": ("posture", E, POSE, "KNEELING", "KNIEN"),
    "kneeling_one_knee": ("posture", E, POSE, "KNEELING ON ONE KNEE", "AUF EINEM KNIE"),
    "lying": ("posture", S, POSE, "LYING", "LIEGEN"),
    "lying_back": ("posture", E, POSE, "LYING ON BACK", "RÜCKENLAGE"),
    "lying_front": ("posture", E, POSE, "LYING ON STOMACH", "BAUCHLAGE"),
    "lying_side": ("posture", E, POSE, "LYING ON SIDE", "SEITENLAGE"),
    "bending_forward": ("posture", S, POSE, "BENDING FORWARD", "VORBEUGEN"),
    "crouching": ("posture", E, POSE, "CROUCHING", "DUCKEN"),
    "upright_upper_body": ("posture", S, POSE, "UPRIGHT (UPPER BODY)", "AUFRECHT (OBERKÖRPER)"),
    "plank_position": ("posture", E, POSE, "PUSH-UP POSITION", "LIEGESTÜTZ-POSITION"),
    "lunge": ("posture", E, POSE, "LUNGE", "AUSFALLSCHRITT"),
    # ---------------- posture modifiers ----------------
    "leaning_forward": ("pose", E, POSE, "LEANING FORWARD", "NACH VORNE GELEHNT"),
    "leaning_back": ("pose", E, POSE, "LEANING BACK", "NACH HINTEN GELEHNT"),
    "leaning_left": ("pose", S, POSE, "LEANING LEFT", "NACH LINKS GELEHNT"),
    "leaning_right": ("pose", S, POSE, "LEANING RIGHT", "NACH RECHTS GELEHNT"),
    "standing_one_leg": ("pose", S, POSE, "STANDING ON ONE LEG", "AUF EINEM BEIN"),
    "arms_crossed": ("pose", E, POSE, "ARMS CROSSED", "ARME VERSCHRÄNKT"),
    "hands_on_hips": ("pose", E, POSE, "HANDS ON HIPS", "HÄNDE IN DEN HÜFTEN"),
    "hands_above_head": ("pose", S, POSE, "HANDS ABOVE HEAD", "HÄNDE ÜBER DEM KOPF"),
    "hands_behind_head": ("pose", E, POSE, "HANDS BEHIND HEAD", "HÄNDE HINTER DEM KOPF"),
    "raising_left_hand": ("pose", S, POSE, "RAISING LEFT HAND", "HEBT LINKE HAND"),
    "raising_right_hand": ("pose", S, POSE, "RAISING RIGHT HAND", "HEBT RECHTE HAND"),
    "raising_both_hands": ("pose", S, POSE, "RAISING BOTH HANDS", "HEBT BEIDE HÄNDE"),
    "t_pose": ("pose", S, POSE, "T-POSE", "T-POSE"),
    "y_pose": ("pose", S, POSE, "Y-POSE", "Y-POSE"),
    "a_pose": ("pose", E, POSE, "A-POSE", "A-POSE"),
    "left_arm_extended": ("pose", S, POSE, "LEFT ARM EXTENDED", "LINKER ARM GESTRECKT"),
    "right_arm_extended": ("pose", S, POSE, "RIGHT ARM EXTENDED", "RECHTER ARM GESTRECKT"),
    "reaching_up": ("pose", S, POSE, "STRETCHING UP", "STRECKT SICH"),
    "hands_together_pose": ("pose", E, POSE, "HANDS TOGETHER", "HÄNDE ZUSAMMEN"),
    "knee_raised_left": ("pose", S, POSE, "LEFT KNEE RAISED", "LINKES KNIE GEHOBEN"),
    "knee_raised_right": ("pose", S, POSE, "RIGHT KNEE RAISED", "RECHTES KNIE GEHOBEN"),
    "torso_twisted": ("pose", E, POSE, "TORSO TWISTED", "OBERKÖRPER GEDREHT"),
    # ---------------- body orientation ----------------
    "facing_camera": ("facing", S, POSE, "FACING CAMERA", "ZUR KAMERA"),
    "back_to_camera": ("facing", E, POSE, "BACK TO CAMERA", "RÜCKEN ZUR KAMERA"),
    "side_on": ("facing", S, POSE, "SIDEWAYS", "SEITLICH"),
    # ---------------- movement ----------------
    "idle": ("movement", S, POSE, "IDLE", "RUHIG"),
    "standing_still": ("movement", S, POSE, "STANDING STILL", "STILLSTEHEN"),
    "walking_slow": ("movement", S, POSE, "WALKING SLOWLY", "LANGSAMES GEHEN"),
    "walking": ("movement", S, POSE, "WALKING", "GEHEN"),
    "walking_fast": ("movement", S, POSE, "WALKING FAST", "SCHNELLES GEHEN"),
    "running": ("movement", S, POSE, "RUNNING", "LAUFEN"),
    "sprinting": ("movement", E, POSE, "SPRINTING", "RENNEN"),
    "jumping": ("movement", S, POSE, "JUMPING", "SPRINGEN"),
    "jumping_repeatedly": ("movement", S, POSE, "JUMPING REPEATEDLY", "WIEDERHOLTES SPRINGEN"),
    "hopping_one_leg": ("movement", E, POSE, "HOPPING ON ONE LEG", "AUF EINEM BEIN HÜPFEN"),
    "sidestepping": ("movement", E, POSE, "SIDESTEPPING", "SEITWÄRTSGEHEN"),
    "walking_backward": ("movement", E, POSE, "WALKING BACKWARD", "RÜCKWÄRTSGEHEN"),
    "approaching_camera": ("movement", S, POSE, "APPROACHING CAMERA", "AUF KAMERA ZU"),
    "moving_away": ("movement", S, POSE, "MOVING AWAY", "VON KAMERA WEG"),
    "walking_in_place": ("movement", E, POSE, "WALKING IN PLACE", "GEHEN AUF DER STELLE"),
    "running_in_place": ("movement", E, POSE, "RUNNING IN PLACE", "LAUFEN AUF DER STELLE"),
    "falling": ("movement", S, POSE, "POSSIBLE FALL", "MÖGLICHER STURZ"),
    "stumbling": ("movement", E, POSE, "STUMBLE-LIKE MOTION", "STOLPERN"),
    "turning": ("movement", E, POSE, "TURNING", "DREHT SICH"),
    # movement direction in IMAGE coordinates (camera view)
    "moving_screen_left": ("direction", S, POSE, "MOVING SCREEN-LEFT", "BEWEGT SICH NACH BILD-LINKS"),
    "moving_screen_right": ("direction", S, POSE, "MOVING SCREEN-RIGHT", "BEWEGT SICH NACH BILD-RECHTS"),
    # ---------------- transitions (short-lived actions) ----------------
    "sitting_down": ("transition", S, POSE, "SITTING DOWN", "SETZT SICH"),
    "standing_up": ("transition", S, POSE, "STANDING UP", "STEHT AUF"),
    "squatting_down": ("transition", S, POSE, "GOING INTO SQUAT", "GEHT IN DIE HOCKE"),
    "rising_from_squat": ("transition", S, POSE, "RISING FROM SQUAT", "AUS DER HOCKE AUF"),
    "bending_down": ("transition", S, POSE, "BENDING DOWN", "BÜCKT SICH"),
    "straightening_up": ("transition", S, POSE, "STRAIGHTENING UP", "RICHTET SICH AUF"),
    "lying_down": ("transition", S, POSE, "LYING DOWN", "LEGT SICH HIN"),
    "getting_up": ("transition", S, POSE, "GETTING UP FROM FLOOR", "STEHT VOM BODEN AUF"),
    "turned_90": ("transition", E, POSE, "TURNED 90°", "90° GEDREHT"),
    "turned_180": ("transition", E, POSE, "TURNED 180°", "180° GEDREHT"),
    "turned_360": ("transition", E, POSE, "FULL TURN 360°", "VOLLE DREHUNG 360°"),
    # ---------------- exercises / repetitive ----------------
    "squat_reps": ("exercise", S, POSE, "SQUATS", "KNIEBEUGEN"),
    "jumping_jacks": ("exercise", E, POSE, "JUMPING JACKS", "HAMPELMANN"),
    "arm_circles": ("exercise", E, POSE, "ARM CIRCLES", "ARMKREISEN"),
    "repetitive_arm_motion": ("exercise", S, POSE, "REPEATED ARM MOVEMENT", "WIEDERHOLTE ARMBEWEGUNG"),
    "repetitive_leg_motion": ("exercise", E, POSE, "REPEATED LEG MOVEMENT", "WIEDERHOLTE BEINBEWEGUNG"),
    "waving_left_hand": ("exercise", S, POSE, "WAVING (LEFT HAND)", "WINKT (LINKS)"),
    "waving_right_hand": ("exercise", S, POSE, "WAVING (RIGHT HAND)", "WINKT (RECHTS)"),
    "leg_raises": ("exercise", S, POSE, "LEG RAISES", "BEINHEBEN"),
    "balance_one_leg": ("exercise", S, POSE, "BALANCING ON ONE LEG", "BALANCE AUF EINEM BEIN"),
    # ---------------- head (pose = coarse, face = face mesh) ----------------
    "head_forward": ("head", S, FACE, "FORWARD", "GERADEAUS"),
    "head_left": ("head", S, FACE, "TURNED LEFT", "NACH LINKS GEDREHT"),
    "head_right": ("head", S, FACE, "TURNED RIGHT", "NACH RECHTS GEDREHT"),
    "head_up": ("head", E, FACE, "UP", "NACH OBEN"),
    "head_down": ("head", E, FACE, "DOWN", "NACH UNTEN"),
    "head_tilted_left": ("head", S, FACE, "TILTED LEFT", "NACH LINKS GENEIGT"),
    "head_tilted_right": ("head", S, FACE, "TILTED RIGHT", "NACH RECHTS GENEIGT"),
    "head_away": ("head", E, POSE, "AWAY FROM CAMERA", "VON KAMERA WEG"),
    "head_turning": ("head", E, FACE, "HEAD TURNING", "KOPF DREHT SICH"),
    "looking_at_camera": ("head", E, FACE, "LOOKING AT CAMERA (APPROX.)", "SCHAUT ZUR KAMERA (CA.)"),
    "nodding": ("head", E, FACE, "NODDING", "NICKT"),
    "shaking_head": ("head", E, FACE, "SHAKING HEAD", "SCHÜTTELT KOPF"),
    "gaze_left": ("gaze", E, FACE, "LEFT", "LINKS"),
    "gaze_right": ("gaze", E, FACE, "RIGHT", "RECHTS"),
    "gaze_up": ("gaze", E, FACE, "UP", "OBEN"),
    "gaze_down": ("gaze", E, FACE, "DOWN", "UNTEN"),
    "gaze_center": ("gaze", E, FACE, "CENTER", "MITTE"),
    "eyes_closed": ("gaze", E, FACE, "EYES CLOSED", "AUGEN GESCHLOSSEN"),
    # ---------------- single-hand gestures ----------------
    "open_hand": ("gesture", S, HANDS, "OPEN HAND", "OFFENE HAND"),
    "fist": ("gesture", S, HANDS, "FIST", "FAUST"),
    "thumbs_up": ("gesture", S, HANDS, "THUMBS UP", "DAUMEN HOCH"),
    "thumbs_down": ("gesture", S, HANDS, "THUMBS DOWN", "DAUMEN RUNTER"),
    "peace": ("gesture", S, HANDS, "PEACE", "PEACE-ZEICHEN"),
    "ok_sign": ("gesture", S, HANDS, "OK SIGN", "OK-ZEICHEN"),
    "pointing_up": ("gesture", S, HANDS, "POINTING UP", "ZEIGT NACH OBEN"),
    "pointing_down": ("gesture", S, HANDS, "POINTING DOWN", "ZEIGT NACH UNTEN"),
    "pointing_left": ("gesture", S, HANDS, "POINTING LEFT", "ZEIGT NACH LINKS"),
    "pointing_right": ("gesture", S, HANDS, "POINTING RIGHT", "ZEIGT NACH RECHTS"),
    "pointing_at_camera": ("gesture", E, HANDS, "POINTING AT CAMERA", "ZEIGT ZUR KAMERA"),
    "rock": ("gesture", S, HANDS, "ROCK SIGN", "ROCK-ZEICHEN"),
    "three_fingers": ("gesture", S, HANDS, "THREE FINGERS", "DREI FINGER"),
    "four_fingers": ("gesture", S, HANDS, "FOUR FINGERS", "VIER FINGER"),
    "palm_facing_camera": ("gesture", E, HANDS, "PALM TO CAMERA", "HANDFLÄCHE ZUR KAMERA"),
    "back_of_hand": ("gesture", E, HANDS, "BACK OF HAND", "HANDRÜCKEN ZUR KAMERA"),
    "hand_over_face": ("gesture", S, HANDS, "HAND IN FRONT OF FACE", "HAND VOR DEM GESICHT"),
    # ---------------- dynamic hand gestures ----------------
    "waving_hand": ("hand_motion", S, HANDS, "WAVING", "WINKEN"),
    "hand_moving_left": ("hand_motion", S, HANDS, "HAND MOVING SCREEN-LEFT", "HAND NACH BILD-LINKS"),
    "hand_moving_right": ("hand_motion", S, HANDS, "HAND MOVING SCREEN-RIGHT", "HAND NACH BILD-RECHTS"),
    "hand_moving_up": ("hand_motion", S, HANDS, "HAND MOVING UP", "HAND NACH OBEN"),
    "hand_moving_down": ("hand_motion", S, HANDS, "HAND MOVING DOWN", "HAND NACH UNTEN"),
    "hand_opening": ("hand_motion", S, HANDS, "HAND OPENING", "HAND ÖFFNET SICH"),
    "hand_closing": ("hand_motion", S, HANDS, "HAND CLOSING", "HAND SCHLIESST SICH"),
    "hand_rotating": ("hand_motion", E, HANDS, "HAND ROTATING", "HAND DREHT SICH"),
    "fingers_moving": ("hand_motion", E, HANDS, "FINGERS MOVING", "FINGER BEWEGEN SICH"),
    # ---------------- two-hand gestures ----------------
    "hands_together": ("two_hands", S, HANDS, "HANDS TOGETHER", "HÄNDE ZUSAMMEN"),
    "clapping": ("two_hands", E, HANDS, "CLAPPING", "KLATSCHEN"),
    "heart_hands": ("two_hands", E, HANDS, "HEART SHAPE", "HERZ-FORM"),
    "pointing_both_hands": ("two_hands", S, HANDS, "POINTING WITH BOTH HANDS", "ZEIGT MIT BEIDEN HÄNDEN"),
    "hands_moving_apart": ("two_hands", S, HANDS, "HANDS MOVING APART", "HÄNDE AUSEINANDER"),
    "hands_moving_together": ("two_hands", S, HANDS, "HANDS MOVING TOGETHER", "HÄNDE ZUEINANDER"),
}

# Things that were requested but are NOT implemented (honest list for the docs/UI)
NOT_IMPLEMENTED: dict[str, str] = {
    "true_3d_eye_tracking": "Exact 3D gaze target (needs calibrated eye tracker / IR camera).",
    "lying_subtype_reliable": "Reliable back/stomach distinction from a single 2D view.",
    "full_identity_tracking": "Long-term identity after leaving the frame (only short re-acquisition, no biometrics).",
}


def is_experimental(key: str) -> bool:
    return LABELS.get(key, ("", E))[1] == E


def label_text(key: str, lang: str = "en") -> str:
    entry = LABELS.get(key)
    if entry is None:
        return key.replace("_", " ").upper()
    return entry[4] if lang == "de" else entry[3]


@dataclass
class Label:
    """One recognised state with a computed confidence (0..1)."""
    key: str
    confidence: float
    detail: str = ""                      # e.g. "x3", "~25°"
    extra: dict = field(default_factory=dict)

    @property
    def experimental(self) -> bool:
        return is_experimental(self.key)

    @property
    def category(self) -> str:
        return LABELS.get(self.key, ("other",))[0]

    def text(self, lang: str = "en", mark_experimental: bool = True) -> str:
        t = label_text(self.key, lang)
        if self.detail:
            t = f"{t} {self.detail}"
        if mark_experimental and self.experimental:
            t += " (EXP)"
        return t
