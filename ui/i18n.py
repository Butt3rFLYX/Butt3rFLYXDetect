"""UI strings in English and German (recognition labels live in core/labels.py)."""
from __future__ import annotations

STRINGS = {
    "system_online": ("SYSTEM ONLINE", "SYSTEM ONLINE"),
    "system_loading": ("SYSTEM BOOTING", "SYSTEM STARTET"),
    "camera_active": ("CAMERA ACTIVE", "KAMERA AKTIV"),
    "camera_lost": ("CAMERA OFFLINE", "KAMERA OFFLINE"),
    "tracked": ("TRACKED PERSONS", "PERSONEN"),
    "posture": ("POSTURE", "HALTUNG"),
    "movement": ("MOVEMENT", "BEWEGUNG"),
    "head": ("HEAD", "KOPF"),
    "gaze": ("GAZE", "BLICK"),
    "facing": ("BODY", "KÖRPER"),
    "left_hand": ("LEFT HAND", "LINKE HAND"),
    "right_hand": ("RIGHT HAND", "RECHTE HAND"),
    "actions": ("ACTIONS", "AKTIONEN"),
    "confidence": ("CONFIDENCE", "KONFIDENZ"),
    "uncertain": ("UNCERTAIN", "UNSICHER"),
    "not_visible": ("NOT VISIBLE", "NICHT SICHTBAR"),
    "unknown": ("UNKNOWN", "UNBEKANNT"),
    "fingers": ("FINGERS", "FINGER"),
    "paused": ("TRACKING PAUSED", "TRACKING PAUSIERT"),
    "no_signal": ("NO SIGNAL", "KEIN SIGNAL"),
    "events": ("EVENT TIMELINE", "EREIGNIS-TIMELINE"),
    "live_analysis": ("LIVE ANALYSIS", "LIVE-ANALYSE"),
    "ai_modules": ("AI MODULES", "KI-MODULE"),
    "settings": ("SETTINGS", "EINSTELLUNGEN"),
    "pause": ("PAUSE", "PAUSE"),
    "record": ("REC", "AUFNAHME"),
    "export": ("EXPORT", "EXPORT"),
    "logging": ("LOGGING", "PROTOKOLL"),
    "no_persons": ("NO PERSONS DETECTED", "KEINE PERSONEN ERKANNT"),
    "head_only": ("HEAD DIRECTION ONLY", "NUR KOPFRICHTUNG"),
    "exp_note": ("(EXP) = experimental detection", "(EXP) = experimentelle Erkennung"),
    "tracked_for": ("TRACKED", "VERFOLGT"),
    "gesture_ctrl": ("GESTURE CONTROL", "GESTENSTEUERUNG"),
}


def tr(key: str, lang: str = "en") -> str:
    v = STRINGS.get(key)
    if v is None:
        return key.upper()
    return v[1] if lang == "de" else v[0]
