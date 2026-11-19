"""Optional control of the program by hand gestures (OFF by default).

A gesture must be held continuously for `hold_seconds` before its command fires,
and after a command there is a cooldown, so random hand movements do not trigger anything.
"""
from __future__ import annotations

COMMANDS = {
    "thumbs_up": "confirm",
    "open_hand": "pause_tracking",
    "fist": "resume_tracking",
    "peace": "toggle_skeleton",
    "hands_above_head": "open_settings",
}


class GestureController:
    def __init__(self, hold_seconds: float = 1.0, cooldown: float = 2.0) -> None:
        self.hold_seconds = hold_seconds
        self.cooldown = cooldown
        self.current: str | None = None
        self.since = 0.0
        self.last_fire = -1e9
        self.fired_for_current = False

    def progress(self, now: float) -> float:
        if self.current is None:
            return 0.0
        return min(1.0, (now - self.since) / self.hold_seconds)

    def update(self, gesture_keys: set[str], now: float) -> str | None:
        """gesture_keys: all gesture/pose keys currently seen. Returns a command or None."""
        g = next((k for k in COMMANDS if k in gesture_keys), None)
        if g != self.current:
            self.current, self.since, self.fired_for_current = g, now, False
            return None
        if g is None or self.fired_for_current:
            return None
        if now - self.since >= self.hold_seconds and now - self.last_fire >= self.cooldown:
            self.last_fire = now
            self.fired_for_current = True
            return COMMANDS[g]
        return None
