"""Turns PersonSnapshots into plain, thread-safe display data for the dashboard."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.action_engine import PersonSnapshot
from core.hand_gestures import HandObservation
from core.labels import Label, label_text

from .i18n import tr

FINGER_LETTERS = (("thumb", "T"), ("index", "I"), ("middle", "M"), ("ring", "R"), ("pinky", "P"))


@dataclass
class Row:
    title: str
    value: str
    confidence: float | None = None
    experimental: bool = False
    dim: bool = False


@dataclass
class PersonView:
    pid: int
    name: str
    confidence: float
    color_state: str
    tracked_for: float
    rows: list[Row] = field(default_factory=list)
    actions: list[tuple[str, bool]] = field(default_factory=list)   # (text, experimental)
    recent_events: list[str] = field(default_factory=list)


def _row(title: str, lab: Label | None, lang: str, fallback: str) -> Row:
    if lab is None:
        return Row(title, fallback, dim=True)
    return Row(title, lab.text(lang, mark_experimental=False), lab.confidence, lab.experimental)


def finger_string(h: HandObservation) -> str:
    return " ".join(letter if h.fingers.get(name) else "·" for name, letter in FINGER_LETTERS)


def hand_row(title: str, h: HandObservation | None, lang: str) -> Row:
    if h is None:
        return Row(title, tr("not_visible", lang), dim=True)
    p = h.primary
    fs = f"[{finger_string(h)}] {h.count}"
    if p is None:
        return Row(title, f"{tr('fingers', lang)} {fs}", h.score)
    extra = ""
    if h.palm is not None:
        extra = " · " + label_text(h.palm.key, lang)
    return Row(title, f"{p.text(lang, False)}  {fs}{extra}", p.confidence, p.experimental)


def build_view(s: PersonSnapshot, lang: str) -> PersonView:
    m = s.motion
    v = PersonView(s.pid, s.name, s.confidence, s.color_state, s.tracked_for, recent_events=s.recent_events[:4])
    v.rows.append(_row(tr("posture", lang), m.posture, lang, tr("unknown", lang)))
    mv = _row(tr("movement", lang), m.movement, lang, "—")
    if m.direction is not None and m.movement is not None:
        mv.value += "  " + ("◄" if m.direction.key == "moving_screen_left" else "►")
    v.rows.append(mv)
    v.rows.append(_row(tr("facing", lang), m.facing, lang, "—"))
    head = s.head
    if head is None:
        v.rows.append(Row(tr("head", lang), tr("not_visible", lang), dim=True))
        v.rows.append(Row(tr("gaze", lang), "—", dim=True))
    else:
        r = _row(tr("head", lang), head.direction, lang, tr("unknown", lang))
        if head.tilt is not None:
            r.value += " · " + head.tilt.text(lang, False)
        if head.source == "pose":       # coarse estimate from the 5 YOLO face keypoints
            r.experimental = True
        v.rows.append(r)
        if head.gaze_status == "ok" and head.gaze is not None:
            v.rows.append(_row(tr("gaze", lang), head.gaze, lang, "—"))
        elif head.eyes_closed is not None:
            v.rows.append(_row(tr("gaze", lang), head.eyes_closed, lang, "—"))
        elif head.gaze_status == "uncertain":
            v.rows.append(Row(tr("gaze", lang), f"{tr('uncertain', lang)} → {tr('head_only', lang)}", dim=True))
        else:
            v.rows.append(Row(tr("gaze", lang), "—", dim=True))
    v.rows.append(hand_row(tr("left_hand", lang), s.hands.get("left"), lang))
    v.rows.append(hand_row(tr("right_hand", lang), s.hands.get("right"), lang))
    acts: list[Label] = list(m.transitions) + list(m.modifiers) + list(m.exercises) + list(s.two_hand)
    if head is not None:
        acts += list(head.motions)
        if head.looking is not None:
            acts.append(head.looking)
    for h in s.hands.values():
        if h is not None:
            acts += h.motions
            acts += [g for g in h.gestures if g.key == "hand_over_face"]
    seen = set()
    for a in acts:
        if a.key in seen:
            continue
        seen.add(a.key)
        v.actions.append((a.text(lang, mark_experimental=False), a.experimental))
    return v
