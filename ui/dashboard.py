"""Dashboard widgets: top status bar, video view, person cards, module status, timeline."""
from __future__ import annotations

import math
import time
from datetime import datetime

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QFrame, QLabel, QListWidget, QListWidgetItem, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from .i18n import tr
from .theme import CYAN, DIM, GREEN, PANEL, RED, STATE_COLORS, TEXT, VIOLET, YELLOW, cyber_font, qcolor
from .view_model import PersonView


class TopBar(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(54)
        self.lang = "en"
        self.fps = 0.0
        self.persons = 0
        self.camera_ok = False
        self.ready = False
        self.mode = "BALANCED"
        self.device = "-"
        self.logging = False
        self.recording = False
        self.paused = False
        self.gesture_ctrl = False
        self._t0 = time.perf_counter()
        t = QTimer(self)
        t.timeout.connect(self.update)
        t.start(50)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        t = time.perf_counter() - self._t0
        g = QLinearGradient(0, 0, 0, H)
        g.setColorAt(0, QColor(10, 16, 34))
        g.setColorAt(1, QColor(4, 6, 14))
        p.fillRect(self.rect(), g)
        p.setPen(QPen(qcolor(CYAN, 140), 1))
        p.drawLine(0, H - 1, W, H - 1)
        # moving highlight on the bottom line
        x = (t * 220) % (W + 300) - 300
        hl = QLinearGradient(x, 0, x + 300, 0)
        hl.setColorAt(0, QColor(0, 240, 255, 0))
        hl.setColorAt(0.5, QColor(0, 240, 255, 255))
        hl.setColorAt(1, QColor(0, 240, 255, 0))
        p.fillRect(QRectF(x, H - 2, 300, 2), hl)
        # logo with subtle glitch
        p.setFont(cyber_font(17, bold=True, spacing=5))
        glitch = int(t * 10) % 37 == 0
        off = 3 if glitch else 1
        p.setPen(QColor(255, 40, 90, 160))
        p.drawText(QPointF(18 - off, 35), "BUTT3RFLYX")
        p.setPen(QColor(0, 240, 255, 160))
        p.drawText(QPointF(18 + off, 35), "BUTT3RFLYX")
        p.setPen(QColor("#ffffff"))
        p.drawText(QPointF(18, 35), "BUTT3RFLYX")
        fm = p.fontMetrics()
        x = 18 + fm.horizontalAdvance("BUTT3RFLYX") + 8
        p.setFont(cyber_font(10, bold=True, spacing=3))
        p.setPen(QColor(VIOLET))
        p.drawText(QPointF(x, 35), "AI")
        x += 50
        chips = [
            (tr("system_online", self.lang) if self.ready else tr("system_loading", self.lang), GREEN if self.ready else YELLOW, True),
            (tr("camera_active", self.lang) if self.camera_ok else tr("camera_lost", self.lang), GREEN if self.camera_ok else RED, True),
            (f"FPS: {self.fps:4.1f}", CYAN if self.fps >= 15 else (YELLOW if self.fps >= 8 else RED), False),
            (f"{tr('tracked', self.lang)}: {self.persons}", CYAN, False),
            (f"MODE: {self.mode}", VIOLET, False),
            (f"DEVICE: {self.device}", VIOLET, False),
        ]
        if self.logging:
            chips.append(("LOG ●", GREEN, True))
        if self.gesture_ctrl:
            chips.append(("GESTURE CTRL", VIOLET, True))
        if self.paused:
            chips.append((tr("paused", self.lang), YELLOW, True))
        if self.recording:
            chips.append(("● REC", RED, True) if int(t * 2) % 2 == 0 else ("  REC", RED, True))
        p.setFont(cyber_font(8.5, bold=True, spacing=1.5))
        fm = p.fontMetrics()
        for text, col, dot in chips:
            w = fm.horizontalAdvance(text) + (30 if dot else 18)
            if x + w > W - 10:
                break
            r = QRectF(x, 14, w, 26)
            p.setPen(QPen(qcolor(col, 150), 1))
            p.setBrush(qcolor(col, 22))
            p.drawRoundedRect(r, 3, 3)
            tx = x + 9
            if dot:
                a = 0.5 + 0.5 * math.sin(t * 4)
                p.setBrush(qcolor(col, int(120 + 135 * a)))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QPointF(x + 12, 27), 4, 4)
                tx = x + 22
            p.setPen(QColor(col))
            p.drawText(QPointF(tx, 31), text)
            x += w + 8
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setFont(cyber_font(9, spacing=2))
        p.setPen(QColor(DIM))
        clock = datetime.now().strftime("%H:%M:%S")
        p.drawText(QPointF(W - fm.horizontalAdvance(clock) - 30, 31), clock)
        p.end()


class VideoView(QWidget):
    """Shows the rendered camera frame with an animated HUD frame around it."""

    def __init__(self) -> None:
        super().__init__()
        self.image: QImage | None = None
        self.message = ""
        self.lang = "en"
        self.reduce = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 300)
        self._t0 = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(33)

    def set_image(self, img: QImage) -> None:
        self.image = img
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        W, H = self.width(), self.height()
        t = time.perf_counter() - self._t0
        p.fillRect(self.rect(), QColor(2, 3, 8))
        target = QRectF(8, 8, W - 16, H - 16)
        if self.image is not None and not self.image.isNull():
            iw, ih = self.image.width(), self.image.height()
            s = min(target.width() / iw, target.height() / ih)
            dw, dh = iw * s, ih * s
            target = QRectF(8 + (target.width() - dw) / 2, 8 + (target.height() - dh) / 2, dw, dh)
            p.drawImage(target, self.image)
        else:
            p.setFont(cyber_font(22, bold=True, spacing=8))
            p.setPen(QColor(RED if self.message else CYAN))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr("no_signal", self.lang))
            if self.message:
                p.setFont(cyber_font(10, spacing=1))
                p.setPen(QColor(TEXT))
                p.drawText(QRectF(40, H / 2 + 30, W - 80, 80),
                           Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, self.message)
        # animated corner frame
        pulse = 0.6 + 0.4 * math.sin(t * 2.2)
        col = QColor(CYAN)
        col.setAlphaF(pulse)
        L = 34
        p.setPen(QPen(col, 2.5))
        r = target.adjusted(-4, -4, 4, 4)
        for (cx, cy, dx, dy) in ((r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1),
                                 (r.left(), r.bottom(), 1, -1), (r.right(), r.bottom(), -1, -1)):
            p.drawLine(QPointF(cx, cy), QPointF(cx + dx * L, cy))
            p.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * L))
        p.setPen(QPen(qcolor(CYAN, 40), 1))
        p.drawRect(r)
        if not self.reduce:
            # slow vertical scan line
            y = r.top() + ((t * 0.18) % 1.0) * r.height()
            g = QLinearGradient(0, y - 40, 0, y)
            g.setColorAt(0, QColor(0, 240, 255, 0))
            g.setColorAt(1, QColor(0, 240, 255, 30))
            p.fillRect(QRectF(r.left(), y - 40, r.width(), 40), g)
            p.setPen(QPen(qcolor(CYAN, 70), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        p.setFont(cyber_font(7.5, spacing=2))
        p.setPen(qcolor(CYAN, 160))
        p.drawText(QPointF(r.left() + 6, r.bottom() - 6), "OPTICAL SENSOR // LIVE // LOCAL PROCESSING ONLY")
        p.end()


class PersonCard(QFrame):
    ROW_H = 19

    def __init__(self) -> None:
        super().__init__()
        self.view: PersonView | None = None
        self.lang = "en"
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_view(self, v: PersonView, lang: str) -> None:
        self.view, self.lang = v, lang
        n_act_lines = max(1, math.ceil(len(v.actions) / 2)) if v.actions else 1
        self.setFixedHeight(46 + self.ROW_H * len(v.rows) + 18 + 16 * n_act_lines + 10)
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        v = self.view
        if v is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        col = QColor(STATE_COLORS.get(v.color_state, CYAN))
        path = QPainterPath()
        path.moveTo(0, 0)
        path.lineTo(W - 14, 0)
        path.lineTo(W - 1, 14)
        path.lineTo(W - 1, H - 1)
        path.lineTo(0, H - 1)
        path.closeSubpath()
        p.fillPath(path, QColor(PANEL))
        p.setPen(QPen(qcolor(col.name(), 120), 1))
        p.drawPath(path)
        p.fillRect(QRectF(0, 0, 3, H), col)
        # header
        p.setFont(cyber_font(11, bold=True, spacing=3))
        p.setPen(col)
        p.drawText(QPointF(12, 22), v.name)
        p.setFont(cyber_font(8, spacing=1))
        p.setPen(QColor(DIM))
        p.drawText(QPointF(12, 37), f"{tr('tracked_for', self.lang)} {v.tracked_for:5.1f}s")
        # confidence bar (real, computed value)
        bw = 110
        bx = W - bw - 16
        p.drawText(QPointF(bx, 15), f"{tr('confidence', self.lang)} {v.confidence * 100:.0f}%")
        p.fillRect(QRectF(bx, 22, bw, 6), QColor(20, 40, 60))
        g = QLinearGradient(bx, 0, bx + bw, 0)
        g.setColorAt(0, QColor(VIOLET))
        g.setColorAt(1, QColor(CYAN))
        p.fillRect(QRectF(bx, 22, bw * max(0.0, min(1.0, v.confidence)), 6), g)
        y = 46
        for row in v.rows:
            y += self.ROW_H
            p.setFont(cyber_font(8, spacing=1.2))
            p.setPen(QColor(DIM))
            p.drawText(QPointF(12, y - 5), row.title)
            p.setFont(cyber_font(8.5, bold=not row.dim, spacing=1))
            c = QColor(DIM) if row.dim else (QColor(YELLOW) if row.experimental else QColor(TEXT))
            p.setPen(c)
            txt = row.value + ("  (EXP)" if row.experimental and not row.dim else "")
            fm = p.fontMetrics()
            avail = W - 120 - (52 if row.confidence is not None else 0)
            txt = fm.elidedText(txt, Qt.TextElideMode.ElideRight, int(avail))
            p.drawText(QPointF(108, y - 5), txt)
            if row.confidence is not None and not row.dim:
                p.setFont(cyber_font(7.5, spacing=0.5))
                p.setPen(qcolor(CYAN, 170))
                p.drawText(QPointF(W - 46, y - 5), f"{row.confidence * 100:3.0f}%")
            p.setPen(QPen(QColor(18, 40, 62), 1))
            p.drawLine(QPointF(10, y), QPointF(W - 10, y))
        y += 16
        p.setFont(cyber_font(8, spacing=1.2))
        p.setPen(QColor(DIM))
        p.drawText(QPointF(12, y), tr("actions", self.lang))
        y += 15
        p.setFont(cyber_font(8, bold=True, spacing=0.8))
        if not v.actions:
            p.setPen(QColor(DIM))
            p.drawText(QPointF(14, y), "—")
        fm = p.fontMetrics()
        x = 14
        for text, exp in v.actions:
            label = text + (" (EXP)" if exp else "")
            w = fm.horizontalAdvance(label) + 14
            if x + w > W - 8 and x > 14:
                x = 14
                y += 16
            r = QRectF(x, y - 11, w, 14)
            c = QColor(YELLOW) if exp else QColor(GREEN)
            p.setPen(QPen(qcolor(c.name(), 120), 1))
            p.setBrush(qcolor(c.name(), 25))
            p.drawRoundedRect(r, 2, 2)
            p.setPen(c)
            p.drawText(QPointF(x + 7, y), label)
            x += w + 6
        p.end()


class PersonPanel(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inner = QWidget()
        self.lay = QVBoxLayout(self.inner)
        self.lay.setContentsMargins(0, 0, 4, 0)
        self.lay.setSpacing(8)
        self.empty = QLabel()
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setStyleSheet(f"color:{DIM}; letter-spacing:2px; padding:30px;")
        self.lay.addWidget(self.empty)
        self.lay.addStretch(1)
        self.setWidget(self.inner)
        self.cards: dict[int, PersonCard] = {}

    def update_views(self, views: list[PersonView], lang: str) -> None:
        self.empty.setText(tr("no_persons", lang))
        self.empty.setVisible(not views)
        ids = {v.pid for v in views}
        for pid in list(self.cards):
            if pid not in ids:
                w = self.cards.pop(pid)
                self.lay.removeWidget(w)
                w.deleteLater()
        for i, v in enumerate(sorted(views, key=lambda x: x.pid)):
            card = self.cards.get(v.pid)
            if card is None:
                card = PersonCard()
                self.cards[v.pid] = card
                self.lay.insertWidget(1 + i, card)
            card.set_view(v, lang)


class ModuleStatus(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.status: dict = {}
        self.timings: dict = {}
        self.lang = "en"
        self.log_count = 0
        self.setFixedHeight(118)

    def set_data(self, status: dict, timings: dict, log_count: int) -> None:
        self.status, self.timings, self.log_count = status, timings, log_count
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        p.setPen(QPen(qcolor(VIOLET, 110), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.setFont(cyber_font(9, bold=True, spacing=3))
        p.setPen(QColor(VIOLET))
        p.drawText(QPointF(10, 18), tr("ai_modules", self.lang))
        rows = [("YOLO11 POSE", "pose", "pose"), ("HAND LANDMARKER", "hands", "hands"),
                ("FACE LANDMARKER", "face", "face"), ("FACE IDENTITY", "identity", None)]
        p.setFont(cyber_font(8, spacing=1))
        y = 38
        for name, key, tkey in rows:
            st = self.status.get(key, "offline")
            ok = st.startswith("online")
            col = GREEN if ok else (DIM if st == "disabled" else (YELLOW if st in ("loading", "offline") else RED))
            p.setBrush(QColor(col))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(14, y - 4), 3.5, 3.5)
            p.setPen(QColor(TEXT))
            p.drawText(QPointF(24, y), name)
            p.setPen(QColor(col))
            short = "ONLINE" if ok else ("DISABLED" if st == "disabled" else ("ERROR" if st.startswith("error") else st.upper()))
            if key == "pose" and ok:
                short += f" ({self.status.get('device', '-')})"
            if tkey and tkey in self.timings and ok:
                short += f"  {self.timings[tkey]:.0f} ms"
            p.drawText(QPointF(170, y), short)
            if st.startswith("error"):
                self.setToolTip(st)
            y += 19
        p.setPen(QColor(DIM))
        p.drawText(QPointF(24, y), f"EVENTS LOGGED: {self.log_count}")
        p.end()


class Timeline(QListWidget):
    COLORS = {"entered": GREEN, "reacquired": CYAN, "left": DIM, "instant": VIOLET, "state": TEXT,
              "command": YELLOW, "system": YELLOW, "ended": DIM}

    def __init__(self) -> None:
        super().__init__()
        self.setFont(cyber_font(8.5, spacing=1))
        self.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def add_events(self, events) -> None:
        for ev in events:
            if ev.event_type == "ended":
                continue
            when = ev.timestamp[11:19] if len(ev.timestamp) >= 19 else ev.timestamp
            conf = f"  [{ev.confidence * 100:.0f}%]" if ev.confidence is not None else ""
            item = QListWidgetItem(f"{when}  ▸  {ev.text}{conf}")
            col = RED if ev.label == "falling" else self.COLORS.get(ev.event_type, TEXT)
            item.setForeground(QColor(col))
            self.insertItem(0, item)
        while self.count() > 400:
            self.takeItem(self.count() - 1)

    def add_system(self, text: str, color: str = YELLOW) -> None:
        item = QListWidgetItem(f"{datetime.now():%H:%M:%S}  ▸  {text}")
        item.setForeground(QColor(color))
        self.insertItem(0, item)


def section_title(text: str, color: str = CYAN) -> QLabel:
    lab = QLabel(text)
    lab.setFont(cyber_font(9.5, bold=True, spacing=3))
    lab.setStyleSheet(f"color:{color}; padding:2px 0;")
    return lab
