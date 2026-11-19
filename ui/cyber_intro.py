"""Butt3rFLYX boot intro - pure QPainter 2D animation (no external assets).

Phase 1  boot terminal + particles          (0.0 - 2.4 s)
Phase 2  particles form a neon butterfly,
         wings flap, then it dissolves      (2.2 - 5.0 s)
Phase 3  glitch logo + REAL system status   (4.8 - 6.8 s)
Phase 4  scan beam transition to the camera (6.8 - 7.8 s)

The status lines show the actual state of the models/camera that are being loaded in the
background while the intro plays. ESC / Space / click skips the intro.
"""
from __future__ import annotations

import math
import random
import time
import wave
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QPolygonF, QRadialGradient
from PySide6.QtWidgets import QWidget

from .theme import BG, CYAN, GREEN, RED, VIOLET, YELLOW, cyber_font

BOOT_LINES = [
    ("INITIALIZING NEURAL NETWORK...", "ui"),
    ("LOADING COMPUTER VISION MODULES...", "pose"),
    ("ACTIVATING MOTION TRACKING...", "hands"),
    ("BOOTING BUTT3RFLYX AI...", "camera"),
]
LOGO = "B U T T 3 R F L Y X"
GLYPHS = "01ABCDEF#$%&@<>/\\|=+*"
T_BUTTERFLY, T_DISSOLVE, T_LOGO, T_STATUS, T_SCAN, T_END = 2.2, 4.5, 4.8, 5.8, 6.8, 7.8
MAX_WAIT_S = 25.0


def butterfly_curve(n: int) -> np.ndarray:
    """Temple H. Fay's butterfly curve, normalised to roughly [-1, 1]."""
    th = np.linspace(0, 12 * math.pi, n)
    r = np.exp(np.cos(th)) - 2 * np.cos(4 * th) - np.sin(th / 12) ** 5
    x, y = np.sin(th) * r, -np.cos(th) * r          # screen y points down -> big wings on top
    pts = np.stack([x, y], axis=1)
    pts[:, 1] -= (pts[:, 1].max() + pts[:, 1].min()) / 2
    return pts / np.abs(pts).max()


def _ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


class CyberIntro(QWidget):
    finished = Signal()

    def __init__(self, status: dict, reduce_effects: bool = False, sound: bool = False, lang: str = "en") -> None:
        super().__init__()
        self.status = status                    # live dict, updated by the loader thread
        self.reduce = reduce_effects
        self.lang = lang
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        rnd = random.Random(7)
        self.rnd = rnd
        n = 260 if reduce_effects else 760
        self.curve = butterfly_curve(2400)
        idx = np.linspace(0, len(self.curve) - 1, n).astype(int)
        self.targets = self.curve[idx] + np.array([[rnd.gauss(0, 0.012), rnd.gauss(0, 0.012)] for _ in range(n)])
        self.start_pos = np.array([[rnd.uniform(-1.6, 1.6), rnd.uniform(-1.0, 1.0)] for _ in range(n)])
        self.spawn_t = np.array([rnd.uniform(0.1, 2.0) for _ in range(n)])
        self.burst_v = np.array([[math.cos(a) * s, math.sin(a) * s] for a, s in
                                 ((rnd.uniform(0, 2 * math.pi), rnd.uniform(0.6, 2.4)) for _ in range(n))])
        self.colors = [QColor(CYAN) if rnd.random() < 0.6 else QColor(VIOLET) for _ in range(n)]
        self.rain = [(rnd.random(), rnd.uniform(0.15, 0.6), "".join(rnd.choice(GLYPHS) for _ in range(28)))
                     for _ in range(0 if reduce_effects else 26)]
        self.circuits = self._make_circuits(rnd)
        self.t0 = time.perf_counter()
        self.t = 0.0
        self.hold_start: float | None = None
        self.done = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)
        self._sound = None
        if sound:
            self._play_sound()

    # ------------------------------------------------------------------
    def _make_circuits(self, rnd) -> list[list[tuple[float, float]]]:
        out = []
        for _ in range(16 if not self.reduce else 6):
            p = self.curve[rnd.randrange(len(self.curve))].copy()
            path = [tuple(p)]
            for _ in range(3):
                if rnd.random() < 0.5:
                    p = p + np.array([math.copysign(rnd.uniform(0.08, 0.22), p[0] or 1), 0])
                else:
                    p = p + np.array([0, rnd.uniform(-0.18, 0.18)])
                path.append(tuple(p))
            out.append(path)
        return out

    def ready(self) -> bool:
        keys = ("pose", "hands", "face", "camera")
        return all(k in self.status and self.status[k] not in ("loading", "opening", "offline") for k in keys)

    def skip(self) -> None:
        if not self.done:
            self.done = True
            self.timer.stop()
            if self._sound is not None:
                self._sound.stop()
            self.finished.emit()

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Space, Qt.Key.Key_Return):
            self.skip()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self.skip()

    def _tick(self) -> None:
        now = time.perf_counter() - self.t0
        # hold before the transition until the models are loaded (bounded)
        if now >= T_SCAN and self.hold_start is None and not self.ready():
            self.hold_start = now
        if self.hold_start is not None:
            if self.ready() or now - self.hold_start > MAX_WAIT_S:
                self.t0 += now - self.hold_start          # resume the timeline
                self.hold_start = None
                now = T_SCAN
            else:
                now = T_SCAN - 0.001
        self.t = now
        if now >= T_END:
            self.skip()
            return
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        self.s = min(W, H) / 1080.0
        p.fillRect(self.rect(), QColor(BG))
        t = self.t
        self._grid(p, W, H, t)
        if not self.reduce:
            self._rain(p, W, H, t)
        self._particles(p, W, H, t)
        if T_BUTTERFLY <= t < T_DISSOLVE + 0.6:
            self._butterfly(p, W, H, t)
        if t < T_BUTTERFLY + 0.6:
            self._terminal(p, W, H, t)
        if T_LOGO <= t:
            self._logo(p, W, H, t)
        if t >= T_STATUS:
            self._status(p, W, H, t)
        if self.hold_start is not None:
            self._waiting(p, W, H)
        if t >= T_SCAN:
            self._scan(p, W, H, t)
        self._scanlines(p, W, H)
        self._vignette(p, W, H)
        p.end()

    def _center(self, W, H):
        return QPointF(W / 2, H / 2)

    def _grid(self, p, W, H, t):
        a = int(18 * _ease(t / 1.5))
        if t >= T_SCAN:
            a = int(a * (1 - _ease((t - T_SCAN) / 0.8)))
        p.setPen(QPen(QColor(0, 240, 255, a), 1))
        step = int(60 * self.s) or 30
        off = int((t * 20) % step)
        for x in range(-step + off, W, step):
            p.drawLine(x, 0, x, H)
        for y in range(-step + off, H, step):
            p.drawLine(0, y, W, y)
        # rotating hexagon rings
        c = self._center(W, H)
        for i, (r, sp, col) in enumerate(((330, 0.15, CYAN), (380, -0.1, VIOLET), (430, 0.06, CYAN))):
            alpha = int(40 * _ease((t - 1.0) / 1.5)) if t < T_SCAN else int(40 * (1 - _ease((t - T_SCAN) / 0.6)))
            if alpha <= 0:
                continue
            pen = QPen(QColor(col), 1.2 * self.s + 0.5)
            q = QColor(col)
            q.setAlpha(alpha)
            pen.setColor(q)
            pen.setStyle(Qt.PenStyle.DashLine if i == 1 else Qt.PenStyle.SolidLine)
            p.setPen(pen)
            rr = r * self.s
            pts = [c + QPointF(rr * math.cos(t * sp + k * math.pi / 3), rr * math.sin(t * sp + k * math.pi / 3))
                   for k in range(7)]
            p.drawPolyline(QPolygonF(pts))

    def _rain(self, p, W, H, t):
        p.setFont(cyber_font(9 * self.s + 3, spacing=0))
        fade = _ease(t / 1.2) * (1 - _ease((t - T_LOGO) / 1.0))
        if fade <= 0:
            return
        lh = 16 * self.s + 6
        for i, (x, speed, chars) in enumerate(self.rain):
            y0 = ((t * speed * H * 0.6) + i * 97) % (H + 400) - 400
            for j, ch in enumerate(chars):
                a = int(70 * fade * (j / len(chars)))
                if a <= 2:
                    continue
                p.setPen(QColor(0, 240, 255, a))
                p.drawText(QPointF(x * W, y0 + j * lh), ch if (j + int(t * 12)) % 7 else self.rnd.choice(GLYPHS))

    def _particle_pos(self, i: int, t: float, W, H):
        c = self._center(W, H)
        scale = 300 * self.s
        flap = 0.3 + 0.7 * abs(math.cos(math.pi * max(0.0, t - T_BUTTERFLY) * 1.1))
        tx, ty = self.targets[i]
        tx *= flap
        sx, sy = self.start_pos[i]
        drift = 0.05 * math.sin(t * 1.3 + i)
        if t < T_BUTTERFLY:
            return c.x() + (sx + drift) * scale * 1.4, c.y() + (sy + drift) * scale * 1.2
        if t < T_DISSOLVE:
            k = _ease((t - T_BUTTERFLY) / 1.1 + (i % 17) * 0.004)
            x = sx * 1.4 + drift
            y = sy * 1.2 + drift
            return c.x() + (x + (tx - x) * k) * scale, c.y() + (y + (ty - y) * k) * scale
        dt = t - T_DISSOLVE
        vx, vy = self.burst_v[i]
        return c.x() + (tx + vx * dt * 1.3) * scale, c.y() + (ty + vy * dt * 1.3 + 0.2 * dt * dt) * scale

    def _particles(self, p, W, H, t):
        if t >= T_DISSOLVE + 1.4:
            return
        p.setPen(Qt.PenStyle.NoPen)
        fade_out = 1 - _ease((t - T_DISSOLVE) / 1.3) if t > T_DISSOLVE else 1.0
        for i in range(len(self.targets)):
            appear = _ease((t - self.spawn_t[i]) / 0.5)
            if appear <= 0:
                continue
            x, y = self._particle_pos(i, t, W, H)
            col = QColor(self.colors[i])
            tw = 0.6 + 0.4 * math.sin(t * 6 + i * 1.7)
            col.setAlphaF(max(0.0, min(1.0, appear * fade_out * tw)))
            r = (1.4 + (i % 3) * 0.7) * self.s + 0.6
            if not self.reduce and i % 4 == 0:
                g = QColor(col)
                g.setAlphaF(col.alphaF() * 0.25)
                p.setBrush(g)
                p.drawEllipse(QPointF(x, y), r * 3.2, r * 3.2)
            p.setBrush(col)
            p.drawEllipse(QPointF(x, y), r, r)

    def _butterfly(self, p, W, H, t):
        c = self._center(W, H)
        scale = 300 * self.s
        reveal = _ease((t - T_BUTTERFLY - 0.4) / 1.0)
        fade = 1 - _ease((t - T_DISSOLVE) / 0.5) if t > T_DISSOLVE else 1.0
        if reveal <= 0 or fade <= 0:
            return
        flap = 0.3 + 0.7 * abs(math.cos(math.pi * (t - T_BUTTERFLY) * 1.1))
        n = int(len(self.curve) * reveal)
        pts = [QPointF(c.x() + x * flap * scale, c.y() + y * scale) for x, y in self.curve[:n:3]]
        if len(pts) < 2:
            return
        poly = QPolygonF(pts)
        grad = QLinearGradient(c.x() - scale, c.y(), c.x() + scale, c.y())
        grad.setColorAt(0, QColor(CYAN))
        grad.setColorAt(0.5, QColor(VIOLET))
        grad.setColorAt(1, QColor(CYAN))
        if not self.reduce:
            glow = QColor(VIOLET)
            glow.setAlphaF(0.18 * fade)
            p.setPen(QPen(glow, 9 * self.s + 2))
            p.drawPolyline(poly)
        pen = QPen(grad, 1.6 * self.s + 0.6)
        p.setOpacity(fade)
        p.setPen(pen)
        p.drawPolyline(poly)
        # body
        p.setPen(QPen(QColor("#ffffff"), 3 * self.s + 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(c + QPointF(0, -0.35 * scale), c + QPointF(0, 0.45 * scale))
        p.setPen(QPen(QColor(CYAN), 1.2 * self.s + 0.5))
        p.drawLine(c + QPointF(0, -0.35 * scale), c + QPointF(-0.18 * scale, -0.62 * scale))
        p.drawLine(c + QPointF(0, -0.35 * scale), c + QPointF(0.18 * scale, -0.62 * scale))
        # circuit traces on the wings
        p.setPen(QPen(QColor(0, 240, 255, 150), 1.1 * self.s + 0.4))
        for path in self.circuits:
            q = [QPointF(c.x() + x * flap * scale, c.y() + y * scale) for x, y in path]
            p.drawPolyline(QPolygonF(q))
            p.setBrush(QColor(GREEN))
            p.drawEllipse(q[-1], 2.5 * self.s + 1, 2.5 * self.s + 1)
            p.setBrush(Qt.BrushStyle.NoBrush)
        # geometric wing panels
        p.setPen(QPen(QColor(180, 76, 255, 90), 1))
        for side in (-1, 1):
            tri = QPolygonF([c, c + QPointF(side * 0.85 * flap * scale, -0.75 * scale),
                             c + QPointF(side * 0.55 * flap * scale, 0.1 * scale)])
            p.drawPolygon(tri)
            tri2 = QPolygonF([c + QPointF(0, 0.1 * scale), c + QPointF(side * 0.6 * flap * scale, 0.75 * scale),
                              c + QPointF(side * 0.3 * flap * scale, 0.3 * scale)])
            p.drawPolygon(tri2)
        p.setOpacity(1.0)

    def _terminal(self, p, W, H, t):
        fade = 1 - _ease((t - T_BUTTERFLY) / 0.6)
        if fade <= 0:
            return
        s = self.s
        p.setFont(cyber_font(13 * s + 3, spacing=2.5 * s + 0.5))
        fm = p.fontMetrics()
        x0, y0 = W * 0.08, H * 0.70
        for i, (line, key) in enumerate(BOOT_LINES):
            start = 0.2 + i * 0.45
            if t < start:
                break
            n = min(len(line), int((t - start) * 38))
            txt = line[:n]
            jitter = 0
            if not self.reduce and self.rnd.random() < 0.04:
                jitter = self.rnd.randint(-8, 8)
            y = y0 + i * (fm.height() + 6 * s)
            col = QColor(CYAN)
            col.setAlphaF(fade)
            p.setPen(col)
            p.drawText(QPointF(x0 + jitter, y), "> " + txt + ("█" if n < len(line) and int(t * 8) % 2 else ""))
            if n == len(line):
                st = "online" if key == "ui" else self.status.get(key, "loading")
                ok = st.startswith("online") or st == "disabled"
                err = st.startswith("error")
                tag, c2 = ("[ OK ]", GREEN) if ok else (("[FAIL]", RED) if err else
                                                        ("[" + "|/-\\"[int(t * 10) % 4] + "...]", YELLOW))
                q = QColor(c2)
                q.setAlphaF(fade)
                p.setPen(q)
                p.drawText(QPointF(x0 + fm.horizontalAdvance("> " + line) + 18 * s, y), tag)
        # loading bar
        prog = _ease(t / T_BUTTERFLY)
        bw, bh = W * 0.32, 4 * s + 2
        by = y0 + len(BOOT_LINES) * (fm.height() + 6 * s) + 10 * s
        p.fillRect(QRectF(x0, by, bw, bh), QColor(0, 240, 255, int(40 * fade)))
        g = QLinearGradient(x0, 0, x0 + bw, 0)
        g.setColorAt(0, QColor(CYAN))
        g.setColorAt(1, QColor(VIOLET))
        p.setOpacity(fade)
        p.fillRect(QRectF(x0, by, bw * prog, bh), g)
        p.setFont(cyber_font(9 * s + 2, spacing=2))
        p.setPen(QColor(CYAN))
        p.drawText(QPointF(x0 + bw + 14 * s, by + bh + 2), f"{prog * 100:5.1f}%")
        p.setOpacity(1.0)

    def _logo(self, p, W, H, t):
        s = self.s
        k = _ease((t - T_LOGO) / 0.6)
        out = 1 - _ease((t - T_SCAN - 0.1) / 0.7) if t > T_SCAN else 1.0
        if out <= 0:
            return
        # scramble -> resolve
        resolve = (t - T_LOGO) / 0.9
        chars = []
        for i, ch in enumerate(LOGO):
            if ch == " " or resolve * len(LOGO) > i:
                chars.append(ch)
            else:
                chars.append(self.rnd.choice(GLYPHS))
        text = "".join(chars)
        f = cyber_font(64 * s + 8, bold=True, spacing=6 * s)
        p.setFont(f)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        x, y = (W - tw) / 2, H * 0.47
        glitch = (not self.reduce) and (self.rnd.random() < 0.12 or (t - T_LOGO) < 0.35)
        off = (6 * s + 2) if glitch else 2 * s
        p.setOpacity(k * out)
        p.setPen(QColor(255, 40, 90, 200))
        p.drawText(QPointF(x - off, y), text)
        p.setPen(QColor(0, 240, 255, 200))
        p.drawText(QPointF(x + off, y), text)
        p.setPen(QColor("#ffffff"))
        p.drawText(QPointF(x, y), text)
        if glitch:
            # displaced horizontal slices
            for _ in range(3):
                sy = y - fm.ascent() + self.rnd.random() * fm.height()
                hgt = self.rnd.uniform(3, 12) * s + 1
                p.save()
                p.setClipRect(QRectF(0, sy, W, hgt))
                p.setPen(QColor(VIOLET))
                p.drawText(QPointF(x + self.rnd.uniform(-30, 30) * s, y), text)
                p.restore()
        # underline
        p.setPen(QPen(QColor(CYAN), 2 * s + 0.5))
        lw = tw * _ease((t - T_LOGO - 0.3) / 0.6)
        p.drawLine(QPointF(W / 2 - lw / 2, y + 18 * s), QPointF(W / 2 + lw / 2, y + 18 * s))
        # subtitles (typed)
        p.setFont(cyber_font(15 * s + 3, spacing=5 * s))
        for j, (sub, col, delay) in enumerate((("ADVANCED AI MOTION TRACKING SYSTEM", CYAN, 0.45),
                                              ("POWERED BY COMPUTER VISION", VIOLET, 0.75))):
            n = int(max(0.0, t - T_LOGO - delay) * 45)
            if n <= 0:
                continue
            txt = sub[:n]
            fm2 = p.fontMetrics()
            p.setPen(QColor(col))
            p.drawText(QPointF((W - fm2.horizontalAdvance(sub)) / 2, y + (55 + 32 * j) * s + 12), txt)
        p.setOpacity(1.0)

    def _status(self, p, W, H, t):
        s = self.s
        out = 1 - _ease((t - T_SCAN - 0.1) / 0.7) if t > T_SCAN else 1.0
        if out <= 0:
            return
        st = self.status
        pose_ok = st.get("pose", "").startswith("online")
        cam = st.get("camera", "opening")
        cam_ok = cam.startswith("online")
        lines = [
            ("NEURAL SYSTEM ONLINE" if pose_ok else
             ("NEURAL SYSTEM LOADING..." if st.get("pose", "loading") == "loading" else "NEURAL SYSTEM ERROR"),
             GREEN if pose_ok else (YELLOW if st.get("pose", "loading") == "loading" else RED)),
            ("CAMERA INITIALIZED" if cam_ok else ("CAMERA INITIALIZING..." if cam in ("opening", "offline") else "CAMERA NOT AVAILABLE"),
             GREEN if cam_ok else (YELLOW if cam in ("opening", "offline") else RED)),
            ("TRACKING SYSTEM READY" if pose_ok and cam_ok else "TRACKING SYSTEM STANDBY", GREEN if pose_ok and cam_ok else YELLOW),
        ]
        p.setFont(cyber_font(12 * s + 3, bold=True, spacing=4 * s))
        fm = p.fontMetrics()
        p.setOpacity(out)
        for i, (txt, col) in enumerate(lines):
            start = T_STATUS + i * 0.3
            if t < start:
                break
            n = int((t - start) * 50)
            shown = txt[:n]
            y = H * 0.70 + i * (fm.height() + 8 * s)
            x = (W - fm.horizontalAdvance(txt)) / 2
            p.setPen(QColor(col))
            p.drawText(QPointF(x, y), shown)
            p.setBrush(QColor(col))
            p.setPen(Qt.PenStyle.NoPen)
            if n >= len(txt):
                p.drawEllipse(QPointF(x - 16 * s, y - fm.ascent() / 2 + 2), 4 * s + 1, 4 * s + 1)
            p.setBrush(Qt.BrushStyle.NoBrush)
        p.setOpacity(1.0)

    def _waiting(self, p, W, H):
        s = self.s
        p.setFont(cyber_font(10 * s + 3, spacing=3 * s))
        p.setPen(QColor(YELLOW))
        dots = "▮" * (1 + int(time.perf_counter() * 4) % 4)
        txt = f"SYNCHRONIZING NEURAL CORES {dots}"
        fm = p.fontMetrics()
        p.drawText(QPointF((W - fm.horizontalAdvance(txt)) / 2, H * 0.90), txt)

    def _scan(self, p, W, H, t):
        k = _ease((t - T_SCAN) / (T_END - T_SCAN - 0.1))
        y = k * H
        grad = QLinearGradient(0, y - 160 * self.s, 0, y)
        grad.setColorAt(0, QColor(0, 240, 255, 0))
        grad.setColorAt(1, QColor(0, 240, 255, 90))
        p.fillRect(QRectF(0, y - 160 * self.s, W, 160 * self.s), grad)
        p.setPen(QPen(QColor("#ffffff"), 2 * self.s + 1))
        p.drawLine(QPointF(0, y), QPointF(W, y))
        p.setPen(QPen(QColor(CYAN), 6 * self.s + 2))
        p.setOpacity(0.35)
        p.drawLine(QPointF(0, y), QPointF(W, y))
        p.setOpacity(1.0)
        p.setFont(cyber_font(8 * self.s + 3, spacing=2))
        p.setPen(QColor(CYAN))
        p.drawText(QPointF(20 * self.s + 4, y - 8), f"SCAN {k * 100:05.1f}%  //  OPENING OPTICAL SENSOR")

    def _scanlines(self, p, W, H):
        if self.reduce:
            return
        p.setPen(QPen(QColor(0, 0, 0, 70), 1))
        for y in range(0, H, 3):
            p.drawLine(0, y, W, y)

    def _vignette(self, p, W, H):
        g = QRadialGradient(QPointF(W / 2, H / 2), max(W, H) * 0.75)
        g.setColorAt(0.55, QColor(0, 0, 0, 0))
        g.setColorAt(1.0, QColor(0, 0, 0, 200))
        p.fillRect(QRectF(0, 0, W, H), g)

    # ------------------------------------------------------------------
    def _play_sound(self) -> None:
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtMultimedia import QSoundEffect
        except ImportError:
            return
        path = Path(__file__).resolve().parent.parent / "assets" / "sounds" / "intro_synth.wav"
        try:
            if not path.exists():
                write_intro_sound(path)
            eff = QSoundEffect(self)
            eff.setSource(QUrl.fromLocalFile(str(path)))
            eff.setVolume(0.35)
            eff.play()
            self._sound = eff
        except Exception:  # noqa: BLE001 - sound is purely optional
            self._sound = None


def write_intro_sound(path: Path, sr: int = 44100) -> None:
    """Synthesises a subtle intro sound (drone, blips, whoosh, chord). No external audio files."""
    dur = T_END
    t = np.arange(int(sr * dur)) / sr
    env_all = np.minimum(1, t / 0.8) * np.minimum(1, (dur - t) / 0.6)
    drone = 0.18 * np.sin(2 * np.pi * 55 * t) + 0.08 * np.sin(2 * np.pi * 110.5 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t))
    out = drone * env_all
    for i in range(len(BOOT_LINES)):
        for j in range(6):
            t0 = 0.2 + i * 0.45 + j * 0.05
            m = (t >= t0) & (t < t0 + 0.03)
            out[m] += 0.08 * np.sin(2 * np.pi * (1800 + 200 * j) * (t[m] - t0)) * np.exp(-(t[m] - t0) * 90)
    rng = np.random.default_rng(3)
    m = (t >= T_DISSOLVE) & (t < T_DISSOLVE + 0.9)
    noise = rng.normal(0, 1, m.sum())
    noise = np.convolve(noise, np.ones(40) / 40, mode="same")
    out[m] += 0.5 * noise * np.exp(-(t[m] - T_DISSOLVE) * 4)
    m = t >= T_LOGO
    for f in (220, 277.2, 329.6, 440):
        out[m] += 0.05 * np.sin(2 * np.pi * f * (t[m] - T_LOGO)) * np.exp(-(t[m] - T_LOGO) * 1.2)
    out = np.clip(out / max(1e-6, np.abs(out).max()) * 0.6, -1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((out * 32767).astype(np.int16).tobytes())


__all__ = ["CyberIntro", "butterfly_curve", "write_intro_sound"]
