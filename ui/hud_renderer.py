"""Draws the futuristic tracking overlay onto the camera frame (QPainter on a QImage).

Runs in the worker thread - QPainter on QImage is thread-safe in Qt.
Analysis always happens on the original image; when the display is mirrored only the
drawing coordinates are mirrored, so left/right of the person stay correct.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF

from core.action_engine import PersonSnapshot
from core.config import AppConfig
from core.geometry import L_EL, L_HIP, L_KNEE, L_SH, NOSE, R_EL, R_HIP, R_KNEE, R_SH, SKELETON
from core.hand_gestures import FINGER_TIPS, HAND_CONNECTIONS
from core.pipeline import FrameAnalysis

from .i18n import tr
from .theme import CYAN, GREEN, RED, STATE_COLORS, TEXT, VIOLET, YELLOW, cyber_font, qcolor

GROUP_TINT = {"torso": 0.0, "head": 0.0, "left_arm": 0.35, "left_leg": 0.2, "right_arm": -0.35, "right_leg": -0.2}
ANGLE_JOINTS = {"l_elbow": L_EL, "r_elbow": R_EL, "l_knee": L_KNEE, "r_knee": R_KNEE,
                "l_shoulder": L_SH, "r_shoulder": R_SH, "l_hip": L_HIP, "r_hip": R_HIP}


def _blend(c1: QColor, c2: QColor, f: float) -> QColor:
    f = max(0.0, min(1.0, f))
    return QColor(int(c1.red() * (1 - f) + c2.red() * f), int(c1.green() * (1 - f) + c2.green() * f),
                  int(c1.blue() * (1 - f) + c2.blue() * f))


def numpy_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def qimage_to_bgr(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), np.uint8, img.bytesPerLine() * h).reshape(h, img.bytesPerLine())
    arr = arr[:, :w * 3].reshape(h, w, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class HudRenderer:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._scan: QImage | None = None

    def _scanlines(self, w: int, h: int) -> QImage:
        if self._scan is None or self._scan.width() != w or self._scan.height() != h:
            img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(Qt.GlobalColor.transparent)
            p = QPainter(img)
            p.setPen(QPen(QColor(0, 0, 0, 38), 1))
            for y in range(0, h, 3):
                p.drawLine(0, y, w, y)
            p.end()
            self._scan = img
        return self._scan

    # ------------------------------------------------------------------
    def render(self, frame_bgr: np.ndarray, fa: FrameAnalysis | None, t_anim: float, paused: bool = False,
               gesture_progress: tuple[str, float] | None = None) -> QImage:
        cfg = self.cfg
        disp = cv2.flip(frame_bgr, 1) if cfg.mirror else frame_bgr
        img = numpy_to_qimage(disp)
        W, H = img.width(), img.height()
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        if not cfg.reduce_effects:
            p.fillRect(0, 0, W, H, QColor(0, 12, 30, 45))
            p.drawImage(0, 0, self._scanlines(W, H))
        self.k = max(0.6, min(2.2, H / 720))          # UI scale for line widths / fonts
        X = (lambda x: W - x) if cfg.mirror else (lambda x: x)
        self.X = X
        if fa is not None and cfg.show_hud:
            for s in fa.persons:
                self._person(p, s, t_anim)
        if paused:
            self._center_banner(p, W, H, tr("paused", cfg.language), YELLOW)
        if gesture_progress is not None and gesture_progress[1] > 0:
            self._gesture_ring(p, W, H, *gesture_progress)
        p.end()
        return img

    # ------------------------------------------------------------------
    def _P(self, xy) -> QPointF:
        return QPointF(float(self.X(xy[0])), float(xy[1]))

    def _glow_line(self, p: QPainter, a: QPointF, b: QPointF, color: QColor, width: float, dashed=False):
        k = self.k
        if not self.cfg.reduce_effects:
            g = QColor(color)
            g.setAlpha(55)
            p.setPen(QPen(g, width * 3.2 * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(a, b)
        pen = QPen(color, width * k, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(a, b)

    def _person(self, p: QPainter, s: PersonSnapshot, t: float) -> None:
        cfg, k = self.cfg, self.k
        base = QColor(STATE_COLORS.get(s.color_state, CYAN))
        kps = s.kps
        T = s.pose.torso_len or max((s.box[3] - s.box[1]) / 3, 20)
        x1, x2 = sorted((self.X(s.box[0]), self.X(s.box[2])))
        y1, y2 = float(s.box[1]), float(s.box[3])

        if cfg.show_boxes:
            self._brackets(p, x1, y1, x2, y2, base, t)
        if cfg.show_trajectories and len(s.motion.trajectory) > 2:
            pts = s.motion.trajectory
            n = len(pts)
            for i in range(1, n):
                c = QColor(VIOLET)
                c.setAlpha(int(200 * i / n))
                p.setPen(QPen(c, 2 * k))
                p.drawLine(self._P(pts[i - 1]), self._P(pts[i]))
        if cfg.show_skeleton:
            for a, b, grp in SKELETON:
                pa, pb = kps.get(a), kps.get(b)
                if pa is None or pb is None:
                    continue
                tint = GROUP_TINT[grp]
                col = _blend(base, QColor(VIOLET if tint > 0 else CYAN), abs(tint)) if s.color_state == "normal" else base
                low = min(kps.conf[a], kps.conf[b]) < 0.55
                if low:
                    col = _blend(col, QColor(YELLOW), 0.6)
                self._glow_line(p, self._P(pa), self._P(pb), col, 2.2, dashed=low)
            for i in range(len(kps.conf)):
                q = kps.get(i)
                if q is None:
                    continue
                r = (3.2 + 1.2 * math.sin(t * 5 + i)) * k
                if not cfg.reduce_effects:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(qcolor(base.name(), 60))
                    p.drawEllipse(self._P(q), r * 2.2, r * 2.2)
                p.setBrush(QColor("#ffffff"))
                p.setPen(QPen(base, 1.5 * k))
                p.drawEllipse(self._P(q), r, r)
            p.setBrush(Qt.BrushStyle.NoBrush)
        if s.pose.hip is not None:
            hp = self._P(s.pose.hip)
            p.setPen(QPen(qcolor(GREEN, 220), 1.5 * k))
            p.drawLine(hp + QPointF(-8 * k, 0), hp + QPointF(8 * k, 0))
            p.drawLine(hp + QPointF(0, -8 * k), hp + QPointF(0, 8 * k))
            p.drawEllipse(hp, 5 * k, 5 * k)
            if cfg.show_vectors and s.motion.velocity is not None and s.motion.speed > 0.3:
                v = s.motion.velocity * 0.35
                tip = s.pose.hip + v
                self._arrow(p, hp, self._P(tip), QColor(GREEN), f"{s.motion.speed:.1f} T/s")
        if cfg.show_angles:
            p.setFont(cyber_font(7.5 * k, spacing=0.5))
            for name, idx in ANGLE_JOINTS.items():
                a = s.pose.joint_angles.get(name)
                q = kps.get(idx)
                if a is None or q is None:
                    continue
                pt = self._P(q) + QPointF(6 * k, -6 * k)
                p.setPen(qcolor(YELLOW, 230))
                p.drawText(pt, f"{a:.0f}°")
        if cfg.show_head_direction and s.head is not None:
            self._head(p, s, T)
        if s.head is not None and s.head.landmarks is not None and cfg.show_face_mesh:
            p.setPen(QPen(qcolor(CYAN, 120), 1.6 * k))
            pts = [self._P(xy) for xy in s.head.landmarks[::3, :2]]
            p.drawPoints(QPolygonF(pts))
        if cfg.show_fingers:
            for side, h in s.hands.items():
                if h is not None:
                    self._hand(p, h, side)
        self._label_panel(p, s, x1, y1, x2, y2, base)

    def _brackets(self, p, x1, y1, x2, y2, col: QColor, t: float):
        k = self.k
        L = min(28 * k, (x2 - x1) / 4, (y2 - y1) / 4)
        pulse = 0.65 + 0.35 * math.sin(t * 3)
        c = QColor(col)
        c.setAlphaF(0.9 * pulse + 0.1)
        thin = QColor(col)
        thin.setAlpha(45)
        p.setPen(QPen(thin, 1))
        p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        p.setPen(QPen(c, 2.6 * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            p.drawLine(QPointF(cx, cy), QPointF(cx + dx * L, cy))
            p.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * L))

    def _arrow(self, p, a: QPointF, b: QPointF, col: QColor, text: str = ""):
        k = self.k
        self._glow_line(p, a, b, col, 2.0)
        ang = math.atan2(b.y() - a.y(), b.x() - a.x())
        s = 9 * k
        head = QPolygonF([b, b - QPointF(s * math.cos(ang - 0.45), s * math.sin(ang - 0.45)),
                          b - QPointF(s * math.cos(ang + 0.45), s * math.sin(ang + 0.45))])
        p.setBrush(col)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(head)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if text:
            p.setFont(cyber_font(7.5 * k, spacing=0.5))
            p.setPen(col)
            p.drawText(b + QPointF(5 * k, -4 * k), text)

    def _head(self, p, s: PersonSnapshot, T: float):
        h = s.head
        nose = s.kps.get(NOSE)
        if h.landmarks is not None:
            nose = h.landmarks[1, :2]
        if nose is None or h.yaw is None:
            return
        # yaw > 0 = person's left = image right (person faces the camera); pitch > 0 = up
        L = 0.55 * T
        dx = math.sin(math.radians(max(-80.0, min(80.0, h.yaw)))) * L
        dy = -math.sin(math.radians(max(-60.0, min(60.0, h.pitch or 0.0)))) * L
        tip = np.array([nose[0] + dx, nose[1] + dy])
        col = QColor(VIOLET) if h.source == "face_mesh" else qcolor(YELLOW)
        self._arrow(p, self._P(nose), self._P(tip), col)

    def _hand(self, p, h, side: str):
        k = self.k
        lm = h.landmarks
        col = QColor(VIOLET) if h.gestures else QColor(CYAN)
        for a, b in HAND_CONNECTIONS:
            self._glow_line(p, self._P(lm[a]), self._P(lm[b]), col, 1.4)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(21):
            p.setBrush(QColor("#ffffff") if i in FINGER_TIPS.values() else col)
            p.drawEllipse(self._P(lm[i]), 2.4 * k, 2.4 * k)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.cfg.show_finger_labels and h.palm_size * 1.0 > 22:
            p.setFont(cyber_font(6.5 * k, spacing=0.8))
            for name, tip in FINGER_TIPS.items():
                ext = h.fingers.get(name, False)
                p.setPen(qcolor(GREEN if ext else TEXT, 235 if ext else 110))
                p.drawText(self._P(lm[tip]) + QPointF(4 * k, -4 * k), name.upper())
        g = h.primary
        if g is not None:
            p.setFont(cyber_font(8 * k, bold=True))
            x0, y0 = min(self.X(x) for x in lm[:, 0]), float(lm[:, 1].min())
            txt = f"{side[0].upper()}: {g.text(self.cfg.language)}"
            fm = p.fontMetrics()
            r = QRectF(x0, y0 - 18 * k, fm.horizontalAdvance(txt) + 10 * k, 15 * k)
            p.fillRect(r, QColor(20, 6, 40, 190))
            p.setPen(QColor(VIOLET))
            p.drawText(r.adjusted(5 * k, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, txt)

    def _label_panel(self, p, s: PersonSnapshot, x1, y1, x2, y2, col: QColor):
        k, lang = self.k, self.cfg.language
        m = s.motion
        lines: list[tuple[str, QColor]] = []
        if m.posture:
            lines.append((m.posture.text(lang), QColor(TEXT)))
        if m.movement:
            c = QColor(RED) if m.movement.key == "falling" else QColor(GREEN)
            lines.append((m.movement.text(lang), c))
        for tr_ in m.transitions[:2]:
            lines.append((tr_.text(lang), QColor(GREEN)))
        for g in (m.exercises + m.modifiers)[:3]:
            lines.append((g.text(lang), QColor(CYAN)))
        for side in ("left", "right"):
            h = s.hands.get(side)
            if h is not None and h.primary is not None:
                lines.append((f"{tr(side + '_hand', lang)}: {h.primary.text(lang)}", QColor(VIOLET)))
        for g in s.two_hand[:1]:
            lines.append((g.text(lang), QColor(VIOLET)))
        if s.head is not None and s.head.direction is not None:
            lines.append((f"{tr('head', lang)}: {s.head.direction.text(lang)}", QColor(TEXT)))
        lines = lines[:8]
        title = f"{s.name}   {s.confidence * 100:.0f}%"
        p.setFont(cyber_font(9.5 * k, bold=True, spacing=2))
        fm_t = p.fontMetrics()
        p.setFont(cyber_font(8 * k, spacing=1))
        fm = p.fontMetrics()
        lh = fm.height() + 2 * k
        w = max([fm_t.horizontalAdvance(title)] + [fm.horizontalAdvance(t) for t, _ in lines]) + 18 * k
        h = fm_t.height() + 6 * k + lh * len(lines) + 6 * k
        W = p.device().width()
        px = x2 + 8 * k if x2 + 8 * k + w < W else max(2.0, x1 - w - 8 * k)
        if px + w > W:
            px = max(2.0, W - w - 2)
        py = max(2.0, y1)
        rect = QRectF(px, py, w, h)
        path = QPainterPath()
        cut = 10 * k
        path.moveTo(rect.left(), rect.top())
        path.lineTo(rect.right() - cut, rect.top())
        path.lineTo(rect.right(), rect.top() + cut)
        path.lineTo(rect.right(), rect.bottom())
        path.lineTo(rect.left(), rect.bottom())
        path.closeSubpath()
        p.fillPath(path, QColor(4, 10, 24, 185))
        edge = QColor(col)
        edge.setAlpha(160)
        p.setPen(QPen(edge, 1.2 * k))
        p.drawPath(path)
        p.fillRect(QRectF(rect.left(), rect.top(), 3 * k, rect.height()), col)
        p.setFont(cyber_font(9.5 * k, bold=True, spacing=2))
        p.setPen(col)
        p.drawText(QPointF(px + 9 * k, py + fm_t.ascent() + 4 * k), title)
        p.setFont(cyber_font(8 * k, spacing=1))
        y = py + fm_t.height() + 6 * k + fm.ascent()
        for text, c in lines:
            p.setPen(c)
            p.drawText(QPointF(px + 9 * k, y), text)
            y += lh
        # connector line
        p.setPen(QPen(qcolor(col.name(), 120), 1 * k, Qt.PenStyle.DotLine))
        anchor_x = x2 if px > x2 else x1
        p.drawLine(QPointF(anchor_x, y1 + 10 * k), QPointF(px if px > x2 else px + w, py + 10 * k))

    def _center_banner(self, p, W, H, text, color):
        k = self.k
        p.fillRect(0, int(H / 2 - 30 * k), W, int(60 * k), QColor(0, 0, 0, 150))
        p.setFont(cyber_font(20 * k, bold=True, spacing=6))
        p.setPen(QColor(color))
        p.drawText(QRectF(0, H / 2 - 30 * k, W, 60 * k), Qt.AlignmentFlag.AlignCenter, text)

    def _gesture_ring(self, p, W, H, command: str, progress: float):
        k = self.k
        c = QPointF(W - 70 * k, H - 70 * k)
        r = 34 * k
        p.setPen(QPen(qcolor(VIOLET, 70), 6 * k))
        p.drawEllipse(c, r, r)
        p.setPen(QPen(QColor(VIOLET), 6 * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), 90 * 16, int(-progress * 360 * 16))
        p.setFont(cyber_font(7 * k, bold=True))
        p.setPen(QColor(TEXT))
        p.drawText(QRectF(c.x() - 80 * k, c.y() + r + 4 * k, 160 * k, 16 * k), Qt.AlignmentFlag.AlignCenter,
                   command.replace("_", " ").upper())


__all__ = ["HudRenderer", "numpy_to_qimage", "qimage_to_bgr"]
