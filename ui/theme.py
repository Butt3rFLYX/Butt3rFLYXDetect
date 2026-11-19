"""Colours, fonts and the Qt stylesheet of the Butt3rFLYX cyber look."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase

BG = "#04050c"
PANEL = "#0a0f1f"
PANEL_2 = "#0e1530"
CYAN = "#00f0ff"
VIOLET = "#b44cff"
GREEN = "#39ff88"
YELLOW = "#ffd23f"
RED = "#ff3355"
TEXT = "#cfefff"
DIM = "#5a7a99"

STATE_COLORS = {"normal": CYAN, "gesture": VIOLET, "success": GREEN, "uncertain": YELLOW, "alarm": RED}

_FONT_FAMILY: str | None = None


def font_family() -> str:
    global _FONT_FAMILY
    if _FONT_FAMILY is None:
        families = set(QFontDatabase.families())
        for f in ("Orbitron", "Rajdhani", "Share Tech Mono", "Menlo", "SF Mono", "Consolas",
                  "DejaVu Sans Mono", "Courier New"):
            if f in families:
                _FONT_FAMILY = f
                break
        else:
            _FONT_FAMILY = QFont().defaultFamily()
    return _FONT_FAMILY


def cyber_font(size: float, bold: bool = False, spacing: float = 1.5) -> QFont:
    f = QFont(font_family())
    f.setPointSizeF(size)
    f.setBold(bold)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


def qcolor(hex_: str, alpha: int = 255) -> QColor:
    c = QColor(hex_)
    c.setAlpha(alpha)
    return c


def stylesheet() -> str:
    ff = font_family()
    return f"""
    QWidget {{ background: {BG}; color: {TEXT}; font-family: "{ff}"; font-size: 12px; }}
    QMainWindow {{ background: {BG}; }}
    QToolTip {{ background: {PANEL_2}; color: {CYAN}; border: 1px solid {CYAN}; }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{ background: {BG}; width: 8px; }}
    QScrollBar::handle:vertical {{ background: #1d3350; border-radius: 4px; min-height: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QListWidget {{ background: rgba(10,15,31,220); border: 1px solid #12324a; border-radius: 4px; padding: 4px; }}
    QListWidget::item {{ padding: 2px 4px; border-bottom: 1px solid #0d1e33; }}
    QPushButton {{ background: {PANEL_2}; color: {CYAN}; border: 1px solid #1e4f6e; border-radius: 3px;
                  padding: 6px 12px; letter-spacing: 1.5px; }}
    QPushButton:hover {{ border-color: {CYAN}; background: #102544; }}
    QPushButton:checked {{ background: #2a1046; border-color: {VIOLET}; color: {VIOLET}; }}
    QPushButton:pressed {{ background: #0b3a55; }}
    QDialog {{ background: {BG}; }}
    QTabWidget::pane {{ border: 1px solid #1e4f6e; top: -1px; background: {PANEL}; }}
    QTabBar::tab {{ background: {PANEL}; color: {DIM}; padding: 8px 16px; border: 1px solid #12324a;
                   border-bottom: none; letter-spacing: 1.5px; }}
    QTabBar::tab:selected {{ color: {CYAN}; border-color: {CYAN}; background: {PANEL_2}; }}
    QGroupBox {{ border: 1px solid #1a3a55; border-radius: 4px; margin-top: 14px; padding: 8px; background: {PANEL}; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {VIOLET}; padding: 0 4px; }}
    QLabel {{ background: transparent; }}
    QCheckBox, QRadioButton {{ background: transparent; spacing: 8px; }}
    QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {CYAN}; background: {BG}; }}
    QCheckBox::indicator:checked {{ background: {CYAN}; }}
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {PANEL_2}; border: 1px solid #1e4f6e;
                  padding: 4px 6px; color: {TEXT}; border-radius: 3px; }}
    QComboBox QAbstractItemView {{ background: {PANEL_2}; color: {TEXT}; selection-background-color: #0b3a55; }}
    QSlider::groove:horizontal {{ height: 4px; background: #12324a; }}
    QSlider::handle:horizontal {{ background: {CYAN}; width: 12px; margin: -5px 0; border-radius: 6px; }}
    QSlider::sub-page:horizontal {{ background: {VIOLET}; }}
    QMessageBox {{ background: {PANEL}; }}
    """
