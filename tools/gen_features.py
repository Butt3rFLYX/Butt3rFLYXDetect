"""Generates docs/features.md from the label registry (core/labels.py) and the test files.

A label counts as "unit-tested" only if its key literally appears in an assertion-bearing
test file - so the table can never claim more than the tests cover.
Run:  python tools/gen_features.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.labels import LABELS, NOT_IMPLEMENTED  # noqa: E402

CATEGORY_DE = {
    "posture": "Körperhaltung", "pose": "Pose-Details", "facing": "Körperausrichtung", "movement": "Bewegung",
    "direction": "Bewegungsrichtung", "transition": "Zustandswechsel", "exercise": "Sport / Wiederholungen",
    "head": "Kopf", "gaze": "Blick (Augen)", "gesture": "Handgeste", "hand_motion": "Handbewegung",
    "two_hands": "Zwei Hände",
}


def main() -> None:
    tests = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_*.py"))
    lines = [
        "# Butt3rFLYX – Übersicht aller Erkennungen",
        "",
        "_Automatisch erzeugt von `tools/gen_features.py` aus `core/labels.py` und `tests/`._",
        "",
        "**Status-Legende**",
        "",
        "- **Implementiert** – regelbasiert umgesetzt, im Interface ohne Markierung.",
        "- **Experimentell** – umgesetzt, aber wegen 2D-Mehrdeutigkeit unzuverlässig; im Interface mit `(EXP)` bzw. gelb markiert.",
        "- **Test** – `✔ synthetisch` = durch automatische Tests mit synthetischen Keypoints/Landmarks abgedeckt; "
        "`—` = implementiert, aber ohne eigenen Unit-Test.",
        "- **Keine** Erkennung wurde bisher mit einer echten Webcam systematisch validiert (siehe `limitations.md`). "
        "Zusätzlich wurden die echten Modelle auf zwei Beispielfotos und einem künstlichen Testvideo geprüft.",
        "",
    ]
    by_cat: dict[str, list] = {}
    for key, (cat, status, model, en, de) in LABELS.items():
        by_cat.setdefault(cat, []).append((key, status, model, en, de))
    total = tested = exp = 0
    for cat, items in by_cat.items():
        lines += [f"## {CATEGORY_DE.get(cat, cat)}", "", "| Erkennung (DE) | Anzeige (EN) | Status | Modell | Test | Schlüssel |",
                  "|---|---|---|---|---|---|"]
        for key, status, model, en, de in items:
            covered = re.search(rf'["\']{re.escape(key)}["\']', tests) is not None
            total += 1
            tested += covered
            exp += status == "experimental"
            st = "Experimentell" if status == "experimental" else "Implementiert"
            lines.append(f"| {de.title()} | {en} | {st} | {model} | {'✔ synthetisch' if covered else '—'} | `{key}` |")
        lines.append("")
    lines += ["## Nicht implementiert", "", "| Funktion | Grund |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in NOT_IMPLEMENTED.items()]
    lines += ["", f"**Summe:** {total} Erkennungen, davon {exp} experimentell, {tested} mit eigenem synthetischem Test.", ""]
    out = ROOT / "docs" / "features.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({total} labels, {tested} tested, {exp} experimental)")


if __name__ == "__main__":
    main()
