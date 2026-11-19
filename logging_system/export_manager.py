"""Export of the in-memory event history of the current session."""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from core.action_engine import EventRecord

from .event_logger import FIELDS


def export_events(events: list[EventRecord], path: Path | str, fmt: str | None = None) -> Path:
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt == "db":
        fmt = "sqlite"
    rows = [{k: ev.as_dict().get(k) for k in FIELDS} for ev in events]
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    elif fmt == "json":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    elif fmt == "txt":
        with open(path, "w", encoding="utf-8") as f:
            for ev, r in zip(events, rows):
                conf = f" ({r['confidence'] * 100:.0f}%)" if r["confidence"] is not None else ""
                f.write(f"{r['timestamp']}  {ev.text}{conf}\n")
    elif fmt == "sqlite":
        con = sqlite3.connect(str(path))
        con.execute("CREATE TABLE IF NOT EXISTS events (" + ", ".join(FIELDS) + ")")
        con.executemany(f"INSERT INTO events VALUES ({', '.join('?' * len(FIELDS))})",
                        [[r[k] for k in FIELDS] for r in rows])
        con.commit()
        con.close()
    else:
        raise ValueError(f"Unknown export format: {fmt}")
    return path
