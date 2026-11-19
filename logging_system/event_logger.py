"""Live event documentation: CSV, JSON Lines, TXT and SQLite.

Only events that were actually recognised are written. Logging is OFF by default and
no camera images are ever stored by the logger.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from core.action_engine import EventRecord

FIELDS = ["timestamp", "person_id", "event_type", "label", "text", "posture", "movement",
          "head_direction", "left_hand", "right_hand", "confidence", "duration_s"]


class EventLogger:
    def __init__(self, log_dir: Path, formats: list[str]) -> None:
        self.log_dir = Path(log_dir)
        self.formats = [f for f in formats if f in ("csv", "json", "txt", "sqlite")]
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._lock = threading.Lock()
        self._files: dict = {}
        self._csv = None
        self._db: sqlite3.Connection | None = None
        self._last_key: set = set()
        self._last_t: float | None = None
        self.count = 0
        self.paths: list[Path] = []
        self._open()

    def _open(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        base = self.log_dir / f"butt3rflyx_{self.session}"
        if "csv" in self.formats:
            p = base.with_suffix(".csv")
            f = open(p, "a", newline="", encoding="utf-8")
            self._csv = csv.DictWriter(f, fieldnames=FIELDS)
            if f.tell() == 0:
                self._csv.writeheader()
            self._files["csv"] = f
            self.paths.append(p)
        if "json" in self.formats:
            p = base.with_suffix(".jsonl")      # JSON Lines: one object per line, crash-safe
            self._files["json"] = open(p, "a", encoding="utf-8")
            self.paths.append(p)
        if "txt" in self.formats:
            p = base.with_suffix(".txt")
            self._files["txt"] = open(p, "a", encoding="utf-8")
            self.paths.append(p)
        if "sqlite" in self.formats:
            p = self.log_dir / "butt3rflyx.db"
            self._db = sqlite3.connect(str(p), check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, "
                + ", ".join(f"{c} {'REAL' if c in ('confidence', 'duration_s') else ('INTEGER' if c == 'person_id' else 'TEXT')}"
                            for c in FIELDS) + ")")
            self._db.commit()
            self.paths.append(p)

    def log(self, events: list[EventRecord]) -> int:
        """Writes events, skipping exact duplicates within the same frame. Returns #written."""
        written = 0
        with self._lock:
            for ev in events:
                key = (ev.person_id, ev.event_type, ev.label)
                if self._last_t == ev.t and key in self._last_key:
                    continue
                if self._last_t != ev.t:
                    self._last_key, self._last_t = set(), ev.t
                self._last_key.add(key)
                d = ev.as_dict()
                row = {k: d.get(k) for k in FIELDS}
                if self._csv is not None:
                    self._csv.writerow(row)
                if "json" in self._files:
                    self._files["json"].write(json.dumps(row, ensure_ascii=False) + "\n")
                if "txt" in self._files:
                    extra = f" ({row['confidence'] * 100:.0f}%)" if row["confidence"] is not None else ""
                    dur = f" [prev. {row['duration_s']:.1f}s]" if row["duration_s"] is not None else ""
                    self._files["txt"].write(f"{row['timestamp']}  {ev.text}{extra}{dur}\n")
                if self._db is not None:
                    self._db.execute(f"INSERT INTO events (session, {', '.join(FIELDS)}) VALUES (?{', ?' * len(FIELDS)})",
                                     [self.session] + [row[k] for k in FIELDS])
                written += 1
            if written:
                for f in self._files.values():
                    f.flush()
                if self._db is not None:
                    self._db.commit()
        self.count += written
        return written

    def close(self) -> None:
        with self._lock:
            for f in self._files.values():
                try:
                    f.close()
                except OSError:
                    pass
            self._files.clear()
            if self._db is not None:
                self._db.close()
                self._db = None
