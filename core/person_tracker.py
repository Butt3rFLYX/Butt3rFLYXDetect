"""Stable person IDs on top of the YOLO/ByteTrack tracker.

ByteTrack sometimes hands out a new raw ID after an occlusion or when a person leaves
the frame for a moment. This layer maps raw IDs to stable display IDs (PERSON 001, ...)
and re-uses a recently lost ID when a new detection appears where that person was
expected to be (position + motion prediction + size similarity).
No biometric data is used or stored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import iou


@dataclass
class Track:
    pid: int
    raw_id: int | None
    box: np.ndarray
    first_seen: float
    last_seen: float
    velocity: np.ndarray
    lost: bool = False

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def height(self) -> float:
        return float(max(self.box[3] - self.box[1], 1.0))


class PersonTracker:
    def __init__(self, reid_timeout: float = 3.0, max_center_dist: float = 0.8, size_tolerance: float = 0.45) -> None:
        self.reid_timeout = reid_timeout
        self.max_center_dist = max_center_dist
        self.size_tolerance = size_tolerance
        self.tracks: dict[int, Track] = {}
        self.raw_to_pid: dict[int, int] = {}
        self.next_pid = 1

    def configure(self, sensitivity: float) -> None:
        """sensitivity 0..1: higher = keeps IDs longer and matches over larger distances."""
        s = float(np.clip(sensitivity, 0.0, 1.0))
        self.reid_timeout = 1.0 + 5.0 * s
        self.max_center_dist = 0.4 + 0.8 * s

    def reset(self) -> None:
        self.tracks.clear()
        self.raw_to_pid.clear()
        self.next_pid = 1

    def update(self, detections: list[tuple[int | None, np.ndarray]], t: float):
        """detections: [(raw_id or None, box xyxy)]. Returns (pids per detection, events).
        events: [(pid, "entered" | "reacquired" | "rematched" | "left")]
        "rematched" = an existing ID was re-associated to a different tracker ID; the analysis
        history should be reset because it may now be a different person."""
        events: list[tuple[int, str]] = []
        pids: list[int | None] = [None] * len(detections)
        claimed: set[int] = set()

        # 1) known raw IDs
        for i, (raw, box) in enumerate(detections):
            if raw is not None and raw in self.raw_to_pid:
                pid = self.raw_to_pid[raw]
                if pid in self.tracks and pid not in claimed:
                    pids[i] = pid
                    claimed.add(pid)

        # 2) unknown detections -> match against unclaimed active / recently lost tracks
        cands = [tr for tr in self.tracks.values() if tr.pid not in claimed and t - tr.last_seen <= self.reid_timeout]
        pairs = []
        for i, (raw, box) in enumerate(detections):
            if pids[i] is not None:
                continue
            box = np.asarray(box, float)
            c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            h = float(max(box[3] - box[1], 1.0))
            for tr in cands:
                dt = min(max(t - tr.last_seen, 0.0), 1.0)
                pred = tr.center + tr.velocity * dt
                d = float(np.linalg.norm(c - pred)) / max(h, tr.height)
                ratio = h / tr.height
                if d <= self.max_center_dist and (1 - self.size_tolerance) <= ratio <= 1 / (1 - self.size_tolerance):
                    score = d - 0.5 * iou(box, tr.box)
                    pairs.append((score, i, tr.pid))
        for _, i, pid in sorted(pairs):
            if pids[i] is not None or pid in claimed:
                continue
            pids[i] = pid
            claimed.add(pid)
            if self.tracks[pid].lost:
                events.append((pid, "reacquired"))

        # 3) new people
        for i in range(len(detections)):
            if pids[i] is None:
                pid = self.next_pid
                self.next_pid += 1
                box = np.asarray(detections[i][1], float)
                self.tracks[pid] = Track(pid, None, box, t, t, np.zeros(2))
                pids[i] = pid
                events.append((pid, "entered"))

        # update state
        for (raw, box), pid in zip(detections, pids):
            tr = self.tracks[pid]
            box = np.asarray(box, float)
            new_c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            dt = t - tr.last_seen
            if 0 < dt < 1.0:
                tr.velocity = 0.7 * tr.velocity + 0.3 * (new_c - tr.center) / dt
            tr.box, tr.last_seen, tr.lost = box, t, False
            if raw is not None:
                if tr.raw_id is not None and tr.raw_id != raw:
                    self.raw_to_pid.pop(tr.raw_id, None)
                    if (pid, "reacquired") not in events:
                        events.append((pid, "rematched"))
                tr.raw_id = raw
                self.raw_to_pid[raw] = pid

        for pid, tr in list(self.tracks.items()):
            if pid in claimed or pid in pids:
                continue
            if not tr.lost:
                tr.lost = True
            if t - tr.last_seen > self.reid_timeout:
                events.append((pid, "left"))
                if tr.raw_id is not None:
                    self.raw_to_pid.pop(tr.raw_id, None)
                del self.tracks[pid]
        return pids, events
