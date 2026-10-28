"""Session-level tracking statistics.

Tracker IDs are assigned by ByteTrack/BoT-SORT. This module never creates IDs;
it only aggregates the IDs that Ultralytics actually returns. A new
``TrackingSession`` is created for every processing session, so unique-ID
statistics reset when a new webcam session or video starts.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class TrackStats:
    """Immutable snapshot of tracking statistics for one inference frame."""

    frame_number: int = 0
    current_objects: int = 0
    unique_ids: int = 0
    class_counts: Mapping[str, int] = field(default_factory=dict)
    unique_class_counts: Mapping[str, int] = field(default_factory=dict)
    untracked_objects: int = 0
    current_ids: tuple[int, ...] = ()
    id_first_frame: Mapping[int, int] = field(default_factory=dict)
    id_last_frame: Mapping[int, int] = field(default_factory=dict)
    id_observations: Mapping[int, int] = field(default_factory=dict)
    id_classes: Mapping[int, str] = field(default_factory=dict)
    inference_frames: int = 0
    total_tracked_observations: int = 0

    def class_rows(self) -> list[dict[str, Any]]:
        """Rows for the class breakdown table (current and unique counts)."""
        names = sorted(set(self.class_counts) | set(self.unique_class_counts))
        rows = [
            {
                "Class": name,
                "Objects now": int(self.class_counts.get(name, 0)),
                "Unique IDs": int(self.unique_class_counts.get(name, 0)),
            }
            for name in names
        ]
        rows.sort(key=lambda row: (-row["Objects now"], -row["Unique IDs"], row["Class"]))
        return rows

    def id_rows(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Rows for the live tracking-ID registry, most recently seen first."""
        ids = sorted(
            self.id_first_frame,
            key=lambda track_id: (-int(self.id_last_frame.get(track_id, 0)), int(track_id)),
        )
        if limit is not None:
            ids = ids[: max(0, int(limit))]
        return [
            {
                "ID": int(track_id),
                "Class": self.id_classes.get(track_id, "object"),
                "First frame": int(self.id_first_frame.get(track_id, 0)),
                "Last frame": int(self.id_last_frame.get(track_id, 0)),
                "Seen (frames)": int(self.id_observations.get(track_id, 0)),
            }
            for track_id in ids
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_number": int(self.frame_number),
            "current_objects": int(self.current_objects),
            "unique_ids": int(self.unique_ids),
            "class_counts": dict(self.class_counts),
            "unique_class_counts": dict(self.unique_class_counts),
            "untracked_objects": int(self.untracked_objects),
            "current_ids": list(self.current_ids),
            "inference_frames": int(self.inference_frames),
            "total_tracked_observations": int(self.total_tracked_observations),
            "id_rows": self.id_rows(),
        }


class TrackingSession:
    """Accumulates per-frame detections for one processing session."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[int, dict[str, Any]] = {}
        self._unique_class_counts: Counter[str] = Counter()
        self._inference_frames = 0
        self._last_stats = TrackStats()

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._unique_class_counts.clear()
            self._inference_frames = 0
            self._last_stats = TrackStats()

    def update(self, detections: Iterable[Any], frame_number: int) -> TrackStats:
        """Aggregate one inference frame's detections.

        Detections with ``track_id is None`` are included in object/class counts
        but deliberately excluded from the unique-ID registry.
        """
        frame = max(0, int(frame_number))
        items = list(detections)

        with self._lock:
            class_counts: Counter[str] = Counter()
            current_ids: set[int] = set()
            untracked = 0
            ids_seen_this_frame: set[int] = set()

            for detection in items:
                class_name = str(getattr(detection, "class_name", None) or "object")
                class_counts[class_name] += 1

                raw_track_id = getattr(detection, "track_id", None)
                if raw_track_id is None:
                    untracked += 1
                    continue
                try:
                    track_id = int(raw_track_id)
                except (TypeError, ValueError):
                    untracked += 1
                    continue

                current_ids.add(track_id)
                record = self._records.get(track_id)
                if record is None:
                    record = {
                        "class_name": class_name,
                        "first_frame": frame,
                        "last_frame": frame,
                        "observations": 0,
                    }
                    self._records[track_id] = record
                    # Unique-class statistics follow the class seen when the ID
                    # was first assigned, which keeps the total consistent.
                    self._unique_class_counts[class_name] += 1

                record["last_frame"] = frame
                if track_id not in ids_seen_this_frame:
                    # Count once per frame even if a tracker anomalously returned
                    # the same ID twice; this keeps "seen frames" meaningful.
                    record["observations"] += 1
                    ids_seen_this_frame.add(track_id)

            self._inference_frames += 1
            stats = TrackStats(
                frame_number=frame,
                current_objects=len(items),
                unique_ids=len(self._records),
                class_counts=dict(class_counts),
                unique_class_counts=dict(self._unique_class_counts),
                untracked_objects=untracked,
                current_ids=tuple(sorted(current_ids)),
                id_first_frame={
                    track_id: int(record["first_frame"])
                    for track_id, record in self._records.items()
                },
                id_last_frame={
                    track_id: int(record["last_frame"])
                    for track_id, record in self._records.items()
                },
                id_observations={
                    track_id: int(record["observations"])
                    for track_id, record in self._records.items()
                },
                id_classes={
                    track_id: str(record["class_name"])
                    for track_id, record in self._records.items()
                },
                inference_frames=self._inference_frames,
                total_tracked_observations=sum(
                    int(record["observations"]) for record in self._records.values()
                ),
            )
            self._last_stats = stats
            return stats

    def snapshot(self) -> TrackStats:
        with self._lock:
            return self._last_stats

    @property
    def unique_ids(self) -> int:
        with self._lock:
            return len(self._records)
