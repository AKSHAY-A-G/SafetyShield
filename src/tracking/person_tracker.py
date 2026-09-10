"""Camera/session-local ByteTrack integration for person detections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace
from typing import Iterable

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from src.detection.person_detector import PERSON_CLASS_ID, PersonDetection


@dataclass(frozen=True, slots=True)
class ByteTrackConfig:
    """Conservative Ultralytics ByteTrack defaults, kept configurable."""

    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.80
    fuse_score: bool = True

    def __post_init__(self) -> None:
        thresholds = (
            self.track_high_thresh,
            self.track_low_thresh,
            self.new_track_thresh,
            self.match_thresh,
        )
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("ByteTrack thresholds must be between 0 and 1")
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("track_low_thresh cannot exceed track_high_thresh")
        if self.track_buffer < 0:
            raise ValueError("track_buffer cannot be negative")


@dataclass(frozen=True, slots=True)
class TrackedPerson:
    """One temporary track observation within a camera/session."""

    camera_id: str
    session_id: str
    track_id: int
    frame_index: int
    timestamp_seconds: float
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def bottom_center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, self.y2)


class PersonTracker:
    """Wrap ByteTrack while exposing IDs local to one camera/session instance."""

    def __init__(
        self,
        camera_id: str,
        session_id: str,
        config: ByteTrackConfig | None = None,
    ) -> None:
        if not camera_id.strip() or not session_id.strip():
            raise ValueError("camera_id and session_id must be non-empty")
        self.camera_id = camera_id
        self.session_id = session_id
        self.config = config or ByteTrackConfig()
        args = SimpleNamespace(tracker_type="bytetrack", **asdict(self.config))
        try:
            self._tracker = BYTETracker(args)
        except Exception as error:
            raise RuntimeError(
                f"ByteTrack initialization failed ({type(error).__name__})"
            ) from None
        self._local_ids: dict[int, int] = {}
        self._next_local_id = 1

    @property
    def unique_track_count(self) -> int:
        """Number of temporary IDs that have appeared in tracker output."""
        return len(self._local_ids)

    def update(
        self,
        detections: Iterable[PersonDetection],
        frame_shape: tuple[int, int] | tuple[int, int, int],
        frame_index: int,
        timestamp_seconds: float,
    ) -> list[TrackedPerson]:
        """Advance ByteTrack once and return active tracks in original coordinates."""
        if len(frame_shape) < 2:
            raise ValueError("frame_shape must contain height and width")
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("frame dimensions must be positive")

        rows = []
        for detection in detections:
            values = (
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
                detection.confidence,
            )
            if not all(math.isfinite(float(value)) for value in values):
                continue
            x1 = max(0, min(width - 1, detection.x1))
            y1 = max(0, min(height - 1, detection.y1))
            x2 = max(0, min(width - 1, detection.x2))
            y2 = max(0, min(height - 1, detection.y2))
            if x2 <= x1 or y2 <= y1 or not 0.0 <= detection.confidence <= 1.0:
                continue
            rows.append((x1, y1, x2, y2, detection.confidence, PERSON_CLASS_ID))

        box_data = np.asarray(rows, dtype=np.float32).reshape((-1, 6))
        results = Boxes(box_data, orig_shape=(height, width))
        tracked_rows = self._tracker.update(results)
        tracks: list[TrackedPerson] = []
        for row in tracked_rows:
            raw_id = int(row[4])
            local_id = self._local_ids.get(raw_id)
            if local_id is None:
                local_id = self._next_local_id
                self._local_ids[raw_id] = local_id
                self._next_local_id += 1
            x1, y1, x2, y2 = (
                max(0, min(width - 1, round(float(row[index]))))
                for index in range(4)
            )
            if x2 <= x1 or y2 <= y1:
                continue
            tracks.append(
                TrackedPerson(
                    camera_id=self.camera_id,
                    session_id=self.session_id,
                    track_id=local_id,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(row[5]),
                )
            )
        return tracks
