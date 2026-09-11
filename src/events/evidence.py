"""Event evidence capture, snapshot persistence, and video clip extraction (EVD-001)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from src.events.audit import (
    AUDIT_ACTION_EVIDENCE_ERROR,
    AUDIT_ACTION_EVIDENCE_SAVED,
    AuditLogger,
)
from src.events.models import SafetyEvent


DEFAULT_PRE_EVENT_SECONDS = 5.0
DEFAULT_POST_EVENT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class ClipCoverage:
    """Requested vs actual time coverage for an extracted evidence video clip."""

    requested_pre_seconds: float
    requested_post_seconds: float
    actual_clip_start_seconds: float
    actual_clip_end_seconds: float
    actual_pre_seconds: float
    actual_post_seconds: float
    clip_frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_pre_seconds": self.requested_pre_seconds,
            "requested_post_seconds": self.requested_post_seconds,
            "actual_clip_start_seconds": self.actual_clip_start_seconds,
            "actual_clip_end_seconds": self.actual_clip_end_seconds,
            "actual_pre_seconds": self.actual_pre_seconds,
            "actual_post_seconds": self.actual_post_seconds,
            "clip_frame_count": self.clip_frame_count,
        }


@dataclass(frozen=True, slots=True)
class EvidenceArtifacts:
    """Paths and coverage metadata for saved evidence items."""

    event_id: str
    event_dir: Path
    raw_snapshot_path: Path
    annotated_snapshot_path: Path
    clip_path: Path
    metadata_path: Path
    coverage: ClipCoverage


def verify_evidence_clip(
    path: Path | str,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> dict[str, Any]:
    """Verify that an extracted evidence video exists, opens, and contains valid frames."""
    clip_path = Path(path).resolve()
    if not clip_path.is_file():
        raise RuntimeError(f"Evidence clip does not exist: {clip_path}")

    capture = cv2.VideoCapture(str(clip_path))
    try:
        opened = capture.isOpened()
        if not opened:
            raise RuntimeError(f"Could not open evidence clip: {clip_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        readable, frame = capture.read()

        if not readable or frame is None:
            raise RuntimeError(f"First frame of evidence clip is unreadable: {clip_path}")
        if frame_count <= 0:
            raise RuntimeError(f"Evidence clip contains 0 frames: {clip_path}")
        if fps <= 0.0 or not math.isfinite(fps):
            raise RuntimeError(f"Evidence clip has invalid FPS ({fps}): {clip_path}")
        if expected_width is not None and width != expected_width:
            raise RuntimeError(
                f"Evidence clip width mismatch: expected {expected_width}, got {width}"
            )
        if expected_height is not None and height != expected_height:
            raise RuntimeError(
                f"Evidence clip height mismatch: expected {expected_height}, got {height}"
            )

        return {
            "opened": True,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "readable_frame": True,
            "dimensions_match": (
                (expected_width is None or width == expected_width)
                and (expected_height is None or height == expected_height)
            ),
        }
    finally:
        capture.release()


def compute_clip_bounds(
    source_timestamp_seconds: float,
    source_duration_seconds: float,
    pre_event_seconds: float = DEFAULT_PRE_EVENT_SECONDS,
    post_event_seconds: float = DEFAULT_POST_EVENT_SECONDS,
) -> tuple[float, float, float, float]:
    """Compute safely clamped start/end times and actual pre/post coverage durations."""
    if not math.isfinite(source_timestamp_seconds) or source_timestamp_seconds < 0:
        raise ValueError(f"Invalid source_timestamp_seconds: {source_timestamp_seconds}")
    if not math.isfinite(source_duration_seconds) or source_duration_seconds <= 0:
        raise ValueError(f"Invalid source_duration_seconds: {source_duration_seconds}")
    if pre_event_seconds < 0 or post_event_seconds < 0:
        raise ValueError("Pre- and post-event durations must be non-negative")

    actual_start = max(0.0, source_timestamp_seconds - pre_event_seconds)
    actual_end = min(source_duration_seconds, source_timestamp_seconds + post_event_seconds)

    # Ensure valid interval even if source_timestamp is at or beyond source_duration
    if actual_end < actual_start:
        actual_end = actual_start

    actual_pre = round(source_timestamp_seconds - actual_start, 3)
    actual_post = round(actual_end - source_timestamp_seconds, 3)

    return (
        round(actual_start, 3),
        round(actual_end, 3),
        max(0.0, actual_pre),
        max(0.0, actual_post),
    )


class EvidenceWriter:
    """Extracts and persists snapshots, video clips, and metadata for safety events."""

    def __init__(
        self,
        evidence_root: Path | str,
        audit_logger: AuditLogger | None = None,
        pre_event_seconds: float = DEFAULT_PRE_EVENT_SECONDS,
        post_event_seconds: float = DEFAULT_POST_EVENT_SECONDS,
    ) -> None:
        self.evidence_root = Path(evidence_root).resolve()
        self.audit_logger = audit_logger
        self.pre_event_seconds = float(pre_event_seconds)
        self.post_event_seconds = float(post_event_seconds)

    def save_event_evidence(
        self,
        event: SafetyEvent,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        source_video_path: Path | str,
        source_duration_seconds: float | None = None,
        source_fps: float | None = None,
    ) -> EvidenceArtifacts:
        """Create snapshots, extract video clip, write metadata, and log audit event.

        Raises FileExistsError if the target event evidence directory already exists.
        """
        event_dir = (
            self.evidence_root
            / event.camera_id
            / event.session_id
            / f"event_{event.event_id}"
        )
        if event_dir.exists():
            error_msg = f"Event evidence directory already exists: {event_dir}"
            if self.audit_logger is not None:
                self.audit_logger.log(
                    event_id=event.event_id,
                    camera_id=event.camera_id,
                    session_id=event.session_id,
                    module_id=event.module_id,
                    action=AUDIT_ACTION_EVIDENCE_ERROR,
                    details={"error": "DUPLICATE_EVENT_DIRECTORY", "path": str(event_dir)},
                )
            raise FileExistsError(error_msg)

        if raw_frame is None or annotated_frame is None:
            raise ValueError("Both raw_frame and annotated_frame must be provided")
        if raw_frame.shape != annotated_frame.shape:
            raise ValueError(
                f"Frame shape mismatch: raw {raw_frame.shape} vs annotated {annotated_frame.shape}"
            )

        try:
            source_path = Path(source_video_path).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"Source video not found: {source_path}")

            event_dir.mkdir(parents=True, exist_ok=False)

            # 1. Write raw and annotated snapshots
            raw_snapshot_path = event_dir / "snapshot_raw.jpg"
            annotated_snapshot_path = event_dir / "snapshot_annotated.jpg"

            success_raw = cv2.imwrite(str(raw_snapshot_path), raw_frame)
            if not success_raw or not raw_snapshot_path.is_file():
                raise RuntimeError(f"Failed to write raw snapshot: {raw_snapshot_path}")

            success_ann = cv2.imwrite(str(annotated_snapshot_path), annotated_frame)
            if not success_ann or not annotated_snapshot_path.is_file():
                raise RuntimeError(f"Failed to write annotated snapshot: {annotated_snapshot_path}")

            # 2. Extract video clip from source video
            clip_path = event_dir / "event_clip.mp4"
            coverage = self._extract_clip(
                source_video_path=source_path,
                event_timestamp_seconds=event.source_timestamp_seconds,
                output_clip_path=clip_path,
                source_duration_seconds=source_duration_seconds,
                source_fps=source_fps,
                expected_width=raw_frame.shape[1],
                expected_height=raw_frame.shape[0],
            )

            # 3. Write metadata JSON
            metadata_path = event_dir / "metadata.json"
            processed_at_utc = datetime.now(timezone.utc).isoformat()
            metadata_content = {
                **event.to_dict(),
                "source_video": source_path.name,
                "processed_at_utc": processed_at_utc,
                "evidence": {
                    "raw_snapshot": raw_snapshot_path.name,
                    "annotated_snapshot": annotated_snapshot_path.name,
                    "video_clip": clip_path.name,
                    **coverage.to_dict(),
                    "resolution": [raw_frame.shape[1], raw_frame.shape[0]],
                },
            }

            # Atomic write of metadata.json
            with tempfile.NamedTemporaryFile(
                mode="w", dir=str(event_dir), delete=False, encoding="utf-8"
            ) as tmp_file:
                json.dump(metadata_content, tmp_file, indent=2, ensure_ascii=False)
                tmp_path = Path(tmp_file.name)
            tmp_path.replace(metadata_path)

            artifacts = EvidenceArtifacts(
                event_id=event.event_id,
                event_dir=event_dir,
                raw_snapshot_path=raw_snapshot_path,
                annotated_snapshot_path=annotated_snapshot_path,
                clip_path=clip_path,
                metadata_path=metadata_path,
                coverage=coverage,
            )

            if self.audit_logger is not None:
                self.audit_logger.log(
                    event_id=event.event_id,
                    camera_id=event.camera_id,
                    session_id=event.session_id,
                    module_id=event.module_id,
                    action=AUDIT_ACTION_EVIDENCE_SAVED,
                    details={
                        "event_dir": str(event_dir.relative_to(self.evidence_root)),
                        "raw_snapshot": raw_snapshot_path.name,
                        "annotated_snapshot": annotated_snapshot_path.name,
                        "clip": clip_path.name,
                        "metadata": metadata_path.name,
                        **coverage.to_dict(),
                    },
                )

            return artifacts

        except Exception as exc:
            if self.audit_logger is not None:
                self.audit_logger.log(
                    event_id=event.event_id,
                    camera_id=event.camera_id,
                    session_id=event.session_id,
                    module_id=event.module_id,
                    action=AUDIT_ACTION_EVIDENCE_ERROR,
                    details={
                        "error": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            raise

    def _extract_clip(
        self,
        source_video_path: Path,
        event_timestamp_seconds: float,
        output_clip_path: Path,
        source_duration_seconds: float | None,
        source_fps: float | None,
        expected_width: int,
        expected_height: int,
    ) -> ClipCoverage:
        """Extract bounded clip from source video and verify reopened file."""
        capture = cv2.VideoCapture(str(source_video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source video: {source_video_path}")

        try:
            fps = source_fps or float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0 or not math.isfinite(fps):
                fps = 30.0

            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = (
                source_duration_seconds
                if source_duration_seconds is not None and source_duration_seconds > 0
                else (total_frames / fps if total_frames > 0 else 0.0)
            )
            if duration <= 0:
                duration = event_timestamp_seconds + self.post_event_seconds

            start_sec, end_sec, act_pre, act_post = compute_clip_bounds(
                source_timestamp_seconds=event_timestamp_seconds,
                source_duration_seconds=duration,
                pre_event_seconds=self.pre_event_seconds,
                post_event_seconds=self.post_event_seconds,
            )

            start_frame = max(0, round(start_sec * fps))
            end_frame = (
                min(total_frames - 1, round(end_sec * fps))
                if total_frames > 0
                else round(end_sec * fps)
            )
            if end_frame < start_frame:
                end_frame = start_frame

            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            writer = cv2.VideoWriter(
                str(output_clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),  # pyrefly: ignore[missing-attribute]
                fps,
                (expected_width, expected_height),
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"Could not create evidence clip writer: {output_clip_path}")

            frames_written = 0
            current_frame_idx = start_frame
            try:
                while current_frame_idx <= end_frame:
                    ret, frame = capture.read()
                    if not ret or frame is None:
                        break
                    writer.write(frame)
                    frames_written += 1
                    current_frame_idx += 1
            finally:
                writer.release()

            if frames_written == 0:
                raise RuntimeError(f"Extracted clip contains 0 frames: {output_clip_path}")

            # Reopen check to verify integrity
            verify_evidence_clip(
                output_clip_path,
                expected_width=expected_width,
                expected_height=expected_height,
            )

            return ClipCoverage(
                requested_pre_seconds=self.pre_event_seconds,
                requested_post_seconds=self.post_event_seconds,
                actual_clip_start_seconds=start_sec,
                actual_clip_end_seconds=end_sec,
                actual_pre_seconds=act_pre,
                actual_post_seconds=act_post,
                clip_frame_count=frames_written,
            )
        finally:
            capture.release()
