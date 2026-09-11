"""Common structured safety event representation for SafetyShield."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Mapping
from uuid import uuid4

from src.rules.temporal import TemporalSafetyEvent
from src.rules.zones import BarEntryEvent


DEFAULT_SCHEMA_VERSION = "1.0"
DEFAULT_SEVERITY = "unclassified"
DEFAULT_STATUS = "new"


def generate_event_id(prefix: str = "evt") -> str:
    """Generate a collision-resistant, credentials-free event identifier."""
    cleaned_prefix = prefix.strip().lower().replace("-", "_") if prefix else "evt"
    return f"{cleaned_prefix}_{uuid4().hex[:16]}"


@dataclass(frozen=True, slots=True)
class SafetyEvent:
    """Standardized event record decoupled from specific detector or rule engines."""

    event_id: str
    camera_id: str
    session_id: str
    module_id: str
    event_type: str
    track_id: int
    source_timestamp_seconds: float
    frame_number: int
    schema_version: str = DEFAULT_SCHEMA_VERSION
    zone_id: str | None = None
    zone_name: str | None = None
    severity: str = DEFAULT_SEVERITY
    status: str = DEFAULT_STATUS
    detection_confidence: float | None = None
    stationary_duration_seconds: float | None = None
    zone_occupancy: int | None = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.event_id or not isinstance(self.event_id, str):
            raise ValueError("event_id must be a non-empty string")
        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string")
        if not self.session_id or not isinstance(self.session_id, str):
            raise ValueError("session_id must be a non-empty string")
        if not self.module_id or not isinstance(self.module_id, str):
            raise ValueError("module_id must be a non-empty string")
        if not self.event_type or not isinstance(self.event_type, str):
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(self.track_id, int) or isinstance(self.track_id, bool) or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        if not isinstance(self.source_timestamp_seconds, (int, float)) or not math.isfinite(self.source_timestamp_seconds) or self.source_timestamp_seconds < 0:
            raise ValueError("source_timestamp_seconds must be a non-negative finite number")
        if not isinstance(self.frame_number, int) or isinstance(self.frame_number, bool) or self.frame_number < 0:
            raise ValueError("frame_number must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dictionary without exposing internal runtime types."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SafetyEvent:
        """Construct a SafetyEvent from a dictionary with validation."""
        required = {
            "event_id",
            "camera_id",
            "session_id",
            "module_id",
            "event_type",
            "track_id",
            "source_timestamp_seconds",
            "frame_number",
        }
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Missing required event fields: {sorted(missing)}")
        return cls(
            event_id=str(data["event_id"]),
            camera_id=str(data["camera_id"]),
            session_id=str(data["session_id"]),
            module_id=str(data["module_id"]),
            event_type=str(data["event_type"]),
            track_id=int(data["track_id"]),
            source_timestamp_seconds=float(data["source_timestamp_seconds"]),
            frame_number=int(data["frame_number"]),
            schema_version=str(data.get("schema_version", DEFAULT_SCHEMA_VERSION)),
            zone_id=str(data["zone_id"]) if data.get("zone_id") is not None else None,
            zone_name=str(data["zone_name"]) if data.get("zone_name") is not None else None,
            severity=str(data.get("severity", DEFAULT_SEVERITY)),
            status=str(data.get("status", DEFAULT_STATUS)),
            detection_confidence=(
                float(data["detection_confidence"])
                if data.get("detection_confidence") is not None
                else None
            ),
            stationary_duration_seconds=(
                float(data["stationary_duration_seconds"])
                if data.get("stationary_duration_seconds") is not None
                else None
            ),
            zone_occupancy=(
                int(data["zone_occupancy"])
                if data.get("zone_occupancy") is not None
                else None
            ),
            created_at_utc=str(
                data.get(
                    "created_at_utc",
                    datetime.now(timezone.utc).isoformat(),
                )
            ),
        )

    @classmethod
    def from_bar_entry_event(
        cls,
        event: BarEntryEvent,
        severity: str = DEFAULT_SEVERITY,
        status: str = DEFAULT_STATUS,
        event_id: str | None = None,
    ) -> SafetyEvent:
        """Adapt a Milestone 3 BAR-001 rule event into a standardized SafetyEvent."""
        return cls(
            event_id=event_id or generate_event_id(f"bar_{event.track_id}"),
            camera_id=event.camera_id,
            session_id=event.session_id,
            module_id=event.module_id,
            event_type=event.event_type,
            track_id=event.track_id,
            source_timestamp_seconds=round(float(event.timestamp_seconds), 3),
            frame_number=event.frame_number,
            zone_id=event.zone_id,
            zone_name=event.zone_name,
            severity=severity,
            status=status,
            detection_confidence=round(float(event.confidence), 4),
            zone_occupancy=None,
        )

    @classmethod
    def from_temporal_safety_event(
        cls,
        event: TemporalSafetyEvent,
        zone_name: str | None = None,
        severity: str = DEFAULT_SEVERITY,
        status: str = DEFAULT_STATUS,
        event_id: str | None = None,
    ) -> SafetyEvent:
        """Adapt a Milestone 4 temporal safety event into a standardized SafetyEvent."""
        prefix = f"{event.module_id.lower().replace('-', '_')}_{event.track_id}"
        return cls(
            event_id=event_id or generate_event_id(prefix),
            camera_id=event.camera_id,
            session_id=event.session_id,
            module_id=event.module_id,
            event_type=event.event_type,
            track_id=event.track_id,
            source_timestamp_seconds=round(float(event.timestamp_seconds), 3),
            frame_number=event.frame_number,
            zone_id=event.zone_id,
            zone_name=zone_name,
            severity=severity,
            status=status,
            stationary_duration_seconds=(
                round(float(event.stationary_duration_seconds), 2)
                if event.stationary_duration_seconds is not None
                else None
            ),
            zone_occupancy=event.current_zone_occupancy,
        )
