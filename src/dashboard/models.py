"""Data transfer models for SafetyShield dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SystemStatusInfo:
    gpu_name: str
    cuda_available: bool
    milestones: dict[str, str]
    python_version: str
    torch_version: str
    streamlit_version: str


@dataclass(frozen=True, slots=True)
class CameraDashboardInfo:
    camera_id: str
    source_type: str
    enabled: bool
    secret_configured: bool
    secret_env_var: str
    has_zone: bool
    zone_count: int
    zone_names: list[str]


@dataclass(frozen=True, slots=True)
class ModuleStatus:
    module_id: str
    label: str
    description: str
    global_enabled: bool
    available: bool
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class RuntimeStatusInfo:
    schema_version: str
    camera_id: str
    session_id: str
    state: str
    updated_at_utc: str
    age_seconds: float
    is_stale: bool
    decoded_width: int | None
    decoded_height: int | None
    source_fps: float | None
    frames_received: int
    frames_processed: int
    frames_dropped: int
    failed_reads: int
    reconnect_count: int
    processing_fps: float
    temporary_track_count: int
    zones_enabled: bool


@dataclass(frozen=True, slots=True)
class DashboardEvent:
    event_id: str
    camera_id: str
    session_id: str
    module_id: str
    event_type: str
    track_id: int | None
    source_timestamp_seconds: float | None
    frame_number: int | None
    zone_id: str | None
    zone_name: str | None
    severity: str
    status: str
    created_at_utc: str | None
    processed_at_utc: str | None
    source_video: str | None
    is_recorded_prototype: bool
    raw_snapshot_path: Path | None
    annotated_snapshot_path: Path | None
    video_clip_path: Path | None
    raw_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AuditEntry:
    event_id: str | None
    camera_id: str | None
    session_id: str | None
    module_id: str | None
    action: str
    audit_timestamp_utc: str | None
    details: dict[str, Any]
