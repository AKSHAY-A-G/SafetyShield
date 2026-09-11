"""Data access and business logic for SafetyShield dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import yaml

from src.dashboard.models import (
    AuditEntry,
    CameraDashboardInfo,
    DashboardEvent,
    ModuleStatus,
    RuntimeStatusInfo,
    SystemStatusInfo,
)

DEFAULT_CONTROLS: dict[str, dict[str, Any]] = {
    "person_detection": {
        "enabled": True,
        "label": "Person Detection",
        "description": "YOLO nano person detector on GPU",
    },
    "person_tracking": {
        "enabled": True,
        "label": "Person Tracking",
        "description": "Camera/session-local ByteTrack",
    },
    "bar_001": {
        "enabled": True,
        "label": "BAR-001 Restricted Zone Entry",
        "description": "Zone boundary entry alert",
    },
    "idt_004": {
        "enabled": True,
        "label": "IDT-004 Zone Occupancy",
        "description": "Restricted zone occupancy counter",
    },
    "exc_002": {
        "enabled": True,
        "label": "EXC-002 Buddy-Required Single Occupancy",
        "description": "Single-worker entry alarm in buddy zone",
    },
    "erg_006": {
        "enabled": True,
        "label": "ERG-006 Prolonged Low Movement",
        "description": "Stationary worker alert",
    },
    "evidence": {
        "enabled": True,
        "label": "Evidence Capture",
        "description": "Snapshots and MP4 clips for safety events",
    },
    "rtsp": {
        "enabled": True,
        "label": "Live RTSP Ingestion",
        "description": "Single-camera low-latency latest-frame reader",
    },
}

MILESTONE_STATES: dict[str, str] = {
    "M0": "COMPLETE",
    "M1": "COMPLETE",
    "M2": "COMPLETE",
    "M3": "COMPLETE",
    "M4": "COMPLETE",
    "M5": "COMPLETE",
    "M6": "COMPLETE",
    "M7": "IN DEVELOPMENT",
}


def get_system_status() -> SystemStatusInfo:
    """Retrieve system, environment and milestone states without heavy inference."""
    gpu_name = "NVIDIA GeForce GTX 1650"
    cuda_available = True
    torch_version = "unavailable"
    streamlit_version = "unavailable"

    try:
        import torch  # type: ignore[import-untyped]
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available and torch.cuda.device_count() > 0:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        import streamlit as st  # type: ignore[import-untyped]
        streamlit_version = st.__version__
    except Exception:
        pass

    return SystemStatusInfo(
        gpu_name=gpu_name,
        cuda_available=cuda_available,
        milestones=dict(MILESTONE_STATES),
        python_version=platform.python_version(),
        torch_version=torch_version,
        streamlit_version=streamlit_version,
    )


def _check_secret_presence(env_var_name: str, env_path: Path | None = None) -> bool:
    """Check if secret exists without ever returning or printing its value."""
    if not env_var_name:
        return False
    if env_var_name in os.environ and bool(os.environ[env_var_name].strip()):
        return True
    if env_path is not None and env_path.exists():
        try:
            content = env_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() == env_var_name:
                    cleaned = value.strip().strip("'\"")
                    return bool(cleaned)
        except Exception:
            return False
    return False


def load_camera_dashboard_info(
    cameras_path: Path,
    zones_path: Path,
    env_path: Path | None = None,
) -> dict[str, CameraDashboardInfo]:
    """Load non-secret camera metadata and zone configuration status."""
    if not cameras_path.exists():
        return {}

    try:
        content = yaml.safe_load(cameras_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(content, dict) or "cameras" not in content:
        return {}

    cameras_raw = content["cameras"]
    if not isinstance(cameras_raw, dict):
        return {}

    # Load zones for presence mapping
    zones_by_camera: dict[str, list[str]] = {}
    if zones_path.exists():
        try:
            zones_raw = yaml.safe_load(zones_path.read_text(encoding="utf-8"))
            if isinstance(zones_raw, dict) and "cameras" in zones_raw:
                for cam_id, cam_cfg in zones_raw["cameras"].items():
                    if isinstance(cam_cfg, dict) and "zones" in cam_cfg:
                        zones_list = cam_cfg["zones"]
                        if isinstance(zones_list, list):
                            names = []
                            for z in zones_list:
                                if isinstance(z, dict):
                                    name = z.get("zone_name") or z.get("zone_id") or "unnamed"
                                    names.append(str(name))
                            zones_by_camera[cam_id] = names
        except Exception:
            pass

    results: dict[str, CameraDashboardInfo] = {}
    for cam_id, data in cameras_raw.items():
        if not isinstance(data, dict):
            continue
        env_var = str(data.get("rtsp_env_var", ""))
        source_type = str(data.get("source_type", "unknown"))
        enabled = bool(data.get("enabled", True))
        secret_present = _check_secret_presence(env_var, env_path)
        zone_names = zones_by_camera.get(cam_id, [])

        results[cam_id] = CameraDashboardInfo(
            camera_id=cam_id,
            source_type=source_type,
            enabled=enabled,
            secret_configured=secret_present,
            secret_env_var=env_var,
            has_zone=len(zone_names) > 0,
            zone_count=len(zone_names),
            zone_names=zone_names,
        )

    return results


def load_module_controls(config_path: Path) -> dict[str, dict[str, Any]]:
    """Load module controls from config, returning defaults if missing."""
    if not config_path.exists():
        return {k: dict(v) for k, v in DEFAULT_CONTROLS.items()}

    try:
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as err:
        raise ValueError(f"Malformed module controls YAML in {config_path}: {err}") from err

    if not isinstance(content, dict) or "modules" not in content:
        raise ValueError(f"Invalid module controls format in {config_path}: missing 'modules' key")

    raw_modules = content["modules"]
    if not isinstance(raw_modules, dict):
        raise ValueError("Invalid module controls format: 'modules' must be a mapping")

    controls: dict[str, dict[str, Any]] = {}
    for mod_key, default_val in DEFAULT_CONTROLS.items():
        if mod_key in raw_modules and isinstance(raw_modules[mod_key], dict):
            mod_dict = dict(default_val)
            if "enabled" in raw_modules[mod_key]:
                mod_dict["enabled"] = bool(raw_modules[mod_key]["enabled"])
            controls[mod_key] = mod_dict
        else:
            controls[mod_key] = dict(default_val)

    return controls


def save_module_controls(config_path: Path, modules: dict[str, dict[str, Any]]) -> None:
    """Validate and atomically persist module controls to YAML."""
    if not isinstance(modules, dict):
        raise ValueError("modules must be a dictionary")

    for key, val in modules.items():
        if not isinstance(val, dict) or "enabled" not in val or not isinstance(val["enabled"], bool):
            raise ValueError(f"Module {key} must have a boolean 'enabled' attribute")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "modules": {
            k: {
                "enabled": v["enabled"],
                "label": v.get("label", DEFAULT_CONTROLS.get(k, {}).get("label", k)),
                "description": v.get("description", DEFAULT_CONTROLS.get(k, {}).get("description", "")),
            }
            for k, v in modules.items()
        },
    }

    tmp_path = config_path.with_suffix(".yaml.tmp")
    tmp_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    tmp_path.replace(config_path)


def evaluate_module_availability(
    modules: dict[str, dict[str, Any]],
    camera_info: CameraDashboardInfo,
) -> dict[str, ModuleStatus]:
    """Evaluate module dependency rules and determine availability for a specific camera."""
    statuses: dict[str, ModuleStatus] = {}

    det_enabled = bool(modules.get("person_detection", {}).get("enabled", False))
    track_enabled = bool(modules.get("person_tracking", {}).get("enabled", False))

    for mod_id, meta in DEFAULT_CONTROLS.items():
        cur = modules.get(mod_id, meta)
        global_enabled = bool(cur.get("enabled", False))
        label = str(cur.get("label", meta["label"]))
        desc = str(cur.get("description", meta["description"]))

        available = True
        unavailable_reason: str | None = None

        if mod_id == "person_detection":
            available = True
        elif mod_id == "person_tracking":
            if not det_enabled:
                available = False
                unavailable_reason = "Requires Person Detection enabled"
        elif mod_id in ("bar_001", "idt_004", "exc_002"):
            if not det_enabled:
                available = False
                unavailable_reason = "Requires Person Detection enabled"
            elif not track_enabled:
                available = False
                unavailable_reason = "Requires Person Tracking enabled"
            elif not camera_info.has_zone:
                available = False
                unavailable_reason = f"Unavailable for {camera_info.camera_id} - zone not configured"
        elif mod_id == "erg_006":
            if not det_enabled:
                available = False
                unavailable_reason = "Requires Person Detection enabled"
            elif not track_enabled:
                available = False
                unavailable_reason = "Requires Person Tracking enabled"
            # ERG-006 does NOT require a zone!
        elif mod_id == "evidence":
            # Requires at least one event rule enabled
            rules_enabled = any(
                bool(modules.get(r, {}).get("enabled", False))
                for r in ("bar_001", "idt_004", "exc_002", "erg_006")
            )
            if not rules_enabled:
                available = False
                unavailable_reason = "Requires at least one safety event rule enabled"
        elif mod_id == "rtsp":
            if camera_info.source_type == "rtsp" and not camera_info.secret_configured:
                available = False
                unavailable_reason = f"RTSP secret ({camera_info.secret_env_var}) not configured"

        statuses[mod_id] = ModuleStatus(
            module_id=mod_id,
            label=label,
            description=desc,
            global_enabled=global_enabled,
            available=available,
            unavailable_reason=unavailable_reason,
        )

    return statuses


def load_runtime_status(
    status_path: Path,
    *,
    clock: Callable[[], datetime] | None = None,
    stale_threshold_seconds: float = 30.0,
) -> RuntimeStatusInfo | None:
    """Load latest runtime status JSON safely, evaluating freshness age."""
    if not status_path.exists():
        return None

    try:
        content = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(content, dict):
        return None

    try:
        updated_str = content.get("updated_at_utc", "")
        updated_dt = datetime.fromisoformat(updated_str)
        if updated_dt.tzinfo is None:
            updated_dt = updated_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

    now_dt = clock() if clock is not None else datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    age_seconds = max(0.0, (now_dt - updated_dt).total_seconds())
    is_stale = age_seconds > stale_threshold_seconds

    return RuntimeStatusInfo(
        schema_version=str(content.get("schema_version", "1.0")),
        camera_id=str(content.get("camera_id", "unknown")),
        session_id=str(content.get("session_id", "unknown")),
        state=str(content.get("state", "unknown")),
        updated_at_utc=updated_str,
        age_seconds=round(age_seconds, 1),
        is_stale=is_stale,
        decoded_width=content.get("decoded_width"),
        decoded_height=content.get("decoded_height"),
        source_fps=content.get("source_fps"),
        frames_received=int(content.get("frames_received", 0)),
        frames_processed=int(content.get("frames_processed", 0)),
        frames_dropped=int(content.get("frames_dropped", 0)),
        failed_reads=int(content.get("failed_reads", 0)),
        reconnect_count=int(content.get("reconnect_count", 0)),
        processing_fps=float(content.get("processing_fps", 0.0)),
        temporary_track_count=int(content.get("temporary_track_count", 0)),
        zones_enabled=bool(content.get("zones_enabled", False)),
    )


def load_recent_events(evidence_dir: Path, limit: int = 50) -> list[DashboardEvent]:
    """Discover and parse recent event metadata files, sorted by event timestamp descending."""
    if not evidence_dir.exists() or not evidence_dir.is_dir():
        return []

    events: list[DashboardEvent] = []

    for meta_file in evidence_dir.rglob("metadata.json"):
        try:
            raw = json.loads(meta_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or "event_id" not in raw:
                continue

            event_dir = meta_file.parent
            ev_data = raw.get("evidence", {})
            raw_snap = event_dir / ev_data.get("raw_snapshot", "snapshot_raw.jpg")
            ann_snap = event_dir / ev_data.get("annotated_snapshot", "snapshot_annotated.jpg")
            clip = event_dir / ev_data.get("video_clip", "event_clip.mp4")

            camera_id = str(raw.get("camera_id", "unknown"))
            source_video = raw.get("source_video")
            is_recorded = bool(source_video or "cam_good_test" in camera_id)

            events.append(
                DashboardEvent(
                    event_id=str(raw.get("event_id")),
                    camera_id=camera_id,
                    session_id=str(raw.get("session_id", "")),
                    module_id=str(raw.get("module_id", "UNKNOWN")),
                    event_type=str(raw.get("event_type", "unknown")),
                    track_id=raw.get("track_id"),
                    source_timestamp_seconds=raw.get("source_timestamp_seconds"),
                    frame_number=raw.get("frame_number"),
                    zone_id=raw.get("zone_id"),
                    zone_name=raw.get("zone_name"),
                    severity=str(raw.get("severity", "unclassified")),
                    status=str(raw.get("status", "new")),
                    created_at_utc=raw.get("created_at_utc"),
                    processed_at_utc=raw.get("processed_at_utc"),
                    source_video=str(source_video) if source_video else None,
                    is_recorded_prototype=is_recorded,
                    raw_snapshot_path=raw_snap if raw_snap.exists() else None,
                    annotated_snapshot_path=ann_snap if ann_snap.exists() else None,
                    video_clip_path=clip if clip.exists() else None,
                    raw_metadata=raw,
                )
            )
        except Exception:
            # Skip corrupt or malformed files safely
            continue

    # Sort primarily by source_timestamp_seconds or created_at_utc descending
    def sort_key(ev: DashboardEvent) -> float:
        if ev.source_timestamp_seconds is not None:
            return float(ev.source_timestamp_seconds)
        if ev.created_at_utc:
            try:
                return datetime.fromisoformat(ev.created_at_utc).timestamp()
            except Exception:
                pass
        return 0.0

    events.sort(key=sort_key, reverse=True)
    return events[:limit]


def load_audit_log(evidence_dir: Path, limit: int = 100) -> list[AuditEntry]:
    """Load audit activity lines safely, discovering audit.jsonl across the evidence hierarchy.

    Requirements satisfied:
    - Recursive discovery under evidence/<camera_id>/<session_id>/audit.jsonl or root/session paths
    - No duplicate entries (deduplication on key tuple)
    - Malformed JSONL lines are skipped without crashing
    - Missing audit file returns empty list gracefully
    - Preserves camera and session context from record or directory hierarchy
    """
    if not evidence_dir.exists():
        return []

    entries: list[AuditEntry] = []
    seen: set[tuple[Any, ...]] = set()

    if evidence_dir.is_file() and evidence_dir.name == "audit.jsonl":
        audit_files = [evidence_dir]
    elif evidence_dir.is_dir():
        audit_files = sorted(evidence_dir.rglob("audit.jsonl"))
    else:
        return []

    for audit_file in audit_files:
        # Determine fallback camera and session context from directory hierarchy
        fallback_camera: str | None = None
        fallback_session: str | None = None
        try:
            rel_parts = audit_file.relative_to(evidence_dir).parts
            if len(rel_parts) >= 3:
                fallback_camera = rel_parts[0]
                fallback_session = rel_parts[1]
            elif len(rel_parts) == 2:
                fallback_session = rel_parts[0]
        except Exception:
            pass

        try:
            lines = audit_file.read_text(encoding="utf-8").splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        continue

                    cam_id = record.get("camera_id") or fallback_camera
                    sess_id = record.get("session_id") or fallback_session
                    ev_id = record.get("event_id")
                    action = str(record.get("action", "UNKNOWN"))
                    ts = record.get("audit_timestamp_utc")

                    dedup_key = (ts, cam_id, sess_id, ev_id, action)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    entries.append(
                        AuditEntry(
                            event_id=ev_id,
                            camera_id=cam_id,
                            session_id=sess_id,
                            module_id=record.get("module_id"),
                            action=action,
                            audit_timestamp_utc=ts,
                            details=record.get("details", {}) if isinstance(record.get("details"), dict) else {},
                        )
                    )
                except Exception:
                    continue
        except Exception:
            continue

    def sort_audit(entry: AuditEntry) -> str:
        return entry.audit_timestamp_utc or ""

    entries.sort(key=sort_audit, reverse=True)
    return entries[:limit]
