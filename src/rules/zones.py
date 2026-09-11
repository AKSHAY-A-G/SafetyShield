"""Configured polygon geometry and camera-local BAR-001/IDT-004 rules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from src.tracking.person_tracker import TrackedPerson


SUPPORTED_ZONE_TYPES = frozenset({"restricted", "counting"})


@dataclass(frozen=True, slots=True)
class ZoneDefinition:
    zone_id: str
    zone_name: str
    zone_type: str
    enabled: bool
    polygon: tuple[tuple[float, float], ...]
    confirm_inside_seconds: float = 0.20
    confirm_outside_seconds: float = 0.20
    state_ttl_seconds: float = 2.0


@dataclass(frozen=True, slots=True)
class CameraZoneConfig:
    camera_id: str
    expected_width: int
    expected_height: int
    fixed_camera: bool
    zones: tuple[ZoneDefinition, ...]


@dataclass(frozen=True, slots=True)
class BarEntryEvent:
    camera_id: str
    session_id: str
    track_id: int
    module_id: str
    event_type: str
    zone_id: str
    zone_name: str
    timestamp_seconds: float
    frame_number: int
    confidence: float


@dataclass(frozen=True, slots=True)
class ZoneFrameResult:
    counts: Mapping[str, int]
    inside_track_ids: Mapping[str, frozenset[int]]
    events: tuple[BarEntryEvent, ...]


@dataclass(slots=True)
class _TrackZoneState:
    confirmed_inside: bool = False
    confirmed_outside: bool = False
    pending_inside_since: float | None = None
    pending_outside_since: float | None = None
    last_seen: float = 0.0


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _polygon_area(polygon: tuple[tuple[float, float], ...]) -> float:
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1])
        )
    ) / 2.0


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if not math.isclose(cross, 0.0, abs_tol=1e-9):
        return False
    return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)


def point_in_polygon(
    point: tuple[float, float], polygon: tuple[tuple[float, float], ...]
) -> bool:
    """Return membership using an inclusive, deterministic polygon boundary."""
    if len(polygon) < 3:
        raise ValueError("polygon must contain at least three vertices")
    if any(_point_on_segment(point, a, b) for a, b in zip(polygon, polygon[1:] + polygon[:1])):
        return True

    px, py = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > py) != (y2 > py):
            crossing_x = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < crossing_x:
                inside = not inside
        previous = current
    return inside


def _require_mapping(value: object, label: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _duration(zone: Mapping[object, object], key: str, default: float) -> float:
    value = zone.get(key, default)
    if not _is_finite_number(value) or float(value) < 0:
        raise ValueError(f"{key} must be a finite non-negative number")
    return float(value)


def _parse_zone(
    raw_zone: object, width: int, height: int, seen_ids: set[str]
) -> ZoneDefinition:
    zone = _require_mapping(raw_zone, "zone")
    zone_id = _require_text(zone.get("zone_id"), "zone_id")
    if zone_id in seen_ids:
        raise ValueError(f"duplicate zone_id: {zone_id}")
    seen_ids.add(zone_id)
    zone_name = _require_text(zone.get("zone_name"), "zone_name")
    zone_type = _require_text(zone.get("zone_type"), "zone_type").lower()
    if zone_type not in SUPPORTED_ZONE_TYPES:
        raise ValueError(f"unsupported zone_type: {zone_type}")
    enabled = zone.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError(f"zone {zone_id} enabled must be true or false")

    raw_polygon = zone.get("polygon")
    if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
        raise ValueError(f"zone {zone_id} polygon must contain at least three vertices")
    points: list[tuple[float, float]] = []
    for raw_point in raw_polygon:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
            raise ValueError(f"zone {zone_id} polygon vertices must be [x, y]")
        x, y = raw_point
        if not _is_finite_number(x) or not _is_finite_number(y):
            raise ValueError(f"zone {zone_id} polygon coordinates must be finite numbers")
        if not 0 <= float(x) < width or not 0 <= float(y) < height:
            raise ValueError(f"zone {zone_id} polygon is outside expected resolution")
        points.append((float(x), float(y)))
    polygon = tuple(points)
    if len(set(polygon)) < 3 or math.isclose(_polygon_area(polygon), 0.0):
        raise ValueError(f"zone {zone_id} polygon must have non-zero area")

    return ZoneDefinition(
        zone_id=zone_id,
        zone_name=zone_name,
        zone_type=zone_type,
        enabled=enabled,
        polygon=polygon,
        confirm_inside_seconds=_duration(zone, "confirm_inside_seconds", 0.20),
        confirm_outside_seconds=_duration(zone, "confirm_outside_seconds", 0.20),
        state_ttl_seconds=_duration(zone, "state_ttl_seconds", 2.0),
    )


def load_camera_zones(
    path: Path, camera_id: str, frame_width: int, frame_height: int
) -> CameraZoneConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Zone configuration not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("Zone configuration is not valid YAML") from error
    root = _require_mapping(raw, "zone configuration")
    cameras = _require_mapping(root.get("cameras"), "cameras")
    camera = _require_mapping(cameras.get(camera_id), f"camera {camera_id}")

    resolution = camera.get("expected_resolution")
    if (
        not isinstance(resolution, (list, tuple))
        or len(resolution) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in resolution)
    ):
        raise ValueError("expected_resolution must be [positive width, positive height]")
    expected_width, expected_height = resolution
    if (expected_width, expected_height) != (frame_width, frame_height):
        raise ValueError(
            "Zone resolution does not match input frame: "
            f"expected {expected_width}x{expected_height}, got {frame_width}x{frame_height}"
        )
    fixed_camera = camera.get("fixed_camera")
    if fixed_camera is not True:
        raise ValueError("Polygon zone rules require fixed_camera: true")
    raw_zones = camera.get("zones")
    if not isinstance(raw_zones, list) or not raw_zones:
        raise ValueError(f"camera {camera_id} must define at least one zone")
    seen_ids: set[str] = set()
    zones = tuple(
        _parse_zone(raw_zone, expected_width, expected_height, seen_ids)
        for raw_zone in raw_zones
    )
    return CameraZoneConfig(
        camera_id=camera_id,
        expected_width=expected_width,
        expected_height=expected_height,
        fixed_camera=fixed_camera,
        zones=zones,
    )


class ZoneRuleEngine:
    """Evaluate enabled zones for one fixed camera/session."""

    def __init__(self, camera_id: str, session_id: str, zones: Iterable[ZoneDefinition]):
        if not camera_id.strip() or not session_id.strip():
            raise ValueError("camera_id and session_id must be non-empty")
        self.camera_id = camera_id
        self.session_id = session_id
        self.zones = tuple(zone for zone in zones if zone.enabled)
        self._states: dict[tuple[int, str], _TrackZoneState] = {}
        self._last_timestamp: float | None = None

    def _expire(self, timestamp_seconds: float) -> None:
        zone_by_id = {zone.zone_id: zone for zone in self.zones}
        expired = [
            key
            for key, state in self._states.items()
            if timestamp_seconds - state.last_seen > zone_by_id[key[1]].state_ttl_seconds
        ]
        for key in expired:
            del self._states[key]

    def update(
        self,
        tracks: Iterable[TrackedPerson],
        timestamp_seconds: float,
        frame_number: int,
    ) -> ZoneFrameResult:
        if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if self._last_timestamp is not None and timestamp_seconds < self._last_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        if frame_number < 0:
            raise ValueError("frame_number cannot be negative")
        self._last_timestamp = timestamp_seconds
        self._expire(timestamp_seconds)

        track_list = list(tracks)
        if len({track.track_id for track in track_list}) != len(track_list):
            raise ValueError("active track IDs must be unique within a frame")
        inside_ids = {zone.zone_id: set() for zone in self.zones}
        events: list[BarEntryEvent] = []

        for track in track_list:
            if track.camera_id != self.camera_id or track.session_id != self.session_id:
                raise ValueError("track camera/session does not match rule engine")
            point = (float(track.bottom_center[0]), float(track.bottom_center[1]))
            for zone in self.zones:
                key = (track.track_id, zone.zone_id)
                state = self._states.setdefault(key, _TrackZoneState(last_seen=timestamp_seconds))
                state.last_seen = timestamp_seconds
                raw_inside = point_in_polygon(point, zone.polygon)
                if raw_inside:
                    state.pending_outside_since = None
                    if not state.confirmed_inside:
                        if state.pending_inside_since is None:
                            state.pending_inside_since = timestamp_seconds
                        if (
                            timestamp_seconds - state.pending_inside_since + 1e-9
                            >= zone.confirm_inside_seconds
                        ):
                            state.confirmed_inside = True
                            state.pending_inside_since = None
                            if zone.zone_type == "restricted" and state.confirmed_outside:
                                events.append(
                                    BarEntryEvent(
                                        camera_id=self.camera_id,
                                        session_id=self.session_id,
                                        track_id=track.track_id,
                                        module_id="BAR-001",
                                        event_type="restricted_zone_entry",
                                        zone_id=zone.zone_id,
                                        zone_name=zone.zone_name,
                                        timestamp_seconds=timestamp_seconds,
                                        frame_number=frame_number,
                                        confidence=track.confidence,
                                    )
                                )
                            state.confirmed_outside = False
                    if state.confirmed_inside:
                        inside_ids[zone.zone_id].add(track.track_id)
                else:
                    state.pending_inside_since = None
                    if state.confirmed_inside:
                        if state.pending_outside_since is None:
                            state.pending_outside_since = timestamp_seconds
                        if (
                            timestamp_seconds - state.pending_outside_since + 1e-9
                            >= zone.confirm_outside_seconds
                        ):
                            state.confirmed_inside = False
                            state.confirmed_outside = True
                            state.pending_outside_since = None
                    elif not state.confirmed_outside:
                        if state.pending_outside_since is None:
                            state.pending_outside_since = timestamp_seconds
                        if (
                            timestamp_seconds - state.pending_outside_since + 1e-9
                            >= zone.confirm_outside_seconds
                        ):
                            state.confirmed_outside = True
                            state.pending_outside_since = None
                    if state.confirmed_inside:
                        inside_ids[zone.zone_id].add(track.track_id)

        frozen_ids = {key: frozenset(value) for key, value in inside_ids.items()}
        return ZoneFrameResult(
            counts={key: len(value) for key, value in frozen_ids.items()},
            inside_track_ids=frozen_ids,
            events=tuple(events),
        )
