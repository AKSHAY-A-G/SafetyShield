"""Camera-local temporal rules for configured zones and tracked people."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from src.tracking.person_tracker import TrackedPerson


@dataclass(frozen=True, slots=True)
class BuddyRequiredConfig:
    enabled: bool
    applicable_zone_ids: tuple[str, ...]
    lone_person_confirm_seconds: float
    reset_confirm_seconds: float

    def __post_init__(self) -> None:
        if not self.applicable_zone_ids or len(set(self.applicable_zone_ids)) != len(
            self.applicable_zone_ids
        ):
            raise ValueError("EXC-002 applicable_zone_ids must be non-empty and unique")
        if self.lone_person_confirm_seconds <= 0 or not math.isfinite(
            self.lone_person_confirm_seconds
        ):
            raise ValueError("EXC-002 confirmation must be finite and positive")
        if self.reset_confirm_seconds < 0 or not math.isfinite(
            self.reset_confirm_seconds
        ):
            raise ValueError("EXC-002 reset confirmation must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LowMovementConfig:
    enabled: bool
    applicable_zone_ids: tuple[str, ...]
    no_movement_seconds: float
    movement_ratio: float
    minimum_movement_pixels: float
    state_ttl_seconds: float

    def __post_init__(self) -> None:
        if not self.applicable_zone_ids or len(set(self.applicable_zone_ids)) != len(
            self.applicable_zone_ids
        ):
            raise ValueError("ERG-006 applicable_zone_ids must be non-empty and unique")
        values = (
            self.no_movement_seconds,
            self.movement_ratio,
            self.minimum_movement_pixels,
            self.state_ttl_seconds,
        )
        if any(value <= 0 or not math.isfinite(value) for value in values):
            raise ValueError("ERG-006 thresholds must be finite and positive")


@dataclass(frozen=True, slots=True)
class TemporalRulesConfig:
    buddy_required: BuddyRequiredConfig
    low_movement: LowMovementConfig


@dataclass(frozen=True, slots=True)
class TemporalSafetyEvent:
    camera_id: str
    session_id: str
    track_id: int
    module_id: str
    event_type: str
    zone_id: str | None
    timestamp_seconds: float
    frame_number: int
    current_zone_occupancy: int | None = None
    stationary_duration_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class TemporalFrameResult:
    events: tuple[TemporalSafetyEvent, ...]
    buddy_status: Mapping[str, str]
    buddy_elapsed_seconds: Mapping[str, float]
    low_movement_elapsed_seconds: Mapping[int, float]


@dataclass(slots=True)
class _BuddyState:
    condition_since: float | None = None
    reset_since: float | None = None
    event_emitted: bool = False


@dataclass(slots=True)
class _MovementState:
    anchor_x: float
    anchor_y: float
    anchor_height: float
    stationary_since: float
    last_seen: float
    event_emitted: bool = False


def _mapping(value: object, label: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false")
    return value


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (float(value) <= 0 if positive else float(value) < 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a finite {qualifier} number")
    return float(value)


def _zone_ids(value: object, label: str, available_zone_ids: set[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} entries must be non-empty text")
        zone_id = item.strip()
        if zone_id in result:
            raise ValueError(f"{label} contains duplicate zone_id: {zone_id}")
        if zone_id not in available_zone_ids:
            raise ValueError(f"{label} references unknown zone_id: {zone_id}")
        result.append(zone_id)
    return tuple(result)


def load_temporal_rules(path: Path, available_zone_ids: Iterable[str]) -> TemporalRulesConfig:
    """Load and validate the two Milestone 4 rule configurations."""
    if not path.is_file():
        raise FileNotFoundError(f"Rule configuration not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("Rule configuration is not valid YAML") from error
    root = _mapping(raw, "rule configuration")
    temporal = _mapping(root.get("temporal_rules"), "temporal_rules")
    buddy = _mapping(temporal.get("EXC-002"), "EXC-002")
    movement = _mapping(temporal.get("ERG-006"), "ERG-006")
    available = set(available_zone_ids)

    return TemporalRulesConfig(
        buddy_required=BuddyRequiredConfig(
            enabled=_boolean(buddy.get("enabled"), "EXC-002 enabled"),
            applicable_zone_ids=_zone_ids(
                buddy.get("applicable_zone_ids"),
                "EXC-002 applicable_zone_ids",
                available,
            ),
            lone_person_confirm_seconds=_number(
                buddy.get("lone_person_confirm_seconds"),
                "EXC-002 lone_person_confirm_seconds",
                positive=True,
            ),
            reset_confirm_seconds=_number(
                buddy.get("reset_confirm_seconds"),
                "EXC-002 reset_confirm_seconds",
            ),
        ),
        low_movement=LowMovementConfig(
            enabled=_boolean(movement.get("enabled"), "ERG-006 enabled"),
            applicable_zone_ids=_zone_ids(
                movement.get("applicable_zone_ids"),
                "ERG-006 applicable_zone_ids",
                available,
            ),
            no_movement_seconds=_number(
                movement.get("no_movement_seconds"),
                "ERG-006 no_movement_seconds",
                positive=True,
            ),
            movement_ratio=_number(
                movement.get("movement_ratio"),
                "ERG-006 movement_ratio",
                positive=True,
            ),
            minimum_movement_pixels=_number(
                movement.get("minimum_movement_pixels"),
                "ERG-006 minimum_movement_pixels",
                positive=True,
            ),
            state_ttl_seconds=_number(
                movement.get("state_ttl_seconds"),
                "ERG-006 state_ttl_seconds",
                positive=True,
            ),
        ),
    )


class TemporalRuleEngine:
    """Evaluate EXC-002 and ERG-006 for one camera/session."""

    def __init__(
        self,
        camera_id: str,
        session_id: str,
        config: TemporalRulesConfig,
    ) -> None:
        if not camera_id.strip() or not session_id.strip():
            raise ValueError("camera_id and session_id must be non-empty")
        self.camera_id = camera_id
        self.session_id = session_id
        self.config = config
        self._buddy_states = {
            zone_id: _BuddyState()
            for zone_id in config.buddy_required.applicable_zone_ids
        }
        self._movement_states: dict[int, _MovementState] = {}
        self._last_timestamp: float | None = None

    def _validate_time(self, timestamp_seconds: float, frame_number: int) -> None:
        if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if self._last_timestamp is not None and timestamp_seconds < self._last_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        if frame_number < 0:
            raise ValueError("frame_number cannot be negative")
        self._last_timestamp = timestamp_seconds

    def _update_buddy(
        self,
        counts: Mapping[str, int],
        inside_track_ids: Mapping[str, frozenset[int]],
        timestamp_seconds: float,
        frame_number: int,
    ) -> tuple[list[TemporalSafetyEvent], dict[str, str], dict[str, float]]:
        events: list[TemporalSafetyEvent] = []
        statuses: dict[str, str] = {}
        elapsed_values: dict[str, float] = {}
        config = self.config.buddy_required
        if not config.enabled:
            return events, statuses, elapsed_values

        for zone_id, state in self._buddy_states.items():
            if zone_id not in counts or zone_id not in inside_track_ids:
                raise ValueError(f"zone result is missing configured zone_id: {zone_id}")
            occupancy = counts[zone_id]
            if occupancy < 0:
                raise ValueError("zone occupancy cannot be negative")
            if occupancy == 1:
                state.reset_since = None
                if state.condition_since is None:
                    state.condition_since = timestamp_seconds
                elapsed = timestamp_seconds - state.condition_since
                elapsed_values[zone_id] = elapsed
                if not state.event_emitted and elapsed + 1e-9 >= config.lone_person_confirm_seconds:
                    track_id = next(iter(inside_track_ids[zone_id]))
                    events.append(
                        TemporalSafetyEvent(
                            camera_id=self.camera_id,
                            session_id=self.session_id,
                            track_id=track_id,
                            module_id="EXC-002",
                            event_type="buddy_required_single_person",
                            zone_id=zone_id,
                            timestamp_seconds=timestamp_seconds,
                            frame_number=frame_number,
                            current_zone_occupancy=occupancy,
                        )
                    )
                    state.event_emitted = True
                statuses[zone_id] = "active" if state.event_emitted else "confirming"
            else:
                if state.condition_since is None:
                    statuses[zone_id] = "clear"
                    elapsed_values[zone_id] = 0.0
                    continue
                if state.reset_since is None:
                    state.reset_since = timestamp_seconds
                reset_elapsed = timestamp_seconds - state.reset_since
                if reset_elapsed + 1e-9 >= config.reset_confirm_seconds:
                    state.condition_since = None
                    state.reset_since = None
                    state.event_emitted = False
                    statuses[zone_id] = "clear"
                    elapsed_values[zone_id] = 0.0
                else:
                    statuses[zone_id] = "resetting"
                    elapsed_values[zone_id] = max(
                        0.0, timestamp_seconds - state.condition_since
                    )
        return events, statuses, elapsed_values

    def _update_movement(
        self,
        tracks: list[TrackedPerson],
        inside_track_ids: Mapping[str, frozenset[int]],
        timestamp_seconds: float,
        frame_number: int,
    ) -> tuple[list[TemporalSafetyEvent], dict[int, float]]:
        events: list[TemporalSafetyEvent] = []
        elapsed_values: dict[int, float] = {}
        config = self.config.low_movement
        if not config.enabled:
            return events, elapsed_values

        expired = [
            track_id
            for track_id, state in self._movement_states.items()
            if timestamp_seconds - state.last_seen > config.state_ttl_seconds
        ]
        for track_id in expired:
            del self._movement_states[track_id]

        eligible_zone_by_track: dict[int, str] = {}
        for zone_id in config.applicable_zone_ids:
            if zone_id not in inside_track_ids:
                raise ValueError(f"zone result is missing configured zone_id: {zone_id}")
            for track_id in inside_track_ids[zone_id]:
                eligible_zone_by_track.setdefault(track_id, zone_id)

        for track in tracks:
            if track.camera_id != self.camera_id or track.session_id != self.session_id:
                raise ValueError("track camera/session does not match temporal rule engine")
            zone_id = eligible_zone_by_track.get(track.track_id)
            if zone_id is None:
                continue
            x, y = map(float, track.bottom_center)
            height = float(track.height)
            state = self._movement_states.get(track.track_id)
            if state is None:
                state = _MovementState(x, y, height, timestamp_seconds, timestamp_seconds)
                self._movement_states[track.track_id] = state
            else:
                threshold = max(
                    config.minimum_movement_pixels,
                    config.movement_ratio * max(state.anchor_height, height),
                )
                displacement = math.hypot(x - state.anchor_x, y - state.anchor_y)
                if displacement > threshold:
                    state.anchor_x = x
                    state.anchor_y = y
                    state.anchor_height = height
                    state.stationary_since = timestamp_seconds
                    state.event_emitted = False
                state.last_seen = timestamp_seconds

            elapsed = timestamp_seconds - state.stationary_since
            elapsed_values[track.track_id] = elapsed
            if not state.event_emitted and elapsed + 1e-9 >= config.no_movement_seconds:
                events.append(
                    TemporalSafetyEvent(
                        camera_id=self.camera_id,
                        session_id=self.session_id,
                        track_id=track.track_id,
                        module_id="ERG-006",
                        event_type="prolonged_low_movement",
                        zone_id=zone_id,
                        timestamp_seconds=timestamp_seconds,
                        frame_number=frame_number,
                        stationary_duration_seconds=elapsed,
                    )
                )
                state.event_emitted = True
        return events, elapsed_values

    def update(
        self,
        tracks: Iterable[TrackedPerson],
        counts: Mapping[str, int],
        inside_track_ids: Mapping[str, frozenset[int]],
        timestamp_seconds: float,
        frame_number: int,
    ) -> TemporalFrameResult:
        self._validate_time(timestamp_seconds, frame_number)
        track_list = list(tracks)
        if len({track.track_id for track in track_list}) != len(track_list):
            raise ValueError("active track IDs must be unique within a frame")
        buddy_events, statuses, buddy_elapsed = self._update_buddy(
            counts, inside_track_ids, timestamp_seconds, frame_number
        )
        movement_events, movement_elapsed = self._update_movement(
            track_list, inside_track_ids, timestamp_seconds, frame_number
        )
        return TemporalFrameResult(
            events=tuple(buddy_events + movement_events),
            buddy_status=statuses,
            buddy_elapsed_seconds=buddy_elapsed,
            low_movement_elapsed_seconds=movement_elapsed,
        )
