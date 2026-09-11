"""Tests for EXC-002 and ERG-006 temporal state semantics."""

from pathlib import Path
import tempfile
import unittest

from src.rules.temporal import (
    BuddyRequiredConfig,
    LowMovementConfig,
    TemporalRuleEngine,
    TemporalRulesConfig,
    load_temporal_rules,
)
from src.tracking.person_tracker import TrackedPerson


ZONE_A = "zone-a"
ZONE_B = "zone-b"


def config(
    *,
    buddy_confirm: float = 1.0,
    buddy_reset: float = 0.5,
    no_movement: float = 2.0,
    ttl: float = 10.0,
) -> TemporalRulesConfig:
    return TemporalRulesConfig(
        buddy_required=BuddyRequiredConfig(True, (ZONE_A, ZONE_B), buddy_confirm, buddy_reset),
        low_movement=LowMovementConfig(True, (ZONE_A,), no_movement, 0.1, 5.0, ttl),
    )


def track(
    track_id: int = 1,
    x: int = 50,
    y: int = 80,
    *,
    height: int = 100,
    camera_id: str = "camera-a",
    session_id: str = "session-a",
) -> TrackedPerson:
    return TrackedPerson(
        camera_id=camera_id,
        session_id=session_id,
        track_id=track_id,
        frame_index=0,
        timestamp_seconds=0.0,
        x1=x - 10,
        y1=y - height,
        x2=x + 10,
        y2=y,
        confidence=0.8,
    )


def zone_state(zone_a_ids=(), zone_b_ids=()):
    inside = {ZONE_A: frozenset(zone_a_ids), ZONE_B: frozenset(zone_b_ids)}
    return {key: len(value) for key, value in inside.items()}, inside


class BuddyRequiredTests(unittest.TestCase):
    @staticmethod
    def buddy_config(**kwargs):
        return config(no_movement=999.0, **kwargs)

    def update(self, engine, timestamp, zone_a_ids=(), zone_b_ids=()):
        counts, inside = zone_state(zone_a_ids, zone_b_ids)
        tracks = [track(value) for value in set(zone_a_ids) | set(zone_b_ids)]
        return engine.update(tracks, counts, inside, timestamp, round(timestamp * 10))

    def test_zero_occupancy_does_not_trigger(self):
        result = self.update(
            TemporalRuleEngine("camera-a", "session-a", self.buddy_config()), 0.0
        )
        self.assertEqual(result.events, ())
        self.assertEqual(result.buddy_status[ZONE_A], "clear")

    def test_one_person_starts_confirmation_but_does_not_trigger_early(self):
        engine = TemporalRuleEngine("camera-a", "session-a", self.buddy_config())
        first = self.update(engine, 0.0, (1,))
        early = self.update(engine, 0.9, (1,))
        self.assertEqual(first.buddy_status[ZONE_A], "confirming")
        self.assertEqual(early.events, ())

    def test_sustained_single_occupancy_emits_once(self):
        engine = TemporalRuleEngine("camera-a", "session-a", self.buddy_config())
        self.update(engine, 0.0, (1,))
        event = self.update(engine, 1.0, (1,))
        later = self.update(engine, 2.0, (1,))
        self.assertEqual([item.module_id for item in event.events], ["EXC-002"])
        self.assertEqual(event.events[0].current_zone_occupancy, 1)
        self.assertEqual(later.events, ())

    def test_two_people_clear_then_allow_new_episode(self):
        engine = TemporalRuleEngine("camera-a", "session-a", self.buddy_config())
        self.update(engine, 0.0, (1,))
        self.update(engine, 1.0, (1,))
        self.update(engine, 1.1, (1, 2))
        cleared = self.update(engine, 1.6, (1, 2))
        self.assertEqual(cleared.buddy_status[ZONE_A], "clear")
        self.update(engine, 2.0, (1,))
        new_event = self.update(engine, 3.0, (1,))
        self.assertEqual([item.module_id for item in new_event.events], ["EXC-002"])

    def test_zero_people_clear_episode(self):
        engine = TemporalRuleEngine("camera-a", "session-a", self.buddy_config())
        self.update(engine, 0.0, (1,))
        self.update(engine, 1.0, (1,))
        self.update(engine, 1.1)
        cleared = self.update(engine, 1.6)
        self.assertEqual(cleared.buddy_status[ZONE_A], "clear")

    def test_short_reset_does_not_split_episode(self):
        engine = TemporalRuleEngine("camera-a", "session-a", self.buddy_config())
        self.update(engine, 0.0, (1,))
        self.update(engine, 1.0, (1,))
        self.update(engine, 1.1)
        resumed = self.update(engine, 1.4, (1,))
        self.assertEqual(resumed.events, ())
        self.assertEqual(resumed.buddy_status[ZONE_A], "active")

    def test_independent_zones_do_not_share_state(self):
        engine = TemporalRuleEngine("camera-a", "session-a", self.buddy_config())
        self.update(engine, 0.0, (1,))
        result = self.update(engine, 1.0, (1,), (2,))
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].zone_id, ZONE_A)
        self.assertEqual(result.buddy_status[ZONE_B], "confirming")

    def test_separate_camera_session_engines_are_isolated(self):
        first = TemporalRuleEngine("camera-a", "session-a", self.buddy_config())
        second = TemporalRuleEngine("camera-a", "session-b", self.buddy_config())
        self.update(first, 0.0, (1,))
        first_event = self.update(first, 1.0, (1,))
        counts, inside = zone_state((1,), ())
        second_result = second.update(
            [track(session_id="session-b")], counts, inside, 1.0, 10
        )
        self.assertEqual(len(first_event.events), 1)
        self.assertEqual(second_result.events, ())


class LowMovementTests(unittest.TestCase):
    @staticmethod
    def movement_config(**kwargs):
        return config(buddy_confirm=999.0, **kwargs)

    def update(self, engine, observation, timestamp, inside=True):
        ids = (observation.track_id,) if observation is not None and inside else ()
        counts, inside_ids = zone_state(ids, ())
        tracks = [] if observation is None else [observation]
        return engine.update(tracks, counts, inside_ids, timestamp, round(timestamp * 10))

    def test_small_jitter_does_not_reset_timer_and_event_waits_for_duration(self):
        engine = TemporalRuleEngine("camera-a", "session-a", self.movement_config())
        self.update(engine, track(x=50), 0.0)
        before = self.update(engine, track(x=58), 1.9)
        event = self.update(engine, track(x=51), 2.0)
        self.assertEqual(before.events, ())
        self.assertEqual(event.events[0].module_id, "ERG-006")
        self.assertAlmostEqual(event.events[0].stationary_duration_seconds, 2.0)

    def test_meaningful_displacement_resets_timer(self):
        engine = TemporalRuleEngine("camera-a", "session-a", self.movement_config())
        self.update(engine, track(x=50), 0.0)
        moved = self.update(engine, track(x=61), 1.5)
        result = self.update(engine, track(x=61), 2.0)
        self.assertEqual(moved.low_movement_elapsed_seconds[1], 0.0)
        self.assertEqual(result.events, ())

    def test_stationary_event_is_deduplicated(self):
        engine = TemporalRuleEngine("camera-a", "session-a", self.movement_config())
        self.update(engine, track(), 0.0)
        event = self.update(engine, track(), 2.0)
        later = self.update(engine, track(), 3.0)
        self.assertEqual([item.module_id for item in event.events], ["ERG-006"])
        self.assertEqual(later.events, ())

    def test_movement_after_event_allows_later_stationary_event(self):
        engine = TemporalRuleEngine("camera-a", "session-a", self.movement_config())
        self.update(engine, track(x=50), 0.0)
        self.update(engine, track(x=50), 2.0)
        self.update(engine, track(x=70), 2.5)
        second = self.update(engine, track(x=70), 4.5)
        self.assertEqual([item.module_id for item in second.events], ["ERG-006"])

    def test_expired_track_history_starts_fresh(self):
        engine = TemporalRuleEngine(
            "camera-a", "session-a", self.movement_config(ttl=1.0)
        )
        self.update(engine, track(), 0.0)
        self.update(engine, None, 1.1)
        fresh = self.update(engine, track(), 2.0)
        self.assertEqual(fresh.low_movement_elapsed_seconds[1], 0.0)
        self.assertEqual(fresh.events, ())

    def test_new_track_id_starts_fresh(self):
        engine = TemporalRuleEngine("camera-a", "session-a", self.movement_config())
        self.update(engine, track(1), 0.0)
        fresh = self.update(engine, track(2), 2.0)
        self.assertEqual(fresh.low_movement_elapsed_seconds[2], 0.0)

    def test_empty_input_does_not_crash(self):
        result = self.update(
            TemporalRuleEngine("camera-a", "session-a", self.movement_config()),
            None,
            0.0,
        )
        self.assertEqual(result.low_movement_elapsed_seconds, {})

    def test_mismatched_camera_session_is_rejected(self):
        engine = TemporalRuleEngine("camera-a", "session-a", self.movement_config())
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.update(engine, track(camera_id="camera-b"), 0.0)

    def test_separate_camera_session_movement_histories_are_isolated(self):
        first = TemporalRuleEngine("camera-a", "session-a", self.movement_config())
        second = TemporalRuleEngine("camera-a", "session-b", self.movement_config())
        self.update(first, track(), 0.0)
        first_event = self.update(first, track(), 2.0)
        counts, inside = zone_state((1,), ())
        second_result = second.update(
            [track(session_id="session-b")], counts, inside, 2.0, 20
        )
        self.assertEqual([item.module_id for item in first_event.events], ["ERG-006"])
        self.assertEqual(second_result.events, ())


class TemporalConfigTests(unittest.TestCase):
    def test_runtime_configuration_preserves_600_second_default(self):
        loaded = load_temporal_rules(Path("config/rules.yaml"), {"restricted_zone_1"})
        self.assertEqual(loaded.low_movement.no_movement_seconds, 600.0)

    def test_unknown_zone_reference_fails_clearly(self):
        content = Path("config/rules.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown zone_id"):
                load_temporal_rules(path, {"another-zone"})

    def test_invalid_threshold_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            LowMovementConfig(True, (ZONE_A,), -1.0, 0.1, 5.0, 1.0)
        content = Path("config/rules.yaml").read_text(encoding="utf-8").replace(
            "no_movement_seconds: 600.0", "no_movement_seconds: -1.0"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive"):
                load_temporal_rules(path, {"restricted_zone_1"})


if __name__ == "__main__":
    unittest.main()
