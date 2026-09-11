"""Unit tests for fixed-camera polygon geometry and zone rule state."""

from pathlib import Path
import tempfile
import unittest

from src.rules.zones import ZoneDefinition, ZoneRuleEngine, load_camera_zones, point_in_polygon
from src.tracking.person_tracker import TrackedPerson


SQUARE = ((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0))


def zone(
    zone_id: str = "restricted-a",
    zone_type: str = "restricted",
    enabled: bool = True,
    confirm_seconds: float = 0.0,
) -> ZoneDefinition:
    return ZoneDefinition(
        zone_id=zone_id,
        zone_name="Restricted A",
        zone_type=zone_type,
        enabled=enabled,
        polygon=SQUARE,
        confirm_inside_seconds=confirm_seconds,
        confirm_outside_seconds=confirm_seconds,
        state_ttl_seconds=2.0,
    )


def track(
    track_id: int,
    bottom_x: int,
    bottom_y: int,
    camera_id: str = "camera-a",
    session_id: str = "session-a",
) -> TrackedPerson:
    return TrackedPerson(
        camera_id=camera_id,
        session_id=session_id,
        track_id=track_id,
        frame_index=0,
        timestamp_seconds=0.0,
        x1=bottom_x - 10,
        y1=bottom_y - 40,
        x2=bottom_x + 10,
        y2=bottom_y,
        confidence=0.8,
    )


class ZoneGeometryTests(unittest.TestCase):
    def test_point_clearly_inside_polygon(self):
        self.assertTrue(point_in_polygon((50.0, 50.0), SQUARE))

    def test_point_clearly_outside_polygon(self):
        self.assertFalse(point_in_polygon((5.0, 50.0), SQUARE))

    def test_boundary_is_deterministically_inside(self):
        self.assertTrue(point_in_polygon((10.0, 50.0), SQUARE))
        self.assertTrue(point_in_polygon((10.0, 10.0), SQUARE))

    def test_bottom_centre_calculation(self):
        self.assertEqual(track(1, 50, 80).bottom_center, (50, 80))

    def test_malformed_polygon_config_fails_clearly(self):
        content = """cameras:
  camera-a:
    expected_resolution: [100, 100]
    fixed_camera: true
    zones:
      - zone_id: broken
        zone_name: Broken
        zone_type: restricted
        enabled: true
        polygon: [[1, 1], [2, 2]]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least three vertices"):
                load_camera_zones(path, "camera-a", 100, 100)

    def test_wrong_resolution_fails_clearly(self):
        content = """cameras:
  camera-a:
    expected_resolution: [100, 100]
    fixed_camera: true
    zones:
      - zone_id: valid
        zone_name: Valid
        zone_type: counting
        enabled: true
        polygon: [[10, 10], [90, 10], [10, 90]]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match input frame"):
                load_camera_zones(path, "camera-a", 200, 100)


class ZoneRuleTests(unittest.TestCase):
    def test_disabled_zone_is_ignored(self):
        engine = ZoneRuleEngine("camera-a", "session-a", [zone(enabled=False)])
        result = engine.update([track(1, 50, 50)], 0.0, 0)
        self.assertEqual(result.counts, {})
        self.assertEqual(result.events, ())

    def test_one_entry_transition_and_remaining_inside_does_not_repeat(self):
        engine = ZoneRuleEngine("camera-a", "session-a", [zone(confirm_seconds=0.1)])
        self.assertEqual(engine.update([track(1, 5, 50)], 0.0, 0).events, ())
        self.assertEqual(engine.update([track(1, 5, 50)], 0.1, 3).events, ())
        self.assertEqual(engine.update([track(1, 50, 50)], 0.2, 6).events, ())
        entered = engine.update([track(1, 50, 50)], 0.3, 9)
        self.assertEqual(len(entered.events), 1)
        self.assertEqual(entered.events[0].module_id, "BAR-001")
        self.assertEqual(engine.update([track(1, 50, 50)], 0.4, 12).events, ())

    def test_unconfirmed_initial_outside_does_not_enable_entry_event(self):
        engine = ZoneRuleEngine("camera-a", "session-a", [zone(confirm_seconds=0.2)])
        engine.update([track(1, 5, 50)], 0.0, 0)
        engine.update([track(1, 50, 50)], 0.1, 3)
        result = engine.update([track(1, 50, 50)], 0.3, 9)
        self.assertEqual(result.counts["restricted-a"], 1)
        self.assertEqual(result.events, ())

    def test_confirmed_exit_then_reentry_produces_another_event(self):
        engine = ZoneRuleEngine("camera-a", "session-a", [zone(confirm_seconds=0.1)])
        # Track first appears inside: confirmed inside but NO entry event
        engine.update([track(1, 50, 50)], 0.0, 0)
        first_confirmed = engine.update([track(1, 50, 50)], 0.1, 3)
        self.assertEqual(len(first_confirmed.events), 0)
        self.assertEqual(first_confirmed.counts["restricted-a"], 1)
        # Track exits outside and confirms exit
        engine.update([track(1, 5, 50)], 0.2, 6)
        exit_confirmed = engine.update([track(1, 5, 50)], 0.3, 9)
        self.assertEqual(len(exit_confirmed.events), 0)
        self.assertEqual(exit_confirmed.counts["restricted-a"], 0)
        # Track re-enters inside and confirms re-entry: emits genuine entry event
        engine.update([track(1, 50, 50)], 0.4, 12)
        reentry = engine.update([track(1, 50, 50)], 0.5, 15)
        self.assertEqual(len(reentry.events), 1)
        self.assertEqual(reentry.events[0].module_id, "BAR-001")
        self.assertEqual(reentry.counts["restricted-a"], 1)

    def test_outside_entry_exit_and_reentry_produces_two_events(self):
        engine = ZoneRuleEngine("camera-a", "session-a", [zone(confirm_seconds=0.1)])
        # Start outside and complete outside confirmation
        self.assertEqual(engine.update([track(1, 5, 50)], 0.0, 0).events, ())
        self.assertEqual(engine.update([track(1, 5, 50)], 0.1, 3).events, ())
        # Move inside and confirm: first entry event
        engine.update([track(1, 50, 50)], 0.2, 6)
        first_entry = engine.update([track(1, 50, 50)], 0.3, 9)
        self.assertEqual(len(first_entry.events), 1)
        self.assertEqual(first_entry.counts["restricted-a"], 1)
        # Move outside and confirm exit
        engine.update([track(1, 5, 50)], 0.4, 12)
        exit_confirmed = engine.update([track(1, 5, 50)], 0.5, 15)
        self.assertEqual(len(exit_confirmed.events), 0)
        self.assertEqual(exit_confirmed.counts["restricted-a"], 0)
        # Re-enter inside and confirm: second entry event
        engine.update([track(1, 50, 50)], 0.6, 18)
        reentry = engine.update([track(1, 50, 50)], 0.7, 21)
        self.assertEqual(len(reentry.events), 1)
        self.assertEqual(reentry.counts["restricted-a"], 1)

    def test_independent_track_ids_have_independent_state(self):
        engine = ZoneRuleEngine("camera-a", "session-a", [zone()])
        engine.update([track(1, 5, 50), track(2, 5, 60)], 0.0, 0)
        result = engine.update([track(1, 50, 50), track(2, 60, 60)], 0.1, 3)
        self.assertEqual({event.track_id for event in result.events}, {1, 2})
        self.assertEqual(result.counts["restricted-a"], 2)

    def test_independent_camera_sessions_do_not_share_state(self):
        first = ZoneRuleEngine("camera-a", "session-a", [zone()])
        second = ZoneRuleEngine("camera-a", "session-b", [zone()])
        first.update([track(1, 5, 50)], 0.0, 0)
        self.assertEqual(len(first.update([track(1, 50, 50)], 0.1, 3).events), 1)
        second_track = track(1, 50, 50, session_id="session-b")
        second.update([track(1, 5, 50, session_id="session-b")], 0.0, 0)
        self.assertEqual(len(second.update([second_track], 0.1, 3).events), 1)

    def test_first_observation_inside_counts_but_is_not_an_entry_transition(self):
        engine = ZoneRuleEngine("camera-a", "session-a", [zone()])
        result = engine.update([track(1, 50, 50)], 0.0, 0)
        self.assertEqual(result.counts["restricted-a"], 1)
        self.assertEqual(result.events, ())

    def test_first_observation_inside_requires_confirmation_for_occupancy_without_entry_event(self):
        engine = ZoneRuleEngine("camera-a", "session-a", [zone(confirm_seconds=0.1)])
        # Pending inside (0.0s elapsed): not yet confirmed occupant
        pending = engine.update([track(1, 50, 50)], 0.0, 0)
        self.assertEqual(pending.counts["restricted-a"], 0)
        self.assertEqual(pending.events, ())
        # Confirmed inside (0.1s elapsed): confirmed occupant, but no entry event
        confirmed = engine.update([track(1, 50, 50)], 0.1, 3)
        self.assertEqual(confirmed.counts["restricted-a"], 1)
        self.assertEqual(confirmed.events, ())

    def test_zone_count_is_current_camera_local_active_tracks(self):
        engine = ZoneRuleEngine("camera-a", "session-a", [zone(zone_type="counting")])
        result = engine.update(
            [track(1, 50, 50), track(2, 5, 50), track(3, 70, 80)], 0.0, 0
        )
        self.assertEqual(result.counts["restricted-a"], 2)
        self.assertEqual(result.events, ())

    def test_empty_tracks_returns_zero_count(self):
        engine = ZoneRuleEngine("camera-a", "session-a", [zone()])
        result = engine.update([], 0.0, 0)
        self.assertEqual(result.counts["restricted-a"], 0)
        self.assertEqual(result.events, ())

    def test_track_from_another_context_is_rejected(self):
        engine = ZoneRuleEngine("camera-a", "session-a", [zone()])
        with self.assertRaisesRegex(ValueError, "does not match"):
            engine.update([track(1, 50, 50, camera_id="camera-b")], 0.0, 0)


if __name__ == "__main__":
    unittest.main()
