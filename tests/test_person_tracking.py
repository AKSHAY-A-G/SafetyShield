"""Lightweight integration tests for camera-local ByteTrack state."""

import unittest
from unittest.mock import patch

from src.detection.person_detector import PersonDetection
from src.tracking.person_tracker import ByteTrackConfig, PersonTracker


def detection(x1=10, y1=10, x2=30, y2=50, confidence=0.90):
    return PersonDetection(x1, y1, x2, y2, confidence)


class PersonTrackerTests(unittest.TestCase):
    def test_tracker_initializes_and_empty_frames_do_not_crash(self):
        tracker = PersonTracker("camera-a", "session-a")
        self.assertEqual(
            tracker.update([], (100, 100, 3), frame_index=0, timestamp_seconds=0.0),
            [],
        )

    def test_tracker_initialization_failure_is_safely_reported(self):
        with patch(
            "src.tracking.person_tracker.BYTETracker",
            side_effect=RuntimeError("synthetic initialization failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "ByteTrack initialization failed \\(RuntimeError\\)"
            ):
                PersonTracker("camera-a", "session-a")

    def test_track_structure_and_coordinates_are_valid(self):
        tracker = PersonTracker("camera-a", "session-a")
        tracks = tracker.update(
            [detection(-5, 5, 110, 90)],
            (80, 100, 3),
            frame_index=0,
            timestamp_seconds=0.0,
        )
        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertEqual((track.camera_id, track.session_id), ("camera-a", "session-a"))
        self.assertEqual(track.track_id, 1)
        self.assertEqual((track.x1, track.y1, track.x2, track.y2), (0, 5, 99, 79))
        self.assertEqual(track.bottom_center, (49, 79))

    def test_same_visible_detection_keeps_local_id(self):
        tracker = PersonTracker("camera-a", "session-a")
        first = tracker.update(
            [detection()], (100, 100), frame_index=0, timestamp_seconds=0.0
        )
        second = tracker.update(
            [detection(11, 10, 31, 50)],
            (100, 100),
            frame_index=1,
            timestamp_seconds=1 / 30,
        )
        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_independent_sessions_expose_independent_local_ids(self):
        first_tracker = PersonTracker("camera-a", "session-a")
        second_tracker = PersonTracker("camera-a", "session-b")
        first = first_tracker.update(
            [detection()], (100, 100), frame_index=0, timestamp_seconds=0.0
        )
        second = second_tracker.update(
            [detection()], (100, 100), frame_index=0, timestamp_seconds=0.0
        )
        self.assertEqual(first[0].track_id, 1)
        self.assertEqual(second[0].track_id, 1)
        self.assertNotEqual(first[0].session_id, second[0].session_id)

    def test_invalid_boxes_are_ignored(self):
        tracker = PersonTracker("camera-a", "session-a")
        tracks = tracker.update(
            [detection(20, 20, 20, 50)],
            (100, 100),
            frame_index=0,
            timestamp_seconds=0.0,
        )
        self.assertEqual(tracks, [])

    def test_invalid_tracker_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            ByteTrackConfig(track_low_thresh=0.5, track_high_thresh=0.25)


if __name__ == "__main__":
    unittest.main()
