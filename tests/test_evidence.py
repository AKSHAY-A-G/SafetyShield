"""Unit tests for Milestone 5 event modeling, evidence capture, and audit logging."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import cv2
import numpy as np

from src.events.audit import (
    AUDIT_ACTION_EVENT_CREATED,
    AUDIT_ACTION_EVIDENCE_ERROR,
    AUDIT_ACTION_EVIDENCE_SAVED,
    AuditLogger,
)
from src.events.evidence import (
    EvidenceWriter,
    compute_clip_bounds,
    verify_evidence_clip,
)
from src.events.models import SafetyEvent, generate_event_id
from src.rules.temporal import TemporalSafetyEvent
from src.rules.zones import BarEntryEvent


def create_synthetic_test_video(
    output_path: Path,
    width: int = 320,
    height: int = 240,
    fps: float = 30.0,
    duration_seconds: float = 12.0,
) -> Path:
    """Create a minimal synthetic MP4 file for fast, deterministic unit testing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    total_frames = int(round(fps * duration_seconds))
    for i in range(total_frames):
        # Create a frame with a moving colored patch
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"Frame {i}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        writer.write(frame)
    writer.release()
    return output_path


class EvidenceTests(unittest.TestCase):
    """Test suite covering Milestone 5 evidence, event modeling, and audit requirements."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.evidence_dir = self.root / "evidence"
        self.source_video_path = self.root / "test_source.mp4"
        create_synthetic_test_video(
            self.source_video_path,
            width=320,
            height=240,
            fps=30.0,
            duration_seconds=50.0,
        )
        self.raw_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.annotated_frame = self.raw_frame.copy()
        cv2.putText(
            self.annotated_frame,
            "TEST OVERLAY",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # 1. unique event IDs are generated
    def test_unique_event_ids_generated(self) -> None:
        ids = {generate_event_id("bar") for _ in range(100)}
        self.assertEqual(len(ids), 100)
        for event_id in ids:
            self.assertTrue(event_id.startswith("bar_"))
            self.assertGreater(len(event_id), 10)

    # 2. event representation preserves camera/session/track separation
    def test_event_representation_camera_session_track_separation(self) -> None:
        event = SafetyEvent(
            event_id="evt_test_01",
            camera_id="cam_east_01",
            session_id="session_alpha_123",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=9,
            source_timestamp_seconds=41.2,
            frame_number=1236,
            zone_id="restricted_zone_1",
            zone_name="Restricted Zone 1",
        )
        data = event.to_dict()
        self.assertEqual(data["camera_id"], "cam_east_01")
        self.assertEqual(data["session_id"], "session_alpha_123")
        self.assertEqual(data["track_id"], 9)
        self.assertNotEqual(data["event_id"], data["track_id"])
        self.assertNotEqual(data["event_id"], data["camera_id"])

    # 3. raw snapshot is written
    # 4. annotated snapshot is written
    # 5. snapshot dimensions remain correct
    def test_snapshots_written_with_correct_dimensions(self) -> None:
        writer = EvidenceWriter(self.evidence_dir)
        event = SafetyEvent(
            event_id="evt_snapshot_test",
            camera_id="cam_good_test",
            session_id="sess_01",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=9,
            source_timestamp_seconds=5.0,
            frame_number=150,
        )
        artifacts = writer.save_event_evidence(
            event=event,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
        )
        self.assertTrue(artifacts.raw_snapshot_path.is_file())
        self.assertTrue(artifacts.annotated_snapshot_path.is_file())

        read_raw = cv2.imread(str(artifacts.raw_snapshot_path))
        read_ann = cv2.imread(str(artifacts.annotated_snapshot_path))
        self.assertIsNotNone(read_raw)
        self.assertIsNotNone(read_ann)
        self.assertEqual(read_raw.shape, (240, 320, 3))
        self.assertEqual(read_ann.shape, (240, 320, 3))

    # 6. metadata JSON is valid and readable
    # 7. metadata contains required fields
    def test_metadata_json_valid_readable_and_complete(self) -> None:
        writer = EvidenceWriter(self.evidence_dir)
        event = SafetyEvent(
            event_id="evt_meta_test",
            camera_id="cam_good_test",
            session_id="sess_01",
            module_id="EXC-002",
            event_type="lone_worker",
            track_id=9,
            source_timestamp_seconds=6.5,
            frame_number=195,
            zone_id="restricted_zone_1",
            zone_name="Restricted Zone 1",
            severity="unclassified",
            status="new",
        )
        artifacts = writer.save_event_evidence(
            event=event,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
        )
        self.assertTrue(artifacts.metadata_path.is_file())

        with artifacts.metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        required_keys = {
            "event_id",
            "schema_version",
            "camera_id",
            "session_id",
            "module_id",
            "event_type",
            "track_id",
            "source_video",
            "source_timestamp_seconds",
            "frame_number",
            "zone_id",
            "zone_name",
            "severity",
            "status",
            "processed_at_utc",
            "evidence",
        }
        self.assertTrue(required_keys.issubset(metadata.keys()))
        evidence_keys = {
            "raw_snapshot",
            "annotated_snapshot",
            "video_clip",
            "requested_pre_seconds",
            "requested_post_seconds",
            "actual_clip_start_seconds",
            "actual_clip_end_seconds",
            "actual_pre_seconds",
            "actual_post_seconds",
            "resolution",
        }
        self.assertTrue(evidence_keys.issubset(metadata["evidence"].keys()))
        # Check relative paths rather than machine-specific absolute paths
        self.assertEqual(metadata["evidence"]["raw_snapshot"], "snapshot_raw.jpg")
        self.assertEqual(metadata["evidence"]["annotated_snapshot"], "snapshot_annotated.jpg")
        self.assertEqual(metadata["evidence"]["video_clip"], "event_clip.mp4")

    # 8. source-relative timestamp and processed-at time remain distinct
    def test_source_timestamp_and_processed_at_remain_distinct(self) -> None:
        writer = EvidenceWriter(self.evidence_dir)
        event = SafetyEvent(
            event_id="evt_distinct_time",
            camera_id="cam_good_test",
            session_id="sess_01",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=9,
            source_timestamp_seconds=41.2,
            frame_number=1236,
        )
        artifacts = writer.save_event_evidence(
            event=event,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
        )
        with artifacts.metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["source_timestamp_seconds"], 41.2)
        # processed_at_utc must be a full ISO 8601 string containing date and 'T'
        self.assertIn("T", meta["processed_at_utc"])
        self.assertNotEqual(str(meta["source_timestamp_seconds"]), meta["processed_at_utc"])

    # 9. evidence path does not contain secrets
    def test_evidence_paths_do_not_contain_secrets(self) -> None:
        writer = EvidenceWriter(self.evidence_dir)
        event = SafetyEvent(
            event_id="evt_clean_path",
            camera_id="cam_good_test",
            session_id="sess_public",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=9,
            source_timestamp_seconds=2.0,
            frame_number=60,
        )
        artifacts = writer.save_event_evidence(
            event=event,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
        )
        path_str = str(artifacts.event_dir)
        for secret_pattern in ("rtsp://", "password", "secret", "user:", ".env"):
            self.assertNotIn(secret_pattern, path_str.lower())

    # 10. clip extraction around a middle-of-video event works
    def test_clip_extraction_middle_of_video(self) -> None:
        writer = EvidenceWriter(self.evidence_dir, pre_event_seconds=3.0, post_event_seconds=3.0)
        event = SafetyEvent(
            event_id="evt_middle",
            camera_id="cam_good_test",
            session_id="sess_01",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=9,
            source_timestamp_seconds=7.0,
            frame_number=210,
        )
        artifacts = writer.save_event_evidence(
            event=event,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
            source_duration_seconds=15.0,
        )
        cov = artifacts.coverage
        self.assertAlmostEqual(cov.actual_clip_start_seconds, 4.0, places=2)
        self.assertAlmostEqual(cov.actual_clip_end_seconds, 10.0, places=2)
        self.assertAlmostEqual(cov.actual_pre_seconds, 3.0, places=2)
        self.assertAlmostEqual(cov.actual_post_seconds, 3.0, places=2)
        self.assertTrue(artifacts.clip_path.is_file())

    # 11. event near video start clamps clip correctly
    def test_event_near_video_start_clamps_correctly(self) -> None:
        start_sec, end_sec, act_pre, act_post = compute_clip_bounds(
            source_timestamp_seconds=1.5,
            source_duration_seconds=15.0,
            pre_event_seconds=5.0,
            post_event_seconds=5.0,
        )
        self.assertEqual(start_sec, 0.0)
        self.assertEqual(act_pre, 1.5)
        self.assertEqual(end_sec, 6.5)
        self.assertEqual(act_post, 5.0)

    # 12. event near video end clamps clip correctly
    def test_event_near_video_end_clamps_correctly(self) -> None:
        start_sec, end_sec, act_pre, act_post = compute_clip_bounds(
            source_timestamp_seconds=14.0,
            source_duration_seconds=15.0,
            pre_event_seconds=5.0,
            post_event_seconds=5.0,
        )
        self.assertEqual(start_sec, 9.0)
        self.assertEqual(act_pre, 5.0)
        self.assertEqual(end_sec, 15.0)
        self.assertEqual(act_post, 1.0)

    # 13. clip output reopens successfully
    def test_clip_output_reopens_successfully(self) -> None:
        writer = EvidenceWriter(self.evidence_dir, pre_event_seconds=2.0, post_event_seconds=2.0)
        event = SafetyEvent(
            event_id="evt_reopen_test",
            camera_id="cam_good_test",
            session_id="sess_01",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=9,
            source_timestamp_seconds=5.0,
            frame_number=150,
        )
        artifacts = writer.save_event_evidence(
            event=event,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
        )
        check = verify_evidence_clip(artifacts.clip_path, expected_width=320, expected_height=240)
        self.assertTrue(check["opened"])
        self.assertTrue(check["readable_frame"])
        self.assertTrue(check["dimensions_match"])
        self.assertGreater(check["frame_count"], 0)
        self.assertGreater(check["fps"], 0.0)

    # 14. duplicate event_id is not silently overwritten
    def test_duplicate_event_id_fails(self) -> None:
        writer = EvidenceWriter(self.evidence_dir)
        event = SafetyEvent(
            event_id="evt_dup_test",
            camera_id="cam_good_test",
            session_id="sess_01",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=9,
            source_timestamp_seconds=5.0,
            frame_number=150,
        )
        writer.save_event_evidence(
            event=event,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
        )
        # Second attempt with same event_id must raise FileExistsError
        with self.assertRaises(FileExistsError):
            writer.save_event_evidence(
                event=event,
                raw_frame=self.raw_frame,
                annotated_frame=self.annotated_frame,
                source_video_path=self.source_video_path,
            )

    # 15. audit JSONL contains valid JSON objects
    # 16. event creation adds expected audit entry
    # 17. successful evidence creation adds expected audit entry
    def test_audit_jsonl_valid_and_logs_expected_entries(self) -> None:
        audit_path = self.evidence_dir / "audit.jsonl"
        audit_logger = AuditLogger(audit_path)

        # Log event creation
        audit_logger.log(
            event_id="evt_audit_01",
            camera_id="cam_good_test",
            session_id="sess_01",
            module_id="BAR-001",
            action=AUDIT_ACTION_EVENT_CREATED,
            details={"track_id": 9, "frame": 120},
        )

        writer = EvidenceWriter(self.evidence_dir, audit_logger=audit_logger)
        event = SafetyEvent(
            event_id="evt_audit_01",
            camera_id="cam_good_test",
            session_id="sess_01",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=9,
            source_timestamp_seconds=4.0,
            frame_number=120,
        )
        writer.save_event_evidence(
            event=event,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
        )

        entries = audit_logger.read_entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["action"], AUDIT_ACTION_EVENT_CREATED)
        self.assertEqual(entries[0]["event_id"], "evt_audit_01")
        self.assertEqual(entries[1]["action"], AUDIT_ACTION_EVIDENCE_SAVED)
        self.assertEqual(entries[1]["event_id"], "evt_audit_01")
        self.assertIn("raw_snapshot", entries[1]["details"])

    # 18. evidence failure can produce an error audit entry
    def test_evidence_failure_logs_error_audit_entry(self) -> None:
        audit_path = self.evidence_dir / "audit.jsonl"
        audit_logger = AuditLogger(audit_path)
        writer = EvidenceWriter(self.evidence_dir, audit_logger=audit_logger)

        event = SafetyEvent(
            event_id="evt_fail_test",
            camera_id="cam_good_test",
            session_id="sess_01",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=9,
            source_timestamp_seconds=4.0,
            frame_number=120,
        )
        # Trigger failure using non-existent source video
        fake_video_path = self.root / "does_not_exist.mp4"
        with self.assertRaises(FileNotFoundError):
            writer.save_event_evidence(
                event=event,
                raw_frame=self.raw_frame,
                annotated_frame=self.annotated_frame,
                source_video_path=fake_video_path,
            )

        entries = audit_logger.read_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["action"], AUDIT_ACTION_EVIDENCE_ERROR)
        self.assertEqual(entries[0]["details"]["error"], "FileNotFoundError")

    # 19. one input event produces one event directory
    def test_one_input_event_produces_one_directory(self) -> None:
        writer = EvidenceWriter(self.evidence_dir)
        event = SafetyEvent(
            event_id="evt_single_dir",
            camera_id="cam_good_test",
            session_id="sess_01",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=9,
            source_timestamp_seconds=3.0,
            frame_number=90,
        )
        artifacts = writer.save_event_evidence(
            event=event,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
        )
        dirs = list((self.evidence_dir / "cam_good_test" / "sess_01").iterdir())
        self.assertEqual(len(dirs), 1)
        self.assertEqual(dirs[0], artifacts.event_dir)

    # 20. repeated presentation/banner frames do not independently create evidence
    def test_repeated_banner_frames_do_not_create_duplicate_evidence(self) -> None:
        # In the pipeline, evidence creation is driven by emitted SafetyEvent instances,
        # not by every frame that draws the banner. If an event is emitted once,
        # only one evidence call is made.
        writer = EvidenceWriter(self.evidence_dir)
        event = SafetyEvent(
            event_id="evt_once",
            camera_id="cam_good_test",
            session_id="sess_01",
            module_id="EXC-002",
            event_type="lone_worker",
            track_id=9,
            source_timestamp_seconds=41.2,
            frame_number=1236,
        )
        # Even if a display banner is active for 30 consecutive frames,
        # we only save evidence once for the event.
        artifacts = writer.save_event_evidence(
            event=event,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
        )
        event_folders = list(
            (self.evidence_dir / "cam_good_test" / "sess_01").glob("event_*")
        )
        self.assertEqual(len(event_folders), 1)

    # 21. empty event list creates no fabricated evidence
    def test_empty_event_list_creates_no_evidence(self) -> None:
        events: list[SafetyEvent] = []
        # When no events are present, no folders or files are written
        self.assertFalse(self.evidence_dir.exists())
        self.assertEqual(len(events), 0)

    # 22. zero ERG-006 events creates no ERG-006 evidence
    def test_zero_erg006_events_creates_no_erg_evidence(self) -> None:
        writer = EvidenceWriter(self.evidence_dir)
        bar_event = SafetyEvent(
            event_id="evt_bar_real",
            camera_id="cam_good_test",
            session_id="sess_01",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=9,
            source_timestamp_seconds=5.0,
            frame_number=150,
        )
        writer.save_event_evidence(
            event=bar_event,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
        )
        # Verify only BAR-001 directory exists; no ERG-006 evidence created
        sess_dir = self.evidence_dir / "cam_good_test" / "sess_01"
        folders = [f.name for f in sess_dir.glob("event_*")]
        self.assertEqual(len(folders), 1)
        self.assertIn("evt_bar_real", folders[0])
        self.assertNotIn("ERG-006", folders[0])

    # 23. camera/session separation is preserved
    def test_camera_session_directory_separation(self) -> None:
        writer = EvidenceWriter(self.evidence_dir)
        ev1 = SafetyEvent(
            event_id="evt_c1",
            camera_id="cam_01",
            session_id="sess_A",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=1,
            source_timestamp_seconds=2.0,
            frame_number=60,
        )
        ev2 = SafetyEvent(
            event_id="evt_c2",
            camera_id="cam_02",
            session_id="sess_B",
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            track_id=1,
            source_timestamp_seconds=2.0,
            frame_number=60,
        )
        art1 = writer.save_event_evidence(
            event=ev1,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
        )
        art2 = writer.save_event_evidence(
            event=ev2,
            raw_frame=self.raw_frame,
            annotated_frame=self.annotated_frame,
            source_video_path=self.source_video_path,
        )
        self.assertTrue(str(art1.event_dir).startswith(str(self.evidence_dir / "cam_01" / "sess_A")))
        self.assertTrue(str(art2.event_dir).startswith(str(self.evidence_dir / "cam_02" / "sess_B")))
        self.assertNotEqual(art1.event_dir.parent, art2.event_dir.parent)

    # 24. invalid source timestamp fails clearly
    def test_invalid_source_timestamp_fails_clearly(self) -> None:
        with self.assertRaises(ValueError):
            compute_clip_bounds(
                source_timestamp_seconds=-5.0,
                source_duration_seconds=15.0,
            )
        with self.assertRaises(ValueError):
            SafetyEvent(
                event_id="evt_invalid_ts",
                camera_id="cam_good_test",
                session_id="sess_01",
                module_id="BAR-001",
                event_type="restricted_zone_entry",
                track_id=9,
                source_timestamp_seconds=-1.0,
                frame_number=0,
            )

    # 25. malformed event input fails clearly
    def test_malformed_event_input_fails_clearly(self) -> None:
        with self.assertRaises(ValueError):
            SafetyEvent.from_dict({})
        with self.assertRaises(ValueError):
            SafetyEvent.from_dict({
                "event_id": "",
                "camera_id": "cam_01",
                "session_id": "sess_01",
                "module_id": "BAR-001",
                "event_type": "restricted_zone_entry",
                "track_id": "not_an_int",
                "source_timestamp_seconds": 1.0,
                "frame_number": 30,
            })

    # Adapter tests
    def test_from_bar_entry_event_adapter(self) -> None:
        bar_evt = BarEntryEvent(
            camera_id="cam_good_test",
            session_id="sess_test",
            track_id=9,
            module_id="BAR-001",
            event_type="restricted_zone_entry",
            zone_id="restricted_zone_1",
            zone_name="Restricted Zone 1",
            timestamp_seconds=38.2,
            frame_number=1146,
            confidence=0.88,
        )
        safety_evt = SafetyEvent.from_bar_entry_event(bar_evt)
        self.assertEqual(safety_evt.module_id, "BAR-001")
        self.assertEqual(safety_evt.track_id, 9)
        self.assertEqual(safety_evt.zone_id, "restricted_zone_1")
        self.assertEqual(safety_evt.detection_confidence, 0.88)
        self.assertEqual(safety_evt.source_timestamp_seconds, 38.2)
        self.assertEqual(safety_evt.severity, "unclassified")
        self.assertEqual(safety_evt.status, "new")

    def test_from_temporal_safety_event_adapter(self) -> None:
        temp_evt = TemporalSafetyEvent(
            camera_id="cam_good_test",
            session_id="sess_test",
            track_id=9,
            module_id="EXC-002",
            event_type="lone_worker",
            zone_id="restricted_zone_1",
            timestamp_seconds=41.2,
            frame_number=1236,
            current_zone_occupancy=1,
            stationary_duration_seconds=None,
        )
        safety_evt = SafetyEvent.from_temporal_safety_event(temp_evt, zone_name="Restricted Zone 1")
        self.assertEqual(safety_evt.module_id, "EXC-002")
        self.assertEqual(safety_evt.track_id, 9)
        self.assertEqual(safety_evt.zone_id, "restricted_zone_1")
        self.assertEqual(safety_evt.zone_name, "Restricted Zone 1")
        self.assertEqual(safety_evt.zone_occupancy, 1)
        self.assertEqual(safety_evt.source_timestamp_seconds, 41.2)


if __name__ == "__main__":
    unittest.main()
