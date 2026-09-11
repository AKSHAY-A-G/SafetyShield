"""Offline unit tests for Milestone 7 SafetyShield dashboard data, logic, and security."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import tempfile
import unittest

import yaml

from src.dashboard.data import (
    DEFAULT_CONTROLS,
    evaluate_module_availability,
    get_system_status,
    load_audit_log,
    load_camera_dashboard_info,
    load_module_controls,
    load_recent_events,
    load_runtime_status,
    save_module_controls,
)
from src.dashboard.models import CameraDashboardInfo


SECRET_URL = "rtsp://admin:supersecret_pass@192.168.1.120:554/live"


class DashboardDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=".")
        self.base = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. camera config loads without exposing RTSP secret
    def test_camera_config_loads_without_exposing_rtsp_secret(self):
        cam_yaml = self.base / "cameras.yaml"
        cam_yaml.write_text(
            yaml.safe_dump({
                "cameras": {
                    "live_cam_1": {
                        "camera_id": "live_cam_1",
                        "source_type": "rtsp",
                        "rtsp_env_var": "SAFETYSHIELD_RTSP_LIVE_CAM_1",
                        "enabled": True,
                    }
                }
            }),
            encoding="utf-8",
        )
        zones_yaml = self.base / "zones.yaml"
        zones_yaml.write_text("cameras: {}\n", encoding="utf-8")

        env_file = self.base / ".env"
        env_file.write_text(f"SAFETYSHIELD_RTSP_LIVE_CAM_1={SECRET_URL}\n", encoding="utf-8")

        cams = load_camera_dashboard_info(cam_yaml, zones_yaml, env_file)
        self.assertIn("live_cam_1", cams)
        cam = cams["live_cam_1"]
        self.assertEqual(cam.camera_id, "live_cam_1")
        self.assertNotIn("secret", str(asdict(cam)).lower().replace("secret_configured", "").replace("secret_env_var", ""))
        self.assertNotIn("supersecret", str(asdict(cam)))
        self.assertNotIn("192.168", str(asdict(cam)))

    # 2. secret presence returns only boolean / safe status
    def test_secret_presence_returns_only_boolean_status(self):
        cam_yaml = self.base / "cameras.yaml"
        cam_yaml.write_text(
            yaml.safe_dump({"cameras": {"live_cam_1": {"camera_id": "live_cam_1", "rtsp_env_var": "TEST_VAR"}}}),
            encoding="utf-8",
        )
        zones_yaml = self.base / "zones.yaml"
        zones_yaml.write_text("cameras: {}\n", encoding="utf-8")

        env_file = self.base / ".env"
        env_file.write_text("TEST_VAR=some_secret_token\n", encoding="utf-8")

        cams = load_camera_dashboard_info(cam_yaml, zones_yaml, env_file)
        self.assertIsInstance(cams["live_cam_1"].secret_configured, bool)
        self.assertTrue(cams["live_cam_1"].secret_configured)
        self.assertEqual(cams["live_cam_1"].secret_env_var, "TEST_VAR")

    # 3. no RTSP URL appears in serialized dashboard camera state
    def test_no_rtsp_url_appears_in_serialized_camera_state(self):
        cam_yaml = self.base / "cameras.yaml"
        cam_yaml.write_text(
            yaml.safe_dump({"cameras": {"live_cam_1": {"camera_id": "live_cam_1", "rtsp_env_var": "SECRET_KEY"}}}),
            encoding="utf-8",
        )
        zones_yaml = self.base / "zones.yaml"
        zones_yaml.write_text("cameras: {}\n", encoding="utf-8")
        env_file = self.base / ".env"
        env_file.write_text(f"SECRET_KEY={SECRET_URL}\n", encoding="utf-8")

        cams = load_camera_dashboard_info(cam_yaml, zones_yaml, env_file)
        serialized = json.dumps([asdict(c) for c in cams.values()])
        self.assertNotIn("rtsp://", serialized)
        self.assertNotIn(SECRET_URL, serialized)

    # 4. module configuration loads
    def test_module_configuration_loads(self):
        ctrl_yaml = self.base / "dashboard_controls.yaml"
        ctrl_yaml.write_text(
            yaml.safe_dump({"schema_version": "1.0", "modules": {"person_detection": {"enabled": False}}}),
            encoding="utf-8",
        )
        mods = load_module_controls(ctrl_yaml)
        self.assertIn("person_detection", mods)
        self.assertFalse(mods["person_detection"]["enabled"])
        self.assertTrue(mods["person_tracking"]["enabled"])  # defaults to True

    # 5. module dependency evaluation works
    def test_module_dependency_evaluation_works(self):
        cam = CameraDashboardInfo(
            camera_id="live_cam_1", source_type="rtsp", enabled=True,
            secret_configured=True, secret_env_var="VAR", has_zone=False,
            zone_count=0, zone_names=[],
        )
        mods = {k: {"enabled": True, "label": k, "description": ""} for k in DEFAULT_CONTROLS}
        statuses = evaluate_module_availability(mods, cam)
        self.assertTrue(statuses["person_detection"].available)
        self.assertTrue(statuses["person_tracking"].available)

    # 6. tracking unavailable if detection disabled
    def test_tracking_unavailable_if_detection_disabled(self):
        cam = CameraDashboardInfo(
            camera_id="c1", source_type="video", enabled=True,
            secret_configured=False, secret_env_var="", has_zone=True,
            zone_count=1, zone_names=["zone1"],
        )
        mods = {k: {"enabled": True, "label": k, "description": ""} for k in DEFAULT_CONTROLS}
        mods["person_detection"]["enabled"] = False
        statuses = evaluate_module_availability(mods, cam)
        self.assertFalse(statuses["person_tracking"].available)
        self.assertIn("Person Detection", statuses["person_tracking"].unavailable_reason)

    # 7. BAR-001 unavailable without zone
    def test_bar_001_unavailable_without_zone(self):
        cam = CameraDashboardInfo(
            camera_id="live_cam_1", source_type="rtsp", enabled=True,
            secret_configured=True, secret_env_var="V", has_zone=False,
            zone_count=0, zone_names=[],
        )
        mods = {k: {"enabled": True, "label": k, "description": ""} for k in DEFAULT_CONTROLS}
        statuses = evaluate_module_availability(mods, cam)
        self.assertFalse(statuses["bar_001"].available)
        self.assertIn("zone not configured", statuses["bar_001"].unavailable_reason)

    # 8. IDT-004 unavailable without zone
    def test_idt_004_unavailable_without_zone(self):
        cam = CameraDashboardInfo(
            camera_id="live_cam_1", source_type="rtsp", enabled=True,
            secret_configured=True, secret_env_var="V", has_zone=False,
            zone_count=0, zone_names=[],
        )
        mods = {k: {"enabled": True, "label": k, "description": ""} for k in DEFAULT_CONTROLS}
        statuses = evaluate_module_availability(mods, cam)
        self.assertFalse(statuses["idt_004"].available)
        self.assertIn("zone not configured", statuses["idt_004"].unavailable_reason)

    # 9. EXC-002 unavailable without zone
    def test_exc_002_unavailable_without_zone(self):
        cam = CameraDashboardInfo(
            camera_id="live_cam_1", source_type="rtsp", enabled=True,
            secret_configured=True, secret_env_var="V", has_zone=False,
            zone_count=0, zone_names=[],
        )
        mods = {k: {"enabled": True, "label": k, "description": ""} for k in DEFAULT_CONTROLS}
        statuses = evaluate_module_availability(mods, cam)
        self.assertFalse(statuses["exc_002"].available)
        self.assertIn("zone not configured", statuses["exc_002"].unavailable_reason)

    # 10. ERG-006 availability does not require zone
    def test_erg_006_availability_does_not_require_zone(self):
        cam = CameraDashboardInfo(
            camera_id="live_cam_1", source_type="rtsp", enabled=True,
            secret_configured=True, secret_env_var="V", has_zone=False,
            zone_count=0, zone_names=[],
        )
        mods = {k: {"enabled": True, "label": k, "description": ""} for k in DEFAULT_CONTROLS}
        statuses = evaluate_module_availability(mods, cam)
        self.assertTrue(statuses["erg_006"].available)
        self.assertIsNone(statuses["erg_006"].unavailable_reason)

    # 11. global-enabled and camera-available states remain distinct
    def test_global_enabled_and_camera_available_states_remain_distinct(self):
        cam = CameraDashboardInfo(
            camera_id="live_cam_1", source_type="rtsp", enabled=True,
            secret_configured=True, secret_env_var="V", has_zone=False,
            zone_count=0, zone_names=[],
        )
        mods = {k: {"enabled": True, "label": k, "description": ""} for k in DEFAULT_CONTROLS}
        statuses = evaluate_module_availability(mods, cam)
        # BAR-001 is globally enabled, but unavailable for this camera
        self.assertTrue(statuses["bar_001"].global_enabled)
        self.assertFalse(statuses["bar_001"].available)

    # 12. module control persistence works if implemented
    def test_module_control_persistence(self):
        ctrl_path = self.base / "controls.yaml"
        mods = {k: {"enabled": False, "label": k, "description": ""} for k in DEFAULT_CONTROLS}
        mods["person_detection"]["enabled"] = True
        save_module_controls(ctrl_path, mods)
        self.assertTrue(ctrl_path.exists())
        loaded = load_module_controls(ctrl_path)
        self.assertTrue(loaded["person_detection"]["enabled"])
        self.assertFalse(loaded["person_tracking"]["enabled"])

    # 13. malformed control config fails safely
    def test_malformed_control_config_fails_safely(self):
        ctrl_path = self.base / "bad_controls.yaml"
        ctrl_path.write_text("not_valid_yaml: [: \n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_module_controls(ctrl_path)

    # 14. runtime status loads
    def test_runtime_status_loads(self):
        status_path = self.base / "status.json"
        now_str = datetime.now(timezone.utc).isoformat()
        status_path.write_text(
            json.dumps({
                "schema_version": "1.0",
                "camera_id": "live_cam_1",
                "session_id": "s1",
                "state": "running",
                "updated_at_utc": now_str,
                "decoded_width": 2560,
                "decoded_height": 1440,
                "source_fps": 20.0,
                "frames_received": 100,
                "frames_processed": 50,
                "frames_dropped": 50,
                "failed_reads": 0,
                "reconnect_count": 0,
                "processing_fps": 10.2,
                "temporary_track_count": 2,
                "zones_enabled": False,
            }),
            encoding="utf-8",
        )
        status = load_runtime_status(status_path)
        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status.camera_id, "live_cam_1")
        self.assertEqual(status.state, "running")
        self.assertEqual(status.decoded_width, 2560)
        self.assertEqual(status.frames_processed, 50)
        self.assertFalse(status.is_stale)

    # 15. malformed runtime status fails safely
    def test_malformed_runtime_status_fails_safely(self):
        status_path = self.base / "bad_status.json"
        status_path.write_text("{broken json", encoding="utf-8")
        status = load_runtime_status(status_path)
        self.assertIsNone(status)

    # 16. stale runtime status is recognized
    def test_stale_runtime_status_is_recognized(self):
        status_path = self.base / "stale_status.json"
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        status_path.write_text(
            json.dumps({
                "schema_version": "1.0",
                "camera_id": "live_cam_1",
                "session_id": "s1",
                "state": "running",
                "updated_at_utc": old_time,
                "frames_received": 10,
                "frames_processed": 5,
            }),
            encoding="utf-8",
        )
        status = load_runtime_status(status_path, stale_threshold_seconds=30.0)
        self.assertIsNotNone(status)
        assert status is not None
        self.assertTrue(status.is_stale)
        self.assertGreater(status.age_seconds, 60.0)

    # 17. evidence metadata discovery works
    def test_evidence_metadata_discovery_works(self):
        ev_dir = self.base / "evidence" / "cam_good_test" / "s1" / "event_1"
        ev_dir.mkdir(parents=True)
        (ev_dir / "snapshot_raw.jpg").write_text("raw", encoding="utf-8")
        (ev_dir / "snapshot_annotated.jpg").write_text("annotated", encoding="utf-8")
        (ev_dir / "metadata.json").write_text(
            json.dumps({
                "event_id": "event_1",
                "camera_id": "cam_good_test",
                "session_id": "s1",
                "module_id": "BAR-001",
                "event_type": "restricted_zone_entry",
                "track_id": 9,
                "source_timestamp_seconds": 38.2,
                "frame_number": 1146,
                "severity": "unclassified",
                "status": "new",
                "created_at_utc": "2026-09-11T07:18:50+00:00",
                "source_video": "cam_good_test.mp4",
                "evidence": {
                    "raw_snapshot": "snapshot_raw.jpg",
                    "annotated_snapshot": "snapshot_annotated.jpg",
                },
            }),
            encoding="utf-8",
        )
        events = load_recent_events(self.base / "evidence")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, "event_1")
        self.assertEqual(events[0].track_id, 9)
        self.assertTrue(events[0].is_recorded_prototype)
        self.assertIsNotNone(events[0].raw_snapshot_path)
        self.assertIsNotNone(events[0].annotated_snapshot_path)

    # 18. events are sorted correctly
    def test_events_are_sorted_correctly(self):
        ev_root = self.base / "evidence" / "cam1" / "s1"
        for idx, t_sec in [(1, 10.0), (2, 50.0), (3, 30.0)]:
            d = ev_root / f"event_{idx}"
            d.mkdir(parents=True)
            (d / "metadata.json").write_text(
                json.dumps({"event_id": f"event_{idx}", "source_timestamp_seconds": t_sec}),
                encoding="utf-8",
            )
        events = load_recent_events(self.base / "evidence")
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].event_id, "event_2")  # 50.0s
        self.assertEqual(events[1].event_id, "event_3")  # 30.0s
        self.assertEqual(events[2].event_id, "event_1")  # 10.0s

    # 19. malformed evidence metadata does not crash loader
    def test_malformed_evidence_metadata_does_not_crash_loader(self):
        ev_root = self.base / "evidence" / "cam1" / "s1"
        good = ev_root / "good_event"
        good.mkdir(parents=True)
        (good / "metadata.json").write_text(json.dumps({"event_id": "good_1"}), encoding="utf-8")
        bad = ev_root / "bad_event"
        bad.mkdir(parents=True)
        (bad / "metadata.json").write_text("invalid json {", encoding="utf-8")

        events = load_recent_events(self.base / "evidence")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, "good_1")

    # 20. missing evidence artifact is reported
    def test_missing_evidence_artifact_is_reported(self):
        ev_dir = self.base / "evidence" / "cam1" / "s1" / "ev1"
        ev_dir.mkdir(parents=True)
        (ev_dir / "metadata.json").write_text(
            json.dumps({
                "event_id": "ev1",
                "evidence": {
                    "raw_snapshot": "missing_raw.jpg",
                    "annotated_snapshot": "missing_ann.jpg",
                    "video_clip": "missing_clip.mp4",
                }
            }),
            encoding="utf-8",
        )
        events = load_recent_events(self.base / "evidence")
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0].raw_snapshot_path)
        self.assertIsNone(events[0].annotated_snapshot_path)
        self.assertIsNone(events[0].video_clip_path)

    # 21. audit JSONL valid lines load
    def test_audit_jsonl_valid_lines_load(self):
        ev_dir = self.base / "evidence" / "cam1" / "s1"
        ev_dir.mkdir(parents=True)
        audit_file = ev_dir / "audit.jsonl"
        audit_file.write_text(
            json.dumps({"event_id": "e1", "action": "EVENT_CREATED", "audit_timestamp_utc": "2026-09-11T10:00:00Z"}) + "\n"
            + json.dumps({"event_id": "e1", "action": "EVIDENCE_SAVED", "audit_timestamp_utc": "2026-09-11T10:01:00Z"}) + "\n",
            encoding="utf-8",
        )
        entries = load_audit_log(self.base / "evidence")
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].action, "EVIDENCE_SAVED")  # newest first
        self.assertEqual(entries[1].action, "EVENT_CREATED")

    # 22. malformed audit line is skipped/reported safely
    def test_malformed_audit_line_is_skipped(self):
        ev_dir = self.base / "evidence" / "cam1" / "s1"
        ev_dir.mkdir(parents=True)
        audit_file = ev_dir / "audit.jsonl"
        audit_file.write_text(
            "corrupt line\n"
            + json.dumps({"event_id": "e1", "action": "EVENT_CREATED", "audit_timestamp_utc": "2026-09-11T10:00:00Z"}) + "\n",
            encoding="utf-8",
        )
        entries = load_audit_log(self.base / "evidence")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].action, "EVENT_CREATED")

    # 23. track ID is represented as temporary/local context
    def test_track_id_is_represented_as_temporary_local_context(self):
        ev_dir = self.base / "evidence" / "cam1" / "s1" / "ev1"
        ev_dir.mkdir(parents=True)
        (ev_dir / "metadata.json").write_text(
            json.dumps({"event_id": "ev1", "camera_id": "cam1", "session_id": "s1", "track_id": 7}),
            encoding="utf-8",
        )
        events = load_recent_events(self.base / "evidence")
        self.assertEqual(events[0].track_id, 7)
        self.assertEqual(events[0].camera_id, "cam1")
        self.assertEqual(events[0].session_id, "s1")

    # 24. dashboard data contains no credential-bearing URL
    def test_dashboard_data_contains_no_credential_bearing_url(self):
        cam_yaml = self.base / "cameras.yaml"
        cam_yaml.write_text(yaml.safe_dump({"cameras": {"c1": {"rtsp_env_var": "VAR"}}}), encoding="utf-8")
        zones_yaml = self.base / "zones.yaml"
        zones_yaml.write_text("cameras: {}\n", encoding="utf-8")
        env_file = self.base / ".env"
        env_file.write_text(f"VAR={SECRET_URL}\n", encoding="utf-8")

        cams = load_camera_dashboard_info(cam_yaml, zones_yaml, env_file)
        events = load_recent_events(self.base / "evidence")
        audit = load_audit_log(self.base / "evidence")

        dumped = f"{cams} {events} {audit}"
        self.assertNotIn("rtsp://", dumped)
        self.assertNotIn("supersecret_pass", dumped)

    # 25. empty evidence directory works
    def test_empty_evidence_directory_works(self):
        empty_dir = self.base / "empty_evidence"
        empty_dir.mkdir()
        self.assertEqual(load_recent_events(empty_dir), [])
        self.assertEqual(load_audit_log(empty_dir), [])

    # 26. missing runtime status works
    def test_missing_runtime_status_works(self):
        missing_path = self.base / "does_not_exist.json"
        self.assertIsNone(load_runtime_status(missing_path))

    # 27. camera without zones reports unavailable zone modules
    def test_camera_without_zones_reports_unavailable_zone_modules(self):
        cam = CameraDashboardInfo(
            camera_id="unconfigured_cam", source_type="rtsp", enabled=True,
            secret_configured=True, secret_env_var="VAR", has_zone=False,
            zone_count=0, zone_names=[],
        )
        mods = {k: {"enabled": True, "label": k, "description": ""} for k in DEFAULT_CONTROLS}
        evals = evaluate_module_availability(mods, cam)
        for zone_mod in ("bar_001", "idt_004", "exc_002"):
            self.assertFalse(evals[zone_mod].available)
            self.assertIn("zone not configured", evals[zone_mod].unavailable_reason)

    # 28. cam_good_test zone is not assigned to live_cam_1
    def test_cam_good_test_zone_is_not_assigned_to_live_cam_1(self):
        cam_yaml = self.base / "cameras.yaml"
        cam_yaml.write_text(
            yaml.safe_dump({
                "cameras": {
                    "live_cam_1": {"camera_id": "live_cam_1", "rtsp_env_var": "VAR"},
                    "cam_good_test": {"camera_id": "cam_good_test"},
                }
            }),
            encoding="utf-8",
        )
        zones_yaml = self.base / "zones.yaml"
        zones_yaml.write_text(
            yaml.safe_dump({
                "cameras": {
                    "cam_good_test": {
                        "zones": [{"zone_id": "restricted_zone_1", "zone_name": "Restricted Zone 1"}]
                    }
                }
            }),
            encoding="utf-8",
        )
        cams = load_camera_dashboard_info(cam_yaml, zones_yaml)
        self.assertTrue(cams["cam_good_test"].has_zone)
        self.assertEqual(cams["cam_good_test"].zone_count, 1)
        self.assertFalse(cams["live_cam_1"].has_zone)
        self.assertEqual(cams["live_cam_1"].zone_count, 0)
        self.assertEqual(cams["live_cam_1"].zone_names, [])


class SystemStatusTests(unittest.TestCase):
    def test_system_status_reports_milestone_and_device(self):
        status = get_system_status()
        self.assertIn("M0", status.milestones)
        self.assertEqual(status.milestones["M6"], "COMPLETE")
        self.assertEqual(status.milestones["M7"], "IN DEVELOPMENT")
        self.assertTrue(bool(status.python_version))
        self.assertTrue(bool(status.streamlit_version))


if __name__ == "__main__":
    unittest.main()
