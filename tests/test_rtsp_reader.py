"""Offline tests for Milestone 6 single-camera RTSP ingestion."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from contextlib import redirect_stderr
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from typing import Callable

import cv2
import numpy as np
import yaml

from scripts.run_rtsp_pipeline import parse_args, run_live_pipeline, save_reference_frame
from src.camera.rtsp_reader import (
    CameraConfigError,
    LatestFrame,
    RTSPCameraConfig,
    RTSPMetrics,
    RTSPReader,
    load_rtsp_camera_config,
    resolve_rtsp_secret,
    zone_status_for_frame,
)


SECRET = "rtsp://example.invalid/synthetic-sensitive-token"


class FakeCapture:
    def __init__(
        self,
        reads: list[
            tuple[bool, np.ndarray | None]
            | Callable[[], tuple[bool, np.ndarray | None]]
        ],
        *,
        opened: bool = True,
        fps: float = 25.0,
    ) -> None:
        self.reads = list(reads)
        self.opened = opened
        self.fps = fps
        self.released = False
        self._lock = threading.Lock()

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self.released:
                return False, None
            if self.reads:
                item = self.reads.pop(0)
                if callable(item):
                    return item()
                return item
        time.sleep(0.002)
        return False, None

    def get(self, propId: int) -> float:
        return self.fps if propId == cv2.CAP_PROP_FPS else 0.0

    def release(self) -> None:
        self.released = True


def frame(width=12, height=8, value=0):
    return np.full((height, width, 3), value, dtype=np.uint8)


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class CameraConfigTests(unittest.TestCase):
    def test_repository_camera_config_contains_env_name_only(self):
        path = Path("config/cameras.yaml")
        text = path.read_text(encoding="utf-8")
        config = load_rtsp_camera_config(path, "live_cam_1")
        self.assertEqual(config.rtsp_env_var, "SAFETYSHIELD_RTSP_LIVE_CAM_1")
        self.assertNotIn("rtsp://", text.lower())
        self.assertNotIn("password", text.lower())

    def test_missing_environment_variable_fails_without_credentials(self):
        config = RTSPCameraConfig(
            "live_cam_1", "rtsp", "SAFETYSHIELD_RTSP_LIVE_CAM_1", True
        )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(CameraConfigError) as caught:
                resolve_rtsp_secret(config)
        message = str(caught.exception)
        self.assertIn("SAFETYSHIELD_RTSP_LIVE_CAM_1", message)
        self.assertNotIn("rtsp://", message)

    def test_serialized_config_does_not_contain_secret(self):
        config = RTSPCameraConfig(
            "live_cam_1", "rtsp", "SAFETYSHIELD_RTSP_LIVE_CAM_1", True
        )
        serialized = json.dumps(
            {name: getattr(config, name) for name in config.__slots__}
        )
        self.assertNotIn(SECRET, serialized)
        self.assertNotIn("password", serialized.lower())

    def test_malformed_camera_config_fails_clearly(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "cameras.yaml"
            path.write_text("cameras: []\n", encoding="utf-8")
            with self.assertRaisesRegex(CameraConfigError, "cameras must be a mapping"):
                load_rtsp_camera_config(path, "live_cam_1")

    def test_cli_has_no_url_or_credential_options(self):
        args = parse_args([])
        self.assertFalse(any("url" in name or "password" in name or "username" in name
                             for name in vars(args)))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--url", SECRET])


class RTSPReaderTests(unittest.TestCase):
    def make_reader(self, captures, logs=None, **kwargs):
        supplied = iter(captures)

        def factory(_source, _backend, _params):
            return next(supplied)

        return RTSPReader(
            "live_cam_1",
            "session-a",
            SECRET,
            capture_factory=factory,
            failed_read_threshold=kwargs.pop("failed_read_threshold", 2),
            backoff_seconds=kwargs.pop("backoff_seconds", (0.0,)),
            max_connect_attempts=kwargs.pop("max_connect_attempts", 3),
            logger=(logs.append if logs is not None else lambda _message: None),
            **kwargs,
        )

    def test_successful_connection_latest_frame_and_native_dimensions(self):
        capture = FakeCapture([(True, frame(13, 9, 7))])
        reader = self.make_reader([capture])
        reader.start()
        self.assertTrue(wait_until(lambda: reader.metrics().frames_received == 1))
        latest = reader.read_latest(0.2)
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.frame.shape, (9, 13, 3))
        metrics = reader.metrics()
        self.assertEqual(metrics.successful_connections, 1)
        self.assertEqual((metrics.decoded_width, metrics.decoded_height), (13, 9))
        self.assertEqual(metrics.reported_source_fps, 25.0)
        self.assertTrue(reader.stop())

    def test_latest_frame_replacement_and_overwrite_accounting(self):
        capture = FakeCapture(
            [(True, frame(value=1)), (True, frame(value=2)), (True, frame(value=3))]
        )
        reader = self.make_reader([capture])
        reader.start()
        self.assertTrue(wait_until(lambda: reader.metrics().frames_received == 3))
        latest = reader.read_latest(0.2)
        assert latest is not None
        self.assertEqual(int(latest.frame[0, 0, 0]), 3)
        reader.mark_processed(latest.sequence)
        metrics = reader.metrics()
        self.assertEqual(metrics.frames_received, 3)
        self.assertEqual(metrics.frames_processed, 1)
        self.assertEqual(metrics.frames_overwritten_or_dropped, 2)
        self.assertTrue(reader.uses_single_frame_slot)
        self.assertFalse(hasattr(reader, "queue"))
        reader.stop()

    def test_logging_and_metrics_do_not_expose_secret(self):
        logs = []
        capture = FakeCapture([(True, frame())])
        reader = self.make_reader([capture], logs=logs)
        reader.start()
        self.assertTrue(wait_until(lambda: reader.metrics().frames_received == 1))
        serialized = json.dumps(reader.metrics().as_dict()) + "\n".join(logs)
        self.assertNotIn(SECRET, serialized)
        self.assertNotIn("password", serialized.lower())
        reader.stop()

    def test_clean_shutdown_is_idempotent_and_releases_capture(self):
        capture = FakeCapture([(True, frame())])
        reader = self.make_reader([capture])
        reader.start()
        self.assertTrue(wait_until(lambda: reader.metrics().frames_received == 1))
        self.assertTrue(reader.stop())
        self.assertTrue(reader.stop())
        self.assertTrue(capture.released)

    def test_temporary_failure_is_counted_then_stream_recovers(self):
        def delayed_frame():
            time.sleep(0.10)
            return True, frame()

        capture = FakeCapture([(False, None), (True, frame()), delayed_frame])
        reader = self.make_reader([capture], failed_read_threshold=2)
        reader.start()
        self.assertTrue(wait_until(lambda: reader.metrics().frames_received >= 1))
        metrics = reader.metrics()
        self.assertGreaterEqual(metrics.failed_reads, 1)
        self.assertEqual(metrics.consecutive_failed_reads, 0)
        self.assertEqual(metrics.reconnect_count, 0)
        reader.stop()

    def test_repeated_failures_trigger_reconnect(self):
        def delayed_frame():
            time.sleep(0.10)
            return True, frame(value=10)

        first = FakeCapture([(False, None), (False, None)])
        second = FakeCapture([(True, frame(value=9)), delayed_frame])
        reader = self.make_reader([first, second])
        reader.start()
        self.assertTrue(wait_until(lambda: reader.metrics().frames_received == 1))
        metrics = reader.metrics()
        self.assertEqual(metrics.connect_attempts, 2)
        self.assertEqual(metrics.successful_connections, 2)
        self.assertGreaterEqual(metrics.reconnect_count, 1)
        self.assertEqual(metrics.consecutive_failed_reads, 0)
        self.assertEqual(metrics.reconnect_delay_seconds, 0.0)
        self.assertTrue(first.released)
        reader.stop()

    def test_connection_failures_are_finite_and_backoff_bounded(self):
        captures = [FakeCapture([], opened=False) for _ in range(3)]
        reader = self.make_reader(
            captures, backoff_seconds=(0.0, 0.01), max_connect_attempts=3
        )
        reader.start()
        self.assertTrue(wait_until(lambda: reader.terminal_failure))
        metrics = reader.metrics()
        self.assertEqual(metrics.connect_attempts, 3)
        self.assertLessEqual(metrics.reconnect_delay_seconds, 0.01)
        self.assertTrue(reader.stop())

    def test_metrics_are_isolated_by_camera_and_session(self):
        one = self.make_reader([FakeCapture([(True, frame())])])
        two = RTSPReader(
            "other_camera", "session-b", SECRET,
            capture_factory=lambda *_: FakeCapture([(True, frame())]),
            backoff_seconds=(0.0,), logger=lambda _message: None,
        )
        one.start()
        two.start()
        self.assertTrue(wait_until(lambda: one.metrics().frames_received == 1))
        self.assertTrue(wait_until(lambda: two.metrics().frames_received == 1))
        self.assertNotEqual(one.metrics().session_id, two.metrics().session_id)
        self.assertNotEqual(one.metrics().camera_id, two.metrics().camera_id)
        one.stop()
        two.stop()

    def test_resolution_change_is_detected(self):
        capture = FakeCapture([(True, frame(12, 8)), (True, frame(16, 10))])
        reader = self.make_reader([capture])
        reader.start()
        self.assertTrue(wait_until(lambda: reader.metrics().frames_received == 2))
        metrics = reader.metrics()
        self.assertEqual(metrics.resolution_changes, 1)
        self.assertEqual((metrics.decoded_width, metrics.decoded_height), (16, 10))
        reader.stop()


class ZoneAndReferenceTests(unittest.TestCase):
    def test_missing_live_zone_does_not_prevent_processing(self):
        enabled, status = zone_status_for_frame(
            Path("config/zones.yaml"), "live_cam_1", 1920, 1080
        )
        self.assertFalse(enabled)
        self.assertEqual(status, "disabled / not configured")

    def test_resolution_mismatch_disables_fixed_zone(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "zones.yaml"
            path.write_text(
                yaml.safe_dump({"cameras": {"live_cam_1": {
                    "expected_resolution": [640, 480], "fixed_camera": True,
                    "zones": [{"zone_id": "safe-test"}],
                }}}), encoding="utf-8"
            )
            enabled, status = zone_status_for_frame(path, "live_cam_1", 800, 600)
        self.assertFalse(enabled)
        self.assertIn("resolution mismatch", status)

    def test_reference_frame_preserves_dimensions(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "reference.jpg"
            original = frame(37, 19, 120)
            save_reference_frame(path, original)
            saved = cv2.imread(str(path))
            self.assertIsNotNone(saved)
            assert saved is not None
            self.assertEqual(saved.shape, original.shape)


class DurationRunnerTests(unittest.TestCase):
    def test_duration_based_runner_stops_without_live_camera(self):
        class StepClock:
            def __init__(self):
                self.value = 0.0

            def __call__(self):
                self.value += 0.02
                return self.value

        class FakeReader:
            terminal_failure = False

            def __init__(self, camera_id, session_id, _source):
                self.camera_id = camera_id
                self.session_id = session_id
                self.received = 0
                self.processed = 0

            def start(self):
                return None

            def read_latest(self, timeout=0.0):
                self.received += 1
                return LatestFrame(self.received, 0.0, frame())

            def mark_processed(self, _sequence):
                self.processed += 1

            def stop(self):
                return True

            def metrics(self):
                return RTSPMetrics(
                    self.camera_id, self.session_id, 1, 1, 0, self.received,
                    self.processed, 0, 0, 0, 0.01, 0.2, 12, 8, 25.0, 0,
                    True, 0.0,
                )

        class FakeDetector:
            def __init__(self, **_kwargs):
                pass

            def detect(self, _frame):
                return []

        class FakeTracker:
            unique_track_count = 0

            def __init__(self, *_args):
                pass

            def update(self, *_args, **_kwargs):
                return []

        clock = StepClock()
        with (
            patch("scripts.run_rtsp_pipeline.torch.cuda.reset_peak_memory_stats"),
            patch("scripts.run_rtsp_pipeline.torch.cuda.synchronize"),
            patch("scripts.run_rtsp_pipeline.torch.cuda.max_memory_allocated", return_value=0),
        ):
            result = run_live_pipeline(
                camera_id="live_cam_1", session_id="test-session", source=SECRET,
                duration_seconds=0.10, output=None, reference_path=None,
                display=False, reader_factory=FakeReader,
                detector_factory=FakeDetector, tracker_factory=FakeTracker,
                clock=clock,
            )
        self.assertTrue(result.clean_shutdown)
        self.assertGreaterEqual(result.acceptance_seconds, 0.10)
        self.assertGreater(result.metrics.frames_processed, 0)


if __name__ == "__main__":
    unittest.main()
