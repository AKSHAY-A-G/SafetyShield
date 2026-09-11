"""Lightweight Milestone 1 tests that do not load a model or process a video."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import cv2

from src.camera.video_reader import VideoReader
from src.detection.person_detector import PersonDetection


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_person_detection.py"
PROJECT_ROOT = SCRIPT.parents[1]
SPEC = importlib.util.spec_from_file_location("run_person_detection", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to create module spec for test")
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)
COMPARISON_SCRIPT = PROJECT_ROOT / "scripts" / "extract_detection_comparisons.py"
COMPARISON_SPEC = importlib.util.spec_from_file_location(
    "extract_detection_comparisons", COMPARISON_SCRIPT
)
if COMPARISON_SPEC is None or COMPARISON_SPEC.loader is None:
    raise RuntimeError("Unable to create module spec for test")
comparison = importlib.util.module_from_spec(COMPARISON_SPEC)
COMPARISON_SPEC.loader.exec_module(comparison)


class VideoReaderTests(unittest.TestCase):
    def test_missing_input_path_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            VideoReader(Path("definitely-not-a-real-video.mp4"))

    def test_reads_metadata_and_normal_eof(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            path = Path(directory) / "tiny.avi"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"MJPG"),  # pyrefly: ignore[missing-attribute]
                12.0,
                (32, 24),
            )
            self.assertTrue(writer.isOpened())
            try:
                import numpy as np

                writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            finally:
                writer.release()

            with VideoReader(path) as reader:
                self.assertEqual((reader.metadata.width, reader.metadata.height), (32, 24))
                self.assertAlmostEqual(reader.metadata.fps, 12.0, places=1)
                self.assertEqual(reader.metadata.frame_count, 1)
                self.assertIsNotNone(reader.read())
                self.assertIsNone(reader.read())


class PersonDetectionTests(unittest.TestCase):
    def test_detection_structure_and_coordinate_clipping(self):
        detection = PersonDetection.from_xyxy(
            (-4.2, 2.4, 110.8, 95.7), 0.876, frame_width=100, frame_height=80
        )
        self.assertEqual((detection.x1, detection.y1, detection.x2, detection.y2), (0, 2, 99, 79))
        self.assertEqual(detection.class_id, 0)
        self.assertEqual(detection.class_name, "Person")
        self.assertAlmostEqual(detection.confidence, 0.876)


class OutputTests(unittest.TestCase):
    def test_output_directory_is_created(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            output = Path(directory) / "nested" / "test.mp4"
            writer = run.create_output_writer(output, 32, 24, 10.0)
            try:
                self.assertTrue(writer.isOpened())
                self.assertTrue(output.parent.is_dir())
            finally:
                writer.release()

    def test_comparison_timestamp_uses_matching_frame_index(self):
        self.assertEqual(comparison.timestamp_to_frame_index(10.0, 30.0), 300)
        self.assertEqual(comparison.timestamp_to_frame_index(1.25, 24.0), 30)
        with self.assertRaises(ValueError):
            comparison.timestamp_to_frame_index(-1.0, 30.0)


if __name__ == "__main__":
    unittest.main()
