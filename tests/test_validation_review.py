from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.ppe.validation_review import ReviewCandidate, exclude_round_one, hamming_distance, perceptual_hash, write_contact_sheets


class ValidationReviewTests(unittest.TestCase):
    def test_timestamp_parser_handles_source_frame_extension(self) -> None:
        source_image = "cam1_clip_t0002000.jpg"
        self.assertEqual(float(Path(source_image).stem.rsplit("_t", maxsplit=1)[1]) / 1000.0, 2.0)

    def test_round_one_filenames_are_excluded(self) -> None:
        candidates = [ReviewCandidate("first.jpg", Path("first.jpg"), 0.0), ReviewCandidate("new.jpg", Path("new.jpg"), 2.0)]
        self.assertEqual([item.filename for item in exclude_round_one(candidates, {"first.jpg"})], ["new.jpg"])

    def test_identical_images_have_zero_perceptual_distance_and_render_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((40, 30, 3), 127, dtype=np.uint8)
            first, second = root / "first.jpg", root / "second.jpg"
            cv2.imwrite(str(first), image)
            cv2.imwrite(str(second), image)
            self.assertEqual(hamming_distance(perceptual_hash(first), perceptual_hash(second)), 0)
            sheets = write_contact_sheets([ReviewCandidate("first.jpg", first, 2.0), ReviewCandidate("second.jpg", second, 4.0)], root / "sheets", priorities={"first.jpg": "vest_candidate"})
            self.assertEqual(len(sheets), 1)
            self.assertTrue(sheets[0].is_file())


if __name__ == "__main__":
    unittest.main()
