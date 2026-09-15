from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.prepare_ppe_test_dataset import priority_for_crop, select_temporally_diverse


class Crop:
    def __init__(self, name: str, group: str, confidence: float, height: int) -> None:
        self.crop_filename = name
        self.crop_path = Path(name)
        self.source_group = group
        self.person_detection_confidence = confidence
        self.crop_y1 = 0
        self.crop_y2 = height


class PPEDataTestPreparationTests(unittest.TestCase):
    def test_selection_round_robins_groups_and_rejects_near_duplicates(self) -> None:
        crops = [Crop("a.jpg", "group_a", 0.9, 450), Crop("b.jpg", "group_a", 0.8, 300), Crop("c.jpg", "group_b", 0.7, 100)]
        with patch("scripts.prepare_ppe_test_dataset.perceptual_hash", side_effect=[1, 1, 255]):
            selected, rejected = select_temporally_diverse(crops, 3)
        self.assertEqual([item.crop_filename for item in selected], ["a.jpg", "c.jpg"])
        self.assertEqual(rejected, 1)

    def test_priority_is_scale_only_and_not_a_ppe_label(self) -> None:
        self.assertEqual(priority_for_crop(Crop("near.jpg", "g", 0.9, 400)), "near_worker_candidate")
        self.assertEqual(priority_for_crop(Crop("far.jpg", "g", 0.9, 100)), "distant_worker_candidate")


if __name__ == "__main__":
    unittest.main()
