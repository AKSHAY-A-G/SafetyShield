"""Focused tests for the copy-only PPE training-pool builder."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from src.ppe.training_pool import build_training_pool
from src.ppe.trainer import check_training_gate


class TrainingPoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp())
        self.public = self.temp / "public"
        self.site = self.temp / "site"
        self.destination = self.temp / "pool"
        self.manifest = self.temp / "public_provenance.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    def _source(self, root: Path, names: list[str], split: str, name: str, row: str) -> None:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_dir / name), np.full((30, 40, 3), 80, dtype=np.uint8))
        (label_dir / f"{Path(name).stem}.txt").write_text(row, encoding="utf-8")
        config = {"names": names, "roboflow": {"project": root.name, "version": 1, "license": "CC BY 4.0", "url": "https://example.invalid"}}
        config["val" if split == "valid" else split] = f"../{split}/images"
        (root / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def test_builds_prefixed_pool_remaps_by_name_and_keeps_training_gate_blocked(self) -> None:
        self._source(self.public, ["vest", "helmet", "no_helmet", "no_vest"], "train", "public.jpg", "0 0.5 0.5 0.2 0.2\n")
        self._source(self.site, ["helmet", "no_helmet", "no_vest", "vest"], "valid", "site.jpg", "2 0.5 0.5 0.2 0.2\n")
        self.manifest.write_text(json.dumps({"groups": [{"group_id": "PUBLIC_DUP_0001", "review_status": "LIKELY_SAME_SOURCE", "members": [{"path": "train/images/public.jpg"}]}]}), encoding="utf-8")

        result = build_training_pool(self.destination, self.public, self.site, self.manifest)

        self.assertEqual(result["total_images"], 2)
        self.assertEqual(result["class_instance_counts"], {"helmet": 0, "no_helmet": 0, "no_vest": 1, "vest": 1})
        self.assertTrue((self.destination / "images" / "public__public.jpg").is_file())
        self.assertEqual((self.destination / "labels" / "public__public.txt").read_text(encoding="utf-8"), "3 0.500000 0.500000 0.200000 0.200000\n")
        record = next(item for item in result["records"] if item["source_type"] == "public")
        self.assertEqual(record["public_provenance"]["candidate_group_id"], "PUBLIC_DUP_0001")
        gate = check_training_gate(self.destination / "data.yaml")
        self.assertFalse(gate.can_train)
        self.assertIn("validation split is empty", gate.reason)

    def test_existing_destination_is_never_overwritten(self) -> None:
        self.destination.mkdir()
        self.manifest.write_text('{"groups": []}', encoding="utf-8")
        with self.assertRaises(FileExistsError):
            build_training_pool(self.destination, self.public, self.site, self.manifest)


if __name__ == "__main__":
    unittest.main()
