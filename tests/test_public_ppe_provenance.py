"""Focused tests for public PPE near-duplicate provenance review."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from src.ppe.public_provenance import build_public_provenance_manifest, create_contact_sheets


class PublicPPEProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp()) / "dataset"
        for split in ("train", "valid", "test"):
            (self.root / split / "images").mkdir(parents=True)
            (self.root / split / "labels").mkdir(parents=True)
        (self.root / "data.yaml").write_text(yaml.safe_dump({"names": ["helmet", "no_helmet", "no_vest", "vest"], "roboflow": {"project": "fixture", "version": 1, "license": "CC BY 4.0"}}), encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.root.parent, ignore_errors=True)

    def _write_image(self, split: str, name: str, image: np.ndarray, label: str) -> None:
        cv2.imwrite(str(self.root / split / "images" / name), image)
        (self.root / split / "labels" / f"{Path(name).stem}.txt").write_text(label, encoding="utf-8")

    def test_cross_split_identical_pixels_form_clear_group_and_render_sheets(self) -> None:
        image = np.full((80, 100, 3), 120, dtype=np.uint8)
        self._write_image("train", "source_01_jpg.rf.a.jpg", image, "0 0.5 0.5 0.4 0.4\n")
        self._write_image("valid", "source_01_jpg.rf.b.jpg", image, "0 0.5 0.5 0.4 0.4\n")
        manifest = build_public_provenance_manifest(self.root)
        self.assertEqual(manifest["summary"]["candidate_pairs_across_splits"], 1)
        self.assertEqual(manifest["summary"]["candidate_groups_total"], 1)
        group = manifest["groups"][0]
        self.assertEqual(group["review_status"], "CLEAR_SAME_SOURCE_NEAR_DUPLICATE")
        self.assertTrue(group["spans_multiple_splits"])
        sheets = create_contact_sheets(self.root, manifest, self.root.parent / "review")
        self.assertEqual(len(sheets), 2)
        self.assertTrue(all(path.is_file() for path in sheets))

    def test_unrelated_images_do_not_create_groups(self) -> None:
        dark = np.random.default_rng(1).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        bright = np.random.default_rng(2).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        self._write_image("train", "dark_jpg.rf.a.jpg", dark, "1 0.5 0.5 0.4 0.4\n")
        self._write_image("test", "bright_jpg.rf.b.jpg", bright, "2 0.5 0.5 0.4 0.4\n")
        manifest = build_public_provenance_manifest(self.root)
        self.assertEqual(manifest["summary"]["candidate_groups_total"], 0)


if __name__ == "__main__":
    unittest.main()
