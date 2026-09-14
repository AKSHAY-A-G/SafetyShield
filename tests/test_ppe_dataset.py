"""Unit tests for SafetyShield Milestone 8 PPE dataset and training preparation tooling."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from src.detection.person_detector import PersonDetection
from src.ppe.constants import (
    DEFAULT_BOX_PADDING_FRACTION,
    DEFAULT_TRAIN_BATCH,
    DEFAULT_TRAIN_IMGSZ,
    DEFAULT_TRAIN_MODEL,
    DEFAULT_TRAIN_PROJECT,
    DEFAULT_TRAIN_WORKERS,
    PPE_CLASSES,
)
from src.ppe.crop_generator import (
    compute_padded_crop_box,
    extract_person_crops_from_frame,
    load_crop_manifest,
    write_crop_manifest,
)
from src.ppe.models import BBoxNormalized, PersonCropInfo, SourceFrameInfo
from src.ppe.splitter import (
    assign_source_group_splits,
    check_source_group_leakage,
    load_split_manifest,
    write_split_manifest,
)
from src.ppe.trainer import check_training_gate
from src.ppe.validator import (
    validate_dataset_yaml,
    validate_label_line,
    validate_ppe_dataset,
)
from src.ppe.visualizer import render_annotation_overlay


class PPEDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # 1. Accepted four-class mapping
    def test_accepted_four_class_mapping(self) -> None:
        expected = {0: "helmet", 1: "no_helmet", 2: "no_vest", 3: "vest"}
        self.assertEqual(dict(PPE_CLASSES), expected)
        self.assertEqual(len(PPE_CLASSES), 4)

    # 2. Unexpected class rejected
    def test_unexpected_class_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_label_line("4 0.5 0.5 0.2 0.2", line_no=1, filename="sample.txt")
        self.assertIn("Unexpected class_id", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_label_line("-1 0.5 0.5 0.2 0.2", line_no=1, filename="sample.txt")
        self.assertIn("Unexpected class_id", str(ctx.exception))

    # 3. Normalized YOLO box validation
    def test_normalized_yolo_box_validation(self) -> None:
        bbox = validate_label_line("0 0.5 0.5 0.2 0.3", line_no=1, filename="sample.txt")
        self.assertEqual(bbox.class_id, 0)
        self.assertAlmostEqual(bbox.x_center, 0.5)
        self.assertAlmostEqual(bbox.y_center, 0.5)
        self.assertAlmostEqual(bbox.width, 0.2)
        self.assertAlmostEqual(bbox.height, 0.3)

    # 4. Negative coordinate rejected
    def test_negative_coordinate_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_label_line("1 -0.05 0.5 0.2 0.2", line_no=1, filename="sample.txt")
        self.assertIn("outside legal range", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_label_line("2 0.5 -0.1 0.2 0.2", line_no=1, filename="sample.txt")
        self.assertIn("outside legal range", str(ctx.exception))

    # 5. Greater than 1 normalized coordinate rejected
    def test_greater_than_one_coordinate_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_label_line("3 1.05 0.5 0.2 0.2", line_no=1, filename="sample.txt")
        self.assertIn("outside legal range", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_label_line("0 0.5 1.5 0.2 0.2", line_no=1, filename="sample.txt")
        self.assertIn("outside legal range", str(ctx.exception))

    # 6. Zero-size box rejected
    def test_zero_size_box_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_label_line("0 0.5 0.5 0.0 0.2", line_no=1, filename="sample.txt")
        self.assertIn("must be positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_label_line("0 0.5 0.5 0.2 -0.1", line_no=1, filename="sample.txt")
        self.assertIn("must be positive", str(ctx.exception))

    # 7. Malformed label rejected
    def test_malformed_label_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_label_line("0 0.5 0.5 0.2", line_no=1, filename="sample.txt")
        self.assertIn("expected 5 values", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_label_line("0 0.5 0.5 0.2 0.2 extra", line_no=1, filename="sample.txt")
        self.assertIn("expected 5 values", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_label_line("abc 0.5 0.5 0.2 0.2", line_no=1, filename="sample.txt")
        self.assertIn("must be an integer", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_label_line("0 nan 0.5 0.2 0.2", line_no=1, filename="sample.txt")
        self.assertIn("is NaN or Inf", str(ctx.exception))

    # 8. Missing label file detected
    def test_missing_label_file_detected(self) -> None:
        d = self.tmp_dir / "dataset"
        img_dir = d / "train" / "images"
        lbl_dir = d / "train" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create valid image
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "img1.jpg"), dummy)
        # Omit label file for img1

        yaml_path = d / "data.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"names": ["helmet", "no_helmet", "no_vest", "vest"]}),
            encoding="utf-8",
        )

        report = validate_ppe_dataset(yaml_path=yaml_path)
        self.assertIn(str(img_dir / "img1.jpg"), report.missing_label_files)

    # 9. Orphan label detected
    def test_orphan_label_detected(self) -> None:
        d = self.tmp_dir / "dataset"
        img_dir = d / "train" / "images"
        lbl_dir = d / "train" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create label without matching image
        (lbl_dir / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        yaml_path = d / "data.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"names": ["helmet", "no_helmet", "no_vest", "vest"]}),
            encoding="utf-8",
        )

        report = validate_ppe_dataset(yaml_path=yaml_path)
        self.assertTrue(any("orphan.txt" in o for o in report.orphan_label_files))

    # 10. Corrupted image handled
    def test_corrupted_image_handled(self) -> None:
        d = self.tmp_dir / "dataset"
        img_dir = d / "train" / "images"
        lbl_dir = d / "train" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create 0-byte corrupt image
        (img_dir / "corrupt.jpg").write_text("not an image", encoding="utf-8")
        (lbl_dir / "corrupt.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        yaml_path = d / "data.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"names": ["helmet", "no_helmet", "no_vest", "vest"]}),
            encoding="utf-8",
        )

        report = validate_ppe_dataset(yaml_path=yaml_path)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("corrupt.jpg" in c for c in report.corrupt_images))

    # 11. Split manifest loads
    def test_split_manifest_loads(self) -> None:
        frames = [
            SourceFrameInfo("f1.jpg", "c1", "clip1", 0.0, 1920, 1080, "g1", Path("f1.jpg")),
            SourceFrameInfo("f2.jpg", "c1", "clip1", 2.0, 1920, 1080, "g1", Path("f2.jpg")),
        ]
        manifest_path = self.tmp_dir / "manifest.csv"
        write_split_manifest(frames, {"f1.jpg": "train", "f2.jpg": "train"}, manifest_path)
        self.assertTrue(manifest_path.exists())

        loaded = load_split_manifest(manifest_path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["source_image"], "f1.jpg")
        self.assertEqual(loaded[0]["split"], "train")
        self.assertEqual(loaded[1]["source_group"], "g1")

    # 12. Source-group leakage detected
    def test_source_group_leakage_detected(self) -> None:
        records_clean = [
            {"source_group": "group_a", "split": "train"},
            {"source_group": "group_b", "split": "val"},
        ]
        self.assertEqual(check_source_group_leakage(records_clean), [])

        records_leaked = [
            {"source_group": "group_a", "split": "train"},
            {"source_group": "group_a", "split": "val"},
        ]
        leaks = check_source_group_leakage(records_leaked)
        self.assertEqual(len(leaks), 1)
        self.assertIn("group_a", leaks[0])

    # 13. Crop preserves source split
    def test_crop_preserves_source_split(self) -> None:
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        frame_info = SourceFrameInfo(
            source_image="frame_01.jpg",
            camera_id="cam_1",
            source_clip="clip_1",
            source_timestamp_seconds=2.0,
            frame_width=1000,
            frame_height=1000,
            source_group="cam1_clip1_b0",
            image_path=Path("frame_01.jpg"),
        )
        dets = [PersonDetection(x1=100, y1=100, x2=300, y2=600, confidence=0.85)]
        out_base = self.tmp_dir / "crops"

        crops = extract_person_crops_from_frame(
            frame=frame,
            frame_info=frame_info,
            detections=dets,
            output_base_dir=out_base,
            split="val",
        )
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].split, "val")
        self.assertTrue((out_base / "val" / crops[0].crop_filename).exists())

    # 14. Crop coordinates clamp to image boundaries
    def test_crop_coordinates_clamp_to_image(self) -> None:
        # Box touching top-left (0, 0)
        x1, y1, x2, y2 = compute_padded_crop_box(
            x1=0, y1=0, x2=200, y2=400, frame_width=1000, frame_height=1000, padding_fraction=0.15
        )
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 0)
        self.assertGreater(x2, 200)
        self.assertGreater(y2, 400)

        # Box touching bottom-right (900, 900, 1000, 1000)
        x1, y1, x2, y2 = compute_padded_crop_box(
            x1=900, y1=900, x2=1000, y2=1000, frame_width=1000, frame_height=1000, padding_fraction=0.15
        )
        self.assertLess(x1, 900)
        self.assertLess(y1, 900)
        self.assertEqual(x2, 1000)
        self.assertEqual(y2, 1000)

    # 15. Crop generated from original image dimensions
    def test_crop_generated_from_original_image_dimensions(self) -> None:
        # Create 1920x1080 native frame with a distinct test patch
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[200:600, 400:700] = 255  # person region

        frame_info = SourceFrameInfo(
            source_image="hd_frame.jpg",
            camera_id="cam_hd",
            source_clip="hd_clip",
            source_timestamp_seconds=1.0,
            frame_width=1920,
            frame_height=1080,
            source_group="g_hd",
            image_path=Path("hd_frame.jpg"),
        )
        dets = [PersonDetection(x1=400, y1=200, x2=700, y2=600, confidence=0.9)]
        out_base = self.tmp_dir / "hd_crops"

        crops = extract_person_crops_from_frame(
            frame=frame,
            frame_info=frame_info,
            detections=dets,
            output_base_dir=out_base,
            split="train",
            padding_fraction=0.10,
        )
        crop_img = cv2.imread(str(crops[0].crop_path))
        self.assertIsNotNone(crop_img)
        # Native dimensions preserved (approx 360x480 with padding), not downscaled to 640
        self.assertEqual(crop_img.shape[0], crops[0].crop_height_px)
        self.assertEqual(crop_img.shape[1], crops[0].crop_width_px)

    # 16. Person detector configuration remains unchanged
    def test_person_detector_configuration_remains_unchanged(self) -> None:
        from src.detection.person_detector import PersonDetector
        # Verify default constants or class init constraints
        self.assertEqual(DEFAULT_BOX_PADDING_FRACTION, 0.12)
        # PersonDetector should enforce imgsz positive and conf in [0, 1]
        with self.assertRaises(ValueError):
            PersonDetector(confidence_threshold=1.5)
        with self.assertRaises(ValueError):
            PersonDetector(image_size=-1)

    # 17. Crop manifest records original dimensions
    def test_crop_manifest_records_original_dimensions(self) -> None:
        info = PersonCropInfo(
            crop_filename="crop1.jpg",
            source_image="src1.jpg",
            camera_id="c1",
            source_clip="clip1",
            source_group="g1",
            split="train",
            person_detection_confidence=0.88,
            person_box_x1=100,
            person_box_y1=100,
            person_box_x2=300,
            person_box_y2=500,
            person_width_px=200,
            person_height_px=400,
            crop_x1=80,
            crop_y1=60,
            crop_x2=320,
            crop_y2=540,
            crop_width_px=240,
            crop_height_px=480,
            crop_path=Path("crop1.jpg"),
        )
        manifest_path = self.tmp_dir / "crop_manifest.csv"
        write_crop_manifest([info], manifest_path)

        loaded = load_crop_manifest(manifest_path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(int(loaded[0]["person_width_px"]), 200)
        self.assertEqual(int(loaded[0]["person_height_px"]), 400)
        self.assertEqual(int(loaded[0]["crop_width_px"]), 240)
        self.assertEqual(int(loaded[0]["crop_height_px"]), 480)

    # 18. Generated dataset paths are portable
    def test_generated_dataset_paths_are_portable(self) -> None:
        yaml_content = Path("config/ppe_dataset.yaml").read_text(encoding="utf-8")
        parsed = yaml.safe_load(yaml_content)
        self.assertNotIn("C:", parsed.get("path", ""))
        self.assertNotIn("\\", parsed.get("path", ""))
        self.assertIn("names", parsed)
        self.assertEqual(len(parsed["names"]), 4)

    # 19. Annotation renderer handles valid labels
    def test_annotation_renderer_handles_valid_labels(self) -> None:
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        boxes = [
            BBoxNormalized(class_id=0, x_center=0.5, y_center=0.2, width=0.2, height=0.1),
            BBoxNormalized(class_id=2, x_center=0.5, y_center=0.6, width=0.4, height=0.4),
        ]
        rendered = render_annotation_overlay(img, boxes)
        self.assertEqual(rendered.shape, (400, 400, 3))
        # Non-zero pixels drawn for overlay
        self.assertGreater(np.count_nonzero(rendered), 0)

    # 20. Empty dataset fails training gate
    def test_empty_dataset_fails_training_gate(self) -> None:
        d = self.tmp_dir / "empty_ds"
        d.mkdir(parents=True)
        yaml_path = d / "data.yaml"
        yaml_path.write_text(
            yaml.safe_dump({
                "names": ["helmet", "no_helmet", "no_vest", "vest"],
                "train": "train",
                "val": "val",
            }),
            encoding="utf-8",
        )
        gate = check_training_gate(yaml_path)
        self.assertFalse(gate.can_train)
        self.assertIn("0 images", gate.reason)

    # 21. Missing validation split fails training gate
    def test_missing_val_split_fails_training_gate(self) -> None:
        d = self.tmp_dir / "no_val_ds"
        (d / "train" / "images").mkdir(parents=True)
        (d / "train" / "labels").mkdir(parents=True)
        (d / "val" / "images").mkdir(parents=True)
        (d / "val" / "labels").mkdir(parents=True)

        # Put 1 image and label in train only
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(d / "train" / "images" / "t1.jpg"), dummy)
        (d / "train" / "labels" / "t1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        yaml_path = d / "data.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"names": ["helmet", "no_helmet", "no_vest", "vest"]}),
            encoding="utf-8",
        )

        gate = check_training_gate(yaml_path)
        self.assertFalse(gate.can_train)
        self.assertIn("validation split is empty", gate.reason)

    # 22. Source-group leakage fails training gate
    def test_source_group_leakage_fails_training_gate(self) -> None:
        d = self.tmp_dir / "leak_ds"
        for split in ("train", "val"):
            (d / split / "images").mkdir(parents=True)
            (d / split / "labels").mkdir(parents=True)
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(d / split / "images" / f"{split}1.jpg"), dummy)
            (d / split / "labels" / f"{split}1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        yaml_path = d / "data.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"names": ["helmet", "no_helmet", "no_vest", "vest"]}),
            encoding="utf-8",
        )

        manifest_path = d / "split_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source_image", "source_group", "split"])
            writer.writeheader()
            # Same source group leaked across train and val
            writer.writerow({"source_image": "train1.jpg", "source_group": "group_alpha", "split": "train"})
            writer.writerow({"source_image": "val1.jpg", "source_group": "group_alpha", "split": "val"})

        gate = check_training_gate(yaml_path, manifest_path=manifest_path)
        self.assertFalse(gate.can_train)
        self.assertIn("leakage", gate.reason.lower())

    # 23. Valid dataset passes training gate
    def test_valid_dataset_passes_training_gate(self) -> None:
        d = self.tmp_dir / "valid_ds"
        for split in ("train", "val"):
            (d / split / "images").mkdir(parents=True)
            (d / split / "labels").mkdir(parents=True)
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(d / split / "images" / f"{split}1.jpg"), dummy)
            (d / split / "labels" / f"{split}1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        yaml_path = d / "data.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"names": ["helmet", "no_helmet", "no_vest", "vest"]}),
            encoding="utf-8",
        )

        manifest_path = d / "split_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source_image", "source_group", "split"])
            writer.writeheader()
            writer.writerow({"source_image": "train1.jpg", "source_group": "group_train", "split": "train"})
            writer.writerow({"source_image": "val1.jpg", "source_group": "group_val", "split": "val"})

        gate = check_training_gate(yaml_path, manifest_path=manifest_path)
        self.assertTrue(gate.can_train)
        self.assertIn("Training gate passed", gate.reason)

    # 24. Training command/config uses separate PPE output directory
    def test_training_config_uses_separate_ppe_output_directory(self) -> None:
        self.assertEqual(DEFAULT_TRAIN_PROJECT, "models/ppe")
        self.assertNotEqual(DEFAULT_TRAIN_PROJECT, ".")
        self.assertNotEqual(DEFAULT_TRAIN_MODEL, DEFAULT_TRAIN_PROJECT)

    # 25. Default training batch remains conservative
    def test_default_training_batch_remains_conservative(self) -> None:
        self.assertEqual(DEFAULT_TRAIN_BATCH, 2)
        self.assertEqual(DEFAULT_TRAIN_IMGSZ, 640)
        self.assertEqual(DEFAULT_TRAIN_WORKERS, 2)

    # 26. No dataset image upload or network action occurs
    def test_no_dataset_image_upload_or_network_action(self) -> None:
        # Verify no external URLs or cloud endpoints are present in constants or validators
        import inspect
        from src.ppe import crop_generator, frame_extractor, trainer, validator, visualizer
        for module in (crop_generator, frame_extractor, trainer, validator, visualizer):
            source = inspect.getsource(module)
            self.assertNotIn("roboflow.com", source)
            self.assertNotIn("api.roboflow", source)
            self.assertNotIn("requests.post", source)
            self.assertNotIn("urllib.request", source)


if __name__ == "__main__":
    unittest.main()
