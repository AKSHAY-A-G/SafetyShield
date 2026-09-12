"""Comprehensive YOLO-format dataset validator for SafetyShield PPE models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import yaml

from src.ppe.constants import PPE_CLASSES
from src.ppe.models import BBoxNormalized, DatasetValidationReport
from src.ppe.splitter import check_source_group_leakage, load_split_manifest


def validate_label_line(line: str, line_no: int, filename: str) -> BBoxNormalized:
    """Validate a single YOLO-format normalized bbox annotation line.

    Expected format: <class_id> <x_center> <y_center> <width> <height>
    Coordinates must be normalized in [0, 1] with strictly positive width and height.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(
            f"{filename}:{line_no} - Malformed label line: expected 5 values, got {len(parts)} ('{line.strip()}')"
        )

    # Validate class_id
    try:
        class_id = int(parts[0])
    except ValueError:
        raise ValueError(f"{filename}:{line_no} - Invalid class_id '{parts[0]}': must be an integer")

    if class_id not in PPE_CLASSES:
        raise ValueError(
            f"{filename}:{line_no} - Unexpected class_id {class_id}. "
            f"Allowed classes are {dict(PPE_CLASSES)}"
        )

    # Validate coordinates
    try:
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    except ValueError as err:
        raise ValueError(f"{filename}:{line_no} - Non-numeric coordinate: {err}") from err

    for name, val in [("x_center", x_center), ("y_center", y_center), ("width", width), ("height", height)]:
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"{filename}:{line_no} - Invalid coordinate '{name}' is NaN or Inf")

    if x_center < 0.0 or x_center > 1.0:
        raise ValueError(f"{filename}:{line_no} - x_center={x_center:.4f} is outside legal range [0.0, 1.0]")
    if y_center < 0.0 or y_center > 1.0:
        raise ValueError(f"{filename}:{line_no} - y_center={y_center:.4f} is outside legal range [0.0, 1.0]")

    if width <= 0.0 or width > 1.0:
        raise ValueError(f"{filename}:{line_no} - width={width:.4f} must be positive and <= 1.0")
    if height <= 0.0 or height > 1.0:
        raise ValueError(f"{filename}:{line_no} - height={height:.4f} must be positive and <= 1.0")

    return BBoxNormalized(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


def validate_dataset_yaml(yaml_path: Path) -> dict:
    """Validate Ultralytics-compatible dataset YAML structure and class names."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as err:
        raise ValueError(f"Malformed YAML in {yaml_path}: {err}") from err

    if not isinstance(content, dict):
        raise ValueError(f"Dataset YAML must be a mapping, got {type(content).__name__}")

    if "names" not in content:
        raise ValueError(f"Missing 'names' key in dataset YAML: {yaml_path}")

    names = content["names"]
    if isinstance(names, dict):
        parsed_names = [names[k] for k in sorted(names.keys())]
    elif isinstance(names, list):
        parsed_names = list(names)
    else:
        raise ValueError(f"'names' must be a list or dict in {yaml_path}")

    expected_names = [PPE_CLASSES[i] for i in range(len(PPE_CLASSES))]
    if parsed_names != expected_names:
        raise ValueError(
            f"Invalid class names in {yaml_path}.\n"
            f"Expected exactly: {expected_names}\n"
            f"Found:            {parsed_names}"
        )

    return content


def validate_ppe_dataset(
    dataset_dir: Path | None = None,
    yaml_path: Path | None = None,
    manifest_path: Path | None = None,
) -> DatasetValidationReport:
    """Perform comprehensive validation on a YOLO-format PPE dataset directory or YAML configuration."""
    if yaml_path is not None and yaml_path.exists():
        try:
            cfg = validate_dataset_yaml(yaml_path)
            resolved_yaml = str(yaml_path)
            root_dir = yaml_path.parent
        except Exception as err:
            return DatasetValidationReport(
                is_valid=False,
                dataset_yaml=str(yaml_path),
                fatal_errors=[f"Dataset YAML validation failed: {err}"],
            )
    elif dataset_dir is not None and dataset_dir.exists():
        # Look for data.yaml in dataset_dir
        candidate_yaml = dataset_dir / "data.yaml"
        if candidate_yaml.exists():
            return validate_ppe_dataset(yaml_path=candidate_yaml, manifest_path=manifest_path)
        resolved_yaml = "None"
        root_dir = dataset_dir
    else:
        return DatasetValidationReport(
            is_valid=False,
            dataset_yaml="None",
            fatal_errors=["Neither valid dataset_dir nor yaml_path was provided"],
        )

    report = DatasetValidationReport(
        is_valid=True,
        dataset_yaml=resolved_yaml,
        classes=[PPE_CLASSES[i] for i in range(len(PPE_CLASSES))],
    )

    # Initialize count tracking
    for c_name in report.classes:
        report.instances_per_class[c_name] = 0

    splits = ["train", "val", "test"]
    split_images: dict[str, list[Path]] = {}
    split_labels: dict[str, list[Path]] = {}
    all_image_stems: dict[str, str] = {}  # stem -> split (collision check)

    for split in splits:
        report.images_per_split[split] = 0
        report.instances_per_split_class[split] = {c_name: 0 for c_name in report.classes}

        # Ultralytics supports root/images/split or root/split/images or root/split
        img_dir = root_dir / "images" / split
        lbl_dir = root_dir / "labels" / split

        if not img_dir.exists():
            img_dir = root_dir / split / "images"
            lbl_dir = root_dir / split / "labels"
        if not img_dir.exists():
            img_dir = root_dir / split
            lbl_dir = root_dir / split

        img_files = []
        if img_dir.exists():
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                img_files.extend(list(img_dir.glob(f"*{ext}")))
                img_files.extend(list(img_dir.glob(f"*{ext.upper()}")))

        split_images[split] = sorted(img_files)
        report.images_per_split[split] = len(img_files)

        # Check filename collisions across splits
        for img_path in img_files:
            stem = img_path.stem
            if stem in all_image_stems:
                report.warnings.append(
                    f"Duplicate filename stem '{stem}' present in both '{all_image_stems[stem]}' and '{split}'"
                )
            else:
                all_image_stems[stem] = split

        # Find labels
        lbl_files = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        split_labels[split] = sorted(lbl_files)

        # Build label lookup
        lbl_by_stem = {p.stem: p for p in lbl_files}
        img_by_stem = {p.stem: p for p in img_files}

        # Check orphan labels (label exists, but image missing)
        for stem, l_path in lbl_by_stem.items():
            if stem not in img_by_stem:
                report.orphan_label_files.append(str(l_path))
                report.warnings.append(f"Orphan label file without image: {l_path.name}")

        # Check each image and its label
        for img_path in img_files:
            stem = img_path.stem
            l_path = lbl_by_stem.get(stem)

            # Test image readability with OpenCV
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                report.corrupt_images.append(str(img_path))
                report.fatal_errors.append(f"Corrupt or unreadable image: {img_path.name}")
                continue

            if l_path is None or not l_path.exists():
                report.missing_label_files.append(str(img_path))
                report.warnings.append(f"Image has missing label file: {img_path.name}")
                continue

            # Read label file
            try:
                content = l_path.read_text(encoding="utf-8").strip()
            except Exception as err:
                report.fatal_errors.append(f"Cannot read label file {l_path.name}: {err}")
                continue

            if not content:
                report.empty_label_files.append(str(l_path))
                continue

            lines = [line.strip() for line in content.splitlines() if line.strip()]
            seen_boxes: set[str] = set()

            for line_no, line in enumerate(lines, start=1):
                try:
                    box = validate_label_line(line, line_no=line_no, filename=l_path.name)
                except ValueError as val_err:
                    report.fatal_errors.append(str(val_err))
                    continue

                class_name = PPE_CLASSES[box.class_id]
                report.total_instances += 1
                report.instances_per_class[class_name] += 1
                report.instances_per_split_class[split][class_name] += 1

                # Check exact duplicate rows
                if line in seen_boxes:
                    report.duplicate_boxes.append(f"{l_path.name}:{line_no} '{line}'")
                    report.warnings.append(f"Duplicate exact box in {l_path.name} line {line_no}: {line}")
                seen_boxes.add(line)

    report.total_images = sum(report.images_per_split.values())

    # Source-group leakage check via split manifest if present
    if manifest_path is not None and manifest_path.exists():
        try:
            records = load_split_manifest(manifest_path)
            leakage_errors = check_source_group_leakage(records)
            if leakage_errors:
                report.source_group_leakage.extend(leakage_errors)
                for lk in leakage_errors:
                    report.fatal_errors.append(f"LEAKAGE: {lk}")
        except Exception as man_err:
            report.warnings.append(f"Failed to inspect split manifest {manifest_path}: {man_err}")

    if report.fatal_errors:
        report.is_valid = False

    return report
