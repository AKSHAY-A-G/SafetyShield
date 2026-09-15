"""Read-only structural and provenance audit for the held-out PPE Test export."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.validation_review import hamming_distance, perceptual_hash
from src.ppe.validator import validate_label_line

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = ["helmet", "no_helmet", "no_vest", "vest"]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def decoded_digest(path: Path) -> str:
    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        raise ValueError(f"Unreadable image: {path}")
    return hashlib.sha256(image.tobytes()).hexdigest()


def original_name(exported_filename: str) -> str:
    stem = exported_filename.split(".rf.", 1)[0]
    root, extension = stem.rsplit("_", 1)
    return f"{root}.{extension}"


def manual_screenshot_record(exported_filename: str, original_filename: str) -> dict[str, object]:
    """Represent only the user-confirmed live-stream provenance; never infer a camera."""
    return {
        "exported_filename": exported_filename,
        "original_screenshot_filename": original_filename,
        "screenshot_timestamp_from_filename": None,
        "source_type": "live_stream",
        "capture_method": "manual_vlc_screenshot",
        "capture_date": "2026-09-15",
        "source_group": "vlc_live_stream_2026-09-15_manual_screenshots",
        "split": "test",
        "provenance_basis": "explicit_user_confirmation",
        "camera_id": None,
        "source_frame_number": None,
        "source_timestamp_seconds": None,
    }


def image_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def overlap(selected: list[dict[str, object]], references: list[Path]) -> dict[str, list[str]]:
    names = {path.name for path in references}
    file_hashes = {digest(path) for path in references}
    decoded_hashes = {decoded_digest(path) for path in references}
    return {
        "filename_collisions": [str(row["exported_filename"]) for row in selected if str(row["exported_filename"]) in names],
        "file_hash_overlaps": [str(row["exported_filename"]) for row in selected if str(row["file_sha256"]) in file_hashes],
        "decoded_image_overlaps": [str(row["exported_filename"]) for row in selected if str(row["decoded_sha256"]) in decoded_hashes],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit held-out PPE Test data without training or inference.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--training-pool", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manual-report", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--expected-image-count", type=int, required=True)
    parser.add_argument("--expected-counts-json", required=True)
    parser.add_argument("--expected-manual-count", type=int, required=True)
    parser.add_argument("--source-export", type=Path)
    parser.add_argument("--schema-reconciliation-manifest", type=Path)
    args = parser.parse_args()
    if "=" in args.expected_counts_json:
        expected_counts = {
            key: int(value)
            for key, value in (part.split("=", 1) for part in args.expected_counts_json.split(","))
        }
    else:
        expected_counts = json.loads(args.expected_counts_json.replace('\\"', '"'))

    data_yaml = args.root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"No data.yaml at export root: {args.root}")
    dataset_yaml = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    class_mapping = dataset_yaml.get("names")
    class_mapping_ok = class_mapping == NAMES
    images_dir, labels_dir = args.root / "test" / "images", args.root / "test" / "labels"
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(path for path in labels_dir.iterdir() if path.suffix.lower() == ".txt")
    train_images = image_files(args.root / "train") if (args.root / "train").exists() else []
    valid_images = image_files(args.root / "valid") if (args.root / "valid").exists() else []

    prepared = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))["records"]
    prepared_by_name = {str(row["crop_filename"]): row for row in prepared}
    records, manual_records = [], []
    missing, corrupt, empty, malformed, duplicate_rows = [], [], [], [], []
    file_groups: dict[str, list[str]] = defaultdict(list)
    decoded_groups: dict[str, list[str]] = defaultdict(list)
    hashes: dict[str, int] = {}
    counts: Counter[str] = Counter()
    for image_path in images:
        exported = image_path.name
        source = original_name(exported)
        try:
            decoded = decoded_digest(image_path)
        except ValueError:
            corrupt.append(exported)
            continue
        file_hash = digest(image_path)
        file_groups[file_hash].append(exported)
        decoded_groups[decoded].append(exported)
        hashes[exported] = perceptual_hash(image_path)
        record: dict[str, object] = {
            "exported_filename": exported,
            "original_filename": source,
            "file_sha256": file_hash,
            "decoded_sha256": decoded,
        }
        if source in prepared_by_name:
            record.update({
                "camera_id": "cam2",
                "clip_id": "cam2_ppe_test_2026-09-15",
                "split": "test",
                "source_group": "cam2_ppe_test_2026-09-15",
                "source_type": "recorded_video",
                "provenance": "test_preparation_manifest",
            })
        elif "_manual_" in source:
            record.update(manual_screenshot_record(exported, source))
            manual_records.append(record.copy())
        else:
            record["provenance"] = "UNRESOLVED"
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing.append(exported)
        else:
            rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                empty.append(label_path.name)
            seen = set()
            for line_number, row in enumerate(rows, 1):
                try:
                    box = validate_label_line(row, line_number, label_path.name)
                    if box.class_id not in range(4):
                        raise ValueError("unexpected class ID")
                    counts[NAMES[box.class_id]] += 1
                except ValueError as error:
                    malformed.append(f"{label_path.name}:{line_number}: {error}")
                if row in seen:
                    duplicate_rows.append(f"{label_path.name}:{line_number}: {row}")
                seen.add(row)
        records.append(record)
    orphan = sorted(path.name for path in labels if path.stem not in {image.stem for image in images})
    actual_counts = {name: counts[name] for name in NAMES}
    prepared_matches = sum(row.get("provenance") == "test_preparation_manifest" for row in records)
    unresolved = [row["exported_filename"] for row in records if row.get("provenance") == "UNRESOLVED"]
    training_overlap = overlap(records, image_files(args.training_pool))
    validation_overlap = overlap(records, image_files(args.validation_root))
    exact_duplicates = {
        "file": [group for group in file_groups.values() if len(group) > 1],
        "decoded": [group for group in decoded_groups.values() if len(group) > 1],
    }
    near = []
    names = sorted(hashes)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            distance = hamming_distance(hashes[left], hashes[right])
            if distance <= 3:
                near.append({"images": [left, right], "hamming_distance": distance})
    structural_ok = not any((missing, corrupt, empty, malformed, orphan, duplicate_rows))
    no_external_overlap = not any(training_overlap.values()) and not any(validation_overlap.values())
    freeze = "FROZEN" if (
        len(images) == args.expected_image_count and not train_images and not valid_images and len(labels) == args.expected_image_count
        and class_mapping_ok and structural_ok and actual_counts == expected_counts
        and no_external_overlap and prepared_matches == args.expected_image_count and len(manual_records) == args.expected_manual_count and not unresolved
    ) else "BLOCKED"
    for record in manual_records:
        record["training_overlap_status"] = "NONE" if not any(training_overlap.values()) else "REVIEW_REQUIRED"
        record["validation_overlap_status"] = "NONE" if not any(validation_overlap.values()) else "REVIEW_REQUIRED"
    manual_report = {
        "status": "NONE" if not manual_records else ("RESOLVED" if len(manual_records) == args.expected_manual_count else "UNRESOLVED"),
        "camera_id": None,
        "reason": "No reliable project metadata established the camera behind the user-confirmed VLC live stream.",
        "images": manual_records,
    }
    report = {
        "dataset_name": args.dataset_name, "export_root": args.root.as_posix(), "format": "Ultralytics YOLO detection",
        "data_yaml": data_yaml.as_posix(), "split_counts": {"train": len(train_images), "valid": len(valid_images), "test": len(images)},
        "label_file_count": len(labels), "class_order": class_mapping, "class_mapping_ok": class_mapping_ok,
        "class_counts": actual_counts, "total_annotations": sum(counts.values()), "invalid_labels": malformed,
        "missing_labels": missing, "orphan_labels": orphan, "corrupt_images": corrupt, "empty_label_files": empty,
        "duplicate_exact_annotation_rows": duplicate_rows, "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_candidates": near, "training_pool_exact_overlap": training_overlap,
        "frozen_validation_exact_overlap": validation_overlap, "prepared_cam2_manifest_matches": prepared_matches,
        "manual_vlc_screenshots": manual_records, "unresolved_images": unresolved, "freeze_status": freeze,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.manual_report.parent.mkdir(parents=True, exist_ok=True)
    args.manual_report.write_text(json.dumps(manual_report, indent=2) + "\n", encoding="utf-8")
    if freeze == "FROZEN":
        manifest = {
            "dataset_name": args.dataset_name, "role": "test", "freeze_status": "FROZEN", "image_count": len(images),
            "label_file_count": len(labels), "class_order": {str(index): name for index, name in enumerate(NAMES)},
            "class_counts": actual_counts, "total_annotations": sum(counts.values()),
            "source_groups": ([
                {"camera_id": "cam2", "clip_id": "cam2_ppe_test_2026-09-15", "source_type": "recorded_video", "image_count": args.expected_image_count, "split": "test"},
            ] if not manual_records else [
                {"camera_id": "cam2", "clip_id": "cam2_ppe_test_2026-09-15", "source_type": "recorded_video", "image_count": prepared_matches, "split": "test"},
                {"camera_id": None, "source_group": "vlc_live_stream_2026-09-15_manual_screenshots", "source_type": "live_stream", "capture_method": "manual_vlc_screenshot", "capture_date": "2026-09-15", "image_count": len(manual_records), "split": "test", "provenance_basis": "explicit_user_confirmation"},
            ]),
            "training_pool_exact_overlap": 0, "validation_exact_overlap": 0,
            "canonical_export_root": args.root.as_posix(),
            "source_export": args.source_export.as_posix() if args.source_export else None,
            "schema_reconciliation": "VERIFIED" if args.schema_reconciliation_manifest else None,
            "schema_reconciliation_manifest": args.schema_reconciliation_manifest.as_posix() if args.schema_reconciliation_manifest else None,
            "preprocessing": "Auto-Orient; 640x640; Fit with black edges / preserve aspect ratio", "augmentation": "none",
            "usage_restrictions": [
                "THIS DATASET MUST NOT BE USED FOR TRAINING.",
                "THIS DATASET MUST NOT BE USED FOR VALIDATION OR MODEL SELECTION.",
                "THIS DATASET MUST NOT BE USED FOR THRESHOLD TUNING.",
                "TEST EVALUATION IS PERMITTED ONLY AFTER MODEL AND OPERATING SETTINGS ARE LOCKED USING TRAINING + VALIDATION.",
            ],
            "limitations": (["helmet ground-truth instances = 0", "Helmet held-out precision/recall cannot be established from this Test version.", "Future independent helmet-rich Test footage may be needed for broader class coverage."] if counts.get("helmet", 0) == 0 else [f"helmet sample count = {counts.get('helmet', 0)}"]),
        }
        args.freeze_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.freeze_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(images), "labels": len(labels), "counts": actual_counts, "prepared_matches": prepared_matches, "manual": len(manual_records), "training_overlap": training_overlap, "validation_overlap": validation_overlap, "freeze": freeze, "near_pairs": len(near)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
