"""Read-only structural/provenance audit for a Roboflow PPE validation export."""
from __future__ import annotations

import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from src.ppe.validation_review import perceptual_hash, hamming_distance
from src.ppe.validator import validate_label_line

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = ["helmet", "no_helmet", "no_vest", "vest"]

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def original_name(exported: str) -> str:
    stem = exported.split(".rf.", 1)[0]
    root, extension = stem.rsplit("_", 1)
    return f"{root}.{extension}"

def resolve_manual_provenance(
    record: dict[str, object], *, camera_id: str, clip_id: str, source_video: str
) -> dict[str, object]:
    """Document confirmed source identity without inventing frame-level metadata."""
    resolved = dict(record)
    resolved.update({
        "provenance": "RESOLVED_BY_EXPLICIT_USER_CONFIRMATION",
        "provenance_basis": "explicit_user_confirmation",
        "camera_id": camera_id,
        "clip_id": clip_id,
        "source_video": source_video,
        "split": "validation",
        "source_group": clip_id,
        "source_frame_number": None,
        "source_timestamp_seconds": None,
    })
    return resolved

def main() -> int:
    p = argparse.ArgumentParser(description="Audit a validation-only Roboflow export; never changes images or labels.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--round1", type=Path, required=True)
    p.add_argument("--round2", type=Path, required=True)
    p.add_argument("--training-pool", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--manual-report", type=Path, required=True)
    p.add_argument("--freeze-manifest", type=Path, required=True)
    p.add_argument("--confirm-manual-source-provenance", action="store_true")
    p.add_argument("--manual-camera-id", default="cam1")
    p.add_argument("--manual-clip-id", default="cam1_ppe_validation_2026-09-15")
    p.add_argument("--manual-source-video", default="cam1_ppe_validation_2026-09-15.mp4.mp4")
    a = p.parse_args()
    images_dir, labels_dir = a.root / "valid" / "images", a.root / "valid" / "labels"
    images = sorted(x for x in images_dir.iterdir() if x.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(x for x in labels_dir.iterdir() if x.suffix.lower() == ".txt")
    prepared = {}
    for manifest_path, round_number in ((a.round1, 1), (a.round2, 2)):
        for row in json.loads(manifest_path.read_text(encoding="utf-8"))["records"]:
            prepared[row["crop_filename"]] = {"round": round_number, **row}
    label_names = {x.stem for x in labels}
    missing, corrupt, empty, malformed, duplicate_rows = [], [], [], [], []
    counts = Counter()
    image_hashes, decoded_hashes, phashes = defaultdict(list), defaultdict(list), {}
    exported_records, manual_additions, unresolved_manual = [], [], []
    for image_path in images:
        export_name, source_name = image_path.name, original_name(image_path.name)
        label = labels_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0: corrupt.append(export_name); continue
        file_hash, decoded_hash = digest(image_path), hashlib.sha256(image.tobytes()).hexdigest()
        image_hashes[file_hash].append(export_name); decoded_hashes[decoded_hash].append(export_name); phashes[export_name] = perceptual_hash(image_path)
        record = {"exported_filename": export_name, "original_filename": source_name, "file_sha256": file_hash, "decoded_sha256": decoded_hash}
        if not label.is_file(): missing.append(export_name)
        else:
            rows = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows: empty.append(label.name)
            seen = set()
            for number, row in enumerate(rows, 1):
                try:
                    box = validate_label_line(row, number, label.name)
                    if box.class_id not in range(4): raise ValueError("unexpected class id")
                    counts[NAMES[box.class_id]] += 1
                except ValueError as err: malformed.append(f"{label.name}:{number}: {err}")
                if row in seen: duplicate_rows.append(f"{label.name}:{number}: {row}")
                seen.add(row)
        if source_name in prepared:
            record["prepared_round"] = prepared[source_name]["round"]
            record["provenance"] = "prepared_cam1_clip_manifest"
        else:
            record["timestamp_or_date_from_filename"] = source_name.replace("Screenshot-", "") if source_name.startswith("Screenshot-") else None
            if a.confirm_manual_source_provenance:
                record = resolve_manual_provenance(
                    record,
                    camera_id=a.manual_camera_id,
                    clip_id=a.manual_clip_id,
                    source_video=a.manual_source_video,
                )
            else:
                record["provenance"] = "PROVENANCE_UNRESOLVED"
                record["camera_id"] = None; record["clip_id"] = None
                unresolved_manual.append(record.copy())
            manual_additions.append(record.copy())
        exported_records.append(record)
    orphan = sorted(x.name for x in labels if x.stem not in {x.stem for x in images})
    train_images = [x for x in a.training_pool.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_SUFFIXES]
    train_names = {x.name for x in train_images}; train_file_hashes = {digest(x) for x in train_images}
    train_decoded_hashes = set()
    for path in train_images:
        image = cv2.imread(str(path))
        if image is not None: train_decoded_hashes.add(hashlib.sha256(image.tobytes()).hexdigest())
    training_overlap = [row["exported_filename"] for row in exported_records if row["exported_filename"] in train_names or row["file_sha256"] in train_file_hashes or row["decoded_sha256"] in train_decoded_hashes]
    exact_groups = {"file": [v for v in image_hashes.values() if len(v) > 1], "decoded": [v for v in decoded_hashes.values() if len(v) > 1]}
    near = []
    keys = sorted(phashes)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            distance = hamming_distance(phashes[left], phashes[right])
            if distance <= 3: near.append({"images": [left, right], "hamming_distance": distance})
    round1_matches = sum(row.get("prepared_round") == 1 for row in exported_records)
    round2_matches = sum(row.get("prepared_round") == 2 for row in exported_records)
    structural_ok = not any((missing, corrupt, empty, malformed, orphan, duplicate_rows))
    expected_counts = {"helmet": 20, "no_helmet": 117, "no_vest": 98, "vest": 50}
    freeze = "FROZEN" if (len(images) == 130 and len(labels) == 130 and structural_ok and dict(counts) == expected_counts and not training_overlap and not unresolved_manual) else "PROVISIONAL"
    manual_status = "RESOLVED" if manual_additions and not unresolved_manual else ("PROVENANCE_UNRESOLVED" if unresolved_manual else "NONE")
    manual_report = {"status": manual_status, "images": manual_additions, "reason": "Source camera and clip identity recorded from explicit user confirmation; exact frame and timestamp are unknown." if manual_status == "RESOLVED" else ("Roboflow-export filenames establish screenshot dates/times but no source camera or clip." if unresolved_manual else None)}
    report = {"dataset_name": "ppe_validation_v1", "export_root": a.root.as_posix(), "format": "Ultralytics YOLO detection", "split_counts": {"train": 0, "valid": len(images), "test": 0}, "label_file_count": len(labels), "class_order": NAMES, "class_counts": dict(counts), "total_annotations": sum(counts.values()), "invalid_labels": malformed, "missing_labels": missing, "orphan_labels": orphan, "corrupt_images": corrupt, "empty_label_files": empty, "duplicate_exact_annotation_rows": duplicate_rows, "exact_duplicate_groups": exact_groups, "near_duplicate_candidates": near, "training_pool_exact_overlap": training_overlap, "round1_matches": round1_matches, "round2_matches": round2_matches, "manual_additions": manual_additions, "manual_provenance_status": manual_status, "freeze_status": freeze}
    a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    a.manual_report.parent.mkdir(parents=True, exist_ok=True); a.manual_report.write_text(json.dumps(manual_report, indent=2) + "\n", encoding="utf-8")
    source_group = {"camera_id": a.manual_camera_id, "clip_id": a.manual_clip_id, "split": "validation", "image_count": len(images)} if not unresolved_manual else {"camera_id": "cam1", "clip_id": "cam1_ppe_validation_2026-09-15", "split": "validation", "prepared_images": round1_matches + round2_matches}
    freeze_manifest = {"dataset_name": "ppe_validation_v1", "role": "validation", "image_count": len(images), "label_file_count": len(labels), "class_order": {str(i): n for i,n in enumerate(NAMES)}, "class_counts": dict(counts), "total_annotations": sum(counts.values()), "source_groups": [source_group], "roboflow_version": 1, "preprocessing": "Auto-Orient; 640x640; Fit with black edges / preserve aspect ratio", "augmentation": "none", "training_pool_exact_overlap": len(training_overlap), "freeze_status": freeze, "freeze_reason": "All required provenance documented; manual screenshot source identity is recorded from explicit user confirmation." if not unresolved_manual else "Eight manually added screenshots lack reliable source camera/clip provenance.", "usage_restrictions": ["THIS DATASET MUST NOT BE USED FOR TRAINING.", "THIS DATASET MUST NOT BE USED AS THE FUTURE TEST SET."]}
    a.freeze_manifest.parent.mkdir(parents=True, exist_ok=True); a.freeze_manifest.write_text(json.dumps(freeze_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(images), "labels": len(labels), "counts": dict(counts), "round1": round1_matches, "round2": round2_matches, "manual": len(manual_additions), "manual_provenance": manual_status, "training_overlap": len(training_overlap), "freeze": freeze, "near_pairs": len(near)}, indent=2))
    return 0
if __name__ == "__main__": raise SystemExit(main())
