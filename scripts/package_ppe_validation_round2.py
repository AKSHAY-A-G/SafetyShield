"""Copy an explicitly human-reviewed PPE validation Round-2 batch; never labels or trains."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.validation_review import ReviewCandidate, file_sha256, write_contact_sheets


def main() -> int:
    parser = argparse.ArgumentParser(description="Package explicit unlabelled Round-2 PPE validation selections.")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument("--all-crops-manifest", type=Path, required=True)
    parser.add_argument("--source-frames-manifest", type=Path, required=True)
    parser.add_argument("--round-one-manifest", type=Path, required=True)
    parser.add_argument("--training-pool", type=Path, required=True)
    parser.add_argument("--upload-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.upload_dir.exists() or args.manifest.exists():
        raise FileExistsError("Refusing to overwrite an existing Round-2 upload directory or manifest")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if not isinstance(selection, list) or not selection:
        raise ValueError("Selection must be a non-empty JSON list")
    selected_names = [entry["crop_filename"] for entry in selection]
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("Round-2 selection contains duplicate crop filenames")
    allowed_priorities = {"helmet_candidate", "vest_candidate", "helmet_and_vest_candidate", "general_validation_candidate"}
    if any(entry.get("review_priority") not in allowed_priorities for entry in selection):
        raise ValueError("Selection has an unsupported review priority")

    with args.all_crops_manifest.open(encoding="utf-8") as stream:
        all_rows = {row["crop_filename"]: row for row in csv.DictReader(stream)}
    round_one = json.loads(args.round_one_manifest.read_text(encoding="utf-8"))
    first_names = {record["crop_filename"] for record in round_one["records"]}
    if set(selected_names) & first_names:
        raise ValueError("Round-2 selection overlaps Round-1 validation crops")
    missing = set(selected_names) - set(all_rows)
    if missing:
        raise ValueError(f"Selection contains crops not present in the generated manifest: {sorted(missing)}")
    training_images = [path for path in args.training_pool.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    training_names = {path.name for path in training_images}
    training_hashes = {file_sha256(path) for path in training_images}

    with args.source_frames_manifest.open(encoding="utf-8") as stream:
        source_records = {row["source_image"]: row for row in csv.DictReader(stream)}
    prepared = []
    for entry in selection:
        row = all_rows[entry["crop_filename"]]
        source = source_records.get(row["source_image"])
        if source is None:
            raise ValueError(f"No source-frame provenance is available for {row['source_image']}")
        crop_path = args.crops_dir / row["crop_filename"]
        digest = file_sha256(crop_path)
        if row["crop_filename"] in training_names or digest in training_hashes:
            raise ValueError(f"Training-pool overlap detected for {row['crop_filename']}")
        prepared.append({
            "crop_filename": row["crop_filename"],
            "original_crop_path": crop_path.as_posix(),
            "source_frame_filename": row["source_image"],
            "camera_id": source["camera_id"],
            "clip_id": source["source_clip"],
            "source_frame_number": round(float(source["source_timestamp_seconds"]) * 30.0),
            "source_timestamp_seconds": float(source["source_timestamp_seconds"]),
            "original_width": int(source["frame_width"]),
            "original_height": int(source["frame_height"]),
            "person_box_xyxy": [int(row[key]) for key in ("person_box_x1", "person_box_y1", "person_box_x2", "person_box_y2")],
            "split": "validation",
            "selection_round": 2,
            "review_priority": entry["review_priority"],
            "sha256": digest,
        })
    args.upload_dir.mkdir(parents=True)
    for record in prepared:
        shutil.copy2(record["original_crop_path"], args.upload_dir / record["crop_filename"])
    priority_counts = {priority: sum(record["review_priority"] == priority for record in prepared) for priority in sorted(allowed_priorities)}
    manifest = {
        "schema_version": "1.0",
        "purpose": "INDEPENDENT_VALIDATION_ROUND_2_HUMAN_ANNOTATION_PREPARATION_ONLY",
        "camera_id": "cam1",
        "clip_id": "cam1_ppe_validation_2026-09-15",
        "split": "validation",
        "selection_round": 2,
        "review_priority_disclaimer": "Review priorities are visual shortlist hints only, not ground-truth PPE labels.",
        "ppe_labels_created": False,
        "round_one_overlap": 0,
        "training_pool_filename_overlap": 0,
        "training_pool_exact_hash_overlap": 0,
        "selected_crop_count": len(prepared),
        "review_priority_counts": priority_counts,
        "records": prepared,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    review_dir = PROJECT_ROOT / "outputs" / "ppe_validation_round2_review" / "selected_round2"
    if review_dir.exists():
        raise FileExistsError(f"Refusing to overwrite selected Round-2 review sheets: {review_dir}")
    sheets = write_contact_sheets(
        [ReviewCandidate(record["crop_filename"], args.crops_dir / record["crop_filename"], float(record["source_timestamp_seconds"])) for record in prepared],
        review_dir,
        priorities={record["crop_filename"]: record["review_priority"] for record in prepared},
    )
    manifest["selected_review_contact_sheets"] = [path.as_posix() for path in sheets]
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(prepared), "priority_counts": priority_counts, "training_overlap": 0, "round_one_overlap": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
