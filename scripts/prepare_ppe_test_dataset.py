"""Prepare an independent, unlabelled PPE Test crop package; never trains or labels."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from collections import defaultdict

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.person_detector import PersonDetector
from src.ppe.constants import DEFAULT_BOX_PADDING_FRACTION
from src.ppe.crop_generator import extract_person_crops_from_frame, write_crop_manifest
from src.ppe.frame_extractor import extract_frames_from_video, write_frame_manifest
from src.ppe.splitter import write_split_manifest
from src.ppe.validation_review import ReviewCandidate, file_sha256, hamming_distance, perceptual_hash, write_contact_sheets

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inspect_complete_video(video_path: Path) -> dict[str, float | int | bool]:
    """Read every frame so container metadata alone cannot pass this check."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        decoded_frames += 1
    capture.release()
    complete = declared_frames > 0 and decoded_frames == declared_frames
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": declared_frames,
        "duration_seconds": declared_frames / fps if fps > 0 else 0.0,
        "decoded_frame_count": decoded_frames,
        "entire_video_readable": complete,
    }


def source_mentions(dataset_root: Path, clip_id: str) -> list[str]:
    """Return existing CSV/JSON manifest files that already mention a logical clip."""
    mentions: list[str] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json"}:
            continue
        try:
            if clip_id in path.read_text(encoding="utf-8"):
                mentions.append(path.as_posix())
        except UnicodeDecodeError:
            continue
    return mentions


def decoded_digest(path: Path) -> str:
    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        raise ValueError(f"Unreadable image while checking overlap: {path}")
    return hashlib.sha256(image.tobytes()).hexdigest()


def priority_for_crop(crop: object) -> str:
    """Use scale-only review hints; these are deliberately not PPE labels."""
    height = int(getattr(crop, "crop_y2")) - int(getattr(crop, "crop_y1"))
    if height >= 400:
        return "near_worker_candidate"
    if height >= 200:
        return "medium_worker_candidate"
    return "distant_worker_candidate"


def select_temporally_diverse(crops: list[object], maximum: int) -> tuple[list[object], int]:
    """Round-robin 30-second groups and reject conservative visual near-duplicates."""
    grouped: dict[str, list[object]] = defaultdict(list)
    hashes: dict[str, int] = {}
    for crop in crops:
        grouped[str(getattr(crop, "source_group"))].append(crop)
        hashes[str(getattr(crop, "crop_filename"))] = perceptual_hash(getattr(crop, "crop_path"))
    for group in grouped.values():
        group.sort(key=lambda item: (-float(getattr(item, "person_detection_confidence")), str(getattr(item, "crop_filename"))))

    selected: list[object] = []
    rejected_near_duplicates = 0
    cursor = 0
    group_names = sorted(grouped)
    while len(selected) < maximum:
        added_this_round = False
        for group_name in group_names:
            if cursor >= len(grouped[group_name]) or len(selected) >= maximum:
                continue
            candidate = grouped[group_name][cursor]
            candidate_hash = hashes[str(getattr(candidate, "crop_filename"))]
            if any(
                hamming_distance(candidate_hash, hashes[str(getattr(previous, "crop_filename"))]) <= 3
                for previous in selected
            ):
                rejected_near_duplicates += 1
                continue
            selected.append(candidate)
            added_this_round = True
        if not added_this_round and all(cursor >= len(grouped[group]) for group in group_names):
            break
        cursor += 1
    return sorted(selected, key=lambda item: str(getattr(item, "crop_filename"))), rejected_near_duplicates


def image_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def overlap_report(selected: list[object], reference_images: list[Path]) -> dict[str, list[str]]:
    reference_names = {path.name for path in reference_images}
    reference_file_hashes = {file_sha256(path) for path in reference_images}
    reference_decoded_hashes = {decoded_digest(path) for path in reference_images}
    names, files, decoded = [], [], []
    for crop in selected:
        name = str(getattr(crop, "crop_filename"))
        path = getattr(crop, "crop_path")
        if name in reference_names:
            names.append(name)
        if file_sha256(path) in reference_file_hashes:
            files.append(name)
        if decoded_digest(path) in reference_decoded_hashes:
            decoded.append(name)
    return {"filename_collisions": names, "file_hash_overlaps": files, "decoded_image_overlaps": decoded}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an independent, unlabelled PPE Test package; never trains or labels.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--camera-id", default="cam2")
    parser.add_argument("--clip-id", default="cam2_ppe_test_2026-09-15")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-pool", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--maximum-selected", type=int, default=100)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "yolo26n.pt")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Source video does not exist: {args.input}")
    if args.output_dir.exists() or args.review_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing Test preparation or review directory")
    if not args.training_pool.is_dir() or not args.validation_root.is_dir():
        raise FileNotFoundError("Training pool and frozen validation export must both exist")
    existing_mentions = source_mentions(PROJECT_ROOT / "data" / "dataset" / "ppe", args.clip_id)
    if existing_mentions:
        raise RuntimeError(f"Test source already appears in existing dataset manifests: {existing_mentions}")

    video = inspect_complete_video(args.input)
    if not video["entire_video_readable"]:
        raise RuntimeError("Source video did not decode completely; no preparation output was created")

    frames_dir = args.output_dir / "source_frames"
    crops_dir = args.output_dir / "person_crops"
    upload_dir = args.output_dir / "roboflow_upload"
    manifests_dir = args.output_dir / "manifests"
    frames = extract_frames_from_video(args.input, args.camera_id, frames_dir, args.interval_seconds, clip_id=args.clip_id)
    if not frames:
        raise RuntimeError("No source frames were extracted")
    write_frame_manifest(frames, manifests_dir / "source_frames_manifest.csv")
    write_split_manifest(frames, {frame.source_image: "test" for frame in frames}, manifests_dir / "split_manifest.csv")

    detector = PersonDetector(args.model, confidence_threshold=0.20, image_size=960, device=0)
    generated: list[object] = []
    for frame_info in frames:
        frame = cv2.imread(str(frame_info.image_path))
        if frame is None:
            raise RuntimeError(f"Extracted source frame is unreadable: {frame_info.source_image}")
        generated.extend(extract_person_crops_from_frame(
            frame, frame_info, detector.detect(frame), crops_dir, split="test", padding_fraction=DEFAULT_BOX_PADDING_FRACTION,
        ))
    write_crop_manifest(generated, manifests_dir / "all_person_crops_manifest.csv")
    selected, near_duplicate_rejections = select_temporally_diverse(generated, args.maximum_selected)
    if not selected:
        raise RuntimeError("No person crops were available for Test review")

    training_overlap = overlap_report(selected, image_files(args.training_pool))
    validation_overlap = overlap_report(selected, image_files(args.validation_root))
    if any(training_overlap.values()) or any(validation_overlap.values()):
        raise RuntimeError(f"Contamination detected: training={training_overlap}; validation={validation_overlap}")

    upload_dir.mkdir(parents=True, exist_ok=False)
    selected_records = []
    frames_by_name = {frame.source_image: frame for frame in frames}
    for crop in selected:
        source = frames_by_name[getattr(crop, "source_image")]
        shutil.copy2(getattr(crop, "crop_path"), upload_dir / getattr(crop, "crop_filename"))
        selected_records.append({
            "crop_filename": getattr(crop, "crop_filename"),
            "source_frame_filename": getattr(crop, "source_image"),
            "source_frame_number": round(source.source_timestamp_seconds * float(video["fps"])),
            "source_timestamp_seconds": source.source_timestamp_seconds,
            "camera_id": args.camera_id,
            "clip_id": args.clip_id,
            "split": "test",
            "source_group": getattr(crop, "source_group"),
            "original_width": source.frame_width,
            "original_height": source.frame_height,
            "person_box_xyxy": [getattr(crop, key) for key in ("person_box_x1", "person_box_y1", "person_box_x2", "person_box_y2")],
            "crop_box_xyxy": [getattr(crop, key) for key in ("crop_x1", "crop_y1", "crop_x2", "crop_y2")],
            "person_detection_confidence": getattr(crop, "person_detection_confidence"),
            "review_priority": priority_for_crop(crop),
            "sha256": file_sha256(getattr(crop, "crop_path")),
        })
    candidates = [ReviewCandidate(row["crop_filename"], upload_dir / row["crop_filename"], float(row["source_timestamp_seconds"])) for row in selected_records]
    sheets = write_contact_sheets(candidates, args.review_dir, priorities={row["crop_filename"]: row["review_priority"] for row in selected_records})

    manifest = {
        "schema_version": "1.0",
        "purpose": "INDEPENDENT_PPE_TEST_HUMAN_ANNOTATION_PREPARATION_ONLY",
        "camera_id": args.camera_id,
        "clip_id": args.clip_id,
        "split": "test",
        "source_video": {"physical_path": args.input.as_posix(), "physical_filename": args.input.name, "metadata": video},
        "sampling_interval_seconds": args.interval_seconds,
        "sampled_source_frame_count": len(frames),
        "generated_person_crop_count": len(generated),
        "selected_upload_crop_count": len(selected_records),
        "selection_method": "30-second source-group round-robin, confidence ordering, conservative pHash near-duplicate rejection (distance <= 3)",
        "near_duplicate_rejections": near_duplicate_rejections,
        "review_priority_disclaimer": "Review priorities are scale-only visual shortlist hints, not PPE ground-truth labels.",
        "training_pool_exact_overlap": training_overlap,
        "frozen_validation_exact_overlap": validation_overlap,
        "ppe_labels_created": False,
        "ppe_auto_labeling_used": False,
        "records": selected_records,
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "test_preparation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (manifests_dir / "selected_crops_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_records[0]))
        writer.writeheader()
        writer.writerows(selected_records)
    print(json.dumps({"video": video, "source_frames": len(frames), "generated": len(generated), "selected": len(selected_records), "training_overlap": training_overlap, "validation_overlap": validation_overlap, "contact_sheets": len(sheets)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
