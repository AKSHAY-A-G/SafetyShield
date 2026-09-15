"""Prepare one independent, unlabelled PPE validation crop package; never trains."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.person_detector import PersonDetector
from src.ppe.constants import DEFAULT_BOX_PADDING_FRACTION
from src.ppe.crop_generator import extract_person_crops_from_frame, write_crop_manifest
from src.ppe.frame_extractor import extract_frames_from_video, write_frame_manifest
from src.ppe.models import SourceFrameInfo
from src.ppe.splitter import write_split_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_video(video_path: Path) -> dict[str, float | int | bool]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        decoded_frames += 1
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": declared_frames,
        "duration_seconds": declared_frames / fps if fps > 0 else 0.0,
        "decoded_frame_count": decoded_frames,
        "entire_video_readable": decoded_frames == declared_frames,
    }


def select_non_repetitive(crops: list, maximum: int) -> list:
    """Keep high-confidence candidates, limiting each 30 s source group to 25 crops.

    Sampling is already two seconds apart.  The group cap further avoids a
    single continuous view dominating the manual annotation batch without
    inventing worker identity or tracking across frames.
    """
    selected = []
    group_counts: dict[str, int] = {}
    for crop in sorted(crops, key=lambda item: (-item.person_detection_confidence, item.crop_filename)):
        if len(selected) >= maximum:
            break
        if group_counts.get(crop.source_group, 0) >= 25:
            continue
        selected.append(crop)
        group_counts[crop.source_group] = group_counts.get(crop.source_group, 0) + 1
    return sorted(selected, key=lambda item: item.crop_filename)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an independent unlabelled PPE validation package; no training occurs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-pool", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--maximum-selected", type=int, default=100)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "yolo26n.pt")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Source video does not exist: {args.input}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing validation output: {args.output_dir}")
    if not args.training_pool.is_dir():
        raise FileNotFoundError(f"Training pool does not exist: {args.training_pool}")

    video = inspect_video(args.input)
    if not video["entire_video_readable"]:
        raise RuntimeError("Source video did not decode completely; no preparation output was created")

    frames_dir = args.output_dir / "source_frames"
    manifests_dir = args.output_dir / "manifests"
    crops_dir = args.output_dir / "crops"
    upload_dir = args.output_dir / "roboflow_upload"
    reports_dir = args.output_dir / "reports"
    frames = extract_frames_from_video(
        args.input, args.camera_id, frames_dir, args.interval_seconds, clip_id=args.clip_id
    )
    if not frames:
        raise RuntimeError("No source frames were extracted")
    write_frame_manifest(frames, manifests_dir / "source_frames_manifest.csv")
    write_split_manifest(frames, {frame.source_image: "validation" for frame in frames}, manifests_dir / "split_manifest.csv")

    detector = PersonDetector(args.model, confidence_threshold=0.20, image_size=960, device=0)
    generated = []
    for frame_info in frames:
        frame = cv2.imread(str(frame_info.image_path))
        if frame is None:
            raise RuntimeError(f"Extracted source frame is unreadable: {frame_info.source_image}")
        generated.extend(extract_person_crops_from_frame(
            frame, frame_info, detector.detect(frame), crops_dir, split="validation",
            padding_fraction=DEFAULT_BOX_PADDING_FRACTION,
        ))
    write_crop_manifest(generated, manifests_dir / "all_person_crops_manifest.csv")
    selected = select_non_repetitive(generated, args.maximum_selected)

    training_images = [path for path in args.training_pool.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    training_names = {path.name for path in training_images}
    training_hashes = {sha256(path) for path in training_images}
    collisions = [crop.crop_filename for crop in selected if crop.crop_filename in training_names]
    overlaps = [crop.crop_filename for crop in selected if sha256(crop.crop_path) in training_hashes]
    if collisions or overlaps:
        raise RuntimeError(f"Training-pool contamination detected; filename collisions={collisions}, exact hash overlaps={overlaps}")

    upload_dir.mkdir(parents=True, exist_ok=False)
    for crop in selected:
        shutil.copy2(crop.crop_path, upload_dir / crop.crop_filename)

    selected_rows = []
    for crop in selected:
        source = next(frame for frame in frames if frame.source_image == crop.source_image)
        selected_rows.append({
            "crop_filename": crop.crop_filename,
            "source_frame_filename": crop.source_image,
            "camera_id": crop.camera_id,
            "clip_id": crop.source_clip,
            "source_frame_number": round(source.source_timestamp_seconds * float(video["fps"])),
            "source_timestamp_seconds": source.source_timestamp_seconds,
            "original_width": source.frame_width,
            "original_height": source.frame_height,
            "person_box_xyxy": [crop.person_box_x1, crop.person_box_y1, crop.person_box_x2, crop.person_box_y2],
            "crop_box_xyxy": [crop.crop_x1, crop.crop_y1, crop.crop_x2, crop.crop_y2],
            "person_detection_confidence": crop.person_detection_confidence,
            "source_group": crop.source_group,
            "split": "validation",
            "sha256": sha256(crop.crop_path),
        })
    manifest = {
        "schema_version": "1.0",
        "purpose": "INDEPENDENT_VALIDATION_ANNOTATION_PREPARATION_ONLY",
        "source_video_filename": args.input.name,
        "source_video_path": args.input.as_posix(),
        "camera_id": args.camera_id,
        "clip_id": args.clip_id,
        "split": "validation",
        "sampling_interval_seconds": args.interval_seconds,
        "source_video": video,
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_crop_count": len(selected_rows),
        "training_pool": args.training_pool.as_posix(),
        "not_part_of_training_pool": True,
        "training_pool_filename_collisions": collisions,
        "training_pool_exact_hash_overlaps": overlaps,
        "ppe_labels": "NOT_YET_CREATED",
        "records": selected_rows,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    csv_fields = [
        "crop_filename", "source_frame_filename", "camera_id", "clip_id", "source_frame_number",
        "source_timestamp_seconds", "original_width", "original_height", "person_box_xyxy",
        "crop_box_xyxy", "person_detection_confidence", "source_group", "split", "sha256",
    ]
    with (manifests_dir / "selected_crops_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(selected_rows)
    (reports_dir / "preparation_report.json").write_text(json.dumps({
        "video": video, "source_frames_sampled": len(frames), "person_crops_generated": len(generated),
        "person_crops_selected": len(selected), "training_pool_overlap": 0,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"video": video, "source_frames": len(frames), "generated": len(generated), "selected": len(selected), "overlap": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
