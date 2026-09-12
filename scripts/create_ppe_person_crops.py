"""Generate original-resolution padded person crops from extracted CCTV frames using PersonDetector."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.person_detector import PersonDetector
from src.ppe.constants import DEFAULT_BOX_PADDING_FRACTION
from src.ppe.crop_generator import extract_person_crops_from_frame, write_crop_manifest
from src.ppe.models import PersonCropInfo, SourceFrameInfo
from src.ppe.splitter import load_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate padded person crops from original-resolution CCTV frames for PPE dataset."
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "source_frames",
        help="Directory containing native-resolution extracted frames.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "manifests" / "split_manifest.csv",
        help="Path to split manifest CSV with source-group assignments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "crops",
        help="Destination directory for generated person crops.",
    )
    parser.add_argument(
        "--crop-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "manifests" / "crop_manifest.csv",
        help="Path to write crop metadata manifest CSV.",
    )
    parser.add_argument(
        "--padding-fraction",
        type=float,
        default=DEFAULT_BOX_PADDING_FRACTION,
        help="Relative box padding fraction (default: 0.12 = 12%%).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "yolo26n.pt",
        help="Path to YOLO nano model checkpoint for person detection.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index for person detector.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.frames_dir.exists():
        print(f"Error: Frames directory not found: {args.frames_dir}", file=sys.stderr)
        return 1

    # Load split manifest if present to preserve source-group splits
    split_lookup: dict[str, dict[str, str]] = {}
    if args.split_manifest.exists():
        records = load_split_manifest(args.split_manifest)
        for r in records:
            split_lookup[r["source_image"]] = r
        print(f"Loaded split assignments for {len(split_lookup)} frames from: {args.split_manifest.name}")
    else:
        print("Note: Split manifest not found; defaulting all crops to 'train' split.")

    # Initialize person detector with accepted settings
    print(f"Initializing PersonDetector (model: {args.model.name}, imgsz=960, conf=0.20, CUDA device={args.device})...")
    try:
        detector = PersonDetector(
            checkpoint=args.model,
            confidence_threshold=0.20,
            image_size=960,
            device=args.device,
        )
    except Exception as err:
        print(f"Failed to initialize person detector on GPU: {err}", file=sys.stderr)
        return 1

    frame_files = sorted(
        list(args.frames_dir.glob("*.jpg")) + list(args.frames_dir.glob("*.png"))
    )
    if not frame_files:
        print(f"No image frames found in: {args.frames_dir}", file=sys.stderr)
        return 1

    print(f"Processing {len(frame_files)} source frames for person crop extraction...")
    all_crops: list[PersonCropInfo] = []

    for f_idx, frame_path in enumerate(frame_files, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        manifest_meta = split_lookup.get(frame_path.name, {})
        split = manifest_meta.get("split", "train")
        source_group = manifest_meta.get("source_group", "unknown_group")
        camera_id = manifest_meta.get("camera_id", "unknown_cam")
        source_clip = manifest_meta.get("source_clip", frame_path.stem)
        timestamp_sec = float(manifest_meta.get("source_timestamp_seconds", 0.0))

        frame_info = SourceFrameInfo(
            source_image=frame_path.name,
            camera_id=camera_id,
            source_clip=source_clip,
            source_timestamp_seconds=timestamp_sec,
            frame_width=w,
            frame_height=h,
            source_group=source_group,
            image_path=frame_path,
        )

        # Detect persons in frame
        detections = detector.detect(frame)
        if not detections:
            continue

        crops = extract_person_crops_from_frame(
            frame=frame,
            frame_info=frame_info,
            detections=detections,
            output_base_dir=args.output_dir,
            split=split,
            padding_fraction=args.padding_fraction,
        )
        all_crops.extend(crops)

        if f_idx % 10 == 0 or f_idx == len(frame_files):
            print(f"  Processed {f_idx}/{len(frame_files)} frames -> {len(all_crops)} person crops generated.")

    print(f"Completed person crop extraction: {len(all_crops)} crops generated.")
    write_crop_manifest(all_crops, args.crop_manifest)
    print(f"Saved crop manifest to: {args.crop_manifest}")

    # Summary by split
    split_counts: dict[str, int] = {}
    for c in all_crops:
        split_counts[c.split] = split_counts.get(c.split, 0) + 1
    print(f"Crops by split: {split_counts}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
