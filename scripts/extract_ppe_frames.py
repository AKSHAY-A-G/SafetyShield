"""Extract native-resolution video frames from local authorized recorded videos for PPE dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.frame_extractor import extract_frames_from_video, write_frame_manifest
from src.ppe.splitter import assign_source_group_splits, write_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract native-resolution frames from local recorded video for PPE dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw_videos" / "cam_good_test.mp4",
        help="Path to local recorded video file.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="cam_good_test",
        help="Camera identifier for source-group metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "source_frames",
        help="Destination directory for extracted frame images.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=2.0,
        help="Sampling interval in seconds between extracted frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to extract.",
    )
    parser.add_argument(
        "--block-duration-seconds",
        type=float,
        default=30.0,
        help="Duration in seconds of contiguous time blocks for source grouping.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "manifests" / "source_frames_manifest.csv",
        help="Path to write extracted frames manifest CSV.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "manifests" / "split_manifest.csv",
        help="Path to write split assignments manifest CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input video does not exist: {args.input}", file=sys.stderr)
        return 1

    print(f"Extracting native-resolution frames from: {args.input.name}")
    print(f"Camera ID: {args.camera_id}")
    print(f"Sampling interval: {args.interval_seconds}s (block duration: {args.block_duration_seconds}s)")
    print(f"Output directory: {args.output_dir}")

    try:
        frames = extract_frames_from_video(
            video_path=args.input,
            camera_id=args.camera_id,
            output_dir=args.output_dir,
            interval_seconds=args.interval_seconds,
            max_frames=args.max_frames,
            block_duration_seconds=args.block_duration_seconds,
        )
    except Exception as err:
        print(f"Extraction failed: {err}", file=sys.stderr)
        return 1

    print(f"Extracted {len(frames)} native-resolution frames.")
    write_frame_manifest(frames, args.manifest)
    print(f"Saved frames manifest to: {args.manifest}")

    # Generate source-group split assignments if multiple groups exist
    unique_groups = {f.source_group for f in frames}
    print(f"Identified {len(unique_groups)} source group(s): {sorted(list(unique_groups))}")

    if len(unique_groups) >= 3:
        img_to_split, group_to_split = assign_source_group_splits(frames)
        write_split_manifest(frames, img_to_split, args.split_manifest)
        print(f"Saved leakage-free split manifest to: {args.split_manifest}")
        for g, sp in sorted(group_to_split.items()):
            print(f"  Source group '{g}' -> {sp}")
    else:
        print(
            f"Note: Only {len(unique_groups)} source group(s) found. "
            "At least 3 source groups are required to produce independent train/val/test splits without leakage. "
            "More diverse source video clips or extended durations are recommended."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
