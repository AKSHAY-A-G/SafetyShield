"""Safe frame extraction utility from local authorized recorded videos for PPE dataset preparation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import cv2

from src.ppe.constants import DEFAULT_SAMPLING_INTERVAL_SECONDS
from src.ppe.models import SourceFrameInfo


def extract_frames_from_video(
    video_path: Path,
    camera_id: str,
    output_dir: Path,
    interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS,
    max_frames: int | None = None,
    block_duration_seconds: float = 30.0,
) -> list[SourceFrameInfo]:
    """Extract frames at configurable time intervals from a local recorded video.

    Preserves original native resolution. Never connects to RTSP or external networks.
    Assigns each frame to a contiguous source group based on camera, clip, and time block.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be strictly positive")
    if block_duration_seconds <= 0:
        raise ValueError("block_duration_seconds must be strictly positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    clip_name = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # check non-positive or NaN
        fps = 30.0  # fallback assumption

    frame_step = max(1, round(fps * interval_seconds))
    extracted_frames: list[SourceFrameInfo] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % frame_step == 0:
            timestamp_sec = frame_idx / fps
            block_idx = int(timestamp_sec // block_duration_seconds)
            source_group = f"{camera_id}_{clip_name}_b{block_idx}"
            
            timestamp_ms = int(timestamp_sec * 1000)
            filename = f"{camera_id}_{clip_name}_t{timestamp_ms:07d}.jpg"
            frame_path = output_dir / filename

            # Write frame in native original resolution
            cv2.imwrite(str(frame_path), frame)

            height, width = frame.shape[:2]
            frame_info = SourceFrameInfo(
                source_image=filename,
                camera_id=camera_id,
                source_clip=clip_name,
                source_timestamp_seconds=round(timestamp_sec, 3),
                frame_width=width,
                frame_height=height,
                source_group=source_group,
                image_path=frame_path,
            )
            extracted_frames.append(frame_info)

            if max_frames is not None and len(extracted_frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return extracted_frames


def write_frame_manifest(frames: Sequence[SourceFrameInfo], manifest_path: Path) -> None:
    """Save extracted frame metadata to CSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_image",
                "camera_id",
                "source_clip",
                "source_timestamp_seconds",
                "frame_width",
                "frame_height",
                "source_group",
            ],
        )
        writer.writeheader()
        for fr in frames:
            writer.writerow({
                "source_image": fr.source_image,
                "camera_id": fr.camera_id,
                "source_clip": fr.source_clip,
                "source_timestamp_seconds": fr.source_timestamp_seconds,
                "frame_width": fr.frame_width,
                "frame_height": fr.frame_height,
                "source_group": fr.source_group,
            })
