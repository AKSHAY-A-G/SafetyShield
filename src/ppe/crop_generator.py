"""Original-resolution person crop generator with configurable padding and quality tracking."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.detection.person_detector import PersonDetection, PersonDetector
from src.ppe.constants import DEFAULT_BOX_PADDING_FRACTION
from src.ppe.models import PersonCropInfo, SourceFrameInfo


def compute_padded_crop_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_width: int,
    frame_height: int,
    padding_fraction: float = DEFAULT_BOX_PADDING_FRACTION,
) -> tuple[int, int, int, int]:
    """Calculate padded bounding box relative to person box, clamped to image boundaries.

    Ensures padding is strictly within [0, frame_width] and [0, frame_height] with no negative coordinates.
    """
    if padding_fraction < 0.0:
        raise ValueError("padding_fraction cannot be negative")

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    pad_x = round(w * padding_fraction)
    pad_y = round(h * padding_fraction)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(frame_width, x2 + pad_x)
    crop_y2 = min(frame_height, y2 + pad_y)

    return crop_x1, crop_y1, crop_x2, crop_y2


def extract_person_crops_from_frame(
    frame: np.ndarray,
    frame_info: SourceFrameInfo,
    detections: Sequence[PersonDetection],
    output_base_dir: Path,
    split: str = "train",
    padding_fraction: float = DEFAULT_BOX_PADDING_FRACTION,
) -> list[PersonCropInfo]:
    """Generate and save padded person crops from an original-resolution frame.

    Saves crops into output_base_dir / split / <crop_filename>.
    Does not upscale small crops and does not invent PPE class labels.
    """
    height, width = frame.shape[:2]
    split_dir = output_base_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    crops_info: list[PersonCropInfo] = []
    base_stem = Path(frame_info.source_image).stem

    for idx, det in enumerate(detections):
        crop_x1, crop_y1, crop_x2, crop_y2 = compute_padded_crop_box(
            det.x1,
            det.y1,
            det.x2,
            det.y2,
            frame_width=width,
            frame_height=height,
            padding_fraction=padding_fraction,
        )

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        if crop_w <= 0 or crop_h <= 0:
            continue

        # Crop from original resolution image
        crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        crop_filename = f"{base_stem}_p{idx:02d}.jpg"
        crop_path = split_dir / crop_filename
        cv2.imwrite(str(crop_path), crop_img)

        info = PersonCropInfo(
            crop_filename=crop_filename,
            source_image=frame_info.source_image,
            camera_id=frame_info.camera_id,
            source_clip=frame_info.source_clip,
            source_group=frame_info.source_group,
            split=split,
            person_detection_confidence=round(float(det.confidence), 4),
            person_box_x1=det.x1,
            person_box_y1=det.y1,
            person_box_x2=det.x2,
            person_box_y2=det.y2,
            person_width_px=det.x2 - det.x1,
            person_height_px=det.y2 - det.y1,
            crop_x1=crop_x1,
            crop_y1=crop_y1,
            crop_x2=crop_x2,
            crop_y2=crop_y2,
            crop_width_px=crop_w,
            crop_height_px=crop_h,
            crop_path=crop_path,
        )
        crops_info.append(info)

    return crops_info


def write_crop_manifest(crops: Sequence[PersonCropInfo], manifest_path: Path) -> None:
    """Save crop metadata manifest to CSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop_filename",
                "source_image",
                "camera_id",
                "source_clip",
                "source_group",
                "split",
                "person_detection_confidence",
                "person_box_x1",
                "person_box_y1",
                "person_box_x2",
                "person_box_y2",
                "person_width_px",
                "person_height_px",
                "crop_x1",
                "crop_y1",
                "crop_x2",
                "crop_y2",
                "crop_width_px",
                "crop_height_px",
            ],
        )
        writer.writeheader()
        for cr in crops:
            writer.writerow({
                "crop_filename": cr.crop_filename,
                "source_image": cr.source_image,
                "camera_id": cr.camera_id,
                "source_clip": cr.source_clip,
                "source_group": cr.source_group,
                "split": cr.split,
                "person_detection_confidence": cr.person_detection_confidence,
                "person_box_x1": cr.person_box_x1,
                "person_box_y1": cr.person_box_y1,
                "person_box_x2": cr.person_box_x2,
                "person_box_y2": cr.person_box_y2,
                "person_width_px": cr.person_width_px,
                "person_height_px": cr.person_height_px,
                "crop_x1": cr.crop_x1,
                "crop_y1": cr.crop_y1,
                "crop_x2": cr.crop_x2,
                "crop_y2": cr.crop_y2,
                "crop_width_px": cr.crop_width_px,
                "crop_height_px": cr.crop_height_px,
            })


def load_crop_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Load crop manifest CSV into dictionaries."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Crop manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
