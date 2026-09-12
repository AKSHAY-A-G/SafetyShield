"""Data models for SafetyShield PPE dataset and training preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceFrameInfo:
    """Metadata for a raw frame extracted from local video."""
    source_image: str
    camera_id: str
    source_clip: str
    source_timestamp_seconds: float
    frame_width: int
    frame_height: int
    source_group: str
    image_path: Path


@dataclass(frozen=True, slots=True)
class PersonCropInfo:
    """Metadata for an original-resolution person crop extracted from a source frame."""
    crop_filename: str
    source_image: str
    camera_id: str
    source_clip: str
    source_group: str
    split: str
    person_detection_confidence: float
    person_box_x1: int
    person_box_y1: int
    person_box_x2: int
    person_box_y2: int
    person_width_px: int
    person_height_px: int
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    crop_width_px: int
    crop_height_px: int
    crop_path: Path


@dataclass(frozen=True, slots=True)
class BBoxNormalized:
    """YOLO-format normalized bounding box (class_id, x_center, y_center, width, height)."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class DatasetValidationReport:
    """Detailed results of YOLO-format PPE dataset validation."""
    is_valid: bool
    dataset_yaml: str
    classes: list[str] = field(default_factory=list)
    total_images: int = 0
    total_instances: int = 0
    images_per_split: dict[str, int] = field(default_factory=dict)
    instances_per_class: dict[str, int] = field(default_factory=dict)
    instances_per_split_class: dict[str, dict[str, int]] = field(default_factory=dict)
    empty_label_files: list[str] = field(default_factory=list)
    missing_label_files: list[str] = field(default_factory=list)
    orphan_label_files: list[str] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)
    duplicate_boxes: list[str] = field(default_factory=list)
    source_group_leakage: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "dataset_yaml": self.dataset_yaml,
            "classes": self.classes,
            "total_images": self.total_images,
            "total_instances": self.total_instances,
            "images_per_split": self.images_per_split,
            "instances_per_class": self.instances_per_class,
            "instances_per_split_class": self.instances_per_split_class,
            "empty_label_files": self.empty_label_files,
            "missing_label_files": self.missing_label_files,
            "orphan_label_files": self.orphan_label_files,
            "corrupt_images": self.corrupt_images,
            "duplicate_boxes": self.duplicate_boxes,
            "source_group_leakage": self.source_group_leakage,
            "fatal_errors": self.fatal_errors,
            "warnings": self.warnings,
        }


@dataclass(frozen=True, slots=True)
class TrainingGateResult:
    """Outcome of pre-training readiness checks."""
    can_train: bool
    reason: str
    dataset_report: DatasetValidationReport | None = None
