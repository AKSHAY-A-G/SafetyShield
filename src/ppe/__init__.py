"""SafetyShield PPE dataset preparation and training package."""

from src.ppe.constants import (
    DEFAULT_BOX_PADDING_FRACTION,
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_TRAIN_BATCH,
    DEFAULT_TRAIN_DEVICE,
    DEFAULT_TRAIN_IMGSZ,
    DEFAULT_TRAIN_MODEL,
    DEFAULT_TRAIN_PROJECT,
    DEFAULT_TRAIN_WORKERS,
    EXPECTED_CLASS_COUNT,
    PPE_CLASSES,
    PPE_CLASS_NAME_TO_ID,
)
from src.ppe.crop_generator import (
    compute_padded_crop_box,
    extract_person_crops_from_frame,
    load_crop_manifest,
    write_crop_manifest,
)
from src.ppe.frame_extractor import (
    extract_frames_from_video,
    write_frame_manifest,
)
from src.ppe.models import (
    BBoxNormalized,
    DatasetValidationReport,
    PersonCropInfo,
    SourceFrameInfo,
    TrainingGateResult,
)
from src.ppe.splitter import (
    assign_source_group_splits,
    check_source_group_leakage,
    load_split_manifest,
    write_split_manifest,
)
from src.ppe.trainer import (
    check_training_gate,
    run_ppe_training,
)
from src.ppe.validator import (
    validate_dataset_yaml,
    validate_label_line,
    validate_ppe_dataset,
)
from src.ppe.visualizer import (
    render_annotation_overlay,
    render_dataset_samples,
)

__all__ = [
    "BBoxNormalized",
    "DatasetValidationReport",
    "PersonCropInfo",
    "SourceFrameInfo",
    "TrainingGateResult",
    "PPE_CLASSES",
    "PPE_CLASS_NAME_TO_ID",
    "EXPECTED_CLASS_COUNT",
    "DEFAULT_BOX_PADDING_FRACTION",
    "DEFAULT_SAMPLING_INTERVAL_SECONDS",
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_TRAIN_IMGSZ",
    "DEFAULT_TRAIN_BATCH",
    "DEFAULT_TRAIN_WORKERS",
    "DEFAULT_TRAIN_DEVICE",
    "DEFAULT_TRAIN_MODEL",
    "DEFAULT_TRAIN_PROJECT",
    "extract_frames_from_video",
    "write_frame_manifest",
    "assign_source_group_splits",
    "check_source_group_leakage",
    "write_split_manifest",
    "load_split_manifest",
    "compute_padded_crop_box",
    "extract_person_crops_from_frame",
    "write_crop_manifest",
    "load_crop_manifest",
    "validate_label_line",
    "validate_dataset_yaml",
    "validate_ppe_dataset",
    "render_annotation_overlay",
    "render_dataset_samples",
    "check_training_gate",
    "run_ppe_training",
]
