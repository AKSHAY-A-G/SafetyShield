"""Constants and default configuration parameters for SafetyShield PPE dataset and training."""

from __future__ import annotations

from typing import Mapping

# The exactly four initial PPE classes for SafetyShield Milestone 8
PPE_CLASSES: Mapping[int, str] = {
    0: "helmet",
    1: "no_helmet",
    # Preserve the class IDs emitted by the manually exported Roboflow data.
    2: "no_vest",
    3: "vest",
}

PPE_CLASS_NAME_TO_ID: Mapping[str, int] = {
    name: class_id for class_id, name in PPE_CLASSES.items()
}

EXPECTED_CLASS_COUNT: int = len(PPE_CLASSES)

# Relative crop padding: 12% relative box padding on width and height
DEFAULT_BOX_PADDING_FRACTION: float = 0.12

# Frame sampling: default interval in seconds for video extraction
DEFAULT_SAMPLING_INTERVAL_SECONDS: float = 2.0

# Leakage-resistant source group split targets
DEFAULT_SPLIT_RATIOS: Mapping[str, float] = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# Conservative training settings designed for GTX 1650 (4 GB VRAM)
DEFAULT_TRAIN_IMGSZ: int = 640
DEFAULT_TRAIN_BATCH: int = 2
DEFAULT_TRAIN_WORKERS: int = 2
DEFAULT_TRAIN_DEVICE: int = 0
DEFAULT_TRAIN_MODEL: str = "yolo26n.pt"
DEFAULT_TRAIN_PROJECT: str = "models/ppe"
DEFAULT_TRAIN_NAME: str = "ppe_yolo26n"
