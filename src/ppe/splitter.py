"""Leakage-resistant train/validation/test splitting at source-group level for PPE dataset."""

from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Mapping, Sequence

from src.ppe.constants import DEFAULT_SPLIT_RATIOS
from src.ppe.models import SourceFrameInfo


def assign_source_group_splits(
    frames: Sequence[SourceFrameInfo],
    split_ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 42,
) -> tuple[dict[str, str], dict[str, str]]:
    """Assign source groups to train/val/test splits, guaranteeing group isolation.

    Returns:
        (image_to_split, group_to_split)
    """
    if not frames:
        raise ValueError("Cannot split an empty sequence of frames")

    # Group frames by source_group
    unique_groups = sorted(list({f.source_group for f in frames}))
    num_groups = len(unique_groups)

    if num_groups < 3:
        # Cannot form 3 independent non-empty splits
        raise ValueError(
            f"At least 3 independent source groups are required to produce non-empty "
            f"train/val/test splits without source leakage. Found only {num_groups} group(s): "
            f"{unique_groups}. More source diversity/clips are required."
        )

    # Deterministic assignment using fixed seed
    rng = random.Random(seed)
    shuffled_groups = list(unique_groups)
    rng.shuffle(shuffled_groups)

    train_ratio = split_ratios.get("train", 0.70)
    val_ratio = split_ratios.get("val", 0.15)
    test_ratio = split_ratios.get("test", 0.15)

    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    n_val = max(1, round(num_groups * val_ratio))
    n_test = max(1, round(num_groups * test_ratio))
    # Ensure train gets remaining and at least 1
    n_train = num_groups - n_val - n_test
    if n_train < 1:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    train_groups = set(shuffled_groups[:n_train])
    val_groups = set(shuffled_groups[n_train : n_train + n_val])
    test_groups = set(shuffled_groups[n_train + n_val :])

    # Assert strict disjointness
    assert not (train_groups & val_groups), "Source group leakage detected between train and val"
    assert not (train_groups & test_groups), "Source group leakage detected between train and test"
    assert not (val_groups & test_groups), "Source group leakage detected between val and test"

    group_to_split: dict[str, str] = {}
    for g in train_groups:
        group_to_split[g] = "train"
    for g in val_groups:
        group_to_split[g] = "val"
    for g in test_groups:
        group_to_split[g] = "test"

    image_to_split: dict[str, str] = {}
    for f in frames:
        image_to_split[f.source_image] = group_to_split[f.source_group]

    return image_to_split, group_to_split


def check_source_group_leakage(records: Sequence[dict[str, str]]) -> list[str]:
    """Inspect records containing 'source_group' and 'split' to detect leakage across splits.

    Returns a list of error strings describing any leaked source groups.
    """
    group_splits: dict[str, set[str]] = {}
    for row in records:
        group = row.get("source_group", "").strip()
        split = row.get("split", "").strip()
        if not group or not split:
            continue
        group_splits.setdefault(group, set()).add(split)

    leakage_errors: list[str] = []
    for group, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            leakage_errors.append(
                f"Source group '{group}' leaked across multiple splits: {sorted(list(splits))}"
            )
    return leakage_errors


def write_split_manifest(
    frames: Sequence[SourceFrameInfo],
    image_to_split: Mapping[str, str],
    manifest_path: Path,
) -> None:
    """Save split assignment manifest to CSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_image",
                "camera_id",
                "source_clip",
                "source_timestamp_seconds",
                "source_group",
                "split",
            ],
        )
        writer.writeheader()
        for fr in frames:
            writer.writerow({
                "source_image": fr.source_image,
                "camera_id": fr.camera_id,
                "source_clip": fr.source_clip,
                "source_timestamp_seconds": fr.source_timestamp_seconds,
                "source_group": fr.source_group,
                "split": image_to_split.get(fr.source_image, "train"),
            })


def load_split_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Load split assignment CSV into dictionaries."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
