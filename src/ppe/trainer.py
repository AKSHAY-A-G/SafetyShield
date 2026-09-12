"""PPE model training runner and pre-training gatekeeper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ppe.constants import (
    DEFAULT_TRAIN_BATCH,
    DEFAULT_TRAIN_DEVICE,
    DEFAULT_TRAIN_IMGSZ,
    DEFAULT_TRAIN_MODEL,
    DEFAULT_TRAIN_NAME,
    DEFAULT_TRAIN_PROJECT,
    DEFAULT_TRAIN_WORKERS,
)
from src.ppe.models import TrainingGateResult
from src.ppe.validator import validate_ppe_dataset


def check_training_gate(
    yaml_path: Path,
    manifest_path: Path | None = None,
) -> TrainingGateResult:
    """Evaluate whether a dataset meets all prerequisites before permitting any training."""
    if not yaml_path.exists():
        return TrainingGateResult(
            can_train=False,
            reason=f"Dataset YAML not found: {yaml_path}",
        )

    report = validate_ppe_dataset(yaml_path=yaml_path, manifest_path=manifest_path)

    if not report.is_valid:
        error_summary = "; ".join(report.fatal_errors[:3])
        return TrainingGateResult(
            can_train=False,
            reason=f"Dataset validation failed with fatal errors: {error_summary}",
            dataset_report=report,
        )

    if report.total_images == 0:
        return TrainingGateResult(
            can_train=False,
            reason="Training gate failed: dataset contains 0 images.",
            dataset_report=report,
        )

    train_count = report.images_per_split.get("train", 0)
    if train_count == 0:
        return TrainingGateResult(
            can_train=False,
            reason="Training gate failed: train split is empty.",
            dataset_report=report,
        )

    val_count = report.images_per_split.get("val", 0)
    if val_count == 0:
        return TrainingGateResult(
            can_train=False,
            reason="Training gate failed: validation split is empty.",
            dataset_report=report,
        )

    if report.total_instances == 0:
        return TrainingGateResult(
            can_train=False,
            reason="Training gate failed: dataset has 0 labelled object instances.",
            dataset_report=report,
        )

    if report.source_group_leakage:
        return TrainingGateResult(
            can_train=False,
            reason=f"Training gate failed: source-group leakage detected: {report.source_group_leakage}",
            dataset_report=report,
        )

    return TrainingGateResult(
        can_train=True,
        reason=(
            f"Training gate passed: dataset is valid and non-empty "
            f"(train={train_count}, val={val_count}, instances={report.total_instances})."
        ),
        dataset_report=report,
    )


def run_ppe_training(
    data_yaml: Path,
    model_weights: str | Path = DEFAULT_TRAIN_MODEL,
    epochs: int = 1,
    imgsz: int = DEFAULT_TRAIN_IMGSZ,
    batch: int = DEFAULT_TRAIN_BATCH,
    workers: int = DEFAULT_TRAIN_WORKERS,
    device: int = DEFAULT_TRAIN_DEVICE,
    project: str | Path = DEFAULT_TRAIN_PROJECT,
    name: str = DEFAULT_TRAIN_NAME,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Execute gated PPE transfer learning using Ultralytics YOLO with conservative settings."""
    # Pre-training gate enforcement
    gate_result = check_training_gate(data_yaml, manifest_path=manifest_path)
    if not gate_result.can_train:
        raise RuntimeError(f"Training gate blocked execution: {gate_result.reason}")

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PPE model training on GTX 1650")
    if device < 0 or device >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device} is not available")

    project_dir = Path(project)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Initialize transfer learning from pretrained nano model
    model = YOLO(str(model_weights))

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=device,
            project=str(project_dir),
            name=name,
            cache=False,
            verbose=True,
        )
        return {"status": "SUCCESS", "results": str(results)}
    except torch.cuda.OutOfMemoryError as oom_err:
        if batch > 1:
            # Automatic fallback to batch=1 on CUDA OOM
            torch.cuda.empty_cache()
            print(f"CUDA OOM with batch={batch}. Retrying with batch=1...")
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=1,
                workers=workers,
                device=device,
                project=str(project_dir),
                name=f"{name}_batch1",
                cache=False,
                verbose=True,
            )
            return {"status": "SUCCESS_BATCH_1", "results": str(results)}
        raise RuntimeError(f"CUDA Out Of Memory during PPE training even with batch=1: {oom_err}") from oom_err
