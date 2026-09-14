"""Train PPE object detector using transfer learning with pre-training gate enforcement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.constants import (
    DEFAULT_TRAIN_BATCH,
    DEFAULT_TRAIN_DEVICE,
    DEFAULT_TRAIN_IMGSZ,
    DEFAULT_TRAIN_MODEL,
    DEFAULT_TRAIN_NAME,
    DEFAULT_TRAIN_PROJECT,
    DEFAULT_TRAIN_WORKERS,
)
from src.ppe.trainer import check_training_gate, run_ppe_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gated PPE transfer learning using conservative hardware settings on GTX 1650."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "config" / "ppe_dataset.yaml",
        help="Path to dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TRAIN_MODEL,
        help="Pretrained YOLO model checkpoint for transfer learning (default: yolo26n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1 for smoke test).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_TRAIN_IMGSZ,
        help="Input image resolution for training (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_TRAIN_BATCH,
        help="Batch size (default: 2; automatically reduced to 1 on CUDA OOM).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=DEFAULT_TRAIN_DEVICE,
        help="CUDA device index (default: 0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_TRAIN_WORKERS,
        help="DataLoader worker count (default: 2).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_TRAIN_PROJECT,
        help="Directory to save trained PPE models (default: models/ppe).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_TRAIN_NAME,
        help="Experiment name subfolder.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "manifests" / "split_manifest.csv",
        help="Path to split manifest CSV for checking source-group leakage.",
    )
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="Only evaluate training gate prerequisites without launching training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("--- Evaluating PPE Pre-Training Gate ---")
    manifest_target = args.manifest if args.manifest.exists() else None
    gate_result = check_training_gate(args.data, manifest_path=manifest_target)

    if not gate_result.can_train:
        print(f"TRAINING GATE FAILED: {gate_result.reason}", file=sys.stderr)
        print(
            "\nTraining is strictly blocked until a valid, non-empty, leakage-free labelled dataset exists.\n"
            "Action required: User annotates crops and exports YOLO dataset to data/dataset/ppe/roboflow_export/.",
            file=sys.stderr,
        )
        return 1

    print(f"TRAINING GATE PASSED: {gate_result.reason}")

    if args.gate_only:
        print("Gate-only check complete. Exiting without training.")
        return 0

    print("\n--- Launching Gated PPE Transfer Learning ---")
    print(f"Model:     {args.model}")
    print(f"Dataset:   {args.data}")
    print(f"Epochs:    {args.epochs}")
    print(f"Imgsz:     {args.imgsz}")
    print(f"Batch:     {args.batch} (with batch=1 OOM fallback)")
    print(f"Workers:   {args.workers}")
    print(f"Device:    CUDA {args.device}")
    print(f"Output:    {args.project / args.name}")

    try:
        outcome = run_ppe_training(
            data_yaml=args.data,
            model_weights=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=args.project,
            name=args.name,
            manifest_path=manifest_target,
        )
        print(f"Training completed successfully with status: {outcome['status']}")
        return 0
    except Exception as err:
        print(f"Training failed: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
