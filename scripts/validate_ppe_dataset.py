"""Validate a YOLO-format PPE dataset and generate a comprehensive audit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.validator import validate_ppe_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a YOLO-format PPE dataset for class adherence, bounding box validity, and leakage."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "config" / "ppe_dataset.yaml",
        help="Path to dataset YAML file.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "roboflow_export",
        help="Path to root dataset directory if YAML is omitted.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "manifests" / "split_manifest.csv",
        help="Path to split manifest CSV for checking source-group leakage.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "reports" / "dataset_report.json",
        help="Path to write JSON validation report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    yaml_target = args.data if args.data.exists() else None
    dir_target = args.dataset_dir if args.dataset_dir.exists() else None
    manifest_target = args.manifest if args.manifest.exists() else None

    if yaml_target is None and dir_target is None:
        print(
            f"Note: Neither dataset YAML ({args.data}) nor dataset directory ({args.dataset_dir}) exists.\n"
            "This is expected before a manual Roboflow dataset export has been placed into the project.",
            file=sys.stderr,
        )
        return 1

    print(f"Validating PPE dataset (YAML: {yaml_target}, Directory: {dir_target})...")
    report = validate_ppe_dataset(
        dataset_dir=dir_target,
        yaml_path=yaml_target,
        manifest_path=manifest_target,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"Saved validation report to: {args.report}")

    print("\n--- PPE Dataset Validation Summary ---")
    print(f"Overall Status:        {'PASS' if report.is_valid else 'FAIL'}")
    print(f"Total Images:          {report.total_images}")
    print(f"Total Object Instances:{report.total_instances}")
    print(f"Images per Split:      {report.images_per_split}")
    print(f"Instances per Class:   {report.instances_per_class}")

    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for w in report.warnings[:5]:
            print(f"  ⚠️ {w}")
        if len(report.warnings) > 5:
            print(f"  ... and {len(report.warnings) - 5} more warnings.")

    if report.fatal_errors:
        print(f"\nFatal Errors ({len(report.fatal_errors)}):", file=sys.stderr)
        for err in report.fatal_errors[:10]:
            print(f"  ❌ {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
