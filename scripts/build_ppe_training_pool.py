"""Build the copy-only SafetyShield PPE training pool; this never starts training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.training_pool import build_training_pool


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a training-only PPE pool from copies of validated source exports.")
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "data/dataset/ppe/training_pool_v1")
    parser.add_argument("--public-root", type=Path, default=PROJECT_ROOT / "data/dataset/ppe/public_base/roboflow_export/ppe_public_base_v1_yolo26")
    parser.add_argument("--site-root", type=Path, default=PROJECT_ROOT / "data/dataset/ppe/roboflow_export/SafetyShield PPE.v1-ppe_smoke_v1.yolo26")
    parser.add_argument("--public-manifest", type=Path, default=PROJECT_ROOT / "config/public_ppe_provenance_manifest.json")
    args = parser.parse_args()
    manifest = build_training_pool(args.destination, args.public_root, args.site_root, args.public_manifest)
    print(f"Training-pool images: {manifest['total_images']}; labels: {manifest['total_labels']}")
    print(f"Class instances: {manifest['class_instance_counts']}")
    print(f"Exact public/site duplicate destinations: {len(manifest['exact_public_site_duplicate_destinations'])}")
    print("Independent validation/test: NOT READY; full training gate: BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
