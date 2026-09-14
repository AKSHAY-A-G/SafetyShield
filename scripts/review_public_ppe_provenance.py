"""Generate the public PPE near-duplicate provenance manifest and contact sheets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.public_provenance import build_public_provenance_manifest, create_contact_sheets, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Review public PPE perceptual near-duplicate candidates without changing source data.")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/dataset/ppe/public_base/roboflow_export/ppe_public_base_v1_yolo26")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "config/public_ppe_provenance_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/ppe_public_provenance_review")
    parser.add_argument("--threshold", type=int, default=3)
    args = parser.parse_args()
    manifest = build_public_provenance_manifest(args.dataset_root, args.threshold)
    write_manifest(manifest, args.manifest)
    sheets = create_contact_sheets(args.dataset_root, manifest, args.output_dir)
    summary = manifest["summary"]
    print(f"Candidate pairs: within={summary['candidate_pairs_within_splits']}, across={summary['candidate_pairs_across_splits']}")
    print(f"Groups: {summary['candidate_groups_total']}; cross-split groups: {summary['cross_split_groups']}; images involved: {summary['images_involved']}")
    print(f"Manifest: {args.manifest}")
    print(f"Contact sheets: {len(sheets)} in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
