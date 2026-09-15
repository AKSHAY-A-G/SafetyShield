"""Generate unlabelled Round-2 PPE validation review sheets; never trains or labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.validation_review import ReviewCandidate, exclude_round_one, write_contact_sheets


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate unlabelled contact sheets for remaining PPE validation candidates.")
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument("--all-crops-manifest", type=Path, required=True)
    parser.add_argument("--round-one-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing review directory: {args.output_dir}")
    round_one = json.loads(args.round_one_manifest.read_text(encoding="utf-8"))
    round_one_names = {record["crop_filename"] for record in round_one["records"]}
    with args.all_crops_manifest.open(encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    candidates = [
        ReviewCandidate(
            filename=row["crop_filename"],
            path=args.crops_dir / row["crop_filename"],
            timestamp_seconds=float(Path(row["source_image"]).stem.rsplit("_t", maxsplit=1)[1]) / 1000.0,
        )
        for row in all_rows
    ]
    missing = [candidate.filename for candidate in candidates if not candidate.path.is_file()]
    if missing:
        raise FileNotFoundError(f"Crops referenced by manifest are missing, e.g. {missing[:3]}")
    remaining = exclude_round_one(candidates, round_one_names)
    sheets = write_contact_sheets(remaining, args.output_dir)
    summary = {
        "total_generated_crops": len(candidates),
        "round_one_crops": len(round_one_names),
        "remaining_candidates": len(remaining),
        "contact_sheets": [path.name for path in sheets],
        "human_selection_required": True,
        "ppe_labels_created": False,
    }
    (args.output_dir / "review_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
