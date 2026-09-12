"""Render bounding box annotation overlays on sample crops for visual label QA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.visualizer import render_dataset_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render visual annotation overlays on sample crops for manual label review."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "roboflow_export",
        help="Root path to exported dataset containing train/val/test splits.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Explicit directory of images to render (overrides dataset-dir).",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Explicit directory of labels to render (overrides dataset-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "ppe" / "reports" / "review_samples",
        help="Directory to save rendered visual QA images.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="Maximum number of sample images to render.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.images_dir is not None and args.labels_dir is not None:
        img_dir = args.images_dir
        lbl_dir = args.labels_dir
    else:
        # Default to inspecting the train split of the roboflow export
        img_dir = args.dataset_dir / "train" / "images"
        lbl_dir = args.dataset_dir / "train" / "labels"
        if not img_dir.exists():
            img_dir = args.dataset_dir / "images" / "train"
            lbl_dir = args.dataset_dir / "labels" / "train"
        if not img_dir.exists():
            img_dir = args.dataset_dir / "train"
            lbl_dir = args.dataset_dir / "train"

    if not img_dir.exists() or not lbl_dir.exists():
        print(
            f"Note: Images directory ({img_dir}) or labels directory ({lbl_dir}) not found.\n"
            "Visual QA renders can be generated once an annotated dataset export is available.",
            file=sys.stderr,
        )
        return 1

    print(f"Rendering annotation overlays from: {img_dir.name}")
    print(f"Output directory: {args.output_dir}")

    rendered = render_dataset_samples(
        images_dir=img_dir,
        labels_dir=lbl_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )

    print(f"Rendered {len(rendered)} review sample(s) with class-color bounding boxes.")
    for r in rendered[:5]:
        print(f"  Saved: {r.name}")
    if len(rendered) > 5:
        print(f"  ... and {len(rendered) - 5} more files in {args.output_dir}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
