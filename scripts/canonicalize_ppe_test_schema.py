"""Create a canonical four-class copy of a verified three-class PPE Test export."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import yaml


SOURCE_NAMES = ["no_helmet", "no_vest", "vest"]
CANONICAL_NAMES = ["helmet", "no_helmet", "no_vest", "vest"]
REMAP = {0: 1, 1: 2, 2: 3}


def remap_label_line(line: str) -> str:
    """Replace only a verified source class ID; leave all coordinates byte-for-byte intact."""
    fields = line.split(maxsplit=1)
    if len(fields) != 2:
        raise ValueError(f"Expected a class ID and four YOLO coordinates: {line!r}")
    source_id = int(fields[0])
    if source_id not in REMAP:
        raise ValueError(f"Unexpected source class ID {source_id}")
    return f"{REMAP[source_id]} {fields[1]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy PPE Test images unchanged and reconcile verified class IDs.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reconciliation-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite canonical dataset: {args.output_root}")
    source_yaml = args.source_root / "data.yaml"
    source_images, source_labels = args.source_root / "test" / "images", args.source_root / "test" / "labels"
    if yaml.safe_load(source_yaml.read_text(encoding="utf-8")).get("names") != SOURCE_NAMES:
        raise ValueError("Source data.yaml does not declare the verified three-class source schema")
    images = sorted(path for path in source_images.iterdir() if path.is_file())
    labels = sorted(path for path in source_labels.iterdir() if path.suffix.lower() == ".txt")
    if len(images) != 100 or len(labels) != 100:
        raise ValueError("Expected exactly 100 source Test images and labels")
    destination_images, destination_labels = args.output_root / "test" / "images", args.output_root / "test" / "labels"
    destination_images.mkdir(parents=True)
    destination_labels.mkdir(parents=True)
    before, after = Counter(), Counter()
    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"Missing label for {image_path.name}")
        shutil.copy2(image_path, destination_images / image_path.name)
        rewritten = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            source_id = int(line.split(maxsplit=1)[0])
            before[source_id] += 1
            remapped = remap_label_line(line)
            after[int(remapped.split(maxsplit=1)[0])] += 1
            rewritten.append(remapped)
        (destination_labels / label_path.name).write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")
    (args.output_root / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
        "nc: 4\nnames: ['helmet', 'no_helmet', 'no_vest', 'vest']\n",
        encoding="utf-8",
    )
    record = {
        "source_dataset": "ppe_test_v3 Roboflow export",
        "source_export_root": args.source_root.as_posix(),
        "canonical_export_root": args.output_root.as_posix(),
        "source_class_mapping_as_observed": {"0": "no_helmet", "1": "no_vest", "2": "vest", "3": "no_instances"},
        "canonical_class_mapping": {str(index): name for index, name in enumerate(CANONICAL_NAMES)},
        "remap": {"0": 1, "1": 2, "2": 3},
        "reason": "Roboflow export numeric class IDs did not align with the declared SafetyShield canonical schema.",
        "boxes_changed": False, "coordinates_changed": False, "images_changed": False,
        "annotations_added": 0, "annotations_removed": 0,
        "total_annotations_before": sum(before.values()), "total_annotations_after": sum(after.values()),
        "source_numeric_class_counts": {str(key): value for key, value in sorted(before.items())},
        "canonical_numeric_class_counts": {str(key): value for key, value in sorted(after.items())},
        "verification_basis": [
            "Source data.yaml declares source IDs 0=no_helmet, 1=no_vest, 2=vest.",
            "Representative manual visual review confirmed uncovered-head, non-vest torso, and high-visibility vest boxes for IDs 0, 1, and 2 respectively.",
        ],
    }
    args.reconciliation_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.reconciliation_manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(images), "labels": len(labels), "before": dict(before), "after": dict(after)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
