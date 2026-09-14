"""Read-only near-duplicate review for the public PPE Roboflow export."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.ppe.models import BBoxNormalized
from src.ppe.validator import validate_label_line
from src.ppe.visualizer import render_annotation_overlay


SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True, slots=True)
class PublicImage:
    split: str
    path: Path
    relative_path: str
    original_stem: str
    width: int
    height: int
    class_counts: dict[str, int]
    pixel_hash: str
    perceptual_hash: int


def _original_stem(filename: str) -> str:
    """Return the source-style Roboflow filename portion before its .rf hash."""
    return filename.split(".rf.", maxsplit=1)[0]


def _perceptual_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(small)[:8, :8].flatten()
    median = float(np.median(coefficients[1:]))
    return sum(1 << index for index, value in enumerate(coefficients) if value > median)


def _load_images(dataset_root: Path, class_names: list[str]) -> list[PublicImage]:
    images: list[PublicImage] = []
    for split in SPLITS:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        for path in paths:
            image = cv2.imread(str(path))
            if image is None or image.size == 0:
                raise ValueError(f"Unreadable image during provenance review: {path.name}")
            label_path = label_dir / f"{path.stem}.txt"
            if not label_path.exists():
                raise ValueError(f"Missing label during provenance review: {path.name}")
            counts = {name: 0 for name in class_names}
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                box = validate_label_line(line, line_no, label_path.name)
                counts[class_names[box.class_id]] += 1
            height, width = image.shape[:2]
            images.append(
                PublicImage(
                    split=split,
                    path=path,
                    relative_path=path.relative_to(dataset_root).as_posix(),
                    original_stem=_original_stem(path.name),
                    width=width,
                    height=height,
                    class_counts=counts,
                    pixel_hash=hashlib.sha256(image.tobytes()).hexdigest(),
                    perceptual_hash=_perceptual_hash(image),
                )
            )
    return images


def _pair_record(left: PublicImage, right: PublicImage) -> dict[str, Any]:
    return {
        "image_a": left.relative_path,
        "image_b": right.relative_path,
        "split_a": left.split,
        "split_b": right.split,
        "original_stem_a": left.original_stem,
        "original_stem_b": right.original_stem,
        "perceptual_hamming_distance": (left.perceptual_hash ^ right.perceptual_hash).bit_count(),
        "same_decoded_pixel_hash": left.pixel_hash == right.pixel_hash,
        "dimensions_a": {"width": left.width, "height": left.height},
        "dimensions_b": {"width": right.width, "height": right.height},
        "class_counts_a": left.class_counts,
        "class_counts_b": right.class_counts,
    }


def build_public_provenance_manifest(dataset_root: Path, threshold: int = 3) -> dict[str, Any]:
    """Build connected near-duplicate groups without modifying dataset files."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    try:
        dataset_reference = dataset_root.relative_to(Path.cwd()).as_posix()
    except ValueError:
        dataset_reference = dataset_root.name
    config = yaml.safe_load((dataset_root / "data.yaml").read_text(encoding="utf-8"))
    raw_names = config.get("names")
    class_names = list(raw_names) if isinstance(raw_names, list) else [raw_names[key] for key in sorted(raw_names)]
    images = _load_images(dataset_root, class_names)
    parent = list(range(len(images)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pairs: list[tuple[int, int, dict[str, Any]]] = []
    for left in range(len(images)):
        for right in range(left + 1, len(images)):
            distance = (images[left].perceptual_hash ^ images[right].perceptual_hash).bit_count()
            if distance <= threshold:
                pair = _pair_record(images[left], images[right])
                pairs.append((left, right, pair))
                union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for index in {item for pair in pairs for item in pair[:2]}:
        components[find(index)].append(index)
    pair_by_component: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for left, right, pair in pairs:
        pair_by_component[find(left)].append(pair)

    groups: list[dict[str, Any]] = []
    for group_number, root in enumerate(sorted(components, key=lambda key: min(images[i].relative_path for i in components[key])), start=1):
        member_indexes = sorted(components[root], key=lambda index: images[index].relative_path)
        group_pairs = sorted(pair_by_component[root], key=lambda pair: (pair["image_a"], pair["image_b"]))
        splits = sorted({images[index].split for index in member_indexes})
        has_pixel_match = any(pair["same_decoded_pixel_hash"] for pair in group_pairs)
        has_related_stem = any(pair["original_stem_a"] == pair["original_stem_b"] for pair in group_pairs)
        if has_pixel_match:
            status = "CLEAR_SAME_SOURCE_NEAR_DUPLICATE"
        elif has_related_stem:
            status = "LIKELY_SAME_SOURCE"
        else:
            status = "REVIEW_REQUIRED"
        groups.append(
            {
                "group_id": f"PUBLIC_DUP_{group_number:04d}",
                "review_status": status,
                "original_splits": splits,
                "spans_multiple_splits": len(splits) > 1,
                "recommended_future_split_handling": "KEEP_WHOLE_GROUP_IN_ONE_SPLIT" if len(splits) > 1 else "KEEP_GROUP_TOGETHER_IF_RESPLIT",
                "members": [
                    {
                        "path": images[index].relative_path,
                        "split": images[index].split,
                        "original_stem": images[index].original_stem,
                        "dimensions": {"width": images[index].width, "height": images[index].height},
                        "class_counts": images[index].class_counts,
                    }
                    for index in member_indexes
                ],
                "candidate_pairs": group_pairs,
            }
        )

    within = sum(1 for _, _, pair in pairs if pair["split_a"] == pair["split_b"])
    across = len(pairs) - within
    return {
        "schema_version": "1.0",
        "dataset": {
            "identity": config.get("roboflow", {}).get("project", "unknown"),
            "roboflow_version": config.get("roboflow", {}).get("version"),
            "license": config.get("roboflow", {}).get("license"),
            "source_url": config.get("roboflow", {}).get("url"),
            "dataset_root": dataset_reference,
            "class_order": class_names,
        },
        "method": {
            "perceptual_hash": "64-bit DCT pHash over a 32x32 grayscale image",
            "hamming_threshold": threshold,
            "warning": "Candidate status is automated evidence, not human provenance confirmation.",
        },
        "summary": {
            "images_scanned": len(images),
            "candidate_pairs_within_splits": within,
            "candidate_pairs_across_splits": across,
            "candidate_groups_total": len(groups),
            "cross_split_groups": sum(group["spans_multiple_splits"] for group in groups),
            "images_involved": sum(len(group["members"]) for group in groups),
        },
        "groups": groups,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write a reviewed, machine-readable provenance manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def create_contact_sheets(dataset_root: Path, manifest: dict[str, Any], output_dir: Path) -> list[Path]:
    """Create plain and labelled contact sheets for candidate groups only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = manifest["dataset"]["class_order"]
    written: list[Path] = []
    for group in manifest["groups"]:
        cards: list[tuple[np.ndarray, str, dict[str, Any]]] = []
        for member in group["members"]:
            image = cv2.imread(str(dataset_root / member["path"]))
            if image is None:
                continue
            cards.append((
                image,
                f"{group['group_id']} | {member['split']} | {Path(member['path']).name}",
                member,
            ))
        for annotated, suffix in ((False, "plain"), (True, "labels")):
            rendered: list[np.ndarray] = []
            for image, caption, member in cards:
                canvas = image.copy()
                if annotated:
                    label_path = dataset_root / Path(member["path"]).parent.parent / "labels" / f"{Path(member['path']).stem}.txt"
                    boxes: list[BBoxNormalized] = []
                    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                        if line.strip():
                            boxes.append(validate_label_line(line, line_no, label_path.name))
                    canvas = render_annotation_overlay(canvas, boxes)
                canvas = cv2.resize(canvas, (320, 240), interpolation=cv2.INTER_AREA)
                card = np.full((282, 320, 3), 255, dtype=np.uint8)
                card[42:282] = canvas
                cv2.putText(card, caption[-48:], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(card, group["review_status"], (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 180), 1, cv2.LINE_AA)
                rendered.append(card)
            if not rendered:
                continue
            columns = 3
            rows = (len(rendered) + columns - 1) // columns
            sheet = np.full((rows * 282, columns * 320, 3), 230, dtype=np.uint8)
            for index, card in enumerate(rendered):
                row, column = divmod(index, columns)
                sheet[row * 282:(row + 1) * 282, column * 320:(column + 1) * 320] = card
            destination = output_dir / f"{group['group_id']}_{suffix}.jpg"
            if not cv2.imwrite(str(destination), sheet):
                raise RuntimeError(f"Could not write contact sheet: {destination}")
            written.append(destination)
    return written
