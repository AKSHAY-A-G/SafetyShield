"""Copy-only construction and validation of the PPE training-only pool."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import yaml

from src.ppe.constants import PPE_CLASSES
from src.ppe.validator import validate_label_line


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TARGET_CLASSES = [PPE_CLASSES[index] for index in range(len(PPE_CLASSES))]


@dataclass(frozen=True, slots=True)
class SourceDataset:
    source_type: str
    root: Path
    prefix: str
    identity: str


def _class_names(data_yaml: Path) -> tuple[list[str], dict[str, Any]]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML is not a mapping: {data_yaml}")
    names = config.get("names")
    parsed = list(names) if isinstance(names, list) else [names[key] for key in sorted(names)] if isinstance(names, dict) else None
    if parsed is None or set(parsed) != set(TARGET_CLASSES) or len(parsed) != len(TARGET_CLASSES):
        raise ValueError(f"{data_yaml.name} must declare exactly these classes by name: {TARGET_CLASSES}")
    return parsed, config


def _split_paths(root: Path, config: dict[str, Any]) -> list[tuple[str, Path, Path]]:
    paths: list[tuple[str, Path, Path]] = []
    for yaml_key, source_split in (("train", "train"), ("val", "valid"), ("test", "test")):
        value = config.get(yaml_key)
        if not isinstance(value, str) or not value:
            continue
        images = Path(value)
        if not images.is_absolute():
            images = root / images
        labels = images.with_name("labels") if images.name == "images" else images / "labels"
        if not images.is_dir() or not labels.is_dir():
            # Roboflow exports often retain ``../train/images`` even when
            # data.yaml is already at the export root.  Prefer the actual,
            # local split layout only when that declared relative path cannot
            # be resolved as written.
            images = root / source_split / "images"
            labels = root / source_split / "labels"
        if not images.is_dir() or not labels.is_dir():
            # The manually validated site export has no test directory despite
            # retaining an unused Roboflow test declaration.  A missing split
            # contains no candidate image and is not fabricated in the pool.
            if yaml_key == "test":
                continue
            raise ValueError(f"Declared source split '{yaml_key}' is missing images or labels under {root.name}")
        paths.append((source_split, images, labels))
    return paths


def _portable(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_hash(image: Any) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _public_group_lookup(manifest_path: Path) -> dict[str, dict[str, str]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    lookup: dict[str, dict[str, str]] = {}
    for group in raw.get("groups", []):
        for member in group.get("members", []):
            lookup[member["path"]] = {
                "candidate_group_id": group["group_id"],
                "review_status": group["review_status"],
            }
    return lookup


def _source_records(source: SourceDataset, public_groups: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    class_names, config = _class_names(source.root / "data.yaml")
    records: list[dict[str, Any]] = []
    for split, images_dir, labels_dir in _split_paths(source.root, config):
        for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"Missing source label: {_portable(image_path)}")
            image = cv2.imread(str(image_path))
            if image is None or image.size == 0:
                raise ValueError(f"Unreadable source image: {_portable(image_path)}")
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                raise ValueError(f"Empty source label: {_portable(label_path)}")
            counts = Counter({name: 0 for name in TARGET_CLASSES})
            remapped: list[str] = []
            seen: set[str] = set()
            for line_no, line in enumerate(lines, start=1):
                box = validate_label_line(line, line_no, label_path.name)
                if line in seen:
                    raise ValueError(f"Duplicate exact source annotation row: {label_path.name}:{line_no}")
                seen.add(line)
                class_name = class_names[box.class_id]
                target_id = TARGET_CLASSES.index(class_name)
                counts[class_name] += 1
                remapped.append(f"{target_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}")
            relative_image = image_path.relative_to(source.root).as_posix()
            provenance = public_groups.get(relative_image, {}) if source.source_type == "public" else {}
            records.append({
                "source_type": source.source_type,
                "source_root": _portable(source.root),
                "original_source_path": _portable(image_path),
                "original_split": split,
                "original_filename": image_path.name,
                "source_label_path": _portable(label_path),
                "destination_image": f"images/{source.prefix}{image_path.name}",
                "destination_label": f"labels/{source.prefix}{image_path.stem}.txt",
                "class_instance_counts": dict(counts),
                "dataset_identity": config.get("roboflow", {}).get("project", source.identity),
                "dataset_version": config.get("roboflow", {}).get("version"),
                "license": config.get("roboflow", {}).get("license"),
                "source_url": config.get("roboflow", {}).get("url"),
                "public_provenance": provenance or None,
                "_image_path": image_path,
                "_label_text": "\n".join(remapped) + "\n",
                "_file_sha256": _sha256(image_path),
                "_decoded_sha256": _decoded_hash(image),
            })
    return records


def build_training_pool(destination: Path, public_root: Path, site_root: Path, public_manifest: Path) -> dict[str, Any]:
    """Create an all-training candidate pool from copies of validated source exports.

    The destination must not exist.  No validation/test directories are created.
    """
    if destination.exists():
        raise FileExistsError(f"Training pool already exists and will not be overwritten: {_portable(destination)}")
    if not public_manifest.is_file():
        raise FileNotFoundError(f"Public provenance manifest is required: {_portable(public_manifest)}")
    public_groups = _public_group_lookup(public_manifest)
    records = _source_records(SourceDataset("public", public_root, "public__", "public"), public_groups)
    records.extend(_source_records(SourceDataset("safetyshield_site", site_root, "site__", "safetyshield_site"), {}))
    destinations = [record["destination_image"] for record in records] + [record["destination_label"] for record in records]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Destination filename collision detected before copying; no files were written")

    public_hashes = {record["_file_sha256"] for record in records if record["source_type"] == "public"}
    public_decoded = {record["_decoded_sha256"] for record in records if record["source_type"] == "public"}
    exact_cross_dataset = [
        record["destination_image"] for record in records
        if record["source_type"] == "safetyshield_site"
        and (record["_file_sha256"] in public_hashes or record["_decoded_sha256"] in public_decoded)
    ]

    (destination / "images").mkdir(parents=True)
    (destination / "labels").mkdir(parents=True)
    for record in records:
        shutil.copy2(record["_image_path"], destination / record["destination_image"])
        (destination / record["destination_label"]).write_text(record["_label_text"], encoding="utf-8")

    data_yaml = {
        "path": _portable(destination),
        "train": "images",
        "names": {index: name for index, name in enumerate(TARGET_CLASSES)},
        "safetyshield_status": "TRAINING_DATA_ONLY_INDEPENDENT_VALIDATION_AND_TEST_NOT_READY",
    }
    (destination / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    totals = Counter()
    cleaned_records: list[dict[str, Any]] = []
    for record in records:
        totals.update(record["class_instance_counts"])
        cleaned_records.append({key: value for key, value in record.items() if not key.startswith("_")})
    manifest = {
        "schema_version": "1.0",
        "pool_identity": "safetyshield_ppe_training_pool_v1",
        "purpose": "TRAINING_DATA_ONLY",
        "independent_validation_data": "NOT_READY",
        "independent_test_data": "NOT_READY",
        "full_training_gate": "BLOCKED",
        "block_reason": "Independent SafetyShield validation/test source groups have not yet been created.",
        "class_order": TARGET_CLASSES,
        "public_provenance_manifest": _portable(public_manifest),
        "exact_public_site_duplicate_destinations": exact_cross_dataset,
        "total_images": len(cleaned_records),
        "total_labels": len(cleaned_records),
        "class_instance_counts": {name: totals[name] for name in TARGET_CLASSES},
        "records": cleaned_records,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (destination / "ATTRIBUTION.md").write_text(
        "# PPE training-pool provenance\n\n"
        "## Public supplemental data\n\n"
        "PPE Computer Vision Dataset, Roboflow Universe; CC BY 4.0. Source URL and export metadata are retained in `manifest.json`. "
        "Public images were supplied by the public Roboflow export, not captured by SafetyShield.\n\n"
        "## SafetyShield site data\n\n"
        "Current SafetyShield site PPE images are local training candidates only. They are not independent validation or test evidence.\n",
        encoding="utf-8",
    )
    return manifest
