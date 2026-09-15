"""Read-only candidate review helpers for unlabelled PPE validation crops."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ReviewCandidate:
    """One unlabelled crop plus only its source provenance."""

    filename: str
    path: Path
    timestamp_seconds: float


def file_sha256(path: Path) -> str:
    """Return the exact file digest without modifying the image."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> int:
    """Return a compact DCT hash used only to flag visual review duplicates."""
    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        raise ValueError(f"Unreadable review crop: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(small)[:8, :8].flatten()
    median = float(np.median(coefficients[1:]))
    return sum(1 << index for index, value in enumerate(coefficients) if value > median)


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def exclude_round_one(candidates: Iterable[ReviewCandidate], round_one_filenames: set[str]) -> list[ReviewCandidate]:
    """Return candidates whose destination names cannot collide with Round 1."""
    return [candidate for candidate in candidates if candidate.filename not in round_one_filenames]


def write_contact_sheets(
    candidates: list[ReviewCandidate],
    output_dir: Path,
    columns: int = 5,
    per_sheet: int = 30,
    priorities: dict[str, str] | None = None,
) -> list[Path]:
    """Create unlabelled review sheets with filename and source timestamp captions."""
    if columns <= 0 or per_sheet <= 0:
        raise ValueError("columns and per_sheet must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    tile_width, tile_height, caption_height = 220, 220, 60
    for start in range(0, len(candidates), per_sheet):
        page = candidates[start:start + per_sheet]
        rows = (len(page) + columns - 1) // columns
        sheet = np.full((rows * (tile_height + caption_height), columns * tile_width, 3), 245, dtype=np.uint8)
        for index, candidate in enumerate(page):
            image = cv2.imread(str(candidate.path))
            if image is None:
                raise ValueError(f"Unreadable review crop: {candidate.path}")
            image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            row, column = divmod(index, columns)
            y, x = row * (tile_height + caption_height), column * tile_width
            sheet[y:y + tile_height, x:x + tile_width] = image
            cv2.putText(sheet, candidate.filename[-31:], (x + 4, y + tile_height + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(sheet, f"t={candidate.timestamp_seconds:.1f}s", (x + 4, y + tile_height + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
            if priorities and candidate.filename in priorities:
                cv2.putText(sheet, priorities[candidate.filename], (x + 4, y + tile_height + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 150), 1, cv2.LINE_AA)
        destination = output_dir / f"remaining_candidates_{start // per_sheet + 1:02d}.jpg"
        if not cv2.imwrite(str(destination), sheet):
            raise RuntimeError(f"Could not write contact sheet: {destination}")
        written.append(destination)
    return written
