"""Annotation visualization overlay utility for manual PPE label quality inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.ppe.constants import PPE_CLASSES
from src.ppe.models import BBoxNormalized
from src.ppe.validator import validate_label_line


# Distinct BGR colors for the four PPE classes
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 220, 0),      # helmet: bright green
    1: (0, 0, 230),      # no_helmet: red
    2: (255, 180, 0),    # vest: cyan/amber
    3: (200, 0, 220),    # no_vest: magenta
}


def render_annotation_overlay(
    image: np.ndarray,
    boxes: Sequence[BBoxNormalized],
) -> np.ndarray:
    """Draw bounding boxes and class label banners over an image.

    Returns a new annotated BGR image copy without altering the input.
    """
    canvas = image.copy()
    height, width = canvas.shape[:2]

    for box in boxes:
        color = CLASS_COLORS.get(box.class_id, (255, 255, 255))
        class_name = PPE_CLASSES.get(box.class_id, f"class_{box.class_id}")

        # Convert normalized coordinates to pixel coordinates
        box_w = box.width * width
        box_h = box.height * height
        x1 = max(0, int(round((box.x_center - box.width / 2.0) * width)))
        y1 = max(0, int(round((box.y_center - box.height / 2.0) * height)))
        x2 = min(width - 1, int(round((box.x_center + box.width / 2.0) * width)))
        y2 = min(height - 1, int(round((box.y_center + box.height / 2.0) * height)))

        # Draw bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=2)

        # Draw label banner
        label_text = f"{class_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        banner_y1 = max(0, y1 - th - baseline - 4)
        banner_y2 = max(th + baseline + 4, y1)
        banner_x2 = min(width - 1, x1 + tw + 6)

        cv2.rectangle(canvas, (x1, banner_y1), (banner_x2, banner_y2), color, thickness=cv2.FILLED)
        cv2.putText(
            canvas,
            label_text,
            (x1 + 3, banner_y2 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),  # black text on bright background
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return canvas


def render_dataset_samples(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    max_samples: int = 30,
) -> list[Path]:
    """Render sample annotations from an images/labels directory into an output QA folder.

    Never modifies original dataset files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_files: list[Path] = []

    img_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not img_paths:
        return rendered_files

    for img_path in img_paths[:max_samples]:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        boxes: list[BBoxNormalized] = []
        try:
            lines = lbl_path.read_text(encoding="utf-8").splitlines()
            for line_no, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                boxes.append(validate_label_line(line, line_no=line_no, filename=lbl_path.name))
        except Exception:
            continue

        annotated = render_annotation_overlay(image, boxes)
        out_path = output_dir / f"qa_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)
        rendered_files.append(out_path)

    return rendered_files
