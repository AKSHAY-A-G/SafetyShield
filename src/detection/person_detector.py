"""Ultralytics YOLO person-only detection on original-resolution frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PERSON_CLASS_ID = 0


@dataclass(frozen=True, slots=True)
class PersonDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = PERSON_CLASS_ID
    class_name: str = "Person"

    @classmethod
    def from_xyxy(
        cls,
        xyxy: tuple[float, float, float, float],
        confidence: float,
        frame_width: int,
        frame_height: int,
    ) -> "PersonDetection":
        """Round and clip model coordinates to the original frame boundaries."""
        x1, y1, x2, y2 = xyxy
        return cls(
            x1=max(0, min(frame_width - 1, round(x1))),
            y1=max(0, min(frame_height - 1, round(y1))),
            x2=max(0, min(frame_width - 1, round(x2))),
            y2=max(0, min(frame_height - 1, round(y2))),
            confidence=float(confidence),
        )


class PersonDetector:
    """Load one official checkpoint and return only COCO person detections."""

    def __init__(
        self,
        checkpoint: str | Path = "yolo26n.pt",
        confidence_threshold: float = 0.20,
        image_size: int = 960,
        device: int = 0,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this Milestone 1 run")
        if device < 0 or device >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {device} is not available")

        self.checkpoint = str(checkpoint)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device
        self.device_name = torch.cuda.get_device_name(device)
        self._model = YOLO(self.checkpoint)
        self.last_inference_ms: float | None = None

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with three channels")

        results: list[Any] = self._model.predict(
            source=frame,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            self.last_inference_ms = None
            return []

        result = results[0]
        speed = getattr(result, "speed", None) or {}
        inference_ms = speed.get("inference")
        self.last_inference_ms = float(inference_ms) if inference_ms is not None else None

        height, width = frame.shape[:2]
        detections: list[PersonDetection] = []
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls[0].item())
            if class_id != PERSON_CLASS_ID:
                continue
            coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
            detections.append(
                PersonDetection.from_xyxy(
                    coordinates,
                    confidence=float(box.conf[0].item()),
                    frame_width=width,
                    frame_height=height,
                )
            )
        return detections
