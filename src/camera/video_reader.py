"""Safe OpenCV reader for local recorded videos."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np


class VideoOpenError(RuntimeError):
    """Raised when a local video cannot be opened or has invalid metadata."""


class VideoReadError(RuntimeError):
    """Raised when decoding stops before the expected end of a video."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int | None

    @property
    def duration_seconds(self) -> float | None:
        if self.frame_count is None:
            return None
        return self.frame_count / self.fps


class VideoReader:
    """Read original-resolution frames and expose validated source metadata."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise VideoOpenError(f"OpenCV could not open input video: {self.path}")

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        raw_frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            self.close()
            raise VideoOpenError("Input video reports invalid frame dimensions")
        if not math.isfinite(fps) or fps <= 0:
            self.close()
            raise VideoOpenError("Input video reports invalid FPS metadata")

        self.metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=raw_frame_count if raw_frame_count > 0 else None,
        )
        self.frames_read = 0

    def read(self) -> np.ndarray | None:
        """Return the next frame, ``None`` at EOF, or raise on an early failure."""
        ok, frame = self._capture.read()
        if ok and frame is not None:
            self.frames_read += 1
            return frame

        expected = self.metadata.frame_count
        if expected is not None and self.frames_read < expected:
            raise VideoReadError(
                f"Frame decoding stopped early after {self.frames_read} of {expected} frames"
            )
        return None

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
