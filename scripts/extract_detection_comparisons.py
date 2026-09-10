"""Extract matched, side-by-side frames from two person-detection videos."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np


DEFAULT_TIMESTAMPS = (10.0, 30.0, 50.0, 70.0, 80.0)


def timestamp_to_frame_index(timestamp_seconds: float, fps: float) -> int:
    if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
        raise ValueError("timestamps must be finite and non-negative")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("video FPS must be finite and positive")
    return round(timestamp_seconds * fps)


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read comparison frame {frame_index}")
    return frame


def add_heading(frame: np.ndarray, heading: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1] - 1, 34), (0, 0, 0), -1)
    cv2.putText(
        result,
        heading,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def extract_comparisons(
    baseline_path: Path,
    higher_path: Path,
    output_directory: Path,
    timestamps: list[float],
) -> list[Path]:
    baseline = cv2.VideoCapture(str(baseline_path))
    higher = cv2.VideoCapture(str(higher_path))
    try:
        if not baseline.isOpened() or not higher.isOpened():
            raise RuntimeError("Both comparison videos must open successfully")

        baseline_fps = float(baseline.get(cv2.CAP_PROP_FPS))
        higher_fps = float(higher.get(cv2.CAP_PROP_FPS))
        baseline_size = (
            int(baseline.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(baseline.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        higher_size = (
            int(higher.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(higher.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if baseline_size != higher_size:
            raise RuntimeError("Comparison videos have different frame dimensions")
        if not math.isclose(baseline_fps, higher_fps, rel_tol=0.0, abs_tol=0.01):
            raise RuntimeError("Comparison videos have different frame rates")

        output_directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for timestamp in timestamps:
            frame_index = timestamp_to_frame_index(timestamp, baseline_fps)
            baseline_frame = add_heading(
                read_frame(baseline, frame_index),
                f"A: imgsz=640 conf=0.25 | {timestamp:.3f}s | frame {frame_index}",
            )
            higher_frame = add_heading(
                read_frame(higher, frame_index),
                f"B: imgsz=960 conf=0.20 | {timestamp:.3f}s | frame {frame_index}",
            )
            comparison = np.hstack((baseline_frame, higher_frame))
            destination = output_directory / f"comparison_{timestamp:06.3f}s.png"
            if not cv2.imwrite(str(destination), comparison):
                raise RuntimeError(f"Could not write comparison image: {destination}")
            written.append(destination)
        return written
    finally:
        baseline.release()
        higher.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--higher", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"))
    parser.add_argument(
        "--timestamps",
        type=float,
        nargs="+",
        default=list(DEFAULT_TIMESTAMPS),
        help="Matching timestamps in seconds (default: 10 30 50 70 80)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = extract_comparisons(
            args.baseline, args.higher, args.output_dir, args.timestamps
        )
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
