"""Create full-resolution A/B tracking contact sheets at matched timestamps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np


DEFAULT_INTERVALS = ((5.0, 15.0), (25.0, 35.0), (75.0, 85.0))
SAMPLE_STEP_SECONDS = 0.5
PART_SECONDS = 5.0
HEADER_HEIGHT = 44


def interval_timestamps(start: float, end: float) -> list[float]:
    if not all(math.isfinite(value) for value in (start, end)):
        raise ValueError("interval bounds must be finite")
    if start < 0 or end <= start:
        raise ValueError("intervals require 0 <= start < end")
    count = math.ceil((end - start) / SAMPLE_STEP_SECONDS)
    return [start + index * SAMPLE_STEP_SECONDS for index in range(count)]


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read comparison frame {frame_index}")
    return frame


def labelled_frame(frame: np.ndarray, label: str) -> np.ndarray:
    header = np.zeros((HEADER_HEIGHT, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        label,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack((header, frame))


def create_sheet(
    baseline: cv2.VideoCapture,
    candidate: cv2.VideoCapture,
    fps: float,
    timestamps: list[float],
) -> np.ndarray:
    if len(timestamps) != 10:
        raise ValueError("each contact-sheet part must contain 10 timestamps")
    paired_tiles: list[tuple[np.ndarray, np.ndarray]] = []
    for timestamp in timestamps:
        frame_index = round(timestamp * fps)
        paired_tiles.append(
            (
                labelled_frame(
                    read_frame(baseline, frame_index),
                    f"A baseline | {timestamp:.1f}s | frame {frame_index}",
                ),
                labelled_frame(
                    read_frame(candidate, frame_index),
                    f"B candidate | {timestamp:.1f}s | frame {frame_index}",
                ),
            )
        )
    rows = []
    for index in range(0, len(paired_tiles), 2):
        first_a, first_b = paired_tiles[index]
        second_a, second_b = paired_tiles[index + 1]
        rows.append(np.hstack((first_a, first_b, second_a, second_b)))
    return np.vstack(rows)


def extract_tracking_comparisons(
    baseline_path: Path,
    candidate_path: Path,
    output_directory: Path,
    intervals: list[tuple[float, float]],
) -> list[Path]:
    baseline = cv2.VideoCapture(str(baseline_path))
    candidate = cv2.VideoCapture(str(candidate_path))
    try:
        if not baseline.isOpened() or not candidate.isOpened():
            raise RuntimeError("Both tracking videos must open successfully")
        baseline_fps = float(baseline.get(cv2.CAP_PROP_FPS))
        candidate_fps = float(candidate.get(cv2.CAP_PROP_FPS))
        baseline_size = (
            int(baseline.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(baseline.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        candidate_size = (
            int(candidate.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(candidate.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if baseline_size != candidate_size:
            raise RuntimeError("Tracking videos have different frame dimensions")
        if not math.isclose(baseline_fps, candidate_fps, abs_tol=0.01):
            raise RuntimeError("Tracking videos have different frame rates")
        if baseline_fps <= 0:
            raise RuntimeError("Tracking videos have invalid FPS metadata")

        output_directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for start, end in intervals:
            timestamps = interval_timestamps(start, end)
            if len(timestamps) != 20 or not math.isclose(end - start, 10.0):
                raise ValueError("each review interval must span exactly 10 seconds")
            for part_index, part_timestamps in enumerate(
                (timestamps[:10], timestamps[10:]), start=1
            ):
                sheet = create_sheet(
                    baseline, candidate, baseline_fps, part_timestamps
                )
                part_name = "a" if part_index == 1 else "b"
                destination = output_directory / (
                    f"tracking_{start:04.0f}_{end:04.0f}_part_{part_name}.jpg"
                )
                if not cv2.imwrite(
                    str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]
                ):
                    raise RuntimeError(f"Could not write contact sheet: {destination}")
                written.append(destination)
        return written
    finally:
        baseline.release()
        candidate.release()


def parse_interval(value: str) -> tuple[float, float]:
    try:
        start_text, end_text = value.split(":", maxsplit=1)
        return float(start_text), float(end_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use START:END, for example 5:15") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--interval",
        action="append",
        type=parse_interval,
        dest="intervals",
        help="10-second START:END interval; repeat for multiple intervals",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    intervals = args.intervals or list(DEFAULT_INTERVALS)
    try:
        written = extract_tracking_comparisons(
            args.baseline, args.candidate, args.output_dir, intervals
        )
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
