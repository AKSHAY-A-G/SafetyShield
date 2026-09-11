"""Run Milestone 1 person detection on a local MP4 using CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.video_reader import VideoReader  # noqa: E402
from src.detection.person_detector import PersonDetection, PersonDetector  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "cam_good_test_person_detected.mp4"


def create_output_writer(
    output_path: str | Path, width: int, height: int, fps: float
) -> cv2.VideoWriter:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # pyrefly: ignore[missing-attribute]
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"OpenCV could not create output video: {path}")
    return writer


def draw_detection(frame, detection: PersonDetection) -> None:
    colour = (0, 220, 0)
    cv2.rectangle(
        frame,
        (detection.x1, detection.y1),
        (detection.x2, detection.y2),
        colour,
        2,
    )
    label = f"Person {detection.confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    label_bottom = max(text_height + baseline + 3, detection.y1)
    label_top = label_bottom - text_height - baseline - 3
    cv2.rectangle(
        frame,
        (detection.x1, label_top),
        (min(frame.shape[1] - 1, detection.x1 + text_width + 4), label_bottom),
        colour,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (detection.x1 + 2, label_bottom - baseline - 2),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_fps(frame, fps: float) -> None:
    cv2.putText(
        frame,
        f"Processing FPS: {fps:.2f}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def verify_output(
    path: str | Path, expected_width: int, expected_height: int
) -> dict[str, int | float | bool]:
    capture = cv2.VideoCapture(str(path))
    try:
        opened = capture.isOpened()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        readable, frame = capture.read() if opened else (False, None)
        return {
            "opened": opened,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "readable_frame": bool(readable and frame is not None),
            "dimensions_match": width == expected_width and height == expected_height,
        }
    finally:
        capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", default="yolo26n.pt")
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_detections = 0
    inference_ms_total = 0.0
    inference_samples = 0

    try:
        with VideoReader(args.input) as reader:
            metadata = reader.metadata
            print(
                "Input: "
                f"{args.input} | {metadata.width}x{metadata.height} | "
                f"{metadata.fps:.3f} FPS | "
                f"frames={metadata.frame_count if metadata.frame_count is not None else 'unknown'}"
            )
            detector = PersonDetector(
                checkpoint=args.checkpoint,
                confidence_threshold=args.confidence,
                image_size=args.imgsz,
                device=args.device,
            )
            print(
                f"Model: {args.checkpoint} | confidence={args.confidence:.2f} | "
                f"imgsz={args.imgsz} | cuda:{args.device} ({detector.device_name})"
            )
            torch.cuda.reset_peak_memory_stats(args.device)
            writer = create_output_writer(
                args.output, metadata.width, metadata.height, metadata.fps
            )
            started = time.perf_counter()
            try:
                while True:
                    frame = reader.read()
                    if frame is None:
                        break
                    detections = detector.detect(frame)
                    total_detections += len(detections)
                    if detector.last_inference_ms is not None:
                        inference_ms_total += detector.last_inference_ms
                        inference_samples += 1
                    for detection in detections:
                        draw_detection(frame, detection)
                    elapsed = time.perf_counter() - started
                    draw_fps(frame, reader.frames_read / elapsed if elapsed > 0 else 0.0)
                    writer.write(frame)
                    if reader.frames_read % 250 == 0:
                        print(f"Processed {reader.frames_read} frames...")
            finally:
                writer.release()

            torch.cuda.synchronize(args.device)
            processing_seconds = time.perf_counter() - started
            average_fps = reader.frames_read / processing_seconds
            peak_mib = torch.cuda.max_memory_allocated(args.device) / (1024**2)

        verification = verify_output(
            args.output, expected_width=metadata.width, expected_height=metadata.height
        )
        print("\nMILESTONE 1 RUN RESULTS")
        print(f"Total frames processed: {reader.frames_read}")
        print(f"Total accepted person detections: {total_detections}")
        print(f"Processing time: {processing_seconds:.3f} seconds")
        print(f"Average end-to-end FPS: {average_fps:.3f}")
        if inference_samples:
            print(
                "Average Ultralytics-reported inference time: "
                f"{inference_ms_total / inference_samples:.3f} ms/frame"
            )
        print(f"Peak PyTorch GPU memory allocated: {peak_mib:.3f} MiB")
        print(f"Output: {args.output}")
        print(f"Output verification: {verification}")

        expected_count = metadata.frame_count
        count_reasonable = expected_count is None or abs(
            int(verification["frame_count"]) - reader.frames_read
        ) <= 1
        if not (
            verification["opened"]
            and verification["readable_frame"]
            and verification["dimensions_match"]
            and count_reasonable
        ):
            print("ERROR: Output verification failed", file=sys.stderr)
            return 1
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
