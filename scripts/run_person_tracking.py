"""Run camera-local ByteTrack person tracking on a recorded MP4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from uuid import uuid4

import cv2
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_person_detection import (  # noqa: E402
    create_output_writer,
    draw_fps,
    verify_output,
)
from src.camera.video_reader import VideoReader  # noqa: E402
from src.detection.person_detector import PersonDetector  # noqa: E402
from src.tracking.person_tracker import (  # noqa: E402
    ByteTrackConfig,
    PersonTracker,
    TrackedPerson,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "cam_good_test_tracked.mp4"
DEFAULT_SAMPLE_DIRECTORY = PROJECT_ROOT / "outputs" / "tracking_samples"
DEFAULT_SAMPLE_TIMESTAMPS = (10.0, 30.0, 50.0, 70.0, 80.0)


def draw_track(frame, track: TrackedPerson) -> None:
    colour = (
        40 + (track.track_id * 67) % 180,
        80 + (track.track_id * 43) % 150,
        80 + (track.track_id * 97) % 150,
    )
    cv2.rectangle(frame, (track.x1, track.y1), (track.x2, track.y2), colour, 2)
    label = f"Track {track.track_id} | {track.confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, scale, thickness
    )
    label_bottom = max(text_height + baseline + 3, track.y1)
    label_top = label_bottom - text_height - baseline - 3
    cv2.rectangle(
        frame,
        (track.x1, label_top),
        (min(frame.shape[1] - 1, track.x1 + text_width + 4), label_bottom),
        colour,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (track.x1 + 2, label_bottom - baseline - 2),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def extract_sample_frames(
    video_path: Path, output_directory: Path, timestamps: list[float]
) -> list[Path]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError("Tracking output could not be opened for sample extraction")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise RuntimeError("Tracking output has invalid FPS metadata")
        output_directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for timestamp in timestamps:
            if timestamp < 0:
                raise ValueError("sample timestamps cannot be negative")
            frame_index = round(timestamp * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read sample frame {frame_index}")
            path = output_directory / f"tracked_{timestamp:06.3f}s_frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Could not write tracking sample: {path}")
            written.append(path)
        return written
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
    parser.add_argument("--camera-id", default="cam_good_test")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--track-high-thresh", type=float, default=0.25)
    parser.add_argument("--track-low-thresh", type=float, default=0.10)
    parser.add_argument("--new-track-thresh", type=float, default=0.25)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.80)
    parser.add_argument("--no-fuse-score", action="store_true")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIRECTORY)
    parser.add_argument(
        "--sample-timestamps",
        type=float,
        nargs="*",
        default=list(DEFAULT_SAMPLE_TIMESTAMPS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_id = args.session_id or uuid4().hex
    total_detections = 0
    inference_ms_total = 0.0
    inference_samples = 0
    maximum_active_tracks = 0

    try:
        tracker_config = ByteTrackConfig(
            track_high_thresh=args.track_high_thresh,
            track_low_thresh=args.track_low_thresh,
            new_track_thresh=args.new_track_thresh,
            track_buffer=args.track_buffer,
            match_thresh=args.match_thresh,
            fuse_score=not args.no_fuse_score,
        )
        with VideoReader(args.input) as reader:
            metadata = reader.metadata
            detector = PersonDetector(
                checkpoint=args.checkpoint,
                confidence_threshold=args.confidence,
                image_size=args.imgsz,
                device=args.device,
            )
            tracker = PersonTracker(args.camera_id, session_id, tracker_config)
            print(
                f"Input: {args.input} | {metadata.width}x{metadata.height} | "
                f"{metadata.fps:.3f} FPS | frames={metadata.frame_count}"
            )
            print(
                f"Detector: {args.checkpoint} | confidence={args.confidence:.2f} | "
                f"imgsz={args.imgsz} | cuda:{args.device} ({detector.device_name})"
            )
            print(f"ByteTrack: {tracker_config}")
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
                    frame_index = reader.frames_read - 1
                    tracks = tracker.update(
                        detections,
                        frame.shape,
                        frame_index=frame_index,
                        timestamp_seconds=frame_index / metadata.fps,
                    )
                    maximum_active_tracks = max(maximum_active_tracks, len(tracks))
                    for track in tracks:
                        draw_track(frame, track)
                    elapsed = time.perf_counter() - started
                    draw_fps(frame, reader.frames_read / elapsed if elapsed else 0.0)
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
            raise RuntimeError("Tracking output verification failed")

        sample_paths = extract_sample_frames(
            args.output, args.sample_dir, args.sample_timestamps
        )
        print("\nMILESTONE 2 RUN RESULTS")
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
        print(f"Unique temporary track IDs generated: {tracker.unique_track_count}")
        print(f"Maximum simultaneous active tracks: {maximum_active_tracks}")
        print(f"Output: {args.output}")
        print(f"Output verification: {verification}")
        print(f"Tracking sample frames: {[str(path) for path in sample_paths]}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
