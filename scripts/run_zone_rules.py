"""Run BAR-001 and IDT-004 polygon rules on a recorded fixed-camera MP4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from uuid import uuid4

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_person_detection import create_output_writer, draw_fps, verify_output  # noqa: E402
from scripts.run_person_tracking import draw_track, extract_sample_frames  # noqa: E402
from src.camera.video_reader import VideoReader  # noqa: E402
from src.detection.person_detector import PersonDetector  # noqa: E402
from src.rules.zones import CameraZoneConfig, ZoneFrameResult, ZoneRuleEngine, load_camera_zones  # noqa: E402
from src.tracking.person_tracker import ByteTrackConfig, PersonTracker, TrackedPerson  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "cam_good_test_zones.mp4"
DEFAULT_SAMPLES = PROJECT_ROOT / "outputs" / "zone_samples"


def draw_zone_overlays(frame: np.ndarray, config: CameraZoneConfig, result: ZoneFrameResult) -> None:
    for zone_index, zone in enumerate(zone for zone in config.zones if zone.enabled):
        points = np.asarray(zone.polygon, dtype=np.int32)
        colour = (0, 80, 255) if zone.zone_type == "restricted" else (255, 180, 0)
        cv2.polylines(frame, [points], True, colour, 3, cv2.LINE_AA)
        anchor_x = max(0, min(frame.shape[1] - 1, int(points[0][0])))
        anchor_y = max(65 + zone_index * 28, min(frame.shape[0] - 10, int(points[0][1])))
        label = f"{zone.zone_name}: {result.counts[zone.zone_id]}"
        cv2.putText(
            frame,
            label,
            (anchor_x, anchor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            2,
            cv2.LINE_AA,
        )


def draw_rule_track(frame: np.ndarray, track: TrackedPerson, result: ZoneFrameResult) -> None:
    inside = any(track.track_id in track_ids for track_ids in result.inside_track_ids.values())
    draw_track(frame, track)
    if inside:
        cv2.rectangle(frame, (track.x1, track.y1), (track.x2, track.y2), (0, 255, 0), 3)
    cv2.circle(frame, track.bottom_center, 5, (0, 255, 255), -1, cv2.LINE_AA)


def draw_event_banner(frame: np.ndarray, labels: list[str]) -> None:
    """Show recently generated entry events long enough for video review."""
    for index, label in enumerate(labels[-3:]):
        top = 50 + index * 38
        cv2.rectangle(frame, (10, top), (min(frame.shape[1] - 10, 650), top + 30), (0, 0, 180), -1)
        cv2.putText(
            frame,
            label,
            (20, top + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--zones", type=Path, default=PROJECT_ROOT / "config" / "zones.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera-id", default="cam_good_test")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--checkpoint", default="yolo26n.pt")
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--track-high-thresh", type=float, default=0.20)
    parser.add_argument("--track-low-thresh", type=float, default=0.10)
    parser.add_argument("--new-track-thresh", type=float, default=0.20)
    parser.add_argument("--track-buffer", type=int, default=45)
    parser.add_argument("--match-thresh", type=float, default=0.80)
    parser.add_argument("--no-fuse-score", action="store_true")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--sample-timestamps", type=float, nargs="*", default=[10, 30, 50, 70, 80])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_id = args.session_id or uuid4().hex
    events = []
    recent_event_labels: list[tuple[float, str]] = []
    maximum_occupancy: dict[str, int] = {}
    try:
        with VideoReader(args.input) as reader:
            metadata = reader.metadata
            zone_config = load_camera_zones(
                args.zones, args.camera_id, metadata.width, metadata.height
            )
            detector = PersonDetector(
                checkpoint=args.checkpoint,
                confidence_threshold=args.confidence,
                image_size=args.imgsz,
                device=args.device,
            )
            tracker_config = ByteTrackConfig(
                track_high_thresh=args.track_high_thresh,
                track_low_thresh=args.track_low_thresh,
                new_track_thresh=args.new_track_thresh,
                track_buffer=args.track_buffer,
                match_thresh=args.match_thresh,
                fuse_score=not args.no_fuse_score,
            )
            tracker = PersonTracker(args.camera_id, session_id, tracker_config)
            rules = ZoneRuleEngine(args.camera_id, session_id, zone_config.zones)
            maximum_occupancy = {
                zone.zone_id: 0 for zone in zone_config.zones if zone.enabled
            }
            print(
                f"Input: {args.input} | {metadata.width}x{metadata.height} | "
                f"{metadata.fps:.3f} FPS | frames={metadata.frame_count}"
            )
            print(
                f"Detector: {args.checkpoint} | confidence={args.confidence:.2f} | "
                f"imgsz={args.imgsz} | cuda:{args.device} ({detector.device_name})"
            )
            print(f"ByteTrack: {tracker.config}")
            print(f"Zones: {[zone.zone_id for zone in zone_config.zones if zone.enabled]}")
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
                    frame_index = reader.frames_read - 1
                    timestamp = frame_index / metadata.fps
                    tracks = tracker.update(
                        detections,
                        frame.shape,
                        frame_index=frame_index,
                        timestamp_seconds=timestamp,
                    )
                    result = rules.update(tracks, timestamp, frame_index)
                    events.extend(result.events)
                    for event in result.events:
                        label = f"BAR-001 ENTRY | {event.zone_name} | Track {event.track_id}"
                        recent_event_labels.append((timestamp + 1.0, label))
                        print(
                            f"BAR-001 event: t={event.timestamp_seconds:.3f}s | "
                            f"frame={event.frame_number} | zone={event.zone_id} | "
                            f"track={event.track_id}"
                        )
                    recent_event_labels = [
                        item for item in recent_event_labels if item[0] >= timestamp
                    ]
                    for zone_id, count in result.counts.items():
                        maximum_occupancy[zone_id] = max(maximum_occupancy[zone_id], count)
                    draw_zone_overlays(frame, zone_config, result)
                    for track in tracks:
                        draw_rule_track(frame, track, result)
                    draw_event_banner(frame, [label for _, label in recent_event_labels])
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
            raise RuntimeError("Zone-rule output verification failed")
        sample_paths = extract_sample_frames(args.output, args.sample_dir, args.sample_timestamps)

        print("\nMILESTONE 3 AUTOMATED RUN RESULTS")
        print(f"Camera/session: {args.camera_id}/{session_id}")
        print(f"Total frames processed: {reader.frames_read}")
        print(f"Processing time: {processing_seconds:.3f} seconds")
        print(f"Average end-to-end FPS: {average_fps:.3f}")
        print(f"BAR-001 entry events generated: {len(events)}")
        print(f"Maximum observed zone occupancy: {maximum_occupancy}")
        print(f"Peak PyTorch GPU memory allocated: {peak_mib:.3f} MiB")
        print(f"Output: {args.output}")
        print(f"Output verification: {verification}")
        print(f"Review frames: {[str(path) for path in sample_paths]}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
