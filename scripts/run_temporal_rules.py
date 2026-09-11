"""Run Milestone 3 zone rules plus Milestone 4 temporal rules on an MP4."""

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
from scripts.run_person_tracking import extract_sample_frames  # noqa: E402
from scripts.run_zone_rules import draw_event_banner, draw_rule_track, draw_zone_overlays  # noqa: E402
from src.camera.video_reader import VideoReader  # noqa: E402
from src.detection.person_detector import PersonDetector  # noqa: E402
from src.rules.temporal import TemporalFrameResult, TemporalRuleEngine, load_temporal_rules  # noqa: E402
from src.rules.zones import ZoneRuleEngine, load_camera_zones  # noqa: E402
from src.tracking.person_tracker import ByteTrackConfig, PersonTracker  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "cam_good_test_temporal_rules.mp4"
DEFAULT_SAMPLES = PROJECT_ROOT / "outputs" / "temporal_rule_samples"


def draw_temporal_status(frame: np.ndarray, result: TemporalFrameResult) -> None:
    lines: list[tuple[str, tuple[int, int, int]]] = []
    for zone_id, status in result.buddy_status.items():
        elapsed = result.buddy_elapsed_seconds.get(zone_id, 0.0)
        if status == "confirming":
            lines.append((f"Buddy Zone: 1 | EXC-002 confirming {elapsed:.1f}s", (0, 215, 255)))
        elif status == "active":
            lines.append(("Buddy Zone: 1 | EXC-002 LONE WORKER", (0, 0, 255)))
        elif status == "resetting":
            lines.append(("EXC-002 clearing", (0, 165, 255)))
        else:
            lines.append(("Buddy Zone: clear", (0, 200, 0)))
    for track_id, elapsed in sorted(result.low_movement_elapsed_seconds.items()):
        lines.append((f"Track {track_id} low movement: {elapsed:.1f}s", (255, 220, 0)))

    for index, (label, colour) in enumerate(lines[:4]):
        y = 30 + index * 30
        cv2.putText(
            frame,
            label,
            (frame.shape[1] - 570, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            colour,
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--zones", type=Path, default=PROJECT_ROOT / "config" / "zones.yaml")
    parser.add_argument("--rules", type=Path, default=PROJECT_ROOT / "config" / "rules.yaml")
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
    parser.add_argument("--sample-timestamps", type=float, nargs="*", default=[35, 38, 40, 42, 45])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_id = args.session_id or uuid4().hex
    bar_events = []
    temporal_events = []
    recent_event_labels: list[tuple[float, str]] = []
    maximum_occupancy: dict[str, int] = {}
    try:
        with VideoReader(args.input) as reader:
            metadata = reader.metadata
            zone_config = load_camera_zones(
                args.zones, args.camera_id, metadata.width, metadata.height
            )
            enabled_zone_ids = {zone.zone_id for zone in zone_config.zones if zone.enabled}
            temporal_config = load_temporal_rules(args.rules, enabled_zone_ids)
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
            zone_rules = ZoneRuleEngine(args.camera_id, session_id, zone_config.zones)
            temporal_rules = TemporalRuleEngine(args.camera_id, session_id, temporal_config)
            maximum_occupancy = {zone_id: 0 for zone_id in enabled_zone_ids}
            print(
                f"Input: {args.input} | {metadata.width}x{metadata.height} | "
                f"{metadata.fps:.3f} FPS | frames={metadata.frame_count}"
            )
            print(
                f"Detector: {args.checkpoint} | confidence={args.confidence:.2f} | "
                f"imgsz={args.imgsz} | cuda:{args.device} ({detector.device_name})"
            )
            print(f"ByteTrack: {tracker.config}")
            print(f"Zones: {sorted(enabled_zone_ids)}")
            print(f"Temporal rules: {temporal_config}")
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
                    zone_result = zone_rules.update(tracks, timestamp, frame_index)
                    temporal_result = temporal_rules.update(
                        tracks,
                        zone_result.counts,
                        zone_result.inside_track_ids,
                        timestamp,
                        frame_index,
                    )
                    bar_events.extend(zone_result.events)
                    temporal_events.extend(temporal_result.events)
                    for event in zone_result.events:
                        recent_event_labels.append(
                            (timestamp + 1.0, f"BAR-001 ENTRY | {event.zone_name} | Track {event.track_id}")
                        )
                        print(
                            f"BAR-001 event: t={timestamp:.3f}s | frame={frame_index} | "
                            f"zone={event.zone_id} | track={event.track_id}"
                        )
                    for event in temporal_result.events:
                        if event.module_id == "EXC-002":
                            label = f"EXC-002 LONE WORKER | {event.zone_id} | Track {event.track_id}"
                        else:
                            label = f"ERG-006 PROLONGED LOW MOVEMENT | Track {event.track_id}"
                        recent_event_labels.append((timestamp + 1.0, label))
                        print(
                            f"{event.module_id} event: t={timestamp:.3f}s | "
                            f"frame={frame_index} | zone={event.zone_id} | track={event.track_id}"
                        )
                    recent_event_labels = [
                        item for item in recent_event_labels if item[0] >= timestamp
                    ]
                    for zone_id, count in zone_result.counts.items():
                        maximum_occupancy[zone_id] = max(maximum_occupancy[zone_id], count)
                    draw_zone_overlays(frame, zone_config, zone_result)
                    for track_item in tracks:
                        draw_rule_track(frame, track_item, zone_result)
                    draw_temporal_status(frame, temporal_result)
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
            raise RuntimeError("Temporal-rule output verification failed")
        exc_events = [event for event in temporal_events if event.module_id == "EXC-002"]
        erg_events = [event for event in temporal_events if event.module_id == "ERG-006"]
        review_timestamps = list(args.sample_timestamps)
        for event in exc_events:
            review_timestamps.extend(
                [max(0.0, event.timestamp_seconds - 2.0), event.timestamp_seconds, event.timestamp_seconds + 0.7]
            )
        review_timestamps = sorted(set(round(value, 3) for value in review_timestamps))
        sample_paths = extract_sample_frames(args.output, args.sample_dir, review_timestamps)

        print("\nMILESTONE 4 AUTOMATED RUN RESULTS")
        print(f"Camera/session: {args.camera_id}/{session_id}")
        print(f"Total frames processed: {reader.frames_read}")
        print(f"Processing time: {processing_seconds:.3f} seconds")
        print(f"Average end-to-end FPS: {average_fps:.3f}")
        print(f"BAR-001 entry events generated: {len(bar_events)}")
        print(f"EXC-002 events generated: {len(exc_events)}")
        print(f"ERG-006 events generated: {len(erg_events)}")
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
