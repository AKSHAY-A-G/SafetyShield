"""Run Milestone 5 complete evidence pipeline on recorded video."""

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

from scripts.run_person_detection import draw_fps  # noqa: E402
from scripts.run_temporal_rules import draw_temporal_status  # noqa: E402
from scripts.run_zone_rules import draw_event_banner, draw_rule_track, draw_zone_overlays  # noqa: E402
from src.camera.video_reader import VideoReader  # noqa: E402
from src.detection.person_detector import PersonDetector  # noqa: E402
from src.events.audit import (  # noqa: E402
    AUDIT_ACTION_EVENT_CREATED,
    AuditLogger,
)
from src.events.evidence import (  # noqa: E402
    DEFAULT_POST_EVENT_SECONDS,
    DEFAULT_PRE_EVENT_SECONDS,
    EvidenceArtifacts,
    EvidenceWriter,
)
from src.events.models import SafetyEvent  # noqa: E402
from src.rules.temporal import TemporalRuleEngine, load_temporal_rules  # noqa: E402
from src.rules.zones import ZoneRuleEngine, load_camera_zones  # noqa: E402
from src.tracking.person_tracker import ByteTrackConfig, PersonTracker  # noqa: E402


DEFAULT_EVIDENCE_ROOT = PROJECT_ROOT / "evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--zones", type=Path, default=PROJECT_ROOT / "config" / "zones.yaml")
    parser.add_argument("--rules", type=Path, default=PROJECT_ROOT / "config" / "rules.yaml")
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_ROOT)
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
    parser.add_argument("--pre-event-seconds", type=float, default=DEFAULT_PRE_EVENT_SECONDS)
    parser.add_argument("--post-event-seconds", type=float, default=DEFAULT_POST_EVENT_SECONDS)
    parser.add_argument("--output", type=Path, default=None, help="Optional full annotated run video")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_id = args.session_id or f"sess_{uuid4().hex[:12]}"
    evidence_root = args.evidence_dir.resolve()
    audit_log_path = evidence_root / args.camera_id / session_id / "audit.jsonl"
    audit_logger = AuditLogger(audit_log_path)
    evidence_writer = EvidenceWriter(
        evidence_root=evidence_root,
        audit_logger=audit_logger,
        pre_event_seconds=args.pre_event_seconds,
        post_event_seconds=args.post_event_seconds,
    )

    captured_events: list[tuple[SafetyEvent, np.ndarray, np.ndarray]] = []
    recent_event_labels: list[tuple[float, str]] = []
    maximum_occupancy: dict[str, int] = {}
    bar_event_count = 0
    exc_event_count = 0
    erg_event_count = 0

    print("==================================================")
    print("SafetyShield - Milestone 5 Evidence Pipeline")
    print("==================================================")
    print(f"Camera ID: {args.camera_id}")
    print(f"Session ID: {session_id}")
    print(f"Evidence Root: {evidence_root}")
    print(f"Audit Log: {audit_log_path}")

    try:
        with VideoReader(args.input) as reader:
            metadata = reader.metadata
            source_duration = (
                metadata.frame_count / metadata.fps
                if metadata.frame_count and metadata.fps
                else 0.0
            )
            print(
                f"Input: {args.input} | {metadata.width}x{metadata.height} | "
                f"{metadata.fps:.3f} FPS | frames={metadata.frame_count} "
                f"(approx {source_duration:.2f}s)"
            )

            zone_config = load_camera_zones(
                args.zones, args.camera_id, metadata.width, metadata.height
            )
            enabled_zone_ids = {zone.zone_id for zone in zone_config.zones if zone.enabled}
            zone_name_map = {zone.zone_id: zone.zone_name for zone in zone_config.zones}
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

            video_writer = None
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                video_writer = cv2.VideoWriter(
                    str(args.output),
                    cv2.VideoWriter_fourcc(*"mp4v"),  # pyrefly: ignore[missing-attribute]
                    metadata.fps,
                    (metadata.width, metadata.height),
                )

            torch.cuda.reset_peak_memory_stats(args.device)
            pipeline_start = time.perf_counter()

            try:
                while True:
                    frame = reader.read()
                    if frame is None:
                        break

                    raw_frame = frame.copy()
                    frame_index = reader.frames_read - 1
                    timestamp = frame_index / metadata.fps

                    detections = detector.detect(frame)
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

                    # Update occupancy
                    for zone_id, count in zone_result.counts.items():
                        maximum_occupancy[zone_id] = max(maximum_occupancy[zone_id], count)

                    # Prepare annotated frame with overlays
                    draw_zone_overlays(frame, zone_config, zone_result)
                    for track_item in tracks:
                        draw_rule_track(frame, track_item, zone_result)
                    draw_temporal_status(frame, temporal_result)

                    # Process newly emitted BAR-001 events
                    newly_emitted_events: list[SafetyEvent] = []
                    for bar_event in zone_result.events:
                        bar_event_count += 1
                        safety_event = SafetyEvent.from_bar_entry_event(bar_event)
                        newly_emitted_events.append(safety_event)
                        recent_event_labels.append(
                            (timestamp + 1.0, f"BAR-001 ENTRY | {bar_event.zone_name} | Track {bar_event.track_id}")
                        )
                        audit_logger.log(
                            event_id=safety_event.event_id,
                            camera_id=safety_event.camera_id,
                            session_id=safety_event.session_id,
                            module_id=safety_event.module_id,
                            action=AUDIT_ACTION_EVENT_CREATED,
                            details={
                                "track_id": safety_event.track_id,
                                "source_timestamp_seconds": safety_event.source_timestamp_seconds,
                                "frame_number": safety_event.frame_number,
                                "zone_id": safety_event.zone_id,
                            },
                        )
                        print(
                            f"[EVENT] BAR-001: t={timestamp:.3f}s | frame={frame_index} | "
                            f"track={bar_event.track_id} | event_id={safety_event.event_id}"
                        )

                    # Process newly emitted temporal events
                    for temp_event in temporal_result.events:
                        zone_name = zone_name_map.get(temp_event.zone_id or "")
                        safety_event = SafetyEvent.from_temporal_safety_event(
                            temp_event, zone_name=zone_name
                        )
                        newly_emitted_events.append(safety_event)
                        if temp_event.module_id == "EXC-002":
                            exc_event_count += 1
                            label = f"EXC-002 LONE WORKER | {temp_event.zone_id} | Track {temp_event.track_id}"
                        else:
                            erg_event_count += 1
                            label = f"ERG-006 PROLONGED LOW MOVEMENT | Track {temp_event.track_id}"
                        recent_event_labels.append((timestamp + 1.0, label))
                        audit_logger.log(
                            event_id=safety_event.event_id,
                            camera_id=safety_event.camera_id,
                            session_id=safety_event.session_id,
                            module_id=safety_event.module_id,
                            action=AUDIT_ACTION_EVENT_CREATED,
                            details={
                                "track_id": safety_event.track_id,
                                "source_timestamp_seconds": safety_event.source_timestamp_seconds,
                                "frame_number": safety_event.frame_number,
                                "zone_id": safety_event.zone_id,
                            },
                        )
                        print(
                            f"[EVENT] {temp_event.module_id}: t={timestamp:.3f}s | "
                            f"frame={frame_index} | track={temp_event.track_id} | "
                            f"event_id={safety_event.event_id}"
                        )

                    recent_event_labels = [
                        item for item in recent_event_labels if item[0] >= timestamp
                    ]
                    draw_event_banner(frame, [label for _, label in recent_event_labels])

                    # Snapshot raw and annotated frames for new events
                    if newly_emitted_events:
                        annotated_snapshot = frame.copy()
                        for ev in newly_emitted_events:
                            captured_events.append((ev, raw_frame, annotated_snapshot))

                    if video_writer is not None:
                        elapsed = time.perf_counter() - pipeline_start
                        draw_fps(frame, reader.frames_read / elapsed if elapsed else 0.0)
                        video_writer.write(frame)

                    if reader.frames_read % 250 == 0:
                        print(f"Processed {reader.frames_read} frames...")

            finally:
                if video_writer is not None:
                    video_writer.release()

            torch.cuda.synchronize(args.device)
            pipeline_time = time.perf_counter() - pipeline_start
            pipeline_fps = reader.frames_read / pipeline_time if pipeline_time > 0 else 0.0
            peak_gpu_mib = torch.cuda.max_memory_allocated(args.device) / (1024**2)

        print("\n--- Main Rule Pipeline Complete ---")
        print(f"Main frames processed: {reader.frames_read}")
        print(f"Main pipeline time: {pipeline_time:.3f} s")
        print(f"Main pipeline FPS: {pipeline_fps:.3f} FPS")
        print(f"Peak GPU memory allocated: {peak_gpu_mib:.3f} MiB")
        print(f"BAR-001 events emitted: {bar_event_count}")
        print(f"EXC-002 events emitted: {exc_event_count}")
        print(f"ERG-006 events emitted: {erg_event_count}")
        print(f"Total events captured for evidence: {len(captured_events)}")

        # -----------------------------------------------------------------
        # Evidence Generation Phase
        # -----------------------------------------------------------------
        print("\n--- Starting Evidence Generation Phase ---")
        evidence_start = time.perf_counter()
        saved_artifacts: list[EvidenceArtifacts] = []

        for event, raw_snap, ann_snap in captured_events:
            print(f"Saving evidence for {event.module_id} (ID: {event.event_id})...")
            artifacts = evidence_writer.save_event_evidence(
                event=event,
                raw_frame=raw_snap,
                annotated_frame=ann_snap,
                source_video_path=args.input,
                source_duration_seconds=source_duration,
                source_fps=metadata.fps,
            )
            saved_artifacts.append(artifacts)

        evidence_time = time.perf_counter() - evidence_start
        print(f"Evidence generation time: {evidence_time:.3f} s")

        # -----------------------------------------------------------------
        # Comprehensive Report
        # -----------------------------------------------------------------
        audit_entries = audit_logger.read_entries()

        print("\n==================================================")
        print("MILESTONE 5 EXECUTION SUMMARY")
        print("==================================================")
        print(f"Camera ID: {args.camera_id}")
        print(f"Session ID: {session_id}")
        print(f"Total Source Frames: {reader.frames_read}")
        print(f"Source Duration: {source_duration:.3f} s")
        print(f"Main Pipeline Processing Time: {pipeline_time:.3f} s")
        print(f"Main Pipeline FPS: {pipeline_fps:.3f} FPS")
        print(f"Evidence Generation Time: {evidence_time:.3f} s")
        print(f"Peak GPU Memory: {peak_gpu_mib:.3f} MiB")
        print(f"BAR-001 Events: {bar_event_count}")
        print(f"EXC-002 Events: {exc_event_count}")
        print(f"ERG-006 Events: {erg_event_count}")
        print(f"Maximum Observed Occupancy: {maximum_occupancy}")
        print(f"Total Audit Entries: {len(audit_entries)}")
        print(f"Audit Log File: {audit_log_path}")

        print("\nSAVED EVIDENCE PACKAGES:")
        for art in saved_artifacts:
            print(f"\n- Event ID: {art.event_id}")
            print(f"  Directory: {art.event_dir}")
            print(f"  Raw Snapshot: {art.raw_snapshot_path.name} (exists: {art.raw_snapshot_path.is_file()})")
            print(f"  Annotated Snapshot: {art.annotated_snapshot_path.name} (exists: {art.annotated_snapshot_path.is_file()})")
            print(f"  Evidence Clip: {art.clip_path.name} (exists: {art.clip_path.is_file()})")
            print(f"  Clip Bounds: [{art.coverage.actual_clip_start_seconds:.2f}s -> {art.coverage.actual_clip_end_seconds:.2f}s]")
            print(f"  Clip Coverage: pre={art.coverage.actual_pre_seconds:.2f}s / post={art.coverage.actual_post_seconds:.2f}s ({art.coverage.clip_frame_count} frames)")
            print(f"  Metadata JSON: {art.metadata_path.name} (exists: {art.metadata_path.is_file()})")

        return 0

    except (FileNotFoundError, RuntimeError, ValueError, FileExistsError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
