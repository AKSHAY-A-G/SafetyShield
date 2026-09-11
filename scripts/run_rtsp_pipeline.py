"""Run person detection and camera/session-local tracking on one live RTSP feed."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import time
from typing import Any, Callable
from uuid import uuid4

import cv2
from dotenv import load_dotenv
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_person_detection import create_output_writer  # noqa: E402
from scripts.run_person_tracking import draw_track  # noqa: E402
from src.camera.rtsp_reader import (  # noqa: E402
    CameraConfigError,
    RTSPMetrics,
    RTSPReader,
    load_rtsp_camera_config,
    resolve_rtsp_secret,
    zone_status_for_frame,
)
from src.detection.person_detector import PersonDetector  # noqa: E402
from src.tracking.person_tracker import ByteTrackConfig, PersonTracker  # noqa: E402


DEFAULT_CAMERAS = PROJECT_ROOT / "config" / "cameras.yaml"
DEFAULT_ZONES = PROJECT_ROOT / "config" / "zones.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "live_cam_1_rtsp_test.mp4"
DEFAULT_REFERENCE = PROJECT_ROOT / "outputs" / "live_cam_1_reference.jpg"


@dataclass(frozen=True, slots=True)
class LiveRunResult:
    metrics: RTSPMetrics
    acceptance_seconds: float
    processing_fps: float
    peak_gpu_mib: float
    person_tracks_observed: int
    zone_status: str
    clean_shutdown: bool
    reference_frame: Path | None
    review_output: Path | None
    review_fps: float | None = None


def save_reference_frame(path: Path, frame) -> Path:
    """Save one unannotated native frame, failing clearly without secret data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not save reference frame: {path}")
    return path


def draw_live_status(frame, camera_id: str, fps: float, zone_status: str) -> None:
    lines = (
        f"{camera_id} | LIVE",
        f"Processing: {fps:.2f} FPS",
        f"Zones: {zone_status}",
    )
    for index, label in enumerate(lines):
        y = 28 + index * 26
        cv2.putText(
            frame,
            label,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 220, 20),
            2,
            cv2.LINE_AA,
        )


def run_live_pipeline(
    *,
    camera_id: str,
    session_id: str,
    source: str,
    duration_seconds: float,
    output: Path | None,
    reference_path: Path | None,
    display: bool,
    output_fps: float | None = None,
    reader_factory: Callable[..., Any] = RTSPReader,
    detector_factory: Callable[..., Any] = PersonDetector,
    tracker_factory: Callable[..., Any] = PersonTracker,
    clock: Callable[[], float] = time.monotonic,
    initial_frame_timeout_seconds: float = 45.0,
) -> LiveRunResult:
    """Run a finite live session; injectable factories keep unit tests offline.

    Authoritative live session elapsed time uses the monotonic clock. The
    optional review video records only processed latest frames. Its writer
    FPS defaults to a prototype cap at 10 FPS (min(source_fps, 10.0)),
    which approximately matches the current GTX 1650 processing rate rather
    than performing exact live-time reconstruction. Real live event time
    must not be derived from review video frame numbers.
    """
    if duration_seconds <= 0 or initial_frame_timeout_seconds <= 0:
        raise ValueError("duration and initial-frame timeout must be positive")
    if output_fps is not None and output_fps <= 0:
        raise ValueError("output_fps must be positive")
    reader = reader_factory(camera_id, session_id, source)
    writer = None
    clean_shutdown = False
    result: LiveRunResult | None = None
    reference_saved: Path | None = None
    output_saved: Path | None = None
    processed = 0
    tracker = None
    zone_status = "disabled / not configured"
    processing_started = 0.0
    try:
        reader.start()
        wait_started = clock()
        first = None
        while clock() - wait_started < initial_frame_timeout_seconds:
            first = reader.read_latest(timeout=0.25)
            if first is not None:
                break
            if reader.terminal_failure:
                break
        if first is None:
            raise RuntimeError(f"No initial frame received from camera {camera_id}")

        raw_height, raw_width = first.frame.shape[:2]
        zone_configured, zone_status = zone_status_for_frame(
            DEFAULT_ZONES, camera_id, raw_width, raw_height
        )
        if zone_configured:
            zone_status = "disabled / Milestone 6 ingestion only"
        # Milestone 6 deliberately does not execute zone/temporal rule engines.
        detector = detector_factory(
            checkpoint="yolo26n.pt", confidence_threshold=0.20, image_size=960, device=0
        )
        tracker = tracker_factory(camera_id, session_id, ByteTrackConfig())
        torch.cuda.reset_peak_memory_stats(0)
        if reference_path is not None:
            reference_saved = save_reference_frame(reference_path, first.frame)
        source_fps = reader.metrics().reported_source_fps
        writer_fps: float | None = None
        if output is not None:
            if output_fps is not None:
                writer_fps = float(output_fps)
            else:
                # Prototype default: cap review playback at 10.0 FPS. This approximately
                # matches the current GTX 1650 processing rate (~10 FPS), providing
                # playback closer to real session duration without claiming exact
                # live-time reconstruction.
                writer_fps = min(source_fps, 10.0) if source_fps is not None else 10.0
            writer = create_output_writer(output, raw_width, raw_height, writer_fps)
            output_saved = output

        processing_started = clock()
        pending = first
        while clock() - processing_started < duration_seconds:
            latest = pending or reader.read_latest(timeout=0.10)
            pending = None
            if latest is None:
                if reader.terminal_failure:
                    raise RuntimeError(f"RTSP reader stopped for camera {camera_id}")
                continue
            native_height, native_width = latest.frame.shape[:2]
            if (native_width, native_height) != (raw_width, raw_height):
                zone_status = "disabled / resolution mismatch"
                if writer is not None:
                    writer.release()
                    writer = None
                    output_saved = None
                    print("WARNING: review recording stopped after resolution change")
                raw_width, raw_height = native_width, native_height
            elapsed = clock() - processing_started
            detections = detector.detect(latest.frame)
            tracks = tracker.update(
                detections,
                latest.frame.shape,
                frame_index=processed,
                timestamp_seconds=elapsed,
            )
            annotated = latest.frame.copy()
            for track in tracks:
                draw_track(annotated, track)
            processed += 1
            reader.mark_processed(latest.sequence)
            fps = processed / max(clock() - processing_started, 1e-9)
            draw_live_status(annotated, camera_id, fps, zone_status)
            if writer is not None:
                writer.write(annotated)
            if display:
                cv2.imshow("SafetyShield live camera", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        torch.cuda.synchronize(0)
        acceptance = clock() - processing_started
        peak_mib = torch.cuda.max_memory_allocated(0) / (1024**2)
        metrics = reader.metrics()
        result = LiveRunResult(
            metrics=metrics,
            acceptance_seconds=acceptance,
            processing_fps=processed / acceptance if acceptance else 0.0,
            peak_gpu_mib=peak_mib,
            person_tracks_observed=0 if tracker is None else tracker.unique_track_count,
            zone_status=zone_status,
            clean_shutdown=False,
            reference_frame=reference_saved,
            review_output=output_saved,
            review_fps=writer_fps,
        )
    finally:
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        clean_shutdown = reader.stop()
    if result is None:
        raise RuntimeError(f"Live pipeline did not produce results for camera {camera_id}")
    return replace(
        result, metrics=reader.metrics(), clean_shutdown=clean_shutdown
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", default="live_cam_1")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--cameras-config", type=Path, default=DEFAULT_CAMERAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="Playback FPS for optional review video (default: min(source_fps, 10.0), approximately matching current GTX 1650 processing rate)",
    )
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument("--save-reference-frame", type=Path, default=DEFAULT_REFERENCE)
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", action="store_true")
    display_group.add_argument("--no-display", action="store_false", dest="display")
    parser.set_defaults(display=False)
    return parser.parse_args(argv)


def _print_results(result: LiveRunResult, clean_shutdown: bool) -> None:
    metrics = result.metrics
    resolution = (
        "unavailable"
        if metrics.decoded_width is None or metrics.decoded_height is None
        else f"{metrics.decoded_width}x{metrics.decoded_height}"
    )
    print("\nMILESTONE 6 LIVE RUN RESULTS")
    print(f"Camera ID: {metrics.camera_id}")
    print(f"Session ID: {metrics.session_id}")
    print("Secret source: SAFETYSHIELD_RTSP_LIVE_CAM_1")
    print("RTSP connection: SUCCESS")
    first_frame = metrics.time_to_first_frame
    print(
        "Time to first frame: "
        + ("unavailable" if first_frame is None else f"{first_frame:.3f} seconds")
    )
    print(f"Decoded resolution: {resolution}")
    print(f"Reported source FPS: {metrics.reported_source_fps}")
    print(f"Acceptance duration: {result.acceptance_seconds:.3f} seconds")
    print(f"Frames received: {metrics.frames_received}")
    print(f"Frames processed: {metrics.frames_processed}")
    print(f"Frames dropped / overwritten: {metrics.frames_overwritten_or_dropped}")
    print(f"Failed reads: {metrics.failed_reads}")
    print(f"Connect attempts: {metrics.connect_attempts}")
    print(f"Successful connections: {metrics.successful_connections}")
    print(f"Reconnect count: {metrics.reconnect_count}")
    print(f"Processing FPS: {result.processing_fps:.3f}")
    print(f"Peak GPU memory: {result.peak_gpu_mib:.3f} MiB")
    print(f"Temporary person tracks observed: {result.person_tracks_observed}")
    print(f"Zone status: {result.zone_status}")
    print("Zone rules: DISABLED")
    print("Temporal rules: DISABLED")
    print("Live evidence: deferred; no live pre/post buffer")
    print(f"Reference frame: {result.reference_frame}")
    print(f"Review material: {result.review_output}")
    if result.review_fps is not None:
        print(f"Review recording FPS: {result.review_fps:.2f}")
    else:
        print("Review recording FPS: disabled")
    print(f"Clean shutdown: {'YES' if clean_shutdown else 'NO'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session_id = args.session_id or f"live_{uuid4().hex[:12]}"
    try:
        # Load runtime values without displaying or inspecting .env contents.
        load_dotenv(PROJECT_ROOT / ".env")
        config = load_rtsp_camera_config(args.cameras_config, args.camera_id)
        source = resolve_rtsp_secret(config)
        print(f"RTSP secret configured: YES ({config.rtsp_env_var})")
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)
        result = run_live_pipeline(
            camera_id=config.camera_id,
            session_id=session_id,
            source=source,
            duration_seconds=args.duration_seconds,
            output=None if args.no_output else args.output,
            reference_path=args.save_reference_frame,
            display=args.display,
            output_fps=args.output_fps,
        )
        _print_results(result, result.clean_shutdown)
        return 0 if result.clean_shutdown else 1
    except KeyboardInterrupt:
        print("Interrupted; clean shutdown requested")
        return 130
    except (CameraConfigError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    except Exception as error:
        print(f"ERROR: live pipeline failed safely ({type(error).__name__})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
