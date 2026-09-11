"""Secret-safe, latest-frame RTSP ingestion for one live camera."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import threading
import time
from typing import Callable, Mapping, Protocol

import cv2
import numpy as np
import yaml


class CameraConfigError(ValueError):
    """Raised when non-secret camera configuration is invalid."""


@dataclass(frozen=True, slots=True)
class RTSPCameraConfig:
    """Non-secret RTSP camera metadata safe to serialize and log."""

    camera_id: str
    source_type: str
    rtsp_env_var: str
    enabled: bool


@dataclass(frozen=True, slots=True)
class LatestFrame:
    """A native decoded frame and its session-relative capture time."""

    sequence: int
    captured_monotonic: float
    frame: np.ndarray


@dataclass(frozen=True, slots=True)
class RTSPMetrics:
    """Secret-free snapshot of reader state."""

    camera_id: str
    session_id: str
    connect_attempts: int
    successful_connections: int
    reconnect_count: int
    frames_received: int
    frames_processed: int
    frames_overwritten_or_dropped: int
    failed_reads: int
    consecutive_failed_reads: int
    time_to_first_frame: float | None
    runtime_seconds: float
    decoded_width: int | None
    decoded_height: int | None
    reported_source_fps: float | None
    resolution_changes: int
    connected: bool
    reconnect_delay_seconds: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class Capture(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def get(self, propId: int) -> float: ...
    def release(self) -> None: ...


CaptureFactory = Callable[[str, int, tuple[int, ...]], Capture]
LogFunction = Callable[[str], None]


def _require_mapping(value: object, label: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise CameraConfigError(f"{label} must be a mapping")
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraConfigError(f"{label} must be non-empty text")
    return value.strip()


def load_rtsp_camera_config(path: Path, camera_id: str) -> RTSPCameraConfig:
    """Load only non-secret camera metadata from YAML."""
    if not path.is_file():
        raise FileNotFoundError(f"Camera configuration not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise CameraConfigError("Camera configuration is not valid YAML") from error
    root = _require_mapping(raw, "camera configuration")
    cameras = _require_mapping(root.get("cameras"), "cameras")
    camera = _require_mapping(cameras.get(camera_id), f"camera {camera_id}")
    configured_id = _require_text(camera.get("camera_id"), "camera_id")
    if configured_id != camera_id:
        raise CameraConfigError("camera_id does not match its configuration key")
    source_type = _require_text(camera.get("source_type"), "source_type").lower()
    if source_type != "rtsp":
        raise CameraConfigError(f"camera {camera_id} source_type must be rtsp")
    env_name = _require_text(camera.get("rtsp_env_var"), "rtsp_env_var")
    if env_name != "SAFETYSHIELD_RTSP_LIVE_CAM_1":
        raise CameraConfigError("live_cam_1 must use the approved RTSP environment variable")
    enabled = camera.get("enabled")
    if not isinstance(enabled, bool):
        raise CameraConfigError(f"camera {camera_id} enabled must be true or false")
    if not enabled:
        raise CameraConfigError(f"camera {camera_id} is disabled")
    return RTSPCameraConfig(configured_id, source_type, env_name, enabled)


def resolve_rtsp_secret(config: RTSPCameraConfig) -> str:
    """Resolve the secret without incorporating its value into an error."""
    value = os.environ.get(config.rtsp_env_var)
    if value is None or not value.strip():
        raise CameraConfigError(
            f"Required environment variable {config.rtsp_env_var} is not configured"
        )
    return value


def zone_status_for_frame(
    zones_path: Path, camera_id: str, frame_width: int, frame_height: int
) -> tuple[bool, str]:
    """Safely decide whether camera-specific fixed geometry can be used."""
    if not zones_path.is_file():
        return False, "disabled / not configured"
    try:
        raw = yaml.safe_load(zones_path.read_text(encoding="utf-8"))
        root = _require_mapping(raw, "zone configuration")
        cameras = _require_mapping(root.get("cameras"), "cameras")
        camera = cameras.get(camera_id)
        if not isinstance(camera, Mapping):
            return False, "disabled / not configured"
        resolution = camera.get("expected_resolution")
        if resolution != [frame_width, frame_height]:
            return False, "disabled / resolution mismatch"
        if camera.get("fixed_camera") is not True:
            return False, "disabled / camera is not fixed"
        zones = camera.get("zones")
        if not isinstance(zones, list) or not zones:
            return False, "disabled / not configured"
        return True, "enabled"
    except (OSError, yaml.YAMLError, CameraConfigError):
        return False, "disabled / invalid configuration"


def _opencv_capture(source: str, backend: int, params: tuple[int, ...]) -> Capture:
    try:
        return cv2.VideoCapture(source, backend, list(params))
    except (TypeError, cv2.error):
        return cv2.VideoCapture(source, backend)


class RTSPReader:
    """Decode on one thread and expose at most one unconsumed native frame."""

    def __init__(
        self,
        camera_id: str,
        session_id: str,
        source: str,
        *,
        capture_factory: CaptureFactory = _opencv_capture,
        open_timeout_ms: int = 5_000,
        read_timeout_ms: int = 5_000,
        failed_read_threshold: int = 5,
        backoff_seconds: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 10.0),
        max_connect_attempts: int = 5,
        clock: Callable[[], float] = time.monotonic,
        logger: LogFunction = print,
    ) -> None:
        if not camera_id.strip() or not session_id.strip() or not source.strip():
            raise ValueError("camera_id, session_id and source must be non-empty")
        if open_timeout_ms <= 0 or read_timeout_ms <= 0:
            raise ValueError("capture timeouts must be positive")
        if failed_read_threshold <= 0 or max_connect_attempts <= 0:
            raise ValueError("failure thresholds must be positive")
        if not backoff_seconds or any(value < 0 for value in backoff_seconds):
            raise ValueError("backoff_seconds must be non-empty and non-negative")
        self.camera_id = camera_id
        self.session_id = session_id
        self._source = source
        self._capture_factory = capture_factory
        self._timeouts = (open_timeout_ms, read_timeout_ms)
        self._failed_read_threshold = failed_read_threshold
        self._backoff_seconds = backoff_seconds
        self._max_connect_attempts = max_connect_attempts
        self._clock = clock
        self._logger = logger
        self._stop_event = threading.Event()
        self._frame_ready = threading.Condition()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._capture: Capture | None = None
        self._latest: LatestFrame | None = None
        self._delivered_sequence: int | None = None
        self._started_at: float | None = None
        self._first_frame_at: float | None = None
        self._connect_attempts = 0
        self._successful_connections = 0
        self._reconnect_count = 0
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_overwritten = 0
        self._failed_reads = 0
        self._consecutive_failed_reads = 0
        self._decoded_size: tuple[int, int] | None = None
        self._reported_source_fps: float | None = None
        self._resolution_changes = 0
        self._connected = False
        self._terminal_failure = False
        self._reconnect_delay_seconds = 0.0

    @property
    def uses_single_frame_slot(self) -> bool:
        return True

    def _capture_params(self) -> tuple[int, ...]:
        params: list[int] = []
        open_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        read_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if isinstance(open_prop, int):
            params.extend((open_prop, self._timeouts[0]))
        if isinstance(read_prop, int):
            params.extend((read_prop, self._timeouts[1]))
        return tuple(params)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._started_at = self._clock()
        self._thread = threading.Thread(
            target=self._run, name=f"rtsp-reader-{self.camera_id}", daemon=False
        )
        self._thread.start()

    def _safe_release(self, capture: Capture | None) -> None:
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            pass

    def _connect(self) -> Capture | None:
        with self._state_lock:
            self._connect_attempts += 1
            attempt = self._connect_attempts
        self._logger(f"Connecting camera {self.camera_id} (attempt {attempt})...")
        try:
            capture = self._capture_factory(
                self._source, getattr(cv2, "CAP_FFMPEG", 0), self._capture_params()
            )
            opened = capture.isOpened()
        except Exception:
            self._logger(f"Camera open failed for {self.camera_id}")
            return None
        if not opened:
            self._safe_release(capture)
            self._logger(f"Camera open failed for {self.camera_id}")
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        with self._state_lock:
            self._capture = capture
            self._successful_connections += 1
            self._connected = True
            self._reported_source_fps = fps if math.isfinite(fps) and fps > 0 else None
        self._logger(f"Connected camera {self.camera_id}")
        return capture

    def _schedule_reconnect(self, capture: Capture | None) -> None:
        self._safe_release(capture)
        with self._state_lock:
            if self._capture is capture:
                self._capture = None
            self._connected = False
            self._reconnect_count += 1

    def _publish(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("invalid decoded frame")
        now = self._clock()
        height, width = frame.shape[:2]
        with self._state_lock:
            old_size = self._decoded_size
            if old_size is not None and old_size != (width, height):
                self._resolution_changes += 1
                self._logger(
                    f"WARNING: decoded resolution changed for {self.camera_id}; "
                    "fixed geometry must remain disabled"
                )
            self._decoded_size = (width, height)
            self._frames_received += 1
            sequence = self._frames_received
            self._consecutive_failed_reads = 0
            self._reconnect_delay_seconds = 0.0
            if self._first_frame_at is None:
                self._first_frame_at = now
        with self._frame_ready:
            if self._latest is not None:
                with self._state_lock:
                    self._frames_overwritten += 1
            self._latest = LatestFrame(sequence, now, frame)
            self._frame_ready.notify_all()

    def _run(self) -> None:
        backoff_index = 0
        consecutive_connection_failures = 0
        capture: Capture | None = None
        try:
            while not self._stop_event.is_set():
                if capture is None:
                    capture = self._connect()
                    if capture is None:
                        consecutive_connection_failures += 1
                        if consecutive_connection_failures >= self._max_connect_attempts:
                            with self._state_lock:
                                self._terminal_failure = True
                            break
                        self._schedule_reconnect(None)
                        delay = self._backoff_seconds[min(backoff_index, len(self._backoff_seconds) - 1)]
                        with self._state_lock:
                            self._reconnect_delay_seconds = delay
                        backoff_index = min(backoff_index + 1, len(self._backoff_seconds) - 1)
                        if self._stop_event.wait(delay):
                            break
                        continue
                try:
                    ok, frame = capture.read()
                except Exception:
                    ok, frame = False, None
                if self._stop_event.is_set():
                    break
                valid_frame = (
                    ok
                    and isinstance(frame, np.ndarray)
                    and frame.ndim == 3
                    and frame.shape[2] == 3
                    and frame.size > 0
                )
                if valid_frame:
                    assert frame is not None
                    self._publish(frame)
                    consecutive_connection_failures = 0
                    backoff_index = 0
                    continue
                with self._state_lock:
                    self._failed_reads += 1
                    self._consecutive_failed_reads += 1
                    failed = self._consecutive_failed_reads
                self._logger(f"Frame read failed for {self.camera_id}")
                if failed < self._failed_read_threshold:
                    if self._stop_event.wait(0.01):
                        break
                    continue
                if self._first_frame_at is None:
                    consecutive_connection_failures += 1
                    if consecutive_connection_failures >= self._max_connect_attempts:
                        with self._state_lock:
                            self._terminal_failure = True
                        break
                self._schedule_reconnect(capture)
                capture = None
                delay = self._backoff_seconds[min(backoff_index, len(self._backoff_seconds) - 1)]
                with self._state_lock:
                    self._reconnect_delay_seconds = delay
                backoff_index = min(backoff_index + 1, len(self._backoff_seconds) - 1)
                if self._stop_event.wait(delay):
                    break
        finally:
            self._safe_release(capture)
            with self._state_lock:
                if self._capture is capture:
                    self._capture = None
                self._connected = False
            with self._frame_ready:
                self._frame_ready.notify_all()

    def read_latest(self, timeout: float = 0.0) -> LatestFrame | None:
        """Consume the newest frame; skipped replaced frames are already counted."""
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = self._clock() + timeout
        with self._frame_ready:
            while self._latest is None and not self._stop_event.is_set():
                if self._terminal_failure:
                    return None
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return None
                self._frame_ready.wait(remaining)
            latest = self._latest
            self._latest = None
            if latest is not None:
                with self._state_lock:
                    if self._delivered_sequence is not None:
                        self._frames_overwritten += 1
                    self._delivered_sequence = latest.sequence
            return latest

    def mark_processed(self, sequence: int) -> None:
        if sequence <= 0:
            raise ValueError("sequence must be positive")
        with self._state_lock:
            if self._delivered_sequence != sequence:
                raise ValueError("sequence is not the currently delivered frame")
            self._frames_processed += 1
            self._delivered_sequence = None

    def metrics(self) -> RTSPMetrics:
        now = self._clock()
        with self._state_lock:
            started = self._started_at
            first = self._first_frame_at
            size = self._decoded_size
            return RTSPMetrics(
                camera_id=self.camera_id,
                session_id=self.session_id,
                connect_attempts=self._connect_attempts,
                successful_connections=self._successful_connections,
                reconnect_count=self._reconnect_count,
                frames_received=self._frames_received,
                frames_processed=self._frames_processed,
                frames_overwritten_or_dropped=self._frames_overwritten,
                failed_reads=self._failed_reads,
                consecutive_failed_reads=self._consecutive_failed_reads,
                time_to_first_frame=(None if started is None or first is None else first - started),
                runtime_seconds=0.0 if started is None else max(0.0, now - started),
                decoded_width=None if size is None else size[0],
                decoded_height=None if size is None else size[1],
                reported_source_fps=self._reported_source_fps,
                resolution_changes=self._resolution_changes,
                connected=self._connected,
                reconnect_delay_seconds=self._reconnect_delay_seconds,
            )

    @property
    def terminal_failure(self) -> bool:
        with self._state_lock:
            return self._terminal_failure

    def stop(self, join_timeout: float = 6.0) -> bool:
        """Request shutdown, release capture to unblock reads, and join boundedly."""
        self._stop_event.set()
        with self._frame_ready:
            if self._latest is not None:
                with self._state_lock:
                    self._frames_overwritten += 1
                self._latest = None
        with self._state_lock:
            capture = self._capture
            if self._delivered_sequence is not None:
                self._frames_overwritten += 1
                self._delivered_sequence = None
        self._safe_release(capture)
        with self._frame_ready:
            self._frame_ready.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, join_timeout))
        return thread is None or not thread.is_alive()

    def __enter__(self) -> "RTSPReader":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
