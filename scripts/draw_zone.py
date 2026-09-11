"""Interactively draw and explicitly save one fixed-camera polygon zone."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml


WINDOW_NAME = "SafetyShield zone drawing"


def read_source_frame(video_path: Path, timestamp_seconds: float) -> tuple[np.ndarray, float]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError("Input video could not be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise RuntimeError("Input video has invalid FPS metadata")
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp_seconds * fps))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("Requested source frame could not be read")
        return frame, fps
    finally:
        capture.release()


def save_zone(
    path: Path,
    camera_id: str,
    width: int,
    height: int,
    zone_id: str,
    zone_name: str,
    zone_type: str,
    polygon: list[tuple[int, int]],
    confirm_inside_seconds: float,
    confirm_outside_seconds: float,
    state_ttl_seconds: float,
) -> None:
    document: object = None
    if path.exists():
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError("Existing zone configuration is not valid YAML") from error
    if document is None:
        document = {"cameras": {}}
    if not isinstance(document, dict):
        raise ValueError("Existing zone configuration must be a mapping")
    cameras = document.setdefault("cameras", {})
    if not isinstance(cameras, dict):
        raise ValueError("Existing cameras value must be a mapping")

    existing_camera = cameras.get(camera_id)
    if existing_camera is None:
        existing_camera = {
            "expected_resolution": [width, height],
            "fixed_camera": True,
            "zones": [],
        }
        cameras[camera_id] = existing_camera
    if not isinstance(existing_camera, dict):
        raise ValueError(f"Existing camera {camera_id} must be a mapping")
    if existing_camera.get("expected_resolution") != [width, height]:
        raise ValueError("Existing camera resolution does not match the selected video")
    if existing_camera.get("fixed_camera") is not True:
        raise ValueError("Existing camera is not configured as fixed")
    zones = existing_camera.setdefault("zones", [])
    if not isinstance(zones, list):
        raise ValueError("Existing camera zones must be a list")

    zone_record = {
        "zone_id": zone_id,
        "zone_name": zone_name,
        "zone_type": zone_type,
        "enabled": True,
        "polygon": [[x, y] for x, y in polygon],
        "confirm_inside_seconds": confirm_inside_seconds,
        "confirm_outside_seconds": confirm_outside_seconds,
        "state_ttl_seconds": state_ttl_seconds,
    }
    for index, existing_zone in enumerate(zones):
        if isinstance(existing_zone, dict) and existing_zone.get("zone_id") == zone_id:
            zones[index] = zone_record
            break
    else:
        zones.append(zone_record)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    temporary_path.replace(path)


def render(frame: np.ndarray, vertices: list[tuple[int, int]], finished: bool) -> np.ndarray:
    display = frame.copy()
    if vertices:
        points = np.asarray(vertices, dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(display, [points], finished, (0, 215, 255), 3, cv2.LINE_AA)
        for index, point in enumerate(vertices, start=1):
            cv2.circle(display, point, 6, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.putText(
                display,
                str(index),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                str(index),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    instructions = (
        "S: SAVE" if finished else "Left click: add | Backspace: undo | Enter: finish"
    )
    cv2.rectangle(display, (0, 0), (display.shape[1] - 1, 42), (0, 0, 0), -1)
    cv2.putText(
        display,
        f"{instructions} | R: reset | Esc/Q: cancel",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def draw_interactively(frame: np.ndarray) -> list[tuple[int, int]] | None:
    state: dict[str, object] = {"vertices": [], "finished": False}

    def mouse_callback(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and not state["finished"]:
            vertices = state["vertices"]
            assert isinstance(vertices, list)
            vertices.append((max(0, min(frame.shape[1] - 1, x)), max(0, min(frame.shape[0] - 1, y))))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    try:
        while True:
            vertices = state["vertices"]
            finished = state["finished"]
            assert isinstance(vertices, list) and isinstance(finished, bool)
            cv2.imshow(WINDOW_NAME, render(frame, vertices, finished))
            key = cv2.waitKeyEx(20)
            if key in (27, ord("q"), ord("Q")):
                return None
            if key in (ord("r"), ord("R")):
                vertices.clear()
                state["finished"] = False
            elif key in (8, 127) and not finished and vertices:
                vertices.pop()
            elif key in (10, 13) and not finished and len(vertices) >= 3:
                state["finished"] = True
            elif key in (ord("s"), ord("S")) and finished:
                return list(vertices)
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    return None
            except cv2.error:
                return None
    finally:
        cv2.destroyWindow(WINDOW_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("config/zones.yaml"))
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--zone-id", required=True)
    parser.add_argument("--zone-name", required=True)
    parser.add_argument("--zone-type", choices=("restricted", "counting"), required=True)
    parser.add_argument("--timestamp", type=float, default=30.0)
    parser.add_argument("--confirm-inside-seconds", type=float, default=0.20)
    parser.add_argument("--confirm-outside-seconds", type=float, default=0.20)
    parser.add_argument("--state-ttl-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.timestamp < 0:
            raise ValueError("timestamp cannot be negative")
        if min(
            args.confirm_inside_seconds,
            args.confirm_outside_seconds,
            args.state_ttl_seconds,
        ) < 0:
            raise ValueError("confirmation and state TTL values cannot be negative")
        frame, fps = read_source_frame(args.input, args.timestamp)
        print(
            f"Frame: {frame.shape[1]}x{frame.shape[0]} at {args.timestamp:.3f}s "
            f"({fps:.3f} FPS source)"
        )
        polygon = draw_interactively(frame)
        if polygon is None:
            print("Cancelled; zone configuration was not changed.")
            return 0
        save_zone(
            args.output,
            args.camera_id,
            frame.shape[1],
            frame.shape[0],
            args.zone_id,
            args.zone_name,
            args.zone_type,
            polygon,
            args.confirm_inside_seconds,
            args.confirm_outside_seconds,
            args.state_ttl_seconds,
        )
        print(f"Saved zone {args.zone_id} with {len(polygon)} vertices to {args.output}")
        print(f"Polygon: {polygon}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
