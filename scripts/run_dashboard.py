"""SafetyShield prototype local operator dashboard.

Run from project root:
    .\\venv\\Scripts\\python.exe -B -m streamlit run scripts\\run_dashboard.py --server.address 127.0.0.1
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # type: ignore[import-untyped]

from src.dashboard.data import (
    DEFAULT_CONTROLS,
    evaluate_module_availability,
    get_system_status,
    load_audit_log,
    load_camera_dashboard_info,
    load_module_controls,
    load_recent_events,
    load_runtime_status,
    save_module_controls,
)


CONFIG_CAMERAS = PROJECT_ROOT / "config" / "cameras.yaml"
CONFIG_ZONES = PROJECT_ROOT / "config" / "zones.yaml"
CONFIG_CONTROLS = PROJECT_ROOT / "config" / "dashboard_controls.yaml"
ENV_PATH = PROJECT_ROOT / ".env"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
RUNTIME_STATUS_DIR = PROJECT_ROOT / "outputs" / "runtime"


def main() -> None:
    st.set_page_config(
        page_title="SafetyShield AI - Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 1. HEADER
    st.title("🛡️ SafetyShield AI")
    st.markdown("### Prototype Safety Monitoring Dashboard")
    st.warning(
        "**PROTOTYPE / DEVELOPMENT MODE**: This dashboard is a local developer/operator "
        "inspection prototype. Events and detections require human review and are not "
        "a substitute for formal site safety procedures or production certified monitoring."
    )

    # 2. SYSTEM STATUS
    st.markdown("---")
    st.markdown("#### 1. System & Hardware Status")
    sys_status = get_system_status()

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    with col_h1:
        st.metric("Target GPU", sys_status.gpu_name)
    with col_h2:
        cuda_badge = "Available (GTX 1650)" if sys_status.cuda_available else "CPU Only"
        st.metric("PyTorch CUDA", cuda_badge)
    with col_h3:
        st.metric("Environment", f"Python {sys_status.python_version}")
    with col_h4:
        st.metric("PyTorch / Streamlit", f"{sys_status.torch_version} / v{sys_status.streamlit_version}")

    # Milestone progression cards
    st.markdown("**Project Milestone Progression**")
    m_cols = st.columns(len(sys_status.milestones))
    for col, (m_id, m_state) in zip(m_cols, sys_status.milestones.items()):
        with col:
            color = "🟢" if m_state == "COMPLETE" else "🟡"
            st.caption(f"**{m_id}**")
            st.write(f"{color} {m_state}")

    # 3. CAMERA STATUS & SELECTION
    st.markdown("---")
    st.markdown("#### 2. Camera Configuration & Selection")
    cameras = load_camera_dashboard_info(CONFIG_CAMERAS, CONFIG_ZONES, ENV_PATH)

    if not cameras:
        st.error(f"No cameras could be loaded from {CONFIG_CAMERAS}.")
        selected_cam_id = None
        selected_cam = None
    else:
        cam_ids = list(cameras.keys())
        selected_cam_id = st.selectbox(
            "Select Active Camera for Inspection",
            options=cam_ids,
            index=0,
            help="Switch between configured cameras to inspect camera-specific status and module availability.",
        )
        selected_cam = cameras[selected_cam_id]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Camera ID", selected_cam.camera_id)
        with c2:
            st.metric("Source Type", selected_cam.source_type.upper())
        with c3:
            secret_badge = "YES (Configured)" if selected_cam.secret_configured else "NO (Missing)"
            st.metric("RTSP Secret Available", secret_badge, help=f"Referenced via: {selected_cam.secret_env_var}")
        with c4:
            zone_badge = f"{selected_cam.zone_count} Zone(s)" if selected_cam.has_zone else "Not Configured"
            st.metric("Zone Geometry", zone_badge)

        if selected_cam.camera_id == "live_cam_1" and not selected_cam.has_zone:
            st.info(
                "ℹ️ **Zone Status for live_cam_1**: No validated polygon is configured for this live view. "
                "The recorded `cam_good_test` polygon is never reused across different viewpoints. "
                "Zone rules (BAR-001, IDT-004, EXC-002) remain safely disabled for this camera."
            )

    # 4. LATEST KNOWN LIVE CAMERA METRICS (RUNTIME STATUS)
    st.markdown("---")
    st.markdown("#### 3. Latest Known Runtime Metrics")
    if selected_cam is not None:
        status_file = RUNTIME_STATUS_DIR / f"{selected_cam.camera_id}_status.json"
        runtime = load_runtime_status(status_file, stale_threshold_seconds=30.0)

        if runtime is None:
            st.info(
                f"No active/recent runtime status file found at `{status_file.name}`.\n\n"
                "Run `scripts\\run_rtsp_pipeline.py` to start live ingestion and stream telemetry."
            )
        else:
            state_color = "🟢" if runtime.state == "running" and not runtime.is_stale else ("🟡" if runtime.is_stale else "⚪")
            st.markdown(
                f"**Pipeline State**: {state_color} `{runtime.state.upper()}` | "
                f"**Session ID**: `{runtime.session_id}` | "
                f"**Last Update**: {runtime.updated_at_utc} ({runtime.age_seconds:.1f}s ago)"
            )
            if runtime.is_stale:
                st.warning(f"⚠️ Telemetry heartbeat is stale (age > 30s). The live pipeline process may be stopped.")

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                res_str = f"{runtime.decoded_width}x{runtime.decoded_height}" if runtime.decoded_width else "N/A"
                st.metric("Decoded Resolution", res_str)
            with m2:
                src_fps = f"{runtime.source_fps:.1f} FPS" if runtime.source_fps else "N/A"
                st.metric("Reported Source FPS", src_fps)
            with m3:
                st.metric("Processing Rate", f"{runtime.processing_fps:.2f} FPS")
            with m4:
                st.metric("Frames Received / Processed", f"{runtime.frames_received} / {runtime.frames_processed}")
            with m5:
                st.metric("Overwritten / Dropped", runtime.frames_dropped, help="Intentional latest-frame drops to prevent lag.")

            m6, m7, m8, m9, m10 = st.columns(5)
            with m6:
                st.metric("Failed Reads", runtime.failed_reads)
            with m7:
                st.metric("Reconnect Count", runtime.reconnect_count)
            with m8:
                st.metric("Temporary Track IDs", runtime.temporary_track_count)
            with m9:
                st.metric("Zone Rules Active", "YES" if runtime.zones_enabled else "DISABLED")
            with m10:
                st.metric("Schema Version", runtime.schema_version)

    # 5. MODULE CONTROLS & AVAILABILITY
    st.markdown("---")
    st.markdown("#### 4. Prototype Module Controls & Availability")
    st.caption("Toggle prototype modules globally and view architectural availability for the currently selected camera.")

    try:
        current_modules = load_module_controls(CONFIG_CONTROLS)
    except Exception as err:
        st.error(f"Error loading module controls: {err}")
        current_modules = {k: dict(v) for k, v in DEFAULT_CONTROLS.items()}

    if selected_cam is not None:
        evaluations = evaluate_module_availability(current_modules, selected_cam)

        with st.form("module_controls_form"):
            updated_modules = {}
            for mod_id, meta in DEFAULT_CONTROLS.items():
                cur = current_modules.get(mod_id, meta)
                ev = evaluations[mod_id]

                c_label, c_toggle, c_avail = st.columns([3, 2, 4])
                with c_label:
                    st.markdown(f"**{meta['label']}**")
                    st.caption(meta["description"])
                with c_toggle:
                    is_enabled = st.toggle(
                        f"Global Enable##{mod_id}",
                        value=bool(cur.get("enabled", True)),
                        key=f"toggle_{mod_id}",
                    )
                    updated_modules[mod_id] = {
                        "enabled": is_enabled,
                        "label": meta["label"],
                        "description": meta["description"],
                    }
                with c_avail:
                    if ev.available:
                        st.success(f"✅ Operational for `{selected_cam.camera_id}`")
                    else:
                        st.warning(f"⚠️ {ev.unavailable_reason}")

            st.caption("ℹ️ *Note: Module changes apply on next pipeline start (restart pipeline to apply).*")
            submitted = st.form_submit_button("Save Module Configuration", width="stretch")
            if submitted:
                try:
                    save_module_controls(CONFIG_CONTROLS, updated_modules)
                    st.success(f"Saved configuration to `{CONFIG_CONTROLS.name}`! Changes apply on next pipeline start.")
                    st.rerun()
                except Exception as save_err:
                    st.error(f"Failed to save controls: {save_err}")

    # 6. RECENT EVIDENCE & EVENTS
    st.markdown("---")
    st.markdown("#### 5. Recent Safety Events & Evidence")
    st.caption("Recorded-video prototype evidence events from Milestone 5 (source: `cam_good_test.mp4`).")

    events = load_recent_events(EVIDENCE_DIR)
    if not events:
        st.info("No recorded evidence events discovered in `evidence/` directory.")
    else:
        event_labels = [
            f"{e.event_id} | {e.module_id} ({e.event_type}) | Track {e.track_id} | t={e.source_timestamp_seconds}s"
            for e in events
        ]
        selected_idx = st.selectbox(
            "Select Event for Detailed Inspection",
            options=range(len(events)),
            format_func=lambda i: event_labels[i],
            index=0,
        )
        selected_event = events[selected_idx]

        ev_c1, ev_c2, ev_c3, ev_c4 = st.columns(4)
        with ev_c1:
            st.metric("Event ID", selected_event.event_id)
        with ev_c2:
            st.metric("Module & Type", f"{selected_event.module_id} ({selected_event.event_type})")
        with ev_c3:
            st.metric("Temporary Track ID", f"Track {selected_event.track_id}", help="Temporary session-local track ID.")
        with ev_c4:
            st.metric("Source Video / Time", f"{selected_event.source_video or 'N/A'} @ {selected_event.source_timestamp_seconds}s")

        # Visual evidence display
        snap_col1, snap_col2 = st.columns(2)
        with snap_col1:
            st.markdown("**Annotated Event Snapshot**")
            if selected_event.annotated_snapshot_path:
                st.image(str(selected_event.annotated_snapshot_path), use_container_width=True)
            else:
                st.warning("Annotated snapshot not available.")

        with snap_col2:
            st.markdown("**Raw Native Snapshot**")
            if selected_event.raw_snapshot_path:
                st.image(str(selected_event.raw_snapshot_path), use_container_width=True)
            else:
                st.warning("Raw native snapshot not available.")

        # Evidence clip
        st.markdown("**Evidence Video Clip**")
        if selected_event.video_clip_path:
            st.video(str(selected_event.video_clip_path))
            st.caption(f"Location: `{selected_event.video_clip_path.name}`")
        else:
            st.info("No video clip available for this event.")

        # Raw metadata expander
        with st.expander("View Full Event Metadata (JSON)"):
            st.json(selected_event.raw_metadata)

    # 7. PROTOTYPE AUDIT LOG
    st.markdown("---")
    st.markdown("#### 6. Prototype Audit Log")
    st.caption("Internal development activity log (prototype audit log; not immutable or forensically certified).")
    audit_entries = load_audit_log(EVIDENCE_DIR)

    if not audit_entries:
        st.info("No audit entries found.")
    else:
        table_data = [
            {
                "Timestamp (UTC)": entry.audit_timestamp_utc or "N/A",
                "Action": entry.action,
                "Module": entry.module_id or "N/A",
                "Event ID": entry.event_id or "N/A",
                "Camera ID": entry.camera_id or "N/A",
                "Details": str(entry.details),
            }
            for entry in audit_entries[:20]
        ]
        st.dataframe(table_data, use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption(
        "SafetyShield AI Architecture © 2026. Local Developer Prototype. "
        "GTX 1650 / CUDA 13.0 / Single-feed low-latency pipeline."
    )


if __name__ == "__main__":
    main()
