"""Streamlit dashboard for real-time YOLO object detection and tracking."""

from __future__ import annotations

import hashlib
import html
import logging
import time
import traceback
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Real-Time Object Detection & Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

from detector import YOLODetector  # noqa: E402
from errors import AppError, ConfigurationError  # noqa: E402
from utils import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE,
    INPUT_VIDEOS_DIR,
    OUTPUT_VIDEOS_DIR,
    SUPPORTED_UPLOAD_TYPES,
    TRACKER_OPTIONS,
    cuda_available,
    dependency_versions,
    device_label,
    discover_models,
    ensure_directories,
    format_bytes,
    format_duration,
    make_output_path,
    probe_video,
    probe_cameras,
    relative_path,
    resolve_model_path,
    save_uploaded_video,
)
from video_processor import (  # noqa: E402
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    ProcessingConfig,
    VideoProcessor,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

ensure_directories()

STATUS_META = {
    "idle": ("Idle", "#475569", "#F1F5F9", "#CBD5E1"),
    "starting": ("Starting", "#0369A1", "#E0F2FE", "#7DD3FC"),
    "running": ("Running", "#047857", "#D1FAE5", "#6EE7B7"),
    "paused": ("Paused", "#B45309", "#FEF3C7", "#FCD34D"),
    "stopping": ("Stopping", "#C2410C", "#FFEDD5", "#FDBA74"),
    "completed": ("Completed", "#047857", "#D1FAE5", "#6EE7B7"),
    "stopped": ("Stopped", "#475569", "#F1F5F9", "#CBD5E1"),
    "error": ("Error", "#B91C1C", "#FEE2E2", "#FCA5A5"),
}

CSS = """
<style>
    :root {
        --ink: #0f172a;
        --muted: #64748b;
        --line: #e2e8f0;
        --card: #ffffff;
        --brand: #4f46e5;
    }
    .block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1500px; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .hero {
        background: radial-gradient(circle at 88% 8%, rgba(99,102,241,.50), transparent 38%),
                    linear-gradient(120deg, #0b1120 0%, #111c34 58%, #172554 100%);
        color: #f8fafc;
        padding: 28px 32px 26px 32px;
        border-radius: 22px;
        margin-bottom: 22px;
        box-shadow: 0 18px 45px rgba(15,23,42,.18);
    }
    .hero-kicker {
        color: #a5b4fc; font-size: .73rem; font-weight: 700;
        letter-spacing: .14em; text-transform: uppercase; margin-bottom: 9px;
    }
    .hero h1 { color: #fff; font-size: 2.18rem; line-height: 1.14; margin: 0 0 10px 0; letter-spacing: -.03em; }
    .hero p { color: #cbd5e1; max-width: 900px; font-size: 1rem; line-height: 1.6; margin: 0 0 16px 0; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip {
        display: inline-block; padding: 5px 11px; border-radius: 999px;
        background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.16);
        color: #e2e8f0; font-size: .74rem; font-weight: 600;
    }
    .session-bar {
        display: flex; align-items: center; justify-content: space-between; gap: 14px;
        background: var(--card); border: 1px solid var(--line); border-radius: 16px;
        padding: 13px 18px; margin-bottom: 16px; box-shadow: 0 5px 18px rgba(15,23,42,.04);
    }
    .session-left { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .session-title { font-weight: 750; color: var(--ink); font-size: .98rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .session-sub { color: var(--muted); font-size: .78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .status-pill {
        display: inline-flex; align-items: center; gap: 7px; padding: 5px 11px;
        border-radius: 999px; border: 1px solid; font-size: .75rem; font-weight: 750;
        letter-spacing: .01em; white-space: nowrap;
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
    div[data-testid="stMetric"] {
        background: var(--card); border: 1px solid var(--line); border-radius: 15px;
        padding: 13px 15px; box-shadow: 0 4px 14px rgba(15,23,42,.035);
    }
    div[data-testid="stMetricLabel"] p { color: var(--muted); font-size: .75rem; font-weight: 650; }
    div[data-testid="stMetricValue"] { color: var(--ink); font-size: 1.46rem; font-weight: 780; }
    .video-placeholder {
        min-height: 315px; display: flex; align-items: center; justify-content: center;
        border: 1px dashed #94a3b8; border-radius: 16px; color: #64748b;
        background: linear-gradient(135deg, #f8fafc, #eef2ff); font-size: .94rem;
        text-align: center; padding: 24px;
    }
    .step-card {
        background: var(--card); border: 1px solid var(--line); border-radius: 16px;
        padding: 18px; height: 100%; box-shadow: 0 5px 16px rgba(15,23,42,.035);
    }
    .step-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 27px; height: 27px; border-radius: 9px; background: #eef2ff;
        color: #4338ca; font-weight: 800; font-size: .8rem; margin-bottom: 10px;
    }
    .step-card h4 { margin: 0 0 6px 0; color: var(--ink); font-size: .96rem; }
    .step-card p { margin: 0; color: var(--muted); font-size: .82rem; line-height: 1.5; }
    .pipeline {
        background: #0b1120; color: #cbd5e1; border-radius: 16px; padding: 16px 18px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: .78rem; line-height: 1.65; overflow-x: auto;
    }
    .pipeline b { color: #a5b4fc; }
    div[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid var(--line); }
    div[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }
    .sidebar-brand { font-weight: 800; color: var(--ink); font-size: 1.03rem; margin-bottom: 2px; }
    .sidebar-sub { color: var(--muted); font-size: .75rem; margin-bottom: 12px; }
    .small-note { color: var(--muted); font-size: .74rem; line-height: 1.45; }
    .stButton > button { border-radius: 10px; font-weight: 650; }
    div[data-testid="stDownloadButton"] > button { border-radius: 10px; font-weight: 700; }
    @media (max-width: 900px) {
        .hero h1 { font-size: 1.65rem; }
        .session-bar { align-items: flex-start; flex-direction: column; }
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

HERO = """
<div class="hero">
    <div class="hero-kicker">Computer Vision · YOLO + Multi-Object Tracking</div>
    <h1>Real-Time Object Detection &amp; Tracking</h1>
    <p>
        Detect objects in every frame, keep a persistent ID on each object with
        ByteTrack or BoT-SORT, and inspect live counts, FPS and tracking statistics.
        Run a webcam session or upload a video and download the annotated result.
    </p>
    <div class="chips">
        <span class="chip">YOLO11 / Ultralytics</span>
        <span class="chip">ByteTrack</span>
        <span class="chip">BoT-SORT</span>
        <span class="chip">OpenCV</span>
        <span class="chip">Streamlit</span>
        <span class="chip">CPU / CUDA</span>
    </div>
</div>
"""


# ---------------------------------------------------------------------------
# Session state and helpers
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "detector": None,
        "detector_key": None,
        "processor": None,
        "last_terminal_render": None,
        "camera_indices": None,
        "upload_signature": None,
        "upload_path": None,
        "upload_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _snapshot() -> dict | None:
    processor = st.session_state.get("processor")
    if processor is None:
        return None
    try:
        return processor.snapshot()
    except Exception:  # noqa: BLE001 - a broken thread should not kill the UI
        LOGGER.exception("Could not read processor snapshot")
        return None


def _is_active(snap: dict | None) -> bool:
    return bool(snap and snap.get("status") in ACTIVE_STATUSES)


def _device_preference(choice: str) -> str:
    if choice == "CPU":
        return "cpu"
    if choice.startswith("CUDA"):
        return "cuda"
    return "auto"


def _get_detector(model_path: str, device_choice: str, active: bool = False) -> YOLODetector:
    """Cache one loaded detector per model/device combination."""
    resolved = resolve_model_path(model_path)
    preference = _device_preference(device_choice)
    key = f"{resolved}|{preference}"
    current = st.session_state.get("detector")
    if (
        current is not None
        and st.session_state.get("detector_key") == key
        and current.is_loaded
    ):
        return current

    if active:
        # Settings are disabled during processing. Never swap a live detector.
        if current is not None and current.is_loaded:
            return current
        raise ConfigurationError("Cannot change the model while processing is active.")

    if current is not None:
        try:
            current.close()
        except Exception:  # noqa: BLE001
            LOGGER.debug("Error closing previous detector", exc_info=True)
    st.session_state.detector = None
    st.session_state.detector_key = None

    detector = YOLODetector(resolved, device=preference)
    detector.load()
    st.session_state.detector = detector
    st.session_state.detector_key = key
    return detector


def _upload_signature(uploaded) -> str:
    buffer = uploaded.getbuffer()
    sample = bytes(buffer[: min(len(buffer), 65536)])
    digest = hashlib.sha1(sample).hexdigest()[:12]
    return f"{uploaded.name}:{uploaded.size}:{digest}"


def _existing_videos() -> list[Path]:
    if not INPUT_VIDEOS_DIR.exists():
        return []
    return sorted(
        [
            path
            for path in INPUT_VIDEOS_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        ],
        key=lambda path: path.name.lower(),
    )


def _status_markup(status: str) -> str:
    label, color, background, border = STATUS_META.get(status, STATUS_META["idle"])
    return (
        f'<span class="status-pill" style="color:{color};background:{background};'
        f'border-color:{border}"><span class="status-dot"></span>{html.escape(label)}</span>'
    )


@st.cache_data(show_spinner=False, max_entries=3)
def _read_bytes_cached(path: str, mtime: float, size: int) -> bytes:
    del mtime, size  # cache key components only
    return Path(path).read_bytes()


# ---------------------------------------------------------------------------
# Main dashboard rendering
# ---------------------------------------------------------------------------
def _render_metrics(snap: dict) -> None:
    stats = snap["stats"]
    resolution = snap.get("resolution")
    resolution_text = f"{resolution[0]} × {resolution[1]}" if resolution else "—"
    status_label = STATUS_META.get(snap["status"], STATUS_META["idle"])[0]
    fps = snap.get("fps") or 0.0
    inference_fps = snap.get("inference_fps") or 0.0

    columns = st.columns(6)
    with columns[0]:
        st.metric("Objects now", f"{stats.current_objects:,}")
    with columns[1]:
        st.metric("Unique tracking IDs", f"{stats.unique_ids:,}")
    with columns[2]:
        st.metric("Processing FPS", f"{fps:.1f}" if fps > 0 else "—")
    with columns[3]:
        st.metric("Frames processed", f"{snap['frame_index']:,}")
    with columns[4]:
        st.metric("Resolution", resolution_text)
    with columns[5]:
        st.metric("Status", status_label)

    source_fps = snap.get("source_fps") or 0.0
    details = (
        f"Elapsed {format_duration(snap.get('elapsed_seconds'))}  ·  "
        f"Inference {inference_fps:.1f} FPS  ·  "
        f"Source {source_fps:.1f} FPS" if source_fps else
        f"Elapsed {format_duration(snap.get('elapsed_seconds'))}  ·  "
        f"Inference {inference_fps:.1f} FPS"
    )
    st.caption(details)


def _render_status_message(snap: dict) -> None:
    status = snap["status"]
    message = snap.get("status_message") or ""
    if status == "error":
        st.error(snap.get("error") or message or "Processing failed.")
        if snap.get("config", {}).get("source_type") == "webcam":
            st.markdown(
                "**Webcam troubleshooting**\n"
                "- Run the app locally on a machine with a webcam (not in a headless/cloud environment).\n"
                "- Close other apps using the camera and allow camera access in OS privacy settings.\n"
                "- Try another camera index or use **Upload video** mode.\n"
                "- On Windows, check *Settings → Privacy & security → Camera*."
            )
        if snap.get("error_detail"):
            with st.expander("Technical details"):
                st.code(str(snap["error_detail"]), language="text")
    elif status == "completed":
        st.success(message or "Processing completed.")
    elif status == "stopped":
        st.info(message or "Processing stopped.")
    elif status == "paused":
        st.warning(message or "Paused. Use Resume to continue.")
    else:
        st.caption(message or "Processing frames...")
    if snap.get("output_error"):
        st.warning(f"Output video note: {snap['output_error']}")


def _render_dashboard(snap: dict, *, live: bool) -> None:
    config = snap.get("config", {})
    source_label = html.escape(str(snap.get("source_label", config.get("source", "source"))))
    status_label = STATUS_META.get(snap["status"], STATUS_META["idle"])[0]
    st.markdown(
        f"""
        <div class="session-bar">
            <div class="session-left">
                {_status_markup(snap["status"])}
                <div>
                    <div class="session-title">{source_label}</div>
                    <div class="session-sub">
                        {html.escape(str(config.get("tracker", "—")))} ·
                        {html.escape(str(config.get("model", "—")))} ·
                        {html.escape(str(config.get("device", "—")))}
                    </div>
                </div>
            </div>
            <div class="small-note">
                Session {html.escape(str(snap.get("session_id", "")))} ·
                {html.escape(status_label)} · IDs reset per session
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_metrics(snap)

    video_column, stats_column = st.columns([2.1, 1])
    with video_column:
        frame = snap.get("latest_rgb")
        if frame is not None:
            st.image(
                frame,
                channels="RGB",
                use_column_width=True,
                output_format="JPEG",
            )
        else:
            st.markdown(
                '<div class="video-placeholder">Waiting for the first processed frame…</div>',
                unsafe_allow_html=True,
            )

        progress = snap.get("progress")
        if progress is not None:
            total = snap.get("effective_total") or snap.get("total_frames") or 0
            percent = max(0.0, min(1.0, float(progress)))
            if total:
                text = (
                    f"Frame {snap['frame_index']:,} of {total:,} "
                    f"({percent * 100:.1f}%)"
                )
            else:
                text = f"Frame {snap['frame_index']:,} ({percent * 100:.1f}%)"
            st.progress(percent, text=text)
        elif snap["status"] in {"running", "paused", "starting"}:
            st.caption("LIVE source — frames are processed as fast as the model allows.")

        _render_status_message(snap)

    with stats_column:
        st.markdown("**Objects by class**")
        rows = snap["stats"].class_rows()
        if rows:
            frame_height = min(330, 34 + 35 * len(rows) + 38)
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
                height=frame_height,
            )
        else:
            st.caption("No objects detected yet.")
        st.caption(
            "Current counts come from the latest inference frame. Unique IDs "
            "are distinct tracker IDs seen in this session."
        )
        if snap["stats"].untracked_objects:
            st.caption(
                f"{snap['stats'].untracked_objects} current detection(s) had no "
                "tracker ID and are not included in unique-ID counts."
            )

    st.markdown("#### Tracking ID registry")
    id_rows = snap["stats"].id_rows(limit=20)
    if id_rows:
        st.dataframe(
            pd.DataFrame(id_rows),
            hide_index=True,
            use_container_width=True,
            height=min(310, 38 + 35 * len(id_rows)),
        )
        shown = min(len(id_rows), 20)
        st.caption(
            f"Showing {shown} most recently seen IDs. A higher 'Seen (frames)' "
            "value means the tracker maintained that object's identity across frames."
        )
    else:
        st.caption(
            "No tracking IDs yet. IDs appear when the tracker assigns them to newly "
            "detected objects (tracking IDs are not fabricated when absent)."
        )

    st.markdown("#### Session configuration")
    detail_columns = st.columns(4)
    detail_columns[0].caption(f"**Source:** {config.get('source', '—')}")
    detail_columns[1].caption(f"**Tracker:** {config.get('tracker', '—')}")
    detail_columns[2].caption(
        f"**Confidence / IoU:** {config.get('confidence', '—')} / {config.get('iou', '—')}"
    )
    detail_columns[3].caption(
        f"**Detection interval:** every {config.get('detect_interval', '—')} frame(s)"
    )
    st.caption(
        "Unique-ID statistics are session-scoped. Starting a new webcam or video "
        "session resets the tracker and therefore resets its IDs."
    )


def _render_output_panel(snap: dict) -> None:
    if snap["status"] not in TERMINAL_STATUSES:
        return
    output_path = snap.get("output_path")
    if not output_path:
        return
    path = Path(output_path)
    st.divider()
    st.subheader("Processed video")
    if snap.get("output_ready") and path.exists():
        size = path.stat().st_size
        st.success(
            "The annotated video is ready. It preserves the source frame rate and "
            "resolution when supported by the MP4 codec."
        )
        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"**File:** `{path.name}`")
            st.caption(
                f"{format_bytes(size)} · {snap.get('output_frames', 0):,} frames · "
                f"codec {snap.get('output_codec') or 'unknown'} · "
                f"`{relative_path(path)}`"
            )
            if snap["status"] == "stopped":
                st.info("This is a partial output from a stopped session.")
        with right:
            try:
                with st.spinner("Preparing download..."):
                    data = _read_bytes_cached(
                        str(path), path.stat().st_mtime, int(size)
                    )
                st.download_button(
                    "Download annotated MP4",
                    data=data,
                    file_name=path.name,
                    mime="video/mp4",
                    use_container_width=True,
                    key=f"download_{snap.get('session_id', uuid.uuid4().hex)}",
                )
            except OSError as exc:
                st.error(f"Could not read the output video for download: {exc}")
    else:
        if snap.get("output_error"):
            st.warning(f"No downloadable output: {snap['output_error']}")
        else:
            st.caption(f"Requested output location: `{relative_path(path)}`")


def _render_welcome(detector: YOLODetector | None) -> None:
    st.markdown("### Start a session")
    columns = st.columns(3)
    cards = [
        (
            "1",
            "Choose a source",
            "Use a local webcam or upload an MP4, AVI, MOV or MKV file. "
            "The included sample video is preselected in Upload video mode.",
        ),
        (
            "2",
            "Configure detection",
            "Pick your YOLO weights, confidence/IoU thresholds, classes, and "
            "ByteTrack or BoT-SORT. The model is loaded once, not per frame.",
        ),
        (
            "3",
            "Start and inspect",
            "Watch boxes, class labels, confidence scores and persistent #IDs. "
            "Pause, resume or stop at any time; uploaded videos can be downloaded.",
        ),
    ]
    for column, (number, title, body) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-num">{number}</div>
                    <h4>{html.escape(title)}</h4>
                    <p>{html.escape(body)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")
    left, right = st.columns([1.55, 1])
    with left:
        st.markdown(
            """
            <div class="pipeline">
                <b>OpenCV frame</b>  →  <b>YOLO detection</b>  →
                <b>ByteTrack / BoT-SORT association</b>  →  persistent #ID<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                <b>Annotated frame</b> (box + class + confidence + ID)  →
                live statistics + optional annotated MP4
            </div>
            """,
            unsafe_allow_html=True,
        )
        if detector is not None and detector.is_loaded:
            info = detector.info
            st.caption(
                f"Loaded model: **{info['model_name']}** · {info['class_count']} classes · "
                f"{info['device_label']} · {info['ultralytics_version']}"
            )
        sample = INPUT_VIDEOS_DIR / "sample_traffic.avi"
        if sample.exists():
            st.success(
                "Included sample video found: `input_videos/sample_traffic.avi`. "
                "Choose **Upload video** in the sidebar to use it immediately."
            )
    with right:
        st.markdown("**Notes**")
        st.markdown(
            "- Webcam mode needs a machine with camera access; use uploaded video "
            "in a headless/cloud environment.\n"
            "- Tracker IDs are session-scoped and reset when a new session starts.\n"
            "- On CPU, inference is slower; reduce inference size or use a GPU.\n"
            "- The model file is reused from `models/yolo11n.pt`; nothing is trained."
        )
        if not cuda_available():
            st.warning(
                "CUDA is not available. CPU inference works but real-time FPS will be lower."
            )


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
_init_state()
st.markdown(HERO, unsafe_allow_html=True)

detector: YOLODetector | None = None
model_error: str | None = None
model_info: dict | None = None

with st.sidebar:
    st.markdown('<div class="sidebar-brand">Control panel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Detection · tracking · session statistics</div>',
        unsafe_allow_html=True,
    )

    snap_before = _snapshot()
    active = _is_active(snap_before)
    st.markdown(_status_markup(snap_before["status"] if snap_before else "idle"), unsafe_allow_html=True)
    st.write("")

    # ------------------------------------------------------------------
    # 1. Model
    # ------------------------------------------------------------------
    with st.expander("1 · Model", expanded=True):
        model_files = discover_models()
        model_labels = [relative_path(path) for path in model_files]
        custom_label = "Custom path…"
        options = model_labels + [custom_label] if model_labels else [custom_label]
        default_index = next(
            (
                index
                for index, label in enumerate(options)
                if "yolo11n" in label.lower()
            ),
            0,
        )
        selected_label = st.selectbox(
            "YOLO weights",
            options=options,
            index=default_index,
            key="model_select",
            disabled=active,
        )
        if selected_label == custom_label:
            model_path_value = st.text_input(
                "Model path",
                value=model_labels[0] if model_labels else DEFAULT_MODEL_RELATIVE,
                key="model_custom_path",
                disabled=active,
                help="Absolute or project-relative path to your existing .pt weights.",
            )
        else:
            model_path_value = selected_label

        has_cuda = cuda_available()
        device_options = ["Auto"] + (["CUDA (GPU)"] if has_cuda else []) + ["CPU"]
        device_choice = st.selectbox(
            "Device",
            options=device_options,
            index=0,
            key="device_choice",
            disabled=active,
            help="Auto uses CUDA when available; otherwise CPU.",
        )

        if not model_files:
            st.caption(
                "No .pt file found in models/. Put your weights at "
                "`models/yolo11n.pt` or enter a custom path above."
            )

        try:
            detector = _get_detector(model_path_value, device_choice, active=active)
            model_info = detector.info
        except AppError as exc:
            model_error = str(exc)
            detector = None
        except Exception as exc:  # noqa: BLE001
            model_error = f"Unexpected model error: {type(exc).__name__}: {exc}"
            LOGGER.exception("Unexpected model loading error")
            detector = None

        if model_error:
            st.error(model_error)
        elif model_info:
            st.success(
                f"{model_info['model_name']} · {model_info['class_count']} classes loaded"
            )
            st.caption(
                f"{model_info['task'].upper()} · {model_info['device_label']} · "
                f"{model_info['parameters'] / 1_000_000:.1f}M params · "
                f"{relative_path(model_info['path'])}"
            )
            if not cuda_available():
                st.warning(
                    "CUDA not detected — using CPU. Inference may run below "
                    "real-time; lower the inference size for more FPS."
                )
            elif not str(model_info["device"]).startswith("cuda"):
                st.caption("CPU mode selected. CUDA is available if you switch Device to Auto/CUDA.")

        if st.button(
            "Reload model",
            use_container_width=True,
            disabled=active,
            key="reload_model",
        ):
            current = st.session_state.get("detector")
            if current is not None:
                try:
                    current.close()
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Error closing detector", exc_info=True)
            st.session_state.detector = None
            st.session_state.detector_key = None
            st.rerun()

    detector_ready = bool(detector is not None and detector.is_loaded)

    # ------------------------------------------------------------------
    # 2. Source
    # ------------------------------------------------------------------
    with st.expander("2 · Video source", expanded=True):
        source_mode = st.radio(
            "Input",
            options=["Webcam", "Upload video"],
            index=1,
            horizontal=True,
            key="source_mode",
            disabled=active,
        )

        camera_index = 0
        mirror = False
        video_path: Path | None = None
        video_info = None
        max_frames_ui = 0
        realtime_ui = False

        if source_mode == "Webcam":
            camera_index = int(
                st.number_input(
                    "Camera index",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                    key="camera_index",
                    disabled=active,
                )
            )
            scan_col, mirror_col = st.columns([1, 1])
            with scan_col:
                if st.button(
                    "Scan cameras",
                    use_container_width=True,
                    disabled=active,
                    key="scan_cameras",
                ):
                    with st.spinner("Checking camera indices..."):
                        st.session_state.camera_indices = probe_cameras(4)
            with mirror_col:
                mirror = st.checkbox(
                    "Mirror view",
                    value=True,
                    key="mirror_view",
                    disabled=active,
                )
            detected = st.session_state.get("camera_indices")
            if detected is not None:
                if detected:
                    st.success("Detected camera indices: " + ", ".join(str(i) for i in detected))
                else:
                    st.warning("No camera opened. Use Upload video mode on this machine.")
            st.caption(
                "Webcam access must be available to the computer running Streamlit. "
                "A headless/cloud server cannot expose your local camera."
            )
        else:
            uploaded = st.file_uploader(
                "Upload a video",
                type=SUPPORTED_UPLOAD_TYPES,
                key="video_upload",
                disabled=active,
                help="MP4, AVI, MOV or MKV. The file is saved under input_videos/uploads/.",
            )
            if uploaded is not None:
                try:
                    signature = _upload_signature(uploaded)
                    if st.session_state.get("upload_signature") != signature:
                        with st.spinner("Saving uploaded video..."):
                            saved = save_uploaded_video(uploaded.name, uploaded.getbuffer())
                        st.session_state.upload_signature = signature
                        st.session_state.upload_path = str(saved)
                        st.session_state.upload_error = None
                    if st.session_state.get("upload_error"):
                        st.error(st.session_state.upload_error)
                    elif st.session_state.get("upload_path"):
                        video_path = Path(st.session_state.upload_path)
                except (AppError, OSError) as exc:
                    st.session_state.upload_signature = signature if "signature" in locals() else None
                    st.session_state.upload_path = None
                    st.session_state.upload_error = str(exc)
                    st.error(str(exc))
            else:
                if st.session_state.get("upload_signature") is not None:
                    st.session_state.upload_signature = None
                    st.session_state.upload_path = None
                    st.session_state.upload_error = None

            existing_files = _existing_videos()
            existing_options = ["(none)"] + [path.name for path in existing_files]
            default_existing = 1 if len(existing_options) > 1 else 0
            selected_existing = st.selectbox(
                "Or choose a video from input_videos/",
                options=existing_options,
                index=default_existing,
                key="existing_video",
                disabled=active,
            )
            existing_path = None
            if selected_existing != "(none)":
                existing_path = INPUT_VIDEOS_DIR / selected_existing

            if video_path is None and existing_path is not None:
                video_path = existing_path
            elif video_path is not None and existing_path is not None:
                st.caption("Uploaded file takes priority over the selected existing file.")

            if video_path is not None:
                try:
                    video_info = probe_video(video_path)
                    st.caption(
                        f"`{video_path.name}` · {video_info.summary()} · "
                        f"{format_bytes(video_path.stat().st_size)}"
                    )
                except (AppError, OSError) as exc:
                    video_info = None
                    video_path = None
                    st.error(str(exc))
            else:
                st.caption("Choose an uploaded or existing video to enable Start.")

            max_frames_ui = int(
                st.number_input(
                    "Frame limit (0 = entire video)",
                    min_value=0,
                    max_value=1_000_000,
                    value=0,
                    step=100,
                    key="max_frames",
                    disabled=active,
                    help="Useful for a quick demonstration. 0 processes the full video.",
                )
            )
            realtime_ui = st.checkbox(
                "Throttle output to source FPS",
                value=False,
                key="realtime_playback",
                disabled=active,
                help="For uploaded video only. Detection is still performed as fast as possible "
                "within the playback pace.",
            )
            st.caption(
                f"Annotated videos are written to `{relative_path(OUTPUT_VIDEOS_DIR)}/` "
                "and can be downloaded when processing finishes."
            )

    # ------------------------------------------------------------------
    # 3. Detection and tracking
    # ------------------------------------------------------------------
    with st.expander("3 · Detection & tracking", expanded=True):
        conf = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            key="confidence_threshold",
            disabled=active or not detector_ready,
            help="Detections below this score are discarded before tracking.",
        )
        iou = st.slider(
            "IoU threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.50,
            step=0.05,
            key="iou_threshold",
            disabled=active or not detector_ready,
            help="Non-maximum suppression overlap threshold used by YOLO.",
        )
        tracker_label = st.radio(
            "Tracker",
            options=list(TRACKER_OPTIONS),
            index=0,
            horizontal=True,
            key="tracker_choice",
            disabled=active or not detector_ready,
            help="ByteTrack is the fast default. BoT-SORT adds global motion compensation "
            "and can be slower, especially on CPU.",
        )
        imgsz = st.selectbox(
            "Inference size",
            options=[416, 512, 640, 800, 960],
            index=2,
            key="inference_size",
            disabled=active or not detector_ready,
            help="Larger values improve small-object detection but reduce FPS.",
        )

        class_ids: tuple[int, ...] = ()
        detect_all = True
        selected_names: list[str] = []
        if detector_ready and detector is not None:
            class_names = [
                name for _, name in sorted(detector.names.items(), key=lambda item: item[0])
            ]
            model_token = hashlib.sha1(
                str(st.session_state.get("detector_key", "model")).encode("utf-8")
            ).hexdigest()[:8]
            detect_all = st.checkbox(
                "Detect all classes",
                value=True,
                key="detect_all_classes",
                disabled=active,
            )
            if not detect_all:
                defaults = [name for name in ("person", "car") if name in class_names]
                if not defaults:
                    defaults = class_names[:3]
                selected_names = st.multiselect(
                    "Classes to include",
                    options=class_names,
                    default=defaults,
                    key=f"class_filter_{model_token}",
                    disabled=active,
                )
                if not selected_names:
                    st.warning("Select at least one class, or enable 'Detect all classes'.")
                name_to_id = {name: class_id for class_id, name in detector.names.items()}
                class_ids = tuple(
                    sorted(
                        name_to_id[name]
                        for name in selected_names
                        if name in name_to_id
                    )
                )
            st.caption(f"Model classes available: {len(class_names)}")
        else:
            st.caption("Load a model to configure class filtering.")

        detect_interval = st.slider(
            "Run detection every N frames",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            key="detect_interval",
            disabled=active or not detector_ready,
            help="Intermediate frames reuse the latest boxes. Tracking state updates only "
            "on inference frames; 1 processes every frame.",
        )
        show_overlay = st.checkbox(
            "Draw statistics overlay on video",
            value=True,
            key="stats_overlay",
            disabled=active,
            help="Adds current object count, unique IDs and FPS to the top-left corner.",
        )

    # ------------------------------------------------------------------
    # 4. Session controls
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Session controls")
    snap_now = _snapshot()
    current_status = snap_now["status"] if snap_now else "idle"
    processor = st.session_state.get("processor")
    # Once a session is terminal its camera/writer and tracker state have been
    # released, even if the Python thread is in its final return instructions.
    thread_busy = bool(
        processor is not None
        and processor.is_alive()
        and not (
            snap_now
            and snap_now.get("status") in TERMINAL_STATUSES
            and snap_now.get("cleanup_complete")
        )
    )

    classes_ok = detect_all or bool(class_ids)
    source_ok = source_mode == "Webcam" or video_path is not None
    can_start = (
        detector_ready
        and not active
        and not thread_busy
        and source_ok
        and classes_ok
    )

    start_col, pause_col = st.columns(2)
    with start_col:
        start_clicked = st.button(
            "▶ Start",
            type="primary",
            use_container_width=True,
            disabled=not can_start,
            key="start_processing",
        )
    with pause_col:
        pause_clicked = st.button(
            "❚❚ Pause",
            use_container_width=True,
            disabled=current_status != "running",
            key="pause_processing",
        )
    resume_col, stop_col = st.columns(2)
    with resume_col:
        resume_clicked = st.button(
            "▶ Resume",
            use_container_width=True,
            disabled=current_status != "paused",
            key="resume_processing",
        )
    with stop_col:
        stop_clicked = st.button(
            "■ Stop",
            use_container_width=True,
            disabled=current_status not in {"starting", "running", "paused"},
            key="stop_processing",
        )

    if not detector_ready:
        st.caption("Start is disabled until the YOLO model loads successfully.")
    elif not source_ok:
        st.caption("Start is disabled until a video source is available.")
    elif not classes_ok:
        st.caption("Start is disabled until at least one class is selected.")
    elif active:
        st.caption("Processing is active. Use Pause, Resume or Stop.")
    elif thread_busy:
        st.caption("Waiting for the previous session to release its resources...")

    if start_clicked:
        config = None
        new_processor = None
        try:
            if detector is None or not detector.is_loaded:
                raise ConfigurationError("The YOLO model is not loaded.")
            if source_mode == "Webcam":
                source = int(camera_index)
                source_type = "webcam"
                output_path = None
                effective_mirror = bool(mirror)
                effective_max_frames = 0
                effective_realtime = False
            else:
                if video_path is None or not Path(video_path).exists():
                    raise ConfigurationError("Choose a valid video file first.")
                source = str(video_path)
                source_type = "video"
                output_path = make_output_path(video_path)
                effective_mirror = False
                effective_max_frames = int(max_frames_ui)
                effective_realtime = bool(realtime_ui)

            info = detector.info
            config = ProcessingConfig(
                source_type=source_type,
                source=source,
                output_path=output_path,
                conf=float(conf),
                iou=float(iou),
                classes=None if detect_all else tuple(class_ids),
                tracker=tracker_label,
                imgsz=int(imgsz),
                detect_interval=int(detect_interval),
                show_overlay=bool(show_overlay),
                mirror=effective_mirror,
                realtime=effective_realtime,
                max_frames=effective_max_frames,
                model_name=info["model_name"],
                device_label=info["device_label"],
            )
            new_processor = VideoProcessor(config, detector)
            new_processor.state.set_status("starting", "Starting processing session...")
            st.session_state.processor = new_processor
            new_processor.start()
        except Exception as exc:  # noqa: BLE001 - show startup errors in the UI
            LOGGER.exception("Could not start processing")
            if new_processor is not None:
                new_processor.state.set_error(
                    f"Could not start processing: {exc}", detail=traceback.format_exc()
                )
            st.error(f"Could not start processing: {exc}")
        else:
            st.rerun()

    if pause_clicked and processor is not None:
        processor.pause()
        st.rerun()
    if resume_clicked and processor is not None:
        processor.resume()
        st.rerun()
    if stop_clicked and processor is not None:
        processor.stop()
        st.rerun()

    st.divider()
    with st.expander("Environment", expanded=False):
        versions = dependency_versions()
        for name, value in versions.items():
            st.caption(f"{name}: {value}")
        st.caption(
            f"Device: {device_label('cuda:0') if cuda_available() else 'CPU only'}"
        )
    st.caption(
        "Tracker IDs are assigned per session and reset on every new start. "
        "Object labels, confidence scores and IDs are drawn on the video."
    )


# ---------------------------------------------------------------------------
# Live page body
# ---------------------------------------------------------------------------
# A full-script refresh is used instead of a fragment. It keeps the sidebar
# status and control states in sync with the worker and avoids the Streamlit
# fragment-rerun race that occurs when a live session becomes terminal.
LIVE_REFRESH_SECONDS = 0.25

snap = _snapshot()
active_now = _is_active(snap)
if active_now:
    _render_dashboard(snap, live=True)
    if snap["status"] in {"starting", "running", "stopping"}:
        time.sleep(LIVE_REFRESH_SECONDS)
        st.rerun()
elif snap is not None:
    _render_dashboard(snap, live=False)
    _render_output_panel(snap)

    # The last automatic refresh can begin while the worker is still
    # "stopping"; its sidebar is rendered before the terminal status arrives.
    # Rerun once on the terminal transition so the sidebar controls re-enable.
    terminal_marker = f"{snap.get('session_id')}:{snap.get('status')}"
    if st.session_state.get("last_terminal_render") != terminal_marker:
        st.session_state.last_terminal_render = terminal_marker
        time.sleep(0.2)
        st.rerun()
else:
    _render_welcome(detector)
