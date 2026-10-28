"""Shared utilities: paths, validation, drawing, formatting and video helpers."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from errors import ConfigurationError, ModelLoadError, OutputVideoError, VideoSourceError

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
INPUT_VIDEOS_DIR = PROJECT_ROOT / "input_videos"
UPLOAD_DIR = INPUT_VIDEOS_DIR / "uploads"
OUTPUT_VIDEOS_DIR = PROJECT_ROOT / "output_videos"

DEFAULT_MODEL_RELATIVE = "models/yolo11n.pt"
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
SUPPORTED_UPLOAD_TYPES = ["mp4", "avi", "mov", "mkv"]

# Friendly names used by the interface -> bundled Ultralytics tracker configs.
TRACKER_OPTIONS = ("ByteTrack", "BoT-SORT")


def ensure_directories() -> None:
    """Create the project's runtime directories if they do not exist."""
    for folder in (MODELS_DIR, INPUT_VIDEOS_DIR, UPLOAD_DIR, OUTPUT_VIDEOS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class VideoInfo:
    """Metadata read from an actual decoded video frame."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str

    @property
    def duration_seconds(self) -> float | None:
        if self.fps > 0 and self.frame_count > 0:
            return self.frame_count / self.fps
        return None

    @property
    def output_fps(self) -> float:
        """A safe frame rate for an output writer."""
        if 1.0 <= self.fps <= 240.0:
            return float(self.fps)
        return 30.0

    @property
    def has_valid_fps(self) -> bool:
        return 1.0 <= self.fps <= 240.0

    def summary(self) -> str:
        duration = format_duration(self.duration_seconds)
        return (
            f"{self.width}x{self.height} | {self.output_fps:.2f} FPS | "
            f"{self.frame_count:,} frames | {duration} | {self.codec}"
        )


def fourcc_to_string(codec_value: int) -> str:
    """Convert OpenCV's numeric FOURCC to a readable four-character code."""
    try:
        value = int(codec_value)
    except (TypeError, ValueError):
        return "unknown"
    if value <= 0:
        return "unknown"
    chars = "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))
    chars = "".join(ch if ch.isprintable() else "?" for ch in chars)
    return chars.strip() or "unknown"


def probe_video(path: str | Path) -> VideoInfo:
    """Open a video, decode its first frame and return validated metadata.

    Raising a friendly :class:`VideoSourceError` for unreadable/unsupported files
    lets the UI fail gracefully rather than passing a broken file to the model.
    """
    video_path = Path(path).expanduser()
    if not video_path.exists():
        raise VideoSourceError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise VideoSourceError(f"Video path is not a file: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
        raise VideoSourceError(
            f"Unsupported video format '{video_path.suffix}'. Supported formats: {supported}."
        )

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoSourceError(
                f"OpenCV could not open '{video_path.name}'. The codec may be unsupported, "
                "or the file may be corrupt. Try re-encoding it to H.264 MP4."
            )
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise VideoSourceError(
                f"OpenCV opened '{video_path.name}' but could not decode a frame. "
                "Try another MP4/AVI/MOV file or re-encode it."
            )
        height, width = frame.shape[:2]
        if width < 2 or height < 2:
            raise VideoSourceError(f"Invalid frame dimensions in '{video_path.name}'.")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps < 0:
            fps = 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        codec = fourcc_to_string(int(capture.get(cv2.CAP_PROP_FOURCC) or 0))
        return VideoInfo(
            path=video_path.resolve(),
            width=int(width),
            height=int(height),
            fps=fps,
            frame_count=max(0, frame_count),
            codec=codec,
        )
    finally:
        capture.release()


def read_video_frame(path: str | Path, frame_index: int = 0) -> np.ndarray:
    """Read a single BGR frame from a video file (used by diagnostics/tests)."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoSourceError(f"Could not open video: {path}")
        if frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise VideoSourceError(f"Could not decode frame {frame_index} from: {path}")
        return frame
    finally:
        capture.release()


def open_capture(source: int | str | Path, prefer_dshow: bool = True) -> cv2.VideoCapture:
    """Open a webcam index or video file with helpful errors.

    On Windows the DirectShow backend is attempted first for webcams because it
    is generally more reliable than Media Foundation for OpenCV camera access.
    """
    if isinstance(source, bool):
        raise VideoSourceError("Camera source must be an integer index, not a boolean.")

    if isinstance(source, int):
        if source < 0:
            raise VideoSourceError("Camera index must be zero or greater.")
        backends: list[tuple[str, int | None]] = []
        if os.name == "nt" and prefer_dshow:
            backends.extend([("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF)])
        backends.append(("default", None))

        for name, backend in backends:
            capture = (
                cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)
            )
            if capture.isOpened():
                try:
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except cv2.error:
                    pass
                LOGGER.info("Opened camera %s using %s backend", source, name)
                return capture
            capture.release()

        raise VideoSourceError(
            f"Could not open camera index {source}. Check that a webcam is connected, "
            "not in use by another application, and that camera permission is enabled "
            "(Windows: Settings > Privacy & security > Camera). In a headless/cloud "
            "environment, use Upload video mode instead."
        )

    video_path = Path(source).expanduser()
    if not video_path.exists():
        raise VideoSourceError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise VideoSourceError(f"Video path is not a file: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoSourceError(
            f"OpenCV could not open '{video_path.name}'. The codec may be unsupported "
            "or the file may be corrupt. Try an H.264 MP4 file."
        )
    return capture


def probe_cameras(max_index: int = 5) -> list[int]:
    """Try camera indices 0..max_index-1 and return those that open.

    This briefly opens each device. It can take a few seconds and should only be
    called from an explicit button click in the interface.
    """
    available: list[int] = []
    for index in range(max(1, int(max_index))):
        capture: cv2.VideoCapture | None = None
        try:
            capture = open_capture(index)
            available.append(index)
        except VideoSourceError:
            continue
        finally:
            if capture is not None:
                capture.release()
    return available


def resolve_device(preference: str = "auto") -> str:
    """Resolve 'auto'/'cpu'/'cuda' into an Ultralytics-compatible device string."""
    pref = (preference or "auto").strip().lower()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency in practice
        if pref == "cpu":
            return "cpu"
        raise ConfigurationError(
            "PyTorch is not installed. Install the dependencies from requirements.txt."
        ) from exc

    if pref == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if pref == "cpu":
        return "cpu"
    if pref in {"cuda", "gpu"} or pref.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ConfigurationError(
                "CUDA was requested but is not available. Install a CUDA-enabled PyTorch "
                "build and verify with `python -c \"import torch; print(torch.cuda.is_available())\"`, "
                "or select CPU/Auto."
            )
        return "cuda:0" if pref in {"cuda", "gpu"} else pref
    raise ConfigurationError(f"Unknown device preference: {preference!r}")


def device_label(device: str) -> str:
    """Human-readable name for a resolved device string."""
    if not device:
        return "Unknown"
    if device.lower().startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                index = 0
                if ":" in device:
                    index = int(device.split(":", 1)[1])
                return f"CUDA GPU ({torch.cuda.get_device_name(index)})"
        except (ImportError, ValueError, RuntimeError):
            pass
        return f"CUDA GPU ({device})"
    return "CPU"


def cuda_available() -> bool:
    """Safely report whether CUDA is available."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False


def discover_models() -> list[Path]:
    """Find .pt weights in models/ and the project root without duplicating paths."""
    found: dict[Path, Path] = {}
    for folder in (MODELS_DIR, PROJECT_ROOT):
        if not folder.exists():
            continue
        for path in folder.glob("*.pt"):
            if path.is_file():
                resolved = path.resolve()
                found[resolved] = path
    return sorted(found.values(), key=lambda p: (p.parent != MODELS_DIR, p.name.lower()))


def resolve_model_path(model_path: str | Path | None) -> Path:
    """Resolve a user-supplied model path, checking sensible fallback locations."""
    raw = str(model_path or DEFAULT_MODEL_RELATIVE).strip().strip('"').strip("'")
    if not raw:
        raw = DEFAULT_MODEL_RELATIVE
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    candidates: list[Path] = []
    if expanded.is_absolute():
        candidates.append(expanded)
    else:
        candidates.extend(
            [
                PROJECT_ROOT / expanded,
                Path.cwd() / expanded,
                MODELS_DIR / expanded.name,
                PROJECT_ROOT / expanded.name,
            ]
        )

    seen: set[Path] = set()
    searched: list[Path] = []
    for candidate in candidates:
        key = candidate.resolve() if candidate.exists() else candidate.absolute()
        if key in seen:
            continue
        seen.add(key)
        searched.append(candidate)
        if candidate.is_file():
            return candidate.resolve()

    search_text = "\n".join(f"  - {path}" for path in searched)
    raise ModelLoadError(
        f"YOLO weights not found: {raw}\n"
        f"Searched:\n{search_text}\n\n"
        "Place your existing weights at models/yolo11n.pt, or choose a custom path in the "
        "sidebar. You can also download the default model with:\n"
        "  python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\"\n"
        "and then move yolo11n.pt into the models/ folder."
    )


def resolve_tracker_config(tracker: str) -> str:
    """Map ByteTrack/BoT-SORT names to the YAML configs bundled with Ultralytics."""
    value = str(tracker or "ByteTrack").strip().lower()
    normalized = value.replace("_", "-").replace(" ", "-")
    if normalized in {"bytetrack", "byte-track", "bytetrack.yaml"}:
        return "bytetrack.yaml"
    if normalized in {"botsort", "bot-sort", "botsort.yaml", "bot-sort.yaml"}:
        return "botsort.yaml"

    # Allow an advanced user to pass their own YAML config path.
    candidate = Path(tracker).expanduser()
    if candidate.suffix.lower() == ".yaml" and candidate.is_file():
        return str(candidate.resolve())
    raise ConfigurationError(
        f"Unknown tracker '{tracker}'. Choose ByteTrack or BoT-SORT."
    )


def tracker_display_name(tracker: str) -> str:
    try:
        return "BoT-SORT" if resolve_tracker_config(tracker) == "botsort.yaml" else "ByteTrack"
    except ConfigurationError:
        return str(tracker)


def sanitize_filename(name: str, fallback: str = "video", max_length: int = 80) -> str:
    """Make a safe, readable file-name component on Windows and Linux."""
    cleaned = re.sub(r"[^\w\-. ]+", "_", str(name), flags=re.UNICODE).strip(" ._-")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = fallback
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if cleaned.split(".")[0].upper() in reserved:
        cleaned = f"_{cleaned}"
    return cleaned[:max_length].rstrip(" ._-") or fallback


def make_output_path(input_path: str | Path, prefix: str | None = None) -> Path:
    """Create a unique MP4 path under output_videos/ for an annotated result."""
    ensure_directories()
    stem = sanitize_filename(prefix or Path(input_path).stem or "video", max_length=60)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = OUTPUT_VIDEOS_DIR / f"{stem}_tracked_{stamp}.mp4"
    counter = 2
    while candidate.exists():
        candidate = OUTPUT_VIDEOS_DIR / f"{stem}_tracked_{stamp}_{counter}.mp4"
        counter += 1
    return candidate


def save_uploaded_video(filename: str, data: bytes | bytearray | memoryview) -> Path:
    """Persist an uploaded video once, using a content hash in its file name."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
        raise VideoSourceError(
            f"Unsupported upload format '{suffix}'. Use one of: {supported}."
        )
    ensure_directories()
    payload = memoryview(data)
    if len(payload) == 0:
        raise VideoSourceError("The uploaded video is empty.")
    digest = hashlib.sha1(payload).hexdigest()[:10]
    stem = sanitize_filename(Path(filename).stem, fallback="upload", max_length=50)
    destination = UPLOAD_DIR / f"{stem}_{digest}{suffix}"
    if not destination.exists():
        with destination.open("wb") as handle:
            handle.write(payload)
        LOGGER.info("Saved uploaded video to %s", destination)
    return destination.resolve()


def create_video_writer(
    output_path: str | Path,
    fps: float,
    size: tuple[int, int],
) -> tuple[cv2.VideoWriter, str]:
    """Create a verified MP4 video writer, trying compatible MP4 codecs."""
    ensure_directories()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(size[0]), int(size[1])
    if width < 2 or height < 2:
        raise OutputVideoError(f"Invalid output resolution: {width}x{height}")
    safe_fps = float(fps) if math.isfinite(float(fps)) and 1.0 <= float(fps) <= 240.0 else 30.0

    for codec_name in ("mp4v", "avc1", "H264"):
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        writer = cv2.VideoWriter(str(output_path), fourcc, safe_fps, (width, height), True)
        if writer.isOpened():
            LOGGER.info("Writing annotated video with %s at %.2f FPS", codec_name, safe_fps)
            return writer, codec_name
        writer.release()

    raise OutputVideoError(
        f"Could not create the output video '{output_path.name}'. "
        "The OpenCV build may lack MP4 codecs. Try reinstalling opencv-python, or use "
        "a different output folder/codec."
    )


def verify_output_video(path: str | Path, expected_size: tuple[int, int] | None = None) -> dict[str, Any]:
    """Confirm an output video exists, decodes, and has valid properties."""
    output_path = Path(path)
    if not output_path.exists() or not output_path.is_file():
        raise OutputVideoError(f"Output video was not created: {output_path}")
    size_bytes = output_path.stat().st_size
    if size_bytes < 512:
        raise OutputVideoError(f"Output video is empty or too small: {output_path}")

    capture = cv2.VideoCapture(str(output_path))
    try:
        if not capture.isOpened():
            raise OutputVideoError(f"Output video cannot be opened: {output_path}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise OutputVideoError(f"Output video cannot be decoded: {output_path}")
        height, width = frame.shape[:2]
        if expected_size is not None:
            expected_w, expected_h = int(expected_size[0]), int(expected_size[1])
            if (width, height) != (expected_w, expected_h):
                raise OutputVideoError(
                    f"Output resolution {width}x{height} does not match expected "
                    f"{expected_w}x{expected_h}."
                )
        return {
            "path": str(output_path.resolve()),
            "size_bytes": int(size_bytes),
            "width": int(width),
            "height": int(height),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        }
    finally:
        capture.release()


def format_duration(seconds: float | None) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return "--:--"
    total = int(round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_bytes(size: int | float | None) -> str:
    if size is None:
        return "--"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024.0:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def relative_path(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except (ValueError, OSError):
        return str(path)


def dependency_versions() -> dict[str, str]:
    """Read installed package versions without importing the heavy packages."""
    packages = {
        "ultralytics": "ultralytics",
        "opencv-python": "opencv-python",
        "numpy": "numpy",
        "streamlit": "streamlit",
        "torch": "torch",
        "lap": "lap",
        "pandas": "pandas",
    }
    result: dict[str, str] = {}
    for label, package in packages.items():
        try:
            result[label] = version(package)
        except PackageNotFoundError:
            result[label] = "not installed"
    return result


# A high-contrast BGR palette. Box colors are stable for a track ID, so the same
# object keeps the same color while its ID remains alive.
_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (94, 63, 244),    # rose
    (246, 130, 59),   # blue
    (129, 185, 16),   # emerald
    (11, 158, 245),   # amber
    (246, 92, 139),   # violet
    (153, 72, 236),   # pink
    (212, 182, 6),    # cyan
    (22, 204, 132),   # lime
    (68, 68, 239),    # red
    (166, 184, 20),   # teal
    (241, 102, 99),   # indigo
    (247, 85, 168),   # purple
    (96, 165, 250),   # light blue
    (73, 183, 119),   # green
    (0, 140, 255),    # orange
    (179, 105, 255),  # magenta
)


def color_for_detection(class_id: int, track_id: int | None = None) -> tuple[int, int, int]:
    key = int(track_id) if track_id is not None else int(class_id)
    return _PALETTE_BGR[key % len(_PALETTE_BGR)]


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = max(3, int(round(font_scale * 6)))
    label_height = text_height + baseline + pad * 2
    frame_height, frame_width = frame.shape[:2]
    x = max(0, min(int(x), max(0, frame_width - text_width - pad * 2)))
    top = int(y) - label_height
    if top < 0:
        top = int(y)
    top = max(0, min(top, max(0, frame_height - label_height)))
    right = min(frame_width, x + text_width + pad * 2)
    bottom = min(frame_height, top + label_height)

    cv2.rectangle(frame, (x, top), (right, bottom), color, -1)
    luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (15, 23, 42) if luminance > 165 else (255, 255, 255)
    cv2.putText(
        frame,
        text,
        (x + pad, top + pad + text_height),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def _blend_rectangle(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int] = (15, 23, 42),
    alpha: float = 0.72,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    height, width = frame.shape[:2]
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    roi = frame[y1:y2, x1:x2]
    dark = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(dark, alpha, roi, 1.0 - alpha, 0, dst=roi)


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Any] | Iterable[Any],
    *,
    show_stats_overlay: bool = True,
    stats: Mapping[str, Any] | Any | None = None,
    fps: float | None = None,
    show_track_ids: bool = True,
) -> np.ndarray:
    """Draw boxes, class labels, confidence scores and track IDs on a BGR frame.

    The frame is annotated in place for efficiency and returned for convenience.
    Detections without a tracker ID are drawn with class/confidence only; no ID
    is invented or displayed.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return frame
    height, width = frame.shape[:2]
    line_width = max(2, int(round(min(height, width) / 400.0)))
    font_scale = max(0.42, min(0.72, min(height, width) / 1200.0))
    text_thickness = 1 if min(height, width) < 900 else 2

    for detection in detections:
        try:
            xyxy = _value(detection, "xyxy")
            class_name = str(_value(detection, "class_name", "object"))
            confidence = float(_value(detection, "confidence", 0.0) or 0.0)
            class_id = int(_value(detection, "class_id", 0) or 0)
            track_id_raw = _value(detection, "track_id", None)
            track_id = int(track_id_raw) if track_id_raw is not None else None

            x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            color = color_for_detection(class_id, track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)
            if show_track_ids and track_id is not None:
                label = f"#{track_id} {class_name} {confidence:.2f}"
            else:
                label = f"{class_name} {confidence:.2f}"
            _draw_label(frame, label, x1, y1, color, font_scale, text_thickness)
        except (TypeError, ValueError, IndexError, cv2.error):
            LOGGER.debug("Skipped malformed detection while drawing", exc_info=True)
            continue

    if show_stats_overlay:
        current_objects = _value(stats, "current_objects", None) if stats is not None else None
        unique_ids = _value(stats, "unique_ids", None) if stats is not None else None
        if current_objects is not None or unique_ids is not None or fps is not None:
            first = (
                f"Objects {int(current_objects or 0)}  |  "
                f"Unique IDs {int(unique_ids or 0)}  |  FPS {(fps or 0.0):.1f}"
            )
            second = f"{width}x{height}  |  Track ID shown as #ID"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.42, min(0.62, min(height, width) / 1400.0))
            (w1, h1), b1 = cv2.getTextSize(first, font, scale, 1)
            (w2, h2), b2 = cv2.getTextSize(second, font, scale, 1)
            pad = 10
            panel_width = max(w1, w2) + pad * 2
            panel_height = h1 + h2 + b1 + b2 + pad * 3
            _blend_rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height))
            cv2.putText(
                frame, first, (10 + pad, 10 + pad + h1),
                font, scale, (248, 250, 252), 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, second, (10 + pad, 10 + pad + h1 + b1 + h2 + 4),
                font, scale, (203, 213, 225), 1, cv2.LINE_AA,
            )

    return frame


class FpsMeter:
    """Small exponential-moving-average FPS meter."""

    def __init__(self, smoothing: float = 0.82) -> None:
        self.smoothing = float(smoothing)
        self._last: float | None = None
        self.value: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            delta = max(now - self._last, 1e-6)
            instantaneous = 1.0 / delta
            if self.value <= 0:
                self.value = instantaneous
            else:
                self.value = (
                    self.smoothing * self.value + (1.0 - self.smoothing) * instantaneous
                )
        self._last = now
        return self.value
