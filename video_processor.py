"""Background video/webcam processing pipeline.

The Streamlit UI never blocks on OpenCV inference: a ``VideoProcessor`` thread
owns the camera/video handle, runs YOLO tracking, writes annotated frames, and
publishes snapshots for the UI to render.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from detector import Detection, YOLODetector
from errors import AppError, ConfigurationError, OutputVideoError, VideoSourceError
from tracker import TrackStats, TrackingSession
from utils import (
    FpsMeter,
    create_video_writer,
    draw_detections,
    open_capture,
    relative_path,
    resolve_tracker_config,
    tracker_display_name,
    verify_output_video,
)

LOGGER = logging.getLogger(__name__)

ACTIVE_STATUSES = frozenset({"starting", "running", "paused", "stopping"})
TERMINAL_STATUSES = frozenset({"completed", "stopped", "error"})


@dataclass
class ProcessingConfig:
    """All settings needed for one processing session."""

    source_type: str
    source: int | str | Path
    output_path: Path | str | None = None
    conf: float = 0.25
    iou: float = 0.5
    classes: Sequence[int] | None = None
    tracker: str = "ByteTrack"
    imgsz: int = 640
    detect_interval: int = 1
    show_overlay: bool = True
    mirror: bool = False
    realtime: bool = False
    max_frames: int = 0
    start_frame: int = 0
    model_name: str = ""
    device_label: str = ""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        self.source_type = str(self.source_type).strip().lower()
        if self.source_type not in {"webcam", "video"}:
            raise ConfigurationError("Source type must be 'webcam' or 'video'.")
        if self.source_type == "webcam" and (
            isinstance(self.source, bool) or not isinstance(self.source, int)
        ):
            raise ConfigurationError("Webcam source must be an integer camera index.")
        if self.source_type == "video" and not isinstance(self.source, (str, Path)):
            raise ConfigurationError("Video source must be a file path.")
        if not 0.0 <= float(self.conf) <= 1.0:
            raise ConfigurationError("Confidence threshold must be between 0 and 1.")
        if not 0.0 <= float(self.iou) <= 1.0:
            raise ConfigurationError("IoU threshold must be between 0 and 1.")
        if int(self.imgsz) < 32:
            raise ConfigurationError("Inference image size must be at least 32.")
        if int(self.detect_interval) < 1:
            raise ConfigurationError("Detection interval must be at least 1 frame.")
        if int(self.max_frames) < 0:
            raise ConfigurationError("Maximum frames cannot be negative.")
        if int(self.start_frame) < 0:
            raise ConfigurationError("Start frame cannot be negative.")

        if self.classes is not None:
            self.classes = tuple(int(value) for value in self.classes)
            if not self.classes:
                raise ConfigurationError(
                    "No classes selected. Select at least one class or detect all classes."
                )
        self.output_path = Path(self.output_path) if self.output_path else None
        self.conf = float(self.conf)
        self.iou = float(self.iou)
        self.imgsz = int(self.imgsz)
        self.detect_interval = int(self.detect_interval)
        self.max_frames = int(self.max_frames)
        self.start_frame = int(self.start_frame)
        self.show_overlay = bool(self.show_overlay)
        self.mirror = bool(self.mirror)
        self.realtime = bool(self.realtime)

        # Validate tracker early so an invalid selection fails before a thread starts.
        resolve_tracker_config(self.tracker)
        if not self.session_id:
            self.session_id = uuid.uuid4().hex[:12]

    @property
    def source_label(self) -> str:
        if self.source_type == "webcam":
            return f"Webcam {self.source}"
        return Path(self.source).name

    def summary(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source": self.source_label,
            "tracker": tracker_display_name(self.tracker),
            "confidence": round(self.conf, 2),
            "iou": round(self.iou, 2),
            "classes": "All classes" if self.classes is None else f"{len(self.classes)} selected",
            "imgsz": self.imgsz,
            "detect_interval": self.detect_interval,
            "overlay": self.show_overlay,
            "realtime": self.realtime,
            "max_frames": self.max_frames,
            "model": self.model_name,
            "device": self.device_label,
            "output": relative_path(self.output_path) if self.output_path else "Not saved",
        }


class ProcessingState:
    """Thread-safe state shared by the worker thread and Streamlit."""

    def __init__(self, config: ProcessingConfig) -> None:
        self._lock = threading.RLock()
        self.session_id = config.session_id
        self.config = config.summary()
        self.source_label = config.source_label
        self.status = "idle"
        self.status_message = "Waiting to start."
        self.error: str | None = None
        self.error_detail: str | None = None
        self.progress: float | None = None
        self.frame_index = 0
        self.total_frames = 0
        self.effective_total = 0
        self.fps = 0.0
        self.inference_fps = 0.0
        self.resolution: tuple[int, int] | None = None
        self.source_fps = 0.0
        self.stats = TrackStats()
        self.latest_rgb: np.ndarray | None = None
        self.frame_serial = 0
        self.output_path: Path | None = config.output_path
        self.output_ready = False
        self.output_size_bytes = 0
        self.output_frames = 0
        self.output_codec = ""
        self.output_error: str | None = None
        self.inference_count = 0
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.resources_released = False
        self.cleanup_complete = False

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self.status in ACTIVE_STATUSES

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self.status in TERMINAL_STATUSES

    def set_status(self, status: str, message: str | None = None) -> None:
        with self._lock:
            if self.status in TERMINAL_STATUSES:
                return
            self.status = status
            if message is not None:
                self.status_message = message

    def set_source(
        self,
        width: int,
        height: int,
        source_fps: float,
        total_frames: int,
        effective_total: int,
    ) -> None:
        with self._lock:
            self.resolution = (int(width), int(height))
            self.source_fps = float(source_fps or 0.0)
            self.total_frames = int(total_frames or 0)
            self.effective_total = int(effective_total or 0)

    def set_output(self, path: Path, codec: str) -> None:
        with self._lock:
            self.output_path = Path(path)
            self.output_codec = str(codec)

    def publish(
        self,
        rgb_frame: np.ndarray,
        *,
        frame_index: int,
        stats: TrackStats,
        fps: float,
        inference_fps: float,
        progress: float | None,
        inference_count: int,
    ) -> None:
        with self._lock:
            self.latest_rgb = rgb_frame
            self.frame_serial += 1
            self.frame_index = int(frame_index)
            self.stats = stats
            self.fps = float(fps or 0.0)
            self.inference_fps = float(inference_fps or 0.0)
            self.progress = None if progress is None else max(0.0, min(1.0, float(progress)))
            self.inference_count = int(inference_count)

    def set_output_info(
        self,
        *,
        path: Path,
        size_bytes: int,
        frames: int,
        codec: str,
    ) -> None:
        with self._lock:
            self.output_path = Path(path)
            self.output_size_bytes = int(size_bytes)
            self.output_frames = int(frames)
            self.output_codec = str(codec or self.output_codec)
            self.output_ready = int(size_bytes) > 0 and int(frames) > 0
            self.output_error = None

    def set_output_error(self, message: str) -> None:
        with self._lock:
            self.output_error = str(message)
            self.output_ready = False

    def set_error(self, message: str, detail: str | None = None) -> None:
        with self._lock:
            self.status = "error"
            self.status_message = str(message)
            self.error = str(message)
            self.error_detail = detail
            self.finished_at = time.time()

    def finish(self, status: str, message: str, progress: float | None = None) -> None:
        with self._lock:
            self.status = str(status)
            self.status_message = str(message)
            if progress is not None:
                self.progress = max(0.0, min(1.0, float(progress)))
            elif self.progress is not None and status == "completed":
                self.progress = 1.0
            self.finished_at = time.time()

    def mark_cleanup(self, *, released: bool, complete: bool = True) -> None:
        with self._lock:
            self.resources_released = bool(released)
            self.cleanup_complete = bool(complete)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            end = self.finished_at or time.time()
            return {
                "session_id": self.session_id,
                "status": self.status,
                "status_message": self.status_message,
                "error": self.error,
                "error_detail": self.error_detail,
                "progress": self.progress,
                "frame_index": self.frame_index,
                "total_frames": self.total_frames,
                "effective_total": self.effective_total,
                "fps": self.fps,
                "inference_fps": self.inference_fps,
                "resolution": self.resolution,
                "source_fps": self.source_fps,
                "stats": self.stats,
                "latest_rgb": self.latest_rgb,
                "frame_serial": self.frame_serial,
                "output_path": str(self.output_path) if self.output_path else None,
                "output_ready": self.output_ready,
                "output_size_bytes": self.output_size_bytes,
                "output_frames": self.output_frames,
                "output_codec": self.output_codec,
                "output_error": self.output_error,
                "inference_count": self.inference_count,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_seconds": max(0.0, end - self.started_at),
                "resources_released": self.resources_released,
                "cleanup_complete": self.cleanup_complete,
                "config": dict(self.config),
                "source_label": self.source_label,
                "active": self.status in ACTIVE_STATUSES,
                "terminal": self.status in TERMINAL_STATUSES,
            }


class VideoProcessor(threading.Thread):
    """One processing session running in a daemon thread."""

    def __init__(
        self,
        config: ProcessingConfig,
        detector: YOLODetector,
        state: ProcessingState | None = None,
    ) -> None:
        super().__init__(name=f"yolo-tracker-{config.session_id}", daemon=True)
        self.config = config
        self.detector = detector
        self.state = state or ProcessingState(config)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._writer_codec = ""
        self._frames_written = 0
        self._width = 0
        self._height = 0

    # ------------------------------------------------------------------
    # UI controls
    # ------------------------------------------------------------------
    def pause(self) -> None:
        if not self.state.is_active or self.state.status in {"paused", "stopping"}:
            return
        self._pause_event.set()
        self.state.set_status("paused", "Paused. The video source is not being read.")

    def resume(self) -> None:
        if not self._pause_event.is_set():
            return
        self._pause_event.clear()
        self.state.set_status("running", "Resumed.")

    def stop(self) -> None:
        if self.state.is_terminal:
            return
        self._stop_event.set()
        self._pause_event.clear()
        self.state.set_status("stopping", "Stopping and releasing video resources...")

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def snapshot(self) -> dict[str, Any]:
        """Read the latest UI-facing state snapshot."""
        return self.state.snapshot()

    # ------------------------------------------------------------------
    # Main thread
    # ------------------------------------------------------------------
    def run(self) -> None:
        final_status = "error"
        final_message = "Processing failed unexpectedly."
        error_detail: str | None = None
        output_verification_error: str | None = None

        try:
            final_status, final_message = self._process()
        except AppError as exc:
            final_status = "error"
            final_message = str(exc)
            error_detail = traceback.format_exc()
            LOGGER.error("Processing error: %s", exc)
        except Exception as exc:  # noqa: BLE001 - never let the worker die silently
            final_status = "error"
            final_message = f"Unexpected processing error: {type(exc).__name__}: {exc}"
            error_detail = traceback.format_exc()
            LOGGER.exception("Unexpected processing error")
        finally:
            self._release_resources()
            try:
                self.detector.reset_tracking()
            except Exception:  # noqa: BLE001
                LOGGER.debug("Could not reset tracking during cleanup", exc_info=True)

            if self.config.output_path is not None and self._frames_written > 0:
                try:
                    info = verify_output_video(
                        self.config.output_path, expected_size=(self._width, self._height)
                    )
                    self.state.set_output_info(
                        path=self.config.output_path,
                        size_bytes=info["size_bytes"],
                        frames=self._frames_written,
                        codec=self._writer_codec,
                    )
                    LOGGER.info(
                        "Verified output video %s (%s, %d frames)",
                        self.config.output_path,
                        info["size_bytes"],
                        self._frames_written,
                    )
                except Exception as exc:  # noqa: BLE001
                    output_verification_error = str(exc)
                    self.state.set_output_error(output_verification_error)
                    LOGGER.error("Output verification failed: %s", exc)
            elif self.config.output_path is not None:
                output_verification_error = "No frames were written to the output video."
                self.state.set_output_error(output_verification_error)

            self.state.mark_cleanup(released=True, complete=True)

            if final_status == "completed" and output_verification_error:
                final_status = "error"
                final_message = output_verification_error
                error_detail = error_detail or output_verification_error
            elif final_status == "stopped" and output_verification_error:
                final_message = f"{final_message} Output note: {output_verification_error}"

            if final_status == "error":
                self.state.set_error(final_message, detail=error_detail)
            else:
                progress = 1.0 if final_status == "completed" else None
                self.state.finish(final_status, final_message, progress=progress)
            LOGGER.info("Processing finished with status=%s: %s", final_status, final_message)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def _process(self) -> tuple[str, str]:
        self.state.set_status("starting", "Opening the video source...")
        self.detector.load()

        self._cap = open_capture(
            self.config.source,
            prefer_dshow=self.config.source_type == "webcam",
        )

        if self.config.source_type == "video" and self.config.start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.config.start_frame))

        first_frame = self._read_first_frame()
        if first_frame is None:
            if self._stop_event.is_set():
                return "stopped", "Stopped before the first frame was processed."
            raise VideoSourceError(
                f"Opened '{self.config.source_label}' but could not read a frame. "
                "The file may be truncated or the camera may be disconnected."
            )

        height, width = first_frame.shape[:2]
        if self.config.mirror:
            first_frame = cv2.flip(first_frame, 1)
        self._width, self._height = int(width), int(height)

        raw_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        source_fps = raw_fps if math.isfinite(raw_fps) and 1.0 <= raw_fps <= 240.0 else 0.0
        raw_total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if raw_total > 0 and self.config.start_frame > 0:
            total_after_seek = max(0, raw_total - self.config.start_frame)
        else:
            total_after_seek = max(0, raw_total)

        if self.config.max_frames > 0:
            effective_total = (
                min(total_after_seek, self.config.max_frames)
                if total_after_seek > 0
                else self.config.max_frames
            )
        else:
            effective_total = total_after_seek

        self.state.set_source(width, height, source_fps, total_after_seek, effective_total)

        if self.config.output_path is not None:
            writer_fps = source_fps if source_fps > 0 else 30.0
            self._writer, self._writer_codec = create_video_writer(
                self.config.output_path,
                fps=writer_fps,
                size=(width, height),
            )
            self.state.set_output(self.config.output_path, self._writer_codec)

        self.detector.reset_tracking()
        session = TrackingSession()
        fps_meter = FpsMeter()
        inference_meter = FpsMeter()
        detections: list[Detection] = []
        stats = session.snapshot()
        inference_count = 0
        frame_index = 0
        frame: np.ndarray | None = first_frame
        loop_started = time.perf_counter()

        if self._stop_event.is_set():
            return self._stopped_message(frame_index)
        if self._pause_event.is_set():
            self.state.set_status("paused", "Paused. The video source is not being read.")
        else:
            self.state.set_status("running", "Processing frames...")
        LOGGER.info(
            "Started processing %s at %dx%d (source FPS=%.2f, total=%d)",
            self.config.source_label,
            width,
            height,
            source_fps,
            total_after_seek,
        )

        while True:
            if self._stop_event.is_set():
                return self._stopped_message(frame_index)
            self._wait_if_paused()
            if self._stop_event.is_set():
                return self._stopped_message(frame_index)

            if frame is None:
                ok, next_frame = self._cap.read()
                if not ok or next_frame is None:
                    if self.config.source_type == "webcam":
                        raise VideoSourceError(
                            "The webcam stream ended unexpectedly. Check the camera "
                            "connection and try again."
                        )
                    break
                frame = next_frame

            if self.config.mirror:
                frame = cv2.flip(frame, 1)
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                # Some cameras change resolution mid-stream. Keep writer dimensions.
                frame = cv2.resize(frame, (self._width, self._height))

            frame_index += 1
            if inference_count == 0 or (frame_index - 1) % self.config.detect_interval == 0:
                detections = self.detector.track(
                    frame,
                    conf=self.config.conf,
                    iou=self.config.iou,
                    classes=self.config.classes,
                    tracker=self.config.tracker,
                    persist=True,
                    imgsz=self.config.imgsz,
                )
                stats = session.update(detections, frame_index)
                inference_count += 1
                inference_meter.tick()

            draw_detections(
                frame,
                detections,
                show_stats_overlay=self.config.show_overlay,
                stats=stats,
                fps=fps_meter.value,
            )

            if self._writer is not None:
                self._writer.write(frame)
            self._frames_written += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processing_fps = fps_meter.tick()
            progress = (
                min(1.0, frame_index / float(effective_total))
                if effective_total > 0
                else None
            )
            self.state.publish(
                rgb,
                frame_index=frame_index,
                stats=stats,
                fps=processing_fps,
                inference_fps=inference_meter.value,
                progress=progress,
                inference_count=inference_count,
            )

            if self.config.max_frames > 0 and frame_index >= self.config.max_frames:
                return (
                    "completed",
                    f"Processed the requested {frame_index} frames.",
                )

            frame = None
            if (
                self.config.realtime
                and self.config.source_type == "video"
                and source_fps > 0
            ):
                target_elapsed = frame_index / source_fps
                actual_elapsed = time.perf_counter() - loop_started
                delay = target_elapsed - actual_elapsed
                if delay > 0:
                    self._stop_event.wait(min(delay, 1.0))

        return "completed", f"Video processing finished. {frame_index} frames processed."

    def _read_first_frame(self) -> np.ndarray | None:
        attempts = 6 if self.config.source_type == "webcam" else 1
        for attempt in range(attempts):
            if self._stop_event.is_set():
                return None
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                return frame
            if attempt < attempts - 1:
                time.sleep(0.12)
        return None

    def _wait_if_paused(self) -> None:
        if not self._pause_event.is_set():
            return
        self.state.set_status("paused", "Paused. The video source is not being read.")
        while self._pause_event.is_set() and not self._stop_event.is_set():
            time.sleep(0.05)
        if not self._stop_event.is_set():
            self.state.set_status("running", "Resumed.")

    def _stopped_message(self, frame_index: int) -> tuple[str, str]:
        self.state.set_status("stopping", "Stopping and releasing video resources...")
        if frame_index == 0:
            return "stopped", "Stopped before any frames were processed."
        return "stopped", f"Stopped by user after {frame_index} frames."

    def _release_resources(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:  # noqa: BLE001
                LOGGER.debug("Error releasing video writer", exc_info=True)
            finally:
                self._writer = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                LOGGER.debug("Error releasing video capture", exc_info=True)
            finally:
                self._cap = None
