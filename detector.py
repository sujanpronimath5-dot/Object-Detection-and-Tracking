"""YOLO detection and multi-object tracking wrapper.

The model is loaded once per detector instance. Consecutive calls to
``track()`` use Ultralytics' ``persist=True`` so tracker state (and therefore
tracking IDs) survives between frames.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from errors import ConfigurationError, ModelLoadError, TrackingError
from utils import (
    device_label,
    resolve_device,
    resolve_model_path,
    resolve_tracker_config,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """A single detected object in one frame."""

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.xyxy[3] - self.xyxy[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "xyxy": tuple(round(float(v), 2) for v in self.xyxy),
            "confidence": round(float(self.confidence), 4),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "track_id": None if self.track_id is None else int(self.track_id),
        }


class YOLODetector:
    """Thin, thread-safe wrapper around an Ultralytics YOLO model."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        imgsz: int = 640,
    ) -> None:
        self.model_path = str(model_path)
        self.device_preference = str(device or "auto")
        self.imgsz = int(imgsz)
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._resolved_path: Path | None = None
        self._device: str | None = None
        self._task: str | None = None
        self._names: dict[int, str] = {}
        self._load_seconds: float | None = None
        self._parameter_count: int = 0
        self._half: bool = False

    # ------------------------------------------------------------------
    # Loading / metadata
    # ------------------------------------------------------------------
    def load(self) -> "YOLODetector":
        """Load the model if needed. Repeated calls are cheap and idempotent."""
        with self._lock:
            if self._model is not None:
                return self

            resolved = resolve_model_path(self.model_path)
            device = resolve_device(self.device_preference)
            started = time.perf_counter()
            try:
                from ultralytics import YOLO  # imported lazily to keep app startup fast

                model = YOLO(str(resolved))
            except Exception as exc:  # noqa: BLE001 - convert library errors to friendly errors
                raise ModelLoadError(
                    f"Failed to load YOLO weights '{resolved.name}': "
                    f"{type(exc).__name__}: {exc}\n"
                    "Check that the file is a valid YOLO .pt checkpoint and that your "
                    "ultralytics/torch installation is compatible."
                ) from exc

            task = str(getattr(model, "task", "") or "").lower()
            if task and task not in {"detect", "segment", "pose", "obb"}:
                raise ModelLoadError(
                    f"The model task '{task}' does not support object detection/tracking. "
                    "Use a detection model such as yolo11n.pt."
                )

            raw_names = getattr(model, "names", {}) or {}
            if isinstance(raw_names, (list, tuple)):
                names = {index: str(name) for index, name in enumerate(raw_names)}
            else:
                names = {int(index): str(name) for index, name in dict(raw_names).items()}

            parameter_count = 0
            try:
                parameter_count = int(sum(p.numel() for p in model.model.parameters()))
            except (AttributeError, TypeError):
                parameter_count = 0

            self._model = model
            self._resolved_path = resolved
            self._device = device
            self._task = task or "detect"
            self._names = names
            self._load_seconds = time.perf_counter() - started
            self._parameter_count = parameter_count
            self._half = device.lower().startswith("cuda")

            if not self._half:
                LOGGER.warning(
                    "CUDA is unavailable; running YOLO inference on CPU. "
                    "Use a smaller --imgsz or a GPU for higher FPS."
                )
            LOGGER.info(
                "Loaded %s on %s in %.2fs (%d classes)",
                resolved.name,
                device_label(device),
                self._load_seconds,
                len(names),
            )
            return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def names(self) -> dict[int, str]:
        with self._lock:
            return dict(self._names)

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def resolved_path(self) -> Path | None:
        return self._resolved_path

    @property
    def info(self) -> dict[str, Any]:
        """Dictionary used by the UI for model/device status."""
        with self._lock:
            resolved = self._resolved_path or Path(self.model_path)
            try:
                import ultralytics

                ultralytics_version = ultralytics.__version__
            except ImportError:
                ultralytics_version = "not installed"
            return {
                "path": str(resolved),
                "model_name": resolved.name,
                "task": self._task or "not loaded",
                "device": self._device or "not loaded",
                "device_label": device_label(self._device or ""),
                "class_count": len(self._names),
                "names": dict(self._names),
                "parameters": self._parameter_count,
                "load_seconds": self._load_seconds,
                "half": self._half,
                "ultralytics_version": ultralytics_version,
            }

    # ------------------------------------------------------------------
    # Tracking / detection
    # ------------------------------------------------------------------
    def reset_tracking(self) -> None:
        """Clear Ultralytics tracker state so a new session starts with new IDs.

        The next ``track(..., persist=True)`` call sees no active trackers and
        creates fresh ByteTrack/BoT-SORT instances.
        """
        with self._lock:
            if self._model is None:
                return
            predictor = getattr(self._model, "predictor", None)
            if predictor is None:
                return
            try:
                predictor.trackers = []
            except (AttributeError, TypeError):
                LOGGER.debug("Could not clear predictor.trackers", exc_info=True)
            try:
                if hasattr(predictor, "vid_path"):
                    predictor.vid_path = [None]
            except (AttributeError, TypeError):
                LOGGER.debug("Could not reset predictor.vid_path", exc_info=True)
            LOGGER.info("Tracker state reset for a new session")

    def track(
        self,
        frame: np.ndarray,
        *,
        conf: float = 0.25,
        iou: float = 0.5,
        classes: Sequence[int] | None = None,
        tracker: str = "ByteTrack",
        persist: bool = True,
        imgsz: int | None = None,
    ) -> list[Detection]:
        """Run detection + tracking on one BGR frame.

        ``persist=True`` is essential: it keeps each track's ID alive across
        consecutive frames. IDs are never synthesized by this wrapper.
        """
        self._validate_frame(frame)
        self._validate_thresholds(conf, iou)
        tracker_config = resolve_tracker_config(tracker)
        kwargs = self._inference_kwargs(conf, iou, classes, imgsz)

        with self._lock:
            self.load()
            started = time.perf_counter()
            try:
                results = self._model.track(
                    frame,
                    persist=bool(persist),
                    tracker=tracker_config,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - expose a useful tracking error
                raise TrackingError(
                    f"YOLO tracking failed: {type(exc).__name__}: {exc}"
                ) from exc
            detections = self._extract_detections(results[0] if results else None)
            LOGGER.debug(
                "track: %d detections, %d with IDs, %.1f ms",
                len(detections),
                sum(1 for d in detections if d.track_id is not None),
                (time.perf_counter() - started) * 1000.0,
            )
            return detections

    def detect(
        self,
        frame: np.ndarray,
        *,
        conf: float = 0.25,
        iou: float = 0.5,
        classes: Sequence[int] | None = None,
        imgsz: int | None = None,
    ) -> list[Detection]:
        """Run one-shot detection (used by verification); no tracking IDs."""
        self._validate_frame(frame)
        self._validate_thresholds(conf, iou)
        kwargs = self._inference_kwargs(conf, iou, classes, imgsz)
        with self._lock:
            self.load()
            try:
                results = self._model.predict(frame, **kwargs)
            except Exception as exc:  # noqa: BLE001
                raise TrackingError(
                    f"YOLO detection failed: {type(exc).__name__}: {exc}"
                ) from exc
            return self._extract_detections(results[0] if results else None)

    def warmup(self, size: int | None = None) -> None:
        """Optional one-frame warmup; useful before benchmarking."""
        dimension = int(size or self.imgsz)
        frame = np.zeros((dimension, dimension, 3), dtype=np.uint8)
        self.detect(frame, conf=0.99, iou=0.5, imgsz=dimension)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _inference_kwargs(
        self,
        conf: float,
        iou: float,
        classes: Sequence[int] | None,
        imgsz: int | None,
    ) -> dict[str, Any]:
        if self._device is None:
            self._device = resolve_device(self.device_preference)
        size = int(imgsz or self.imgsz)
        if size < 32:
            raise ConfigurationError("Inference image size must be at least 32 pixels.")
        kwargs: dict[str, Any] = {
            "conf": float(conf),
            "iou": float(iou),
            "imgsz": size,
            "device": self._device,
            "max_det": 300,
            "verbose": False,
        }
        if self._half:
            # Ultralytics 8.4 replaced the deprecated `half` flag with `quantize`.
            kwargs["quantize"] = 16
        if classes is not None:
            values = [int(value) for value in classes]
            if not values:
                raise ConfigurationError(
                    "No object classes are selected. Select at least one class or "
                    "enable 'Detect all classes'."
                )
            kwargs["classes"] = values
        return kwargs

    @staticmethod
    def _validate_thresholds(conf: float, iou: float) -> None:
        if not 0.0 <= float(conf) <= 1.0:
            raise ConfigurationError("Confidence threshold must be between 0 and 1.")
        if not 0.0 <= float(iou) <= 1.0:
            raise ConfigurationError("IoU threshold must be between 0 and 1.")

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame is None or not isinstance(frame, np.ndarray):
            raise TrackingError("No video frame was supplied to the detector.")
        if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            raise TrackingError(
                "Expected a BGR color frame with shape (height, width, 3)."
            )

    def _extract_detections(self, result: Any | None) -> list[Detection]:
        if result is None:
            return []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        try:
            xyxy_values = boxes.xyxy.detach().cpu().numpy()
            confidence_values = boxes.conf.detach().cpu().numpy()
            class_values = boxes.cls.detach().cpu().numpy().astype(int)
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise TrackingError(f"Could not read YOLO box data: {exc}") from exc

        id_values: list[int] | None = None
        raw_ids = getattr(boxes, "id", None)
        if raw_ids is not None:
            try:
                id_values = [int(value) for value in raw_ids.detach().cpu().numpy().tolist()]
            except (AttributeError, RuntimeError, TypeError, ValueError):
                id_values = None

        result_names = getattr(result, "names", None)
        if isinstance(result_names, dict):
            names = {int(key): str(value) for key, value in result_names.items()}
        elif isinstance(result_names, (list, tuple)):
            names = {index: str(value) for index, value in enumerate(result_names)}
        else:
            names = self._names

        detections: list[Detection] = []
        for index in range(len(xyxy_values)):
            try:
                class_id = int(class_values[index])
                track_id = id_values[index] if id_values is not None and index < len(id_values) else None
                detections.append(
                    Detection(
                        xyxy=(
                            float(xyxy_values[index][0]),
                            float(xyxy_values[index][1]),
                            float(xyxy_values[index][2]),
                            float(xyxy_values[index][3]),
                        ),
                        confidence=float(confidence_values[index]),
                        class_id=class_id,
                        class_name=names.get(class_id, str(class_id)),
                        track_id=track_id,
                    )
                )
            except (IndexError, TypeError, ValueError):
                LOGGER.debug("Skipped malformed YOLO detection at index %s", index, exc_info=True)
        return detections

    def close(self) -> None:
        """Release model references (used when switching weights)."""
        with self._lock:
            if self._model is not None:
                try:
                    predictor = getattr(self._model, "predictor", None)
                    if predictor is not None:
                        predictor.trackers = []
                except (AttributeError, TypeError):
                    pass
            self._model = None
            self._resolved_path = None
            self._device = None
            self._task = None
            self._names = {}
            self._load_seconds = None
            self._parameter_count = 0
            self._half = False
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
            LOGGER.info("Detector closed")


def detections_to_dicts(detections: Iterable[Detection]) -> list[dict[str, Any]]:
    return [detection.to_dict() for detection in detections]
