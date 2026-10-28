"""Unit tests for configuration, tracking statistics, drawing and video I/O."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import utils
from detector import Detection
from errors import ConfigurationError, ModelLoadError, OutputVideoError, VideoSourceError
from tracker import TrackStats, TrackingSession
from utils import (
    FpsMeter,
    color_for_detection,
    create_video_writer,
    draw_detections,
    format_bytes,
    format_duration,
    make_output_path,
    probe_video,
    resolve_device,
    resolve_model_path,
    resolve_tracker_config,
    sanitize_filename,
    save_uploaded_video,
    verify_output_video,
)
from video_processor import ProcessingConfig as VideoProcessingConfig


# ---------------------------------------------------------------------------
# ProcessingConfig
# ---------------------------------------------------------------------------
def test_processing_config_normalizes_values(synthetic_video: Path, tmp_path: Path) -> None:
    config = VideoProcessingConfig(
        source_type=" VIDEO ",
        source=synthetic_video,
        output_path=tmp_path / "out.mp4",
        conf=0.3,
        iou=0.6,
        classes=[0, 2],
        tracker="BoT-SORT",
        imgsz=512,
        detect_interval=2,
        max_frames=5,
        start_frame=3,
        model_name="yolo11n.pt",
        device_label="CPU",
    )
    assert config.source_type == "video"
    assert config.classes == (0, 2)
    assert config.output_path == tmp_path / "out.mp4"
    assert config.source_label == synthetic_video.name
    summary = config.summary()
    assert summary["tracker"] == "BoT-SORT"
    assert summary["classes"] == "2 selected"
    assert summary["confidence"] == 0.3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_type": "image", "source": "frame.jpg"},
        {"source_type": "webcam", "source": "0"},
        {"source_type": "video", "source": 0},
        {"source_type": "video", "source": "clip.mp4", "conf": 1.5},
        {"source_type": "video", "source": "clip.mp4", "iou": -0.1},
        {"source_type": "video", "source": "clip.mp4", "imgsz": 16},
        {"source_type": "video", "source": "clip.mp4", "detect_interval": 0},
        {"source_type": "video", "source": "clip.mp4", "max_frames": -1},
        {"source_type": "video", "source": "clip.mp4", "start_frame": -1},
        {"source_type": "video", "source": "clip.mp4", "classes": []},
        {"source_type": "video", "source": "clip.mp4", "tracker": "NotATracker"},
    ],
)
def test_processing_config_rejects_invalid_settings(kwargs: dict) -> None:
    with pytest.raises(ConfigurationError):
        VideoProcessingConfig(**kwargs)


def test_tracker_display_name_and_aliases(tmp_path: Path) -> None:
    assert resolve_tracker_config("ByteTrack") == "bytetrack.yaml"
    assert resolve_tracker_config("byte_track") == "bytetrack.yaml"
    assert resolve_tracker_config("BoT-SORT") == "botsort.yaml"
    assert resolve_tracker_config("bot_sort") == "botsort.yaml"
    custom = tmp_path / "custom_tracker.yaml"
    custom.write_text("tracker_type: bytetrack\n", encoding="utf-8")
    assert resolve_tracker_config(str(custom)) == str(custom.resolve())
    with pytest.raises(ConfigurationError):
        resolve_tracker_config("SORT-ish")


# ---------------------------------------------------------------------------
# Tracking statistics
# ---------------------------------------------------------------------------
def test_tracking_session_counts_ids_classes_and_observations() -> None:
    session = TrackingSession()
    first = session.update(
        [
            Detection((10, 10, 40, 60), 0.91, 2, "car", 7),
            Detection((50, 20, 80, 90), 0.77, 0, "person", None),
        ],
        frame_number=1,
    )
    assert first.current_objects == 2
    assert first.unique_ids == 1
    assert first.untracked_objects == 1
    assert first.class_counts == {"car": 1, "person": 1}
    assert first.unique_class_counts == {"car": 1}
    assert first.current_ids == (7,)
    assert first.inference_frames == 1

    second = session.update(
        [
            Detection((11, 12, 42, 63), 0.93, 2, "car", 7),
            Detection((100, 30, 130, 95), 0.70, 2, "car", 8),
            Detection((5, 5, 20, 20), 0.65, 0, "person", "not-an-id"),
        ],
        frame_number=2,
    )
    assert second.current_objects == 3
    assert second.unique_ids == 2
    assert second.untracked_objects == 1
    assert second.class_counts == {"car": 2, "person": 1}
    assert second.unique_class_counts == {"car": 2}
    assert second.id_observations[7] == 2
    assert second.id_observations[8] == 1
    assert second.id_first_frame[7] == 1
    assert second.id_last_frame[7] == 2
    assert second.total_tracked_observations == 3

    rows = second.class_rows()
    car_row = next(row for row in rows if row["Class"] == "car")
    assert car_row["Objects now"] == 2
    assert car_row["Unique IDs"] == 2
    person_row = next(row for row in rows if row["Class"] == "person")
    assert person_row["Objects now"] == 1
    assert person_row["Unique IDs"] == 0

    # The same tracker ID appearing twice in one frame is counted once.
    duplicate = session.update(
        [
            Detection((11, 12, 42, 63), 0.93, 2, "car", 7),
            Detection((12, 12, 43, 63), 0.90, 2, "car", 7),
        ],
        frame_number=3,
    )
    assert duplicate.id_observations[7] == 3
    assert duplicate.unique_ids == 2

    snapshot = session.snapshot()
    assert snapshot.frame_number == 3
    assert [row["ID"] for row in snapshot.id_rows(limit=1)] == [7]
    assert snapshot.to_dict()["current_ids"] == [7]

    session.reset()
    cleared = session.snapshot()
    assert cleared == TrackStats()
    assert cleared.unique_ids == 0
    assert cleared.class_rows() == []


# ---------------------------------------------------------------------------
# Detection data model and basic helpers
# ---------------------------------------------------------------------------
def test_detection_geometry_and_serialization() -> None:
    detection = Detection((10.123, 20.456, 40.789, 80.987), 0.91234, 2, "car", 15)
    assert detection.width == pytest.approx(30.666)
    assert detection.height == pytest.approx(60.531)
    payload = detection.to_dict()
    assert payload == {
        "xyxy": (10.12, 20.46, 40.79, 80.99),
        "confidence": 0.9123,
        "class_id": 2,
        "class_name": "car",
        "track_id": 15,
    }
    assert color_for_detection(2, 15) == color_for_detection(2, 15)


def test_device_resolution_auto_cpu_and_cuda_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ConfigurationError):
        resolve_device("cuda")
    with pytest.raises(ConfigurationError):
        resolve_device("quantum")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda:0"
    assert resolve_device("cuda") == "cuda:0"
    assert resolve_device("cuda:1") == "cuda:1"


def test_filename_sanitization_and_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert sanitize_filename("CON") == "_CON"
    assert sanitize_filename("  ???  ", fallback="video") == "video"
    assert "/" not in sanitize_filename("a/b c?.mp4")
    assert len(sanitize_filename("x" * 200, max_length=20)) <= 20

    monkeypatch.setattr(utils, "OUTPUT_VIDEOS_DIR", tmp_path / "outputs")
    first = make_output_path("demo.mp4", prefix="session one")
    assert first.parent == tmp_path / "outputs"
    assert first.suffix == ".mp4"
    assert "session_one" in first.name
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"x")
    second = make_output_path("demo.mp4", prefix="session one")
    assert second != first


def test_formatting_helpers() -> None:
    assert format_duration(None) == "--:--"
    assert format_duration(-5) == "--:--"
    assert format_duration(0) == "0:00"
    assert format_duration(65) == "1:05"
    assert format_duration(3725) == "1:02:05"
    assert format_bytes(None) == "--"
    assert format_bytes(512) == "512 B"
    assert format_bytes(1536) == "1.5 KB"
    assert format_bytes(5 * 1024 * 1024) == "5.0 MB"


def test_fps_meter_smoothing(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([0.0, 0.1, 0.3])
    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(ticks))
    meter = FpsMeter(smoothing=0.5)
    assert meter.tick() == 0.0
    assert meter.tick() == pytest.approx(10.0)
    assert meter.tick() == pytest.approx(7.5)


def test_draw_detections_draws_boxes_and_overlay() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = [
        Detection((20, 30, 120, 180), 0.88, 0, "person", 3),
        Detection((150, 60, 280, 200), 0.72, 2, "car", None),
    ]
    annotated = draw_detections(
        frame,
        detections,
        show_stats_overlay=True,
        stats={"current_objects": 2, "unique_ids": 1},
        fps=17.5,
    )
    assert annotated is frame
    assert float(np.mean(cv2.absdiff(frame, np.zeros_like(frame)))) > 1.0
    assert draw_detections(frame, [], show_stats_overlay=False) is frame

    # Malformed objects are skipped instead of crashing the processing loop.
    draw_detections(frame, [{"xyxy": (1, 2)}, object()], show_stats_overlay=False)


# ---------------------------------------------------------------------------
# Video and upload I/O
# ---------------------------------------------------------------------------
def test_probe_video_and_errors(synthetic_video: Path, tmp_path: Path) -> None:
    info = probe_video(synthetic_video)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(15.0, abs=1.0)
    assert 85 <= info.frame_count <= 95
    assert info.has_valid_fps
    assert info.duration_seconds is not None

    with pytest.raises(VideoSourceError):
        probe_video(tmp_path / "missing.mp4")
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not a video", encoding="utf-8")
    with pytest.raises(VideoSourceError):
        probe_video(unsupported)


def test_writer_round_trip_and_verification(tmp_path: Path) -> None:
    output = tmp_path / "written.mp4"
    writer, codec = create_video_writer(output, fps=30.0, size=(320, 240))
    assert codec
    for index in range(12):
        frame = np.full((240, 320, 3), (index * 10, 60, 120), dtype=np.uint8)
        cv2.putText(frame, str(index), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()

    info = verify_output_video(output, expected_size=(320, 240))
    assert info["size_bytes"] > 512
    assert info["width"] == 320
    assert info["height"] == 240
    assert info["frame_count"] == pytest.approx(12, abs=1)
    assert info["fps"] == pytest.approx(30.0, abs=1.0)

    with pytest.raises(OutputVideoError):
        verify_output_video(output, expected_size=(640, 480))
    with pytest.raises(OutputVideoError):
        verify_output_video(tmp_path / "absent.mp4")


def test_uploaded_video_is_saved_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(utils, "UPLOAD_DIR", tmp_path / "uploads")
    payload = b"fake-mp4-bytes"
    first = save_uploaded_video("My Clip.mp4", payload)
    second = save_uploaded_video("My Clip.mp4", payload)
    assert first == second
    assert first.exists()
    assert first.read_bytes() == payload
    assert first.parent == tmp_path / "uploads"

    with pytest.raises(VideoSourceError):
        save_uploaded_video("clip.txt", payload)
    with pytest.raises(VideoSourceError):
        save_uploaded_video("empty.mp4", b"")


def test_resolve_model_path(tmp_path: Path) -> None:
    weights = tmp_path / "custom.pt"
    weights.write_bytes(b"placeholder")
    assert resolve_model_path(weights) == weights.resolve()
    with pytest.raises(ModelLoadError):
        resolve_model_path(tmp_path / "missing.pt")


def test_open_capture_rejects_missing_video_and_bad_camera(tmp_path: Path) -> None:
    with pytest.raises(VideoSourceError):
        utils.open_capture(tmp_path / "missing.mp4")
    with pytest.raises(VideoSourceError):
        utils.open_capture(-1)
