"""Integration tests for real YOLO inference, tracking and the video pipeline."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import cv2
import pytest

from detector import Detection, YOLODetector
from utils import open_capture, probe_video, read_video_frame, verify_output_video
from video_processor import ProcessingConfig, VideoProcessor


class StubDetector:
    """Fast deterministic detector used to test pipeline control flow."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = float(delay)
        self.load_calls = 0
        self.reset_calls = 0
        self.track_calls = 0

    def load(self) -> "StubDetector":
        self.load_calls += 1
        return self

    def reset_tracking(self) -> None:
        self.reset_calls += 1

    def track(self, frame, **kwargs) -> list[Detection]:
        self.track_calls += 1
        if self.delay:
            time.sleep(self.delay)
        height, width = frame.shape[:2]
        return [
            Detection(
                (width * 0.2, height * 0.2, width * 0.6, height * 0.8),
                0.9,
                0,
                "person",
                1,
            )
        ]


def _wait_for(predicate, timeout: float = 10.0, interval: float = 0.01) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Real model
# ---------------------------------------------------------------------------
def test_real_model_detects_sample_frame(detector: YOLODetector, sample_video: Path) -> None:
    frame = read_video_frame(sample_video, 480)
    detections = detector.detect(frame, conf=0.2, iou=0.5, imgsz=640)
    assert detections, "Expected at least one object in the sample frame."
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        assert x2 > x1 and y2 > y1
        assert 0.0 <= detection.confidence <= 1.0
        assert detection.class_name
        assert detection.track_id is None  # one-shot detection must not invent IDs


def test_real_tracking_keeps_an_id_across_frames(
    detector: YOLODetector, sample_video: Path
) -> None:
    capture = open_capture(sample_video)
    seen_ids: Counter[int] = Counter()
    frames = 0
    try:
        assert capture.set(cv2.CAP_PROP_POS_FRAMES, 420)
        detector.reset_tracking()
        for _ in range(20):
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            detections = detector.track(
                frame,
                conf=0.2,
                iou=0.5,
                tracker="ByteTrack",
                persist=True,
                imgsz=640,
            )
            for detection in detections:
                if detection.track_id is not None:
                    seen_ids[int(detection.track_id)] += 1
    finally:
        capture.release()
        detector.reset_tracking()

    assert frames == 20
    assert seen_ids, "The tracker did not assign any IDs."
    most_common_id, count = seen_ids.most_common(1)[0]
    assert count >= 10, f"ID #{most_common_id} was not persistent ({count}/20 frames)."


def test_botsort_tracker_runs(detector: YOLODetector, sample_video: Path) -> None:
    capture = open_capture(sample_video)
    frames = 0
    detections = 0
    ids: set[int] = set()
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 420)
        detector.reset_tracking()
        for _ in range(8):
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            results = detector.track(
                frame,
                conf=0.2,
                iou=0.5,
                tracker="BoT-SORT",
                persist=True,
                imgsz=640,
            )
            detections += len(results)
            ids.update(d.track_id for d in results if d.track_id is not None)
    finally:
        capture.release()
        detector.reset_tracking()

    assert frames == 8
    assert detections > 0
    assert ids


def test_video_processor_end_to_end(
    detector: YOLODetector, sample_video: Path, tmp_path: Path
) -> None:
    source_info = probe_video(sample_video)
    output = tmp_path / "annotated.mp4"
    config = ProcessingConfig(
        source_type="video",
        source=sample_video,
        output_path=output,
        conf=0.2,
        iou=0.5,
        tracker="ByteTrack",
        imgsz=640,
        detect_interval=1,
        show_overlay=True,
        start_frame=420,
        max_frames=15,
        model_name="yolo11n.pt",
        device_label="CPU",
    )
    processor = VideoProcessor(config, detector)
    processor.start()
    processor.join(timeout=90)

    assert not processor.is_alive(), "Processor did not finish in time."
    snap = processor.snapshot()
    assert snap["status"] == "completed", snap["error"] or snap["status_message"]
    assert snap["frame_index"] == 15
    assert snap["resolution"] == (source_info.width, source_info.height)
    assert snap["source_fps"] == pytest.approx(source_info.output_fps, abs=0.5)
    assert snap["stats"].unique_ids >= 1
    assert snap["resources_released"] is True
    assert snap["cleanup_complete"] is True
    assert snap["output_ready"] is True
    assert snap["output_error"] is None
    assert snap["output_frames"] == 15

    info = verify_output_video(output, expected_size=(source_info.width, source_info.height))
    assert info["size_bytes"] > 1024
    assert info["frame_count"] == 15
    assert info["fps"] == pytest.approx(source_info.output_fps, abs=1.0)


def test_video_processor_missing_video(tmp_path: Path, detector: YOLODetector) -> None:
    output = tmp_path / "should_not_exist.mp4"
    config = ProcessingConfig(
        source_type="video",
        source=tmp_path / "missing.mp4",
        output_path=output,
        max_frames=5,
    )
    processor = VideoProcessor(config, detector)
    processor.start()
    processor.join(timeout=30)

    assert not processor.is_alive()
    snap = processor.snapshot()
    assert snap["status"] == "error"
    assert "not found" in (snap["error"] or "").lower()
    assert snap["resources_released"] is True
    assert snap["cleanup_complete"] is True
    assert not output.exists()


# ---------------------------------------------------------------------------
# Pipeline control flow (no real model required)
# ---------------------------------------------------------------------------
def test_detection_interval_skips_inference_but_writes_all_frames(
    synthetic_video: Path, tmp_path: Path
) -> None:
    stub = StubDetector()
    config = ProcessingConfig(
        source_type="video",
        source=synthetic_video,
        output_path=tmp_path / "interval.mp4",
        detect_interval=3,
        max_frames=10,
        show_overlay=False,
    )
    processor = VideoProcessor(config, stub)
    processor.start()
    processor.join(timeout=30)

    assert not processor.is_alive()
    snap = processor.snapshot()
    assert snap["status"] == "completed", snap["error"]
    assert snap["frame_index"] == 10
    assert snap["output_frames"] == 10
    # Frames 1, 4, 7 and 10 run inference when the interval is three.
    assert stub.track_calls == 4
    assert snap["stats"].inference_frames == 4
    assert snap["stats"].current_objects == 1


def test_processor_pause_resume_controls(synthetic_video: Path) -> None:
    stub = StubDetector()
    processor = VideoProcessor(
        ProcessingConfig(source_type="video", source=synthetic_video), stub
    )
    processor.state.set_status("running", "Processing frames...")

    processor.pause()
    assert processor.snapshot()["status"] == "paused"
    processor.resume()
    assert processor.snapshot()["status"] == "running"
    processor.stop()
    assert processor.snapshot()["status"] == "stopping"
    assert processor.stop_requested is True


def test_processor_stop_releases_resources(
    long_synthetic_video: Path, tmp_path: Path
) -> None:
    stub = StubDetector(delay=0.01)
    output = tmp_path / "partial.mp4"
    processor = VideoProcessor(
        ProcessingConfig(
            source_type="video",
            source=long_synthetic_video,
            output_path=output,
            max_frames=0,
            show_overlay=False,
        ),
        stub,
    )
    processor.start()
    started = _wait_for(
        lambda: processor.snapshot()["frame_index"] >= 2,
        timeout=10,
    )
    assert started, "Processing did not start in time."
    processor.stop()
    processor.join(timeout=15)

    assert not processor.is_alive()
    snap = processor.snapshot()
    assert snap["status"] == "stopped"
    assert snap["frame_index"] >= 2
    assert snap["resources_released"] is True
    assert snap["cleanup_complete"] is True
    assert output.exists() and output.stat().st_size > 512
    # The partial video is still a valid, decodable MP4.
    assert verify_output_video(output)["frame_count"] >= 1
