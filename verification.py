"""End-to-end verification for the object detection and tracking project.

Run this after installation to check:
  * dependency/model loading,
  * detection on a sample frame,
  * tracking IDs across a short video,
  * both ByteTrack and BoT-SORT configurations,
  * annotated output video generation and properties,
  * session statistics correctness,
  * camera/writer cleanup on success and on an invalid input.

Examples
--------
    python verification.py
    python verification.py --video input_videos/sample_traffic.avi --start-frame 420 --frames 60
    python verification.py --camera 0

The default chooses the included sample video and a segment that contains a
tracked vehicle. For your own video, pass --video and optionally --start-frame.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detector import Detection, YOLODetector
from errors import AppError
from tracker import TrackingSession
from utils import (
    INPUT_VIDEOS_DIR,
    OUTPUT_VIDEOS_DIR,
    SUPPORTED_VIDEO_EXTENSIONS,
    cuda_available,
    dependency_versions,
    draw_detections,
    ensure_directories,
    format_bytes,
    make_output_path,
    open_capture,
    probe_video,
    relative_path,
    resolve_model_path,
)
from video_processor import ProcessingConfig, VideoProcessor

LOGGER = logging.getLogger("verification")
DEFAULT_SAMPLE = INPUT_VIDEOS_DIR / "sample_traffic.avi"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    duration: float
    required: bool = True
    skipped: bool = False


class VerificationRunner:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def run(self, name: str, function, *, required: bool = True) -> CheckResult:
        print(f"\n[CHECK] {name}")
        started = time.perf_counter()
        try:
            detail = function()
            elapsed = time.perf_counter() - started
            result = CheckResult(name, True, str(detail or "OK"), elapsed, required)
            print(f"  PASS  {result.detail}  ({elapsed:.2f}s)")
        except Exception as exc:  # noqa: BLE001 - this is a test runner
            elapsed = time.perf_counter() - started
            result = CheckResult(name, False, f"{type(exc).__name__}: {exc}", elapsed, required)
            print(f"  FAIL  {result.detail}  ({elapsed:.2f}s)")
            traceback.print_exc()
        self.results.append(result)
        return result

    def skip(self, name: str, detail: str) -> None:
        print(f"\n[CHECK] {name}")
        print(f"  SKIP  {detail}")
        self.results.append(CheckResult(name, True, detail, 0.0, required=False, skipped=True))


def _find_default_video() -> Path | None:
    if DEFAULT_SAMPLE.exists():
        return DEFAULT_SAMPLE
    if not INPUT_VIDEOS_DIR.exists():
        return None
    candidates = sorted(
        path
        for path in INPUT_VIDEOS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )
    return candidates[0] if candidates else None


def _choose_detection_frame(capture_path: Path, frame_count: int) -> tuple[int, np.ndarray]:
    """Scan several frames and return the first one with a detection."""
    if frame_count <= 0:
        indices = [0]
    elif capture_path.resolve() == DEFAULT_SAMPLE.resolve():
        indices = [480, 120, 600, 840, 0]
    else:
        indices = sorted(
            {
                0,
                *[max(0, min(frame_count - 1, int(frame_count * fraction))) for fraction in (0.1, 0.25, 0.4, 0.6, 0.8)],
            }
        )
    last_frame = None
    last_index = 0
    for index in indices:
        capture = cv2.VideoCapture(str(capture_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open {capture_path.name}")
            if index > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            continue
        last_frame = frame
        last_index = index
        detections = DETECTOR.detect(frame, conf=0.20, iou=0.50, classes=None, imgsz=640)
        if detections:
            return index, frame
    if last_frame is None:
        raise RuntimeError(f"Could not decode any frame from {capture_path.name}")
    return last_index, last_frame


def check_environment() -> str:
    versions = dependency_versions()
    details = ", ".join(f"{name}={value}" for name, value in versions.items())
    print(f"  Python {platform.python_version()} on {platform.platform()}")
    print(f"  {details}")
    print(f"  CUDA available: {cuda_available()}")
    missing = [name for name, value in versions.items() if value == "not installed"]
    if missing:
        raise RuntimeError(
            "Missing packages: " + ", ".join(missing) + ". Run: pip install -r requirements.txt"
        )
    return f"Python {platform.python_version()}; " + ", ".join(
        f"{name} {value}" for name, value in versions.items()
    )


def check_model() -> str:
    global DETECTOR
    started = time.perf_counter()
    DETECTOR = YOLODetector(MODEL_PATH, device="auto")
    DETECTOR.load()
    elapsed = time.perf_counter() - started
    info = DETECTOR.info
    if info["class_count"] <= 0:
        raise RuntimeError("The model reports zero object classes.")
    if info["task"] not in {"detect", "segment", "pose", "obb"}:
        raise RuntimeError(f"Unsupported task: {info['task']}")
    print(
        f"  {info['model_name']} | task={info['task']} | classes={info['class_count']} | "
        f"device={info['device_label']} | params={info['parameters']:,}"
    )
    return (
        f"Loaded {info['model_name']} on {info['device_label']} in {elapsed:.2f}s "
        f"({info['class_count']} classes)"
    )


def check_detection() -> str:
    if VIDEO_PATH is None:
        raise RuntimeError("No sample video available for the detection check.")
    info = probe_video(VIDEO_PATH)
    index, frame = _choose_detection_frame(VIDEO_PATH, info.frame_count)
    detections = DETECTOR.detect(frame, conf=0.20, iou=0.50, classes=None, imgsz=640)
    if not detections:
        raise RuntimeError(
            "No objects were detected in the sampled video frames. "
            "Use a video containing visible objects or lower the confidence threshold."
        )

    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        if not (x2 > x1 and y2 > y1):
            raise RuntimeError(f"Invalid bounding box: {detection}")
        if not 0.0 <= detection.confidence <= 1.0:
            raise RuntimeError(f"Invalid confidence: {detection.confidence}")
        if not detection.class_name:
            raise RuntimeError(f"Missing class name: {detection}")

    session = TrackingSession()
    stats = session.update(detections, index + 1)
    annotated = draw_detections(
        frame.copy(),
        detections,
        show_stats_overlay=True,
        stats=stats,
        fps=12.3,
    )
    difference = float(np.mean(cv2.absdiff(frame, annotated)))
    if difference <= 0.01:
        raise RuntimeError("draw_detections did not modify the image; labels/boxes are missing.")
    OUTPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    annotated_path = OUTPUT_VIDEOS_DIR / "verification_detection.jpg"
    if not cv2.imwrite(str(annotated_path), annotated):
        raise RuntimeError(f"Could not save annotated sample image: {annotated_path}")

    labels = ", ".join(
        f"{d.class_name} {d.confidence:.2f}" for d in detections[:6]
    )
    scan_output = OUTPUT_VIDEOS_DIR / "verification_detection.jpg"
    print(f"  Frame {index}: {len(detections)} detection(s): {labels}")
    print(f"  Annotated image: {relative_path(scan_output)}")
    return (
        f"{len(detections)} detections on frame {index} ({labels}); "
        f"annotated image saved to {relative_path(annotated_path)}"
    )


def check_unique_id_statistics() -> str:
    session = TrackingSession()
    first = session.update(
        [
            Detection((1, 1, 20, 20), 0.9, 2, "car", 7),
            Detection((2, 2, 21, 21), 0.8, 0, "person", None),
        ],
        1,
    )
    if first.unique_ids != 1 or first.current_objects != 2 or first.untracked_objects != 1:
        raise RuntimeError(f"Unexpected first-frame statistics: {first.to_dict()}")
    second = session.update(
        [
            Detection((1, 1, 20, 20), 0.9, 2, "car", 7),
            Detection((3, 3, 22, 22), 0.7, 2, "car", 8),
        ],
        2,
    )
    if second.unique_ids != 2 or second.id_observations.get(7) != 2:
        raise RuntimeError(f"Unique ID de-duplication failed: {second.to_dict()}")
    if second.current_objects != 2 or second.class_counts.get("car") != 2:
        raise RuntimeError(f"Class counts failed: {second.to_dict()}")
    session.reset()
    if session.snapshot().unique_ids != 0:
        raise RuntimeError("TrackingSession.reset() did not clear unique IDs.")
    print(
        "  Verified: untracked detections excluded, repeated IDs counted once, "
        "class counts correct, reset clears session IDs."
    )
    return "Unique IDs de-duplicated correctly; untracked detections excluded; reset works"


def check_tracking_and_output() -> str:
    if VIDEO_PATH is None:
        raise RuntimeError("No video available for the tracking check.")
    info = probe_video(VIDEO_PATH)
    start_frame = START_FRAME
    remaining = max(0, info.frame_count - start_frame) if info.frame_count else FRAMES
    expected_frames = min(FRAMES, remaining) if remaining > 0 else FRAMES
    output_path = make_output_path(VIDEO_PATH, prefix="verification")

    config = ProcessingConfig(
        source_type="video",
        source=VIDEO_PATH,
        output_path=output_path,
        conf=0.20,
        iou=0.50,
        tracker="ByteTrack",
        imgsz=640,
        detect_interval=1,
        show_overlay=True,
        start_frame=start_frame,
        max_frames=FRAMES,
        model_name=DETECTOR.info["model_name"],
        device_label=DETECTOR.info["device_label"],
    )
    processor = VideoProcessor(config, DETECTOR)
    started = time.perf_counter()
    processor.start()
    processor.join(timeout=max(120, FRAMES * 3))
    elapsed = time.perf_counter() - started
    if processor.is_alive():
        processor.stop()
        processor.join(timeout=15)
        raise RuntimeError("Tracking did not finish within the timeout.")

    snap = processor.snapshot()
    print(f"  Status: {snap['status']} | message: {snap['status_message']}")
    print(
        f"  Frames: {snap['frame_index']} | FPS: {snap['fps']:.2f} | "
        f"inference FPS: {snap['inference_fps']:.2f} | elapsed: {elapsed:.2f}s"
    )
    print(
        f"  Current objects: {snap['stats'].current_objects} | "
        f"unique IDs: {snap['stats'].unique_ids} | classes: {dict(snap['stats'].class_counts)}"
    )
    if snap["status"] != "completed":
        raise RuntimeError(f"Expected completed status, got {snap['status']}: {snap['error']}")
    if snap["frame_index"] <= 0:
        raise RuntimeError("No frames were processed.")
    if expected_frames and snap["frame_index"] != expected_frames:
        raise RuntimeError(
            f"Expected {expected_frames} processed frames, got {snap['frame_index']}."
        )
    if not snap["resources_released"] or not snap["cleanup_complete"]:
        raise RuntimeError("Resources were not marked as released after processing.")

    id_rows = snap["stats"].id_rows()
    if not id_rows:
        raise RuntimeError(
            "No tracking IDs were assigned. Ensure objects are visible and the "
            "confidence threshold is low enough."
        )
    most_stable = max(id_rows, key=lambda row: row["Seen (frames)"])
    print(
        f"  Most stable ID: #{most_stable['ID']} ({most_stable['Class']}) seen in "
        f"{most_stable['Seen (frames)']} frames"
    )
    if most_stable["Seen (frames)"] < min(3, snap["frame_index"]):
        raise RuntimeError("Tracking IDs were not maintained across frames.")

    if not snap["output_ready"]:
        raise RuntimeError(f"Output video was not ready: {snap['output_error']}")
    output = Path(snap["output_path"])
    if not output.exists() or output.stat().st_size <= 1024:
        raise RuntimeError(f"Output video missing or too small: {output}")
    output_info = probe_video(output)
    print(
        f"  Output: {relative_path(output)} | {format_bytes(output.stat().st_size)} | "
        f"{output_info.summary()}"
    )
    if (output_info.width, output_info.height) != (info.width, info.height):
        raise RuntimeError(
            f"Output resolution {output_info.width}x{output_info.height} differs from "
            f"source {info.width}x{info.height}."
        )
    if info.has_valid_fps and abs(output_info.output_fps - info.output_fps) > 1.0:
        raise RuntimeError(
            f"Output FPS {output_info.output_fps:.2f} differs from source {info.output_fps:.2f}."
        )
    preview_frame = read_output_preview(output)
    if preview_frame is None or float(preview_frame.mean()) <= 0.0:
        raise RuntimeError("Output video frames could not be decoded / are blank.")
    preview_path = OUTPUT_VIDEOS_DIR / "verification_tracked_preview.jpg"
    cv2.imwrite(str(preview_path), preview_frame)
    print(f"  Preview frame: {relative_path(preview_path)}")

    return (
        f"{snap['frame_index']} frames, {snap['stats'].unique_ids} unique IDs, "
        f"stable ID #{most_stable['ID']} in {most_stable['Seen (frames)']} frames; "
        f"output {format_bytes(output.stat().st_size)}"
    )


def read_output_preview(path: Path) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 2:
            capture.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def check_botsort_configuration() -> str:
    if VIDEO_PATH is None:
        return "BoT-SORT config check skipped (no video provided)."
    reference = probe_video(VIDEO_PATH)
    capture = open_capture(VIDEO_PATH)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
        DETECTOR.reset_tracking()
        detections = 0
        frames_seen = 0
        ids: set[int] = set()
        for _ in range(12):
            ok, frame = capture.read()
            if not ok:
                break
            frames_seen += 1
            results = DETECTOR.track(
                frame,
                conf=0.20,
                iou=0.50,
                tracker="BoT-SORT",
                persist=True,
                imgsz=640,
            )
            detections += len(results)
            ids.update(d.track_id for d in results if d.track_id is not None)
        print(
            f"  BoT-SORT ran for {frames_seen} frames, "
            f"{detections} detections, {len(ids)} track IDs"
        )
    finally:
        capture.release()
        DETECTOR.reset_tracking()
    if detections == 0:
        print("  Note: the sampled segment had no detections; tracker config still initialized.")
    return (
        f"BoT-SORT tracker config initialized and ran ({detections} detections, "
        f"{len(ids)} IDs)"
    )


def check_error_cleanup() -> str:
    missing = OUTPUT_VIDEOS_DIR / "_verification_should_not_exist.mp4"
    if missing.exists():
        missing.unlink()
    config = ProcessingConfig(
        source_type="video",
        source=OUTPUT_VIDEOS_DIR / "definitely_missing_video.mp4",
        output_path=missing,
        conf=0.25,
        tracker="ByteTrack",
        max_frames=10,
    )
    processor = VideoProcessor(config, DETECTOR)
    processor.start()
    processor.join(timeout=20)
    if processor.is_alive():
        processor.stop()
        processor.join(timeout=10)
        raise RuntimeError("Processor stayed alive after an invalid video input.")
    snap = processor.snapshot()
    print(
        f"  Status: {snap['status']} | released={snap['resources_released']} | "
        f"cleanup={snap['cleanup_complete']} | error={snap['error']}"
    )
    if snap["status"] != "error":
        raise RuntimeError(f"Expected error status for a missing video, got {snap['status']}.")
    if not snap["resources_released"] or not snap["cleanup_complete"]:
        raise RuntimeError("Cleanup flags were not set after the error.")
    if missing.exists():
        raise RuntimeError("An output file was created despite the invalid input.")
    return "Invalid video failed gracefully; resources released; no output left behind"


def check_webcam() -> str:
    config = ProcessingConfig(
        source_type="webcam",
        source=CAMERA_INDEX,
        output_path=None,
        conf=0.25,
        tracker="ByteTrack",
        max_frames=20,
        model_name=DETECTOR.info["model_name"],
        device_label=DETECTOR.info["device_label"],
    )
    processor = VideoProcessor(config, DETECTOR)
    processor.start()
    processor.join(timeout=60)
    if processor.is_alive():
        processor.stop()
        processor.join(timeout=10)
        raise RuntimeError("Webcam processing did not stop.")
    snap = processor.snapshot()
    print(
        f"  Camera {CAMERA_INDEX}: status={snap['status']} | frames={snap['frame_index']} | "
        f"resolution={snap['resolution']} | released={snap['resources_released']}"
    )
    if snap["status"] != "completed":
        raise RuntimeError(f"Webcam session failed: {snap['error'] or snap['status_message']}")
    if snap["frame_index"] <= 0:
        raise RuntimeError("The webcam did not return any frames.")
    if not snap["resources_released"]:
        raise RuntimeError("The camera resource was not released.")
    return f"Webcam {CAMERA_INDEX} processed {snap['frame_index']} frames and released cleanly"


def _print_summary(runner: VerificationRunner) -> int:
    print("\n" + "=" * 78)
    print("VERIFICATION SUMMARY")
    print("=" * 78)
    required_failures = 0
    for result in runner.results:
        if result.skipped:
            marker = "SKIP"
        elif result.passed:
            marker = "PASS"
        else:
            marker = "FAIL"
            if result.required:
                required_failures += 1
        print(f"  [{marker}] {result.name}: {result.detail}")
    total = len(runner.results)
    passed = sum(1 for result in runner.results if result.passed and not result.skipped)
    skipped = sum(1 for result in runner.results if result.skipped)
    print("-" * 78)
    print(f"  {passed}/{total - skipped} checks passed, {skipped} skipped")
    print(f"  Model: {MODEL_PATH}")
    if VIDEO_PATH:
        print(f"  Video: {VIDEO_PATH}")
        print(f"  Start frame: {START_FRAME} | frames: {FRAMES}")
    print(f"  Artifacts: {relative_path(OUTPUT_VIDEOS_DIR)}")
    print("=" * 78)
    if required_failures:
        print(f"RESULT: FAILED ({required_failures} required check(s) failed)")
        return 1
    print("RESULT: PASSED (required checks)")
    return 0


DETECTOR: YOLODetector
MODEL_PATH: Path
VIDEO_PATH: Path | None
START_FRAME: int
FRAMES: int
CAMERA_INDEX: int | None


def main() -> int:
    global MODEL_PATH, VIDEO_PATH, START_FRAME, FRAMES, CAMERA_INDEX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/yolo11n.pt", help="Path to YOLO .pt weights")
    parser.add_argument("--video", default=None, help="Video for tracking/output checks")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="First frame for the tracking check (default: 420 for the included sample)",
    )
    parser.add_argument("--frames", type=int, default=60, help="Frames to process (default: 60)")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Optional webcam index to test (e.g. 0). Skipped if omitted.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    ensure_directories()

    try:
        MODEL_PATH = resolve_model_path(args.model)
    except AppError as exc:
        print(f"ERROR: {exc}")
        return 1
    VIDEO_PATH = Path(args.video).expanduser().resolve() if args.video else _find_default_video()
    if VIDEO_PATH is not None and not VIDEO_PATH.exists():
        print(f"ERROR: video not found: {VIDEO_PATH}")
        return 1
    is_sample = bool(VIDEO_PATH and VIDEO_PATH.resolve() == DEFAULT_SAMPLE.resolve())
    START_FRAME = args.start_frame if args.start_frame is not None else (420 if is_sample else 0)
    FRAMES = max(1, int(args.frames))
    CAMERA_INDEX = args.camera

    if VIDEO_PATH is not None:
        try:
            probe = probe_video(VIDEO_PATH)
            if probe.frame_count and START_FRAME >= probe.frame_count:
                print(
                    f"WARNING: start frame {START_FRAME} is past the end; "
                    "using frame 0 instead."
                )
                START_FRAME = 0
        except AppError as exc:
            print(f"ERROR: {exc}")
            return 1

    print("=" * 78)
    print("YOLO OBJECT DETECTION + TRACKING VERIFICATION")
    print("=" * 78)
    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH if VIDEO_PATH else 'none'}")
    print(f"Camera test: {'index ' + str(CAMERA_INDEX) if CAMERA_INDEX is not None else 'not requested'}")
    print(f"Tracking window: start_frame={START_FRAME}, max_frames={FRAMES}")

    runner = VerificationRunner()
    runner.run("Environment and dependencies", check_environment)
    model_result = runner.run("Model loading", check_model, required=True)
    if not model_result.passed:
        print("\nCannot continue without a loaded model.")
        return _print_summary(runner)

    if VIDEO_PATH is None:
        runner.skip(
            "Detection on a sample video frame",
            "No sample video found. Add an MP4/AVI/MOV file to input_videos/ or pass --video.",
        )
    else:
        runner.run("Detection on a sample video frame", check_detection)
    runner.run("Unique-ID statistics semantics", check_unique_id_statistics)
    if VIDEO_PATH is None:
        runner.skip(
            "Tracking IDs and annotated output video",
            "No sample video found. Add an MP4/AVI/MOV file to input_videos/ or pass --video.",
        )
        runner.skip("BoT-SORT tracker configuration", "No sample video available.")
    else:
        runner.run("Tracking IDs and annotated output video", check_tracking_and_output)
        runner.run("BoT-SORT tracker configuration", check_botsort_configuration)
    runner.run("Graceful error handling and resource cleanup", check_error_cleanup)

    if CAMERA_INDEX is None:
        runner.skip(
            "Webcam pipeline",
            "Not requested. Run locally with `python verification.py --camera 0` to test a camera.",
        )
    else:
        runner.run("Webcam pipeline", check_webcam)

    if VIDEO_PATH is None:
        print(
            "\nNo video was available for the video checks. Add your own MP4/AVI/MOV file "
            "to input_videos/ or pass --video path."
        )

    # Close the model before summary so any GPU memory is released.
    try:
        DETECTOR.close()
    except Exception:  # noqa: BLE001
        pass

    return _print_summary(runner)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nVerification interrupted.")
        raise SystemExit(130)
