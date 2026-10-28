"""Shared pytest fixtures for the object-detection project."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector import YOLODetector  # noqa: E402
from utils import INPUT_VIDEOS_DIR, MODELS_DIR  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def model_path(project_root: Path) -> Path:
    path = MODELS_DIR / "yolo11n.pt"
    if not path.is_file():
        pytest.skip("models/yolo11n.pt is not available; run the unit tests or add weights.")
    return path


@pytest.fixture(scope="session")
def sample_video() -> Path:
    path = INPUT_VIDEOS_DIR / "sample_traffic.avi"
    if not path.is_file():
        pytest.skip("input_videos/sample_traffic.avi is not available.")
    return path


@pytest.fixture(scope="session")
def detector(model_path: Path):
    model = YOLODetector(model_path, device="cpu")
    model.load()
    yield model
    model.close()


def _write_synthetic_video(path: Path, frame_count: int = 90) -> Path:
    width, height, fps = 320, 240, 15.0
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot write an MP4 test video.")

    try:
        for index in range(frame_count):
            frame = np.full((height, width, 3), (24, 28, 38), dtype=np.uint8)
            # Static background detail so encoded frames are never completely blank.
            cv2.line(frame, (0, 190), (width, 205), (70, 80, 95), 2)
            cv2.circle(frame, (270, 55), 24, (90, 120, 160), -1)
            # Moving object: a bright rectangle travelling across the frame.
            left = 10 + (index * 2) % max(1, width - 70)
            cv2.rectangle(frame, (left, 90), (left + 55, 155), (230, 230, 230), -1)
            cv2.putText(
                frame,
                f"frame {index + 1}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 220, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    return _write_synthetic_video(tmp_path / "synthetic_scene.mp4")


@pytest.fixture
def long_synthetic_video(tmp_path: Path) -> Path:
    return _write_synthetic_video(tmp_path / "synthetic_long.mp4", frame_count=240)
