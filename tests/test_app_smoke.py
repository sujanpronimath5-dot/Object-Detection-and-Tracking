"""Smoke test that the Streamlit dashboard renders with the bundled model."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utils import MODELS_DIR


def test_dashboard_renders_without_exceptions(project_root: Path) -> None:
    if not (MODELS_DIR / "yolo11n.pt").is_file():
        pytest.skip("models/yolo11n.pt is not available for the dashboard smoke test.")

    app = AppTest.from_file(str(project_root / "app.py"), default_timeout=120).run()

    assert not app.exception, [str(error.value) for error in app.exception]
    labels = [button.label for button in app.button]
    assert "▶ Start" in labels
    assert "■ Stop" in labels
    assert any("Real-Time Object Detection" in block.value for block in app.markdown)
