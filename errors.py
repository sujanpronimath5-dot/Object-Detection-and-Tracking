"""Application-specific exceptions with user-facing messages.

Keeping these separate from the processing code makes it easy for the
Streamlit interface to show a useful error instead of a raw traceback.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for errors that are safe and useful to show to the user."""


class ConfigurationError(AppError):
    """A setting or combination of settings is invalid."""


class ModelLoadError(AppError):
    """The YOLO weights could not be found or loaded."""


class VideoSourceError(AppError):
    """A webcam or video file could not be opened/decoded."""


class OutputVideoError(AppError):
    """An annotated output video could not be created or verified."""


class TrackingError(AppError):
    """YOLO detection/tracking failed for a frame."""
