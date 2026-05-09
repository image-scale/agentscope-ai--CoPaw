# -*- coding: utf-8 -*-
"""Local model management for CoPaw."""

from .download import (
    DownloadStatus,
    DownloadSnapshot,
    DownloadResult,
    ProgressTracker,
    start_download,
    apply_result,
)
from .model_manager import (
    DownloadSource,
    LocalModelInfo,
    ModelManager,
    BYTES_PER_GB,
)

__all__ = [
    "DownloadStatus",
    "DownloadSnapshot",
    "DownloadResult",
    "ProgressTracker",
    "start_download",
    "apply_result",
    "DownloadSource",
    "LocalModelInfo",
    "ModelManager",
    "BYTES_PER_GB",
]
