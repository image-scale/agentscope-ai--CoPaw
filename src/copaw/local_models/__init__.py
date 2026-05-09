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
from .llamacpp import (
    LlamaCppBackend,
    DownloadCancelled,
    LLAMA_CPP_RELEASE_URL,
    LLAMA_CPP_DEFAULT_TAG,
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
    "LlamaCppBackend",
    "DownloadCancelled",
    "LLAMA_CPP_RELEASE_URL",
    "LLAMA_CPP_DEFAULT_TAG",
]
