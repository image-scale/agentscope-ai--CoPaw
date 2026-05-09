# -*- coding: utf-8 -*-
"""Local model manager for downloading and managing LLM models.

Provides model discovery, download management, and local storage tracking
for models from HuggingFace and ModelScope repositories.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import threading
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from pydantic import Field

from ..providers.base import ModelInfo
from ..settings import SECRET_DIR
from .download import DownloadStatus, ProgressTracker, start_download

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3


class DownloadSource(str, Enum):
    """Source for model downloads."""

    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    AUTO = "auto"


class LocalModelInfo(ModelInfo):
    """Extended model info for locally stored models."""

    size_bytes: int = Field(
        default=0,
        description="Model size in bytes",
    )
    downloaded: bool = Field(
        default=False,
        description="Whether model is fully downloaded",
    )
    source: DownloadSource = Field(
        default=DownloadSource.AUTO,
        description="Preferred download source",
    )


class ModelManager:
    """Manager for downloading and tracking local LLM models.

    Provides methods to:
    - Get model recommendations based on system memory
    - Check download status of models
    - List and remove downloaded models
    - Track download progress
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        tmp_dir: Path | None = None,
    ) -> None:
        """Initialize the model manager.

        Args:
            model_dir: Directory for storing downloaded models.
            tmp_dir: Directory for temporary download files.
        """
        base_dir = SECRET_DIR / "local_models"
        self._model_dir = model_dir or base_dir / "models"
        self._tmp_dir = tmp_dir or base_dir / "tmp"
        self._lock = threading.RLock()
        self._progress = ProgressTracker()
        self._downloading_model: str | None = None

    @property
    def model_dir(self) -> Path:
        """Get the model storage directory."""
        return self._model_dir

    @property
    def tmp_dir(self) -> Path:
        """Get the temporary download directory."""
        return self._tmp_dir

    def get_model_path(self, model_id: str) -> Path:
        """Get the expected local path for a model.

        Args:
            model_id: Model repository ID (e.g., "org/model-name").

        Returns:
            Path where the model would be stored.
        """
        return self._model_dir.joinpath(*model_id.split("/"))

    def is_downloaded(self, model_id: str) -> bool:
        """Check if a model is downloaded and ready to use.

        Args:
            model_id: Model repository ID.

        Returns:
            True if model exists with GGUF files.
        """
        model_path = self.get_model_path(model_id)
        if not model_path.exists():
            return False
        return any(model_path.glob("*.gguf"))

    def get_recommended_models(self) -> list[LocalModelInfo]:
        """Get model recommendations based on system memory.

        Returns:
            List of recommended models for current hardware.
        """
        memory_gb = self._detect_available_memory()

        if memory_gb < 4:
            return []

        if memory_gb <= 8:
            models = [
                LocalModelInfo(
                    id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                    name="Qwen2.5-1.5B-Instruct",
                    size_bytes=int(1.6 * BYTES_PER_GB),
                    source=DownloadSource.HUGGINGFACE,
                ),
                LocalModelInfo(
                    id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                    name="Qwen2.5-3B-Instruct",
                    size_bytes=int(3.0 * BYTES_PER_GB),
                    source=DownloadSource.HUGGINGFACE,
                ),
            ]
        elif memory_gb <= 16:
            models = [
                LocalModelInfo(
                    id="Qwen/Qwen2.5-7B-Instruct-GGUF",
                    name="Qwen2.5-7B-Instruct",
                    size_bytes=int(5.5 * BYTES_PER_GB),
                    source=DownloadSource.HUGGINGFACE,
                ),
                LocalModelInfo(
                    id="microsoft/Phi-3-mini-4k-instruct-gguf",
                    name="Phi-3-mini-4k-instruct",
                    size_bytes=int(4.0 * BYTES_PER_GB),
                    source=DownloadSource.HUGGINGFACE,
                ),
            ]
        else:
            models = [
                LocalModelInfo(
                    id="Qwen/Qwen2.5-14B-Instruct-GGUF",
                    name="Qwen2.5-14B-Instruct",
                    size_bytes=int(10.0 * BYTES_PER_GB),
                    source=DownloadSource.HUGGINGFACE,
                ),
                LocalModelInfo(
                    id="mistralai/Mistral-7B-Instruct-v0.3-GGUF",
                    name="Mistral-7B-Instruct",
                    size_bytes=int(7.5 * BYTES_PER_GB),
                    source=DownloadSource.HUGGINGFACE,
                ),
            ]

        for model in models:
            model.downloaded = self.is_downloaded(model.id)

        return models

    def list_downloaded_models(self) -> list[LocalModelInfo]:
        """List all downloaded local models.

        Returns:
            List of downloaded model information.
        """
        if not self._model_dir.exists():
            return []

        models: list[LocalModelInfo] = []
        for model_dir in self._iter_model_directories():
            repo_id = self._path_to_repo_id(model_dir)
            size_bytes = self._calculate_directory_size(model_dir)
            models.append(
                LocalModelInfo(
                    id=repo_id,
                    name=repo_id,
                    size_bytes=size_bytes,
                    downloaded=True,
                )
            )

        return models

    def remove_model(self, model_id: str) -> bool:
        """Remove a downloaded model.

        Args:
            model_id: Model repository ID to remove.

        Returns:
            True if model was removed, False if not found.
        """
        model_path = self.get_model_path(model_id)
        if not model_path.exists():
            return False

        shutil.rmtree(model_path, ignore_errors=True)
        self._cleanup_empty_parents(model_path.parent)
        return True

    def get_download_progress(self) -> dict[str, Any]:
        """Get the current download progress.

        Returns:
            Dictionary with download status and progress information.
        """
        return self._progress.snapshot()

    def is_download_active(self) -> bool:
        """Check if a download is currently in progress.

        Returns:
            True if downloading.
        """
        with self._lock:
            status = self._progress.get_status()
            return status in {
                DownloadStatus.PENDING,
                DownloadStatus.DOWNLOADING,
                DownloadStatus.CANCELING,
            }

    def prepare_download(
        self,
        model_id: str,
        source: DownloadSource | None = None,
        total_bytes: int | None = None,
    ) -> Path:
        """Prepare to start a model download.

        Sets up progress tracking and directories. The actual download
        should be performed separately (e.g., using huggingface_hub).

        Args:
            model_id: Model repository ID.
            source: Download source preference.
            total_bytes: Expected total size.

        Returns:
            Path to the staging directory for download.

        Raises:
            RuntimeError: If a download is already in progress.
        """
        with self._lock:
            if self.is_download_active():
                raise RuntimeError("A download is already in progress")

            resolved_source = source or self._resolve_source()
            self._downloading_model = model_id

            staging_dir = self._tmp_dir / model_id.replace("/", "_")
            staging_dir.mkdir(parents=True, exist_ok=True)

            start_download(
                self._progress,
                total_bytes=total_bytes,
                model_name=model_id,
                source=resolved_source.value,
            )

            return staging_dir

    def complete_download(self, model_id: str, staging_dir: Path) -> Path:
        """Complete a download by moving staged files to final location.

        Args:
            model_id: Model repository ID.
            staging_dir: Directory containing downloaded files.

        Returns:
            Final model directory path.
        """
        final_dir = self.get_model_path(model_id)
        final_dir.parent.mkdir(parents=True, exist_ok=True)

        if final_dir.exists():
            shutil.rmtree(final_dir)

        shutil.move(str(staging_dir), str(final_dir))

        with self._lock:
            self._downloading_model = None
            downloaded_bytes = self._calculate_directory_size(final_dir)
            self._progress.mark_completed(
                local_path=str(final_dir),
                downloaded_bytes=downloaded_bytes,
            )

        return final_dir

    def fail_download(self, error: str) -> None:
        """Mark current download as failed.

        Args:
            error: Error message describing the failure.
        """
        with self._lock:
            self._downloading_model = None
            self._progress.mark_failed(error)

    def cancel_download(self) -> bool:
        """Request cancellation of the current download.

        Returns:
            True if cancellation was initiated.
        """
        with self._lock:
            if not self.is_download_active():
                return False

            self._progress.mark_canceling()
            return True

    def confirm_cancellation(self, staging_dir: Path | None = None) -> None:
        """Confirm download cancellation and clean up.

        Args:
            staging_dir: Staging directory to clean up.
        """
        with self._lock:
            self._downloading_model = None
            self._progress.mark_cancelled()

        if staging_dir and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    def update_progress(self, downloaded_bytes: int) -> None:
        """Update download progress.

        Args:
            downloaded_bytes: Number of bytes downloaded so far.
        """
        self._progress.update_progress(downloaded_bytes)

    def probe_huggingface(self) -> bool:
        """Check if HuggingFace Hub is reachable.

        Returns:
            True if HuggingFace is accessible.
        """
        try:
            response = httpx.get(
                "https://huggingface.co",
                follow_redirects=True,
                timeout=5,
            )
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    def probe_modelscope(self) -> bool:
        """Check if ModelScope is reachable.

        Returns:
            True if ModelScope is accessible.
        """
        try:
            response = httpx.get(
                "https://modelscope.cn",
                follow_redirects=True,
                timeout=5,
            )
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    def _resolve_source(self) -> DownloadSource:
        """Resolve AUTO source to specific provider."""
        if self.probe_huggingface():
            return DownloadSource.HUGGINGFACE
        return DownloadSource.MODELSCOPE

    def _detect_available_memory(self) -> float:
        """Detect available system memory in GB."""
        try:
            if platform.system() == "Linux":
                return self._get_linux_memory()
            elif platform.system() == "Darwin":
                return self._get_macos_memory()
            elif platform.system() == "Windows":
                return self._get_windows_memory()
        except Exception:
            pass
        return 8.0

    def _get_linux_memory(self) -> float:
        """Get memory on Linux via /proc/meminfo."""
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
        return 8.0

    def _get_macos_memory(self) -> float:
        """Get memory on macOS via sysctl."""
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / BYTES_PER_GB
        return 8.0

    def _get_windows_memory(self) -> float:
        """Get memory on Windows via ctypes."""
        import ctypes

        kernel32 = ctypes.windll.kernel32
        c_ulong = ctypes.c_ulong

        class MEMORYSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength", c_ulong),
                ("dwMemoryLoad", c_ulong),
                ("dwTotalPhys", c_ulong),
                ("dwAvailPhys", c_ulong),
                ("dwTotalPageFile", c_ulong),
                ("dwAvailPageFile", c_ulong),
                ("dwTotalVirtual", c_ulong),
                ("dwAvailVirtual", c_ulong),
            ]

        status = MEMORYSTATUS()
        status.dwLength = ctypes.sizeof(MEMORYSTATUS)
        kernel32.GlobalMemoryStatus(ctypes.byref(status))
        return status.dwTotalPhys / BYTES_PER_GB

    def _iter_model_directories(self) -> list[Path]:
        """Iterate over directories containing downloaded models."""
        if not self._model_dir.exists():
            return []

        candidates: list[Path] = []
        for entry in sorted(self._model_dir.rglob("*")):
            if not entry.is_dir():
                continue
            if self._is_temp_directory(entry):
                continue
            if not any(entry.glob("*.gguf")):
                continue
            if not self._is_model_root(entry):
                continue
            candidates.append(entry)

        selected: list[Path] = []
        for candidate in sorted(candidates, key=lambda p: len(p.parts)):
            if any(candidate.is_relative_to(parent) for parent in selected):
                continue
            selected.append(candidate)
        return selected

    def _is_temp_directory(self, path: Path) -> bool:
        """Check if path is a temporary download directory."""
        try:
            relative = path.relative_to(self._model_dir)
            return any(
                part.startswith(".") or part.endswith(".downloading")
                for part in relative.parts
            )
        except ValueError:
            return True

    def _is_model_root(self, path: Path) -> bool:
        """Check if path looks like a model root directory."""
        visible = [c for c in path.iterdir() if not c.name.startswith(".")]
        return any(not c.is_dir() for c in visible)

    def _path_to_repo_id(self, model_dir: Path) -> str:
        """Convert model directory path to repository ID."""
        relative = model_dir.relative_to(self._model_dir)
        return "/".join(relative.parts)

    @staticmethod
    def _calculate_directory_size(path: Path) -> int:
        """Calculate total size of files in a directory."""
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    def _cleanup_empty_parents(self, path: Path) -> None:
        """Remove empty parent directories up to model_dir."""
        while path != self._model_dir and path.exists():
            if any(path.iterdir()):
                return
            path.rmdir()
            path = path.parent
