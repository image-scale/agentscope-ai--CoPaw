# -*- coding: utf-8 -*-
"""Download progress tracking for local model downloads.

This module provides thread-safe progress tracking for background download tasks,
with support for status transitions, speed calculation, and cancellation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any


class DownloadStatus(str, Enum):
    """Download task lifecycle states."""

    IDLE = "idle"
    PENDING = "pending"
    DOWNLOADING = "downloading"
    CANCELING = "canceling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class DownloadSnapshot:
    """Immutable snapshot of download progress state.

    Attributes:
        status: Current download lifecycle status.
        model_name: Name of the model being downloaded.
        downloaded_bytes: Number of bytes downloaded so far.
        total_bytes: Total size of the download (if known).
        speed_bytes_per_sec: Current download speed.
        source: Download source (e.g., "huggingface", "modelscope").
        error: Error message if download failed.
        local_path: Path where the file was saved.
    """

    status: DownloadStatus = DownloadStatus.IDLE
    model_name: str | None = None
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    speed_bytes_per_sec: float = 0.0
    source: str | None = None
    error: str | None = None
    local_path: str | None = None


@dataclass(frozen=True)
class DownloadResult:
    """Terminal result of a download task.

    Attributes:
        status: Final status (COMPLETED, FAILED, or CANCELLED).
        local_path: Path to downloaded file (if successful).
        error: Error message (if failed).
    """

    status: DownloadStatus
    local_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Serialize to dictionary for cross-thread communication."""
        return {
            "status": self.status.value,
            "local_path": self.local_path,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DownloadResult:
        """Create from serialized dictionary."""
        return cls(
            status=DownloadStatus(data["status"]),
            local_path=data.get("local_path"),
            error=data.get("error"),
        )


class ProgressTracker:
    """Thread-safe download progress tracker.

    Provides methods to update download progress, track transfer speed,
    and transition between lifecycle states.
    """

    def __init__(self) -> None:
        """Initialize an idle progress tracker."""
        self._lock = threading.Lock()
        self._progress = DownloadSnapshot()
        self._last_bytes = 0
        self._last_time = time.monotonic()

    def reset(
        self,
        *,
        status: DownloadStatus = DownloadStatus.IDLE,
        total_bytes: int | None = None,
        model_name: str | None = None,
        source: str | None = None,
        error: str | None = None,
        local_path: str | None = None,
    ) -> DownloadSnapshot:
        """Reset progress to initial state.

        Args:
            status: Initial status.
            total_bytes: Expected total size.
            model_name: Name of the model.
            source: Download source.
            error: Initial error (usually None).
            local_path: Initial path (usually None).

        Returns:
            The new progress snapshot.
        """
        with self._lock:
            self._progress = DownloadSnapshot(
                status=status,
                model_name=model_name,
                downloaded_bytes=0,
                total_bytes=total_bytes,
                speed_bytes_per_sec=0.0,
                source=source,
                error=error,
                local_path=local_path,
            )
            self._last_bytes = 0
            self._last_time = time.monotonic()
            return self._progress

    def update_status(
        self,
        status: DownloadStatus,
        *,
        error: str | None = None,
        local_path: str | None = None,
        model_name: str | None = None,
        source: str | None = None,
        total_bytes: int | None = None,
    ) -> DownloadSnapshot:
        """Update lifecycle status and optional metadata.

        Args:
            status: New status.
            error: Error message (for failed status).
            local_path: Local path (for completed status).
            model_name: Model name to update.
            source: Source to update.
            total_bytes: Total bytes to update.

        Returns:
            The updated progress snapshot.
        """
        with self._lock:
            current = self._progress

            new_total = total_bytes if total_bytes is not None else current.total_bytes
            new_model = model_name if model_name is not None else current.model_name
            new_source = source if source is not None else current.source
            new_error = error if error is not None else current.error
            new_path = local_path if local_path is not None else current.local_path

            new_speed = current.speed_bytes_per_sec
            if status in {
                DownloadStatus.CANCELING,
                DownloadStatus.CANCELLED,
                DownloadStatus.COMPLETED,
                DownloadStatus.FAILED,
            }:
                new_speed = 0.0

            self._progress = replace(
                current,
                status=status,
                model_name=new_model,
                total_bytes=new_total,
                speed_bytes_per_sec=new_speed,
                source=new_source,
                error=new_error,
                local_path=new_path,
            )
            return self._progress

    def update_progress(
        self,
        downloaded_bytes: int,
        *,
        total_bytes: int | None = None,
        model_name: str | None = None,
        source: str | None = None,
    ) -> DownloadSnapshot:
        """Update downloaded bytes and calculate transfer speed.

        Args:
            downloaded_bytes: Bytes downloaded so far.
            total_bytes: Total size (if known).
            model_name: Model name.
            source: Download source.

        Returns:
            The updated progress snapshot.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = max(now - self._last_time, 1e-6)
            speed = max(0.0, (downloaded_bytes - self._last_bytes) / elapsed)

            current = self._progress
            new_total = total_bytes if total_bytes is not None else current.total_bytes
            new_model = model_name if model_name is not None else current.model_name
            new_source = source if source is not None else current.source

            self._progress = replace(
                current,
                model_name=new_model,
                downloaded_bytes=downloaded_bytes,
                total_bytes=new_total,
                speed_bytes_per_sec=speed,
                source=new_source,
            )
            self._last_bytes = downloaded_bytes
            self._last_time = now
            return self._progress

    def mark_cancelled(self) -> DownloadSnapshot:
        """Mark download as cancelled."""
        return self.update_status(DownloadStatus.CANCELLED)

    def mark_canceling(self) -> DownloadSnapshot:
        """Mark download as being cancelled."""
        return self.update_status(DownloadStatus.CANCELING)

    def mark_failed(
        self,
        error: str,
        *,
        status: DownloadStatus = DownloadStatus.FAILED,
    ) -> DownloadSnapshot:
        """Mark download as failed.

        Args:
            error: Error message describing the failure.
            status: Status to set (defaults to FAILED).

        Returns:
            The updated progress snapshot.
        """
        return self.update_status(status, error=error)

    def mark_completed(
        self,
        *,
        local_path: str,
        downloaded_bytes: int | None = None,
    ) -> DownloadSnapshot:
        """Mark download as completed.

        Args:
            local_path: Path where the file was saved.
            downloaded_bytes: Final byte count (if known).

        Returns:
            The updated progress snapshot.
        """
        with self._lock:
            progress = self._progress

        if downloaded_bytes is not None:
            progress = self.update_progress(downloaded_bytes)

        return self.update_status(
            DownloadStatus.COMPLETED,
            local_path=local_path,
            total_bytes=progress.total_bytes,
        )

    def get_status(self) -> DownloadStatus:
        """Get the current download status."""
        with self._lock:
            return self._progress.status

    def get_progress(self) -> DownloadSnapshot:
        """Get the current progress snapshot."""
        with self._lock:
            return self._progress

    def snapshot(self) -> dict[str, Any]:
        """Get a dictionary snapshot of current progress.

        Returns:
            Dictionary with all progress fields and status as string.
        """
        with self._lock:
            data = asdict(self._progress)
        data["status"] = self.get_status().value
        return data


def start_download(
    tracker: ProgressTracker,
    *,
    total_bytes: int | None = None,
    model_name: str | None = None,
    source: str | None = None,
) -> None:
    """Initialize tracker for a new download.

    Args:
        tracker: Progress tracker to initialize.
        total_bytes: Expected total size.
        model_name: Name of the model.
        source: Download source.
    """
    tracker.reset(
        status=DownloadStatus.PENDING,
        total_bytes=total_bytes,
        model_name=model_name,
        source=source,
    )
    tracker.update_status(DownloadStatus.DOWNLOADING)


def apply_result(
    tracker: ProgressTracker,
    result: DownloadResult,
    *,
    downloaded_bytes: int | None = None,
) -> DownloadResult:
    """Apply a terminal result to a progress tracker.

    Args:
        tracker: Progress tracker to update.
        result: Terminal download result.
        downloaded_bytes: Final byte count (for completed downloads).

    Returns:
        The same result passed in.
    """
    if result.status == DownloadStatus.COMPLETED:
        if result.local_path is None:
            raise RuntimeError("Completed result must include local_path")
        tracker.mark_completed(
            local_path=result.local_path,
            downloaded_bytes=downloaded_bytes,
        )
        return result

    if result.status == DownloadStatus.CANCELLED:
        tracker.mark_cancelled()
        return result

    tracker.mark_failed(
        result.error or "Download failed",
        status=result.status,
    )
    return result
