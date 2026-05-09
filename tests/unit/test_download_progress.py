# -*- coding: utf-8 -*-
"""Tests for copaw.local_models.download module - download progress tracking."""

import pytest

from copaw.local_models.download import (
    DownloadStatus,
    DownloadSnapshot,
    DownloadResult,
    ProgressTracker,
    start_download,
    apply_result,
)


class TestDownloadStatus:
    """Tests for DownloadStatus enum."""

    def test_status_values(self):
        """Status enum has expected values."""
        assert DownloadStatus.IDLE.value == "idle"
        assert DownloadStatus.PENDING.value == "pending"
        assert DownloadStatus.DOWNLOADING.value == "downloading"
        assert DownloadStatus.COMPLETED.value == "completed"
        assert DownloadStatus.FAILED.value == "failed"
        assert DownloadStatus.CANCELLED.value == "cancelled"

    def test_status_is_string_enum(self):
        """Status can be used as string."""
        assert DownloadStatus.DOWNLOADING.value == "downloading"
        assert DownloadStatus.DOWNLOADING == "downloading"


class TestDownloadResult:
    """Tests for DownloadResult dataclass."""

    def test_create_completed_result(self):
        """Create a completed download result."""
        result = DownloadResult(
            status=DownloadStatus.COMPLETED,
            local_path="/tmp/model",
        )
        assert result.status == DownloadStatus.COMPLETED
        assert result.local_path == "/tmp/model"
        assert result.error is None

    def test_create_failed_result(self):
        """Create a failed download result."""
        result = DownloadResult(
            status=DownloadStatus.FAILED,
            error="Network error",
        )
        assert result.status == DownloadStatus.FAILED
        assert result.error == "Network error"
        assert result.local_path is None

    def test_to_dict(self):
        """Result can be serialized to dict."""
        result = DownloadResult(
            status=DownloadStatus.COMPLETED,
            local_path="/tmp/model",
        )
        data = result.to_dict()
        assert data["status"] == "completed"
        assert data["local_path"] == "/tmp/model"

    def test_from_dict(self):
        """Result can be deserialized from dict."""
        data = {
            "status": "completed",
            "local_path": "/tmp/model",
        }
        result = DownloadResult.from_dict(data)
        assert result.status == DownloadStatus.COMPLETED
        assert result.local_path == "/tmp/model"

    def test_round_trip_serialization(self):
        """Result survives round-trip serialization."""
        original = DownloadResult(
            status=DownloadStatus.FAILED,
            error="Test error",
        )
        restored = DownloadResult.from_dict(original.to_dict())
        assert restored == original


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_initial_state_is_idle(self):
        """Tracker starts in idle state."""
        tracker = ProgressTracker()
        assert tracker.get_status() == DownloadStatus.IDLE

    def test_reset_clears_progress(self):
        """reset() clears all progress data."""
        tracker = ProgressTracker()
        tracker.update_progress(1000, total_bytes=5000)

        tracker.reset()

        progress = tracker.get_progress()
        assert progress.downloaded_bytes == 0
        assert progress.total_bytes is None

    def test_reset_with_parameters(self):
        """reset() accepts initial values."""
        tracker = ProgressTracker()
        tracker.reset(
            status=DownloadStatus.PENDING,
            total_bytes=10000,
            model_name="test-model",
            source="huggingface",
        )

        progress = tracker.get_progress()
        assert progress.status == DownloadStatus.PENDING
        assert progress.total_bytes == 10000
        assert progress.model_name == "test-model"
        assert progress.source == "huggingface"

    def test_update_status(self):
        """update_status() changes status."""
        tracker = ProgressTracker()
        tracker.update_status(DownloadStatus.DOWNLOADING)

        assert tracker.get_status() == DownloadStatus.DOWNLOADING

    def test_update_status_with_error(self):
        """update_status() can set error message."""
        tracker = ProgressTracker()
        tracker.update_status(DownloadStatus.FAILED, error="Connection lost")

        progress = tracker.get_progress()
        assert progress.status == DownloadStatus.FAILED
        assert progress.error == "Connection lost"

    def test_update_progress(self):
        """update_progress() updates byte count."""
        tracker = ProgressTracker()
        tracker.update_status(DownloadStatus.DOWNLOADING)

        tracker.update_progress(500, total_bytes=1000)

        progress = tracker.get_progress()
        assert progress.downloaded_bytes == 500
        assert progress.total_bytes == 1000

    def test_update_progress_calculates_speed(self):
        """update_progress() calculates transfer speed."""
        tracker = ProgressTracker()
        tracker.update_status(DownloadStatus.DOWNLOADING)

        tracker.update_progress(0)
        tracker.update_progress(1000)

        progress = tracker.get_progress()
        assert progress.speed_bytes_per_sec > 0

    def test_mark_cancelled(self):
        """mark_cancelled() sets cancelled status."""
        tracker = ProgressTracker()
        tracker.update_status(DownloadStatus.DOWNLOADING)

        tracker.mark_cancelled()

        assert tracker.get_status() == DownloadStatus.CANCELLED

    def test_mark_canceling(self):
        """mark_canceling() sets canceling status."""
        tracker = ProgressTracker()
        tracker.update_status(DownloadStatus.DOWNLOADING)

        tracker.mark_canceling()

        assert tracker.get_status() == DownloadStatus.CANCELING

    def test_mark_failed(self):
        """mark_failed() sets failed status with error."""
        tracker = ProgressTracker()
        tracker.update_status(DownloadStatus.DOWNLOADING)

        tracker.mark_failed("Download timeout")

        progress = tracker.get_progress()
        assert progress.status == DownloadStatus.FAILED
        assert progress.error == "Download timeout"

    def test_mark_completed(self):
        """mark_completed() sets completed status."""
        tracker = ProgressTracker()
        tracker.update_status(DownloadStatus.DOWNLOADING)

        tracker.mark_completed(local_path="/tmp/model.bin")

        progress = tracker.get_progress()
        assert progress.status == DownloadStatus.COMPLETED
        assert progress.local_path == "/tmp/model.bin"

    def test_mark_completed_with_bytes(self):
        """mark_completed() can finalize byte count."""
        tracker = ProgressTracker()
        tracker.update_status(DownloadStatus.DOWNLOADING)
        tracker.update_progress(500, total_bytes=1000)

        tracker.mark_completed(local_path="/tmp/model.bin", downloaded_bytes=1000)

        progress = tracker.get_progress()
        assert progress.downloaded_bytes == 1000

    def test_snapshot(self):
        """snapshot() returns dict with all progress fields."""
        tracker = ProgressTracker()
        tracker.reset(
            status=DownloadStatus.DOWNLOADING,
            model_name="test-model",
            total_bytes=5000,
        )
        tracker.update_progress(2500)

        snapshot = tracker.snapshot()

        assert snapshot["status"] == "downloading"
        assert snapshot["model_name"] == "test-model"
        assert snapshot["downloaded_bytes"] == 2500
        assert snapshot["total_bytes"] == 5000

    def test_terminal_status_resets_speed(self):
        """Terminal statuses reset speed to zero."""
        tracker = ProgressTracker()
        tracker.update_status(DownloadStatus.DOWNLOADING)
        tracker.update_progress(1000)

        tracker.update_status(DownloadStatus.COMPLETED)

        progress = tracker.get_progress()
        assert progress.speed_bytes_per_sec == 0.0


class TestStartDownload:
    """Tests for start_download helper function."""

    def test_start_download_initializes_tracker(self):
        """start_download() sets up tracker for new download."""
        tracker = ProgressTracker()

        start_download(
            tracker,
            total_bytes=10000,
            model_name="llama-3",
            source="huggingface",
        )

        progress = tracker.get_progress()
        assert progress.status == DownloadStatus.DOWNLOADING
        assert progress.model_name == "llama-3"
        assert progress.total_bytes == 10000
        assert progress.source == "huggingface"


class TestApplyResult:
    """Tests for apply_result helper function."""

    def test_apply_completed_result(self):
        """apply_result() marks tracker as completed."""
        tracker = ProgressTracker()
        start_download(tracker, total_bytes=100)

        result = DownloadResult(
            status=DownloadStatus.COMPLETED,
            local_path="/tmp/model",
        )
        apply_result(tracker, result, downloaded_bytes=100)

        progress = tracker.get_progress()
        assert progress.status == DownloadStatus.COMPLETED
        assert progress.local_path == "/tmp/model"
        assert progress.downloaded_bytes == 100

    def test_apply_failed_result(self):
        """apply_result() marks tracker as failed."""
        tracker = ProgressTracker()
        start_download(tracker, total_bytes=100)

        result = DownloadResult(
            status=DownloadStatus.FAILED,
            error="Network error",
        )
        apply_result(tracker, result)

        progress = tracker.get_progress()
        assert progress.status == DownloadStatus.FAILED
        assert progress.error == "Network error"

    def test_apply_cancelled_result(self):
        """apply_result() marks tracker as cancelled."""
        tracker = ProgressTracker()
        start_download(tracker, total_bytes=100)

        result = DownloadResult(status=DownloadStatus.CANCELLED)
        apply_result(tracker, result)

        assert tracker.get_status() == DownloadStatus.CANCELLED

    def test_apply_completed_without_path_raises(self):
        """apply_result() raises if completed without local_path."""
        tracker = ProgressTracker()
        start_download(tracker, total_bytes=100)

        result = DownloadResult(status=DownloadStatus.COMPLETED)

        with pytest.raises(RuntimeError, match="local_path"):
            apply_result(tracker, result)
