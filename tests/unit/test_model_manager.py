# -*- coding: utf-8 -*-
"""Tests for copaw.local_models.model_manager module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from copaw.local_models.model_manager import (
    DownloadSource,
    LocalModelInfo,
    ModelManager,
    BYTES_PER_GB,
)
from copaw.local_models.download import DownloadStatus


class TestDownloadSource:
    """Tests for DownloadSource enum."""

    def test_source_values(self):
        """DownloadSource has expected values."""
        assert DownloadSource.HUGGINGFACE.value == "huggingface"
        assert DownloadSource.MODELSCOPE.value == "modelscope"
        assert DownloadSource.AUTO.value == "auto"

    def test_source_comparison(self):
        """DownloadSource can be compared with strings."""
        assert DownloadSource.HUGGINGFACE == "huggingface"
        assert DownloadSource.MODELSCOPE == "modelscope"


class TestLocalModelInfo:
    """Tests for LocalModelInfo dataclass."""

    def test_create_basic_info(self):
        """LocalModelInfo stores basic model data."""
        info = LocalModelInfo(
            id="org/model",
            name="Model Name",
        )
        assert info.id == "org/model"
        assert info.name == "Model Name"
        assert info.size_bytes == 0
        assert info.downloaded is False

    def test_create_with_all_fields(self):
        """LocalModelInfo stores all fields correctly."""
        info = LocalModelInfo(
            id="org/model",
            name="Model Name",
            size_bytes=1024,
            downloaded=True,
            source=DownloadSource.HUGGINGFACE,
        )
        assert info.size_bytes == 1024
        assert info.downloaded is True
        assert info.source == DownloadSource.HUGGINGFACE

    def test_inherits_from_model_info(self):
        """LocalModelInfo extends ModelInfo."""
        info = LocalModelInfo(
            id="org/model",
            name="Model Name",
            supports_multimodal=True,
        )
        assert info.supports_multimodal is True


class TestModelManagerInit:
    """Tests for ModelManager initialization."""

    def test_default_directories(self, tmp_path):
        """ModelManager uses default directories."""
        with patch("copaw.local_models.model_manager.SECRET_DIR", tmp_path):
            manager = ModelManager()

        assert manager.model_dir == tmp_path / "local_models" / "models"
        assert manager.tmp_dir == tmp_path / "local_models" / "tmp"

    def test_custom_directories(self, tmp_path):
        """ModelManager accepts custom directories."""
        model_dir = tmp_path / "custom_models"
        tmp_dir = tmp_path / "custom_tmp"

        manager = ModelManager(model_dir=model_dir, tmp_dir=tmp_dir)

        assert manager.model_dir == model_dir
        assert manager.tmp_dir == tmp_dir


class TestModelManagerPaths:
    """Tests for ModelManager path handling."""

    def test_get_model_path_simple(self, tmp_path):
        """get_model_path returns correct path for simple ID."""
        manager = ModelManager(model_dir=tmp_path / "models")

        path = manager.get_model_path("org/model")

        assert path == tmp_path / "models" / "org" / "model"

    def test_get_model_path_nested(self, tmp_path):
        """get_model_path handles nested IDs."""
        manager = ModelManager(model_dir=tmp_path / "models")

        path = manager.get_model_path("org/sub/model")

        assert path == tmp_path / "models" / "org" / "sub" / "model"


class TestModelManagerIsDownloaded:
    """Tests for ModelManager.is_downloaded method."""

    def test_returns_false_for_missing_directory(self, tmp_path):
        """is_downloaded returns False when directory missing."""
        manager = ModelManager(model_dir=tmp_path / "models")

        result = manager.is_downloaded("org/model")

        assert result is False

    def test_returns_false_for_empty_directory(self, tmp_path):
        """is_downloaded returns False for empty directory."""
        manager = ModelManager(model_dir=tmp_path / "models")
        model_dir = tmp_path / "models" / "org" / "model"
        model_dir.mkdir(parents=True)

        result = manager.is_downloaded("org/model")

        assert result is False

    def test_returns_false_without_gguf(self, tmp_path):
        """is_downloaded returns False without GGUF files."""
        manager = ModelManager(model_dir=tmp_path / "models")
        model_dir = tmp_path / "models" / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        result = manager.is_downloaded("org/model")

        assert result is False

    def test_returns_true_with_gguf(self, tmp_path):
        """is_downloaded returns True with GGUF files."""
        manager = ModelManager(model_dir=tmp_path / "models")
        model_dir = tmp_path / "models" / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").write_bytes(b"GGUF")

        result = manager.is_downloaded("org/model")

        assert result is True


class TestModelManagerRecommendations:
    """Tests for ModelManager.get_recommended_models method."""

    def test_returns_empty_for_low_memory(self, tmp_path):
        """get_recommended_models returns empty for <4GB."""
        manager = ModelManager(model_dir=tmp_path / "models")
        manager._detect_available_memory = lambda: 2.0

        models = manager.get_recommended_models()

        assert models == []

    def test_returns_small_models_for_8gb(self, tmp_path):
        """get_recommended_models returns small models for <=8GB."""
        manager = ModelManager(model_dir=tmp_path / "models")
        manager._detect_available_memory = lambda: 8.0

        models = manager.get_recommended_models()

        assert len(models) == 2
        assert all(m.size_bytes < 4 * BYTES_PER_GB for m in models)

    def test_returns_medium_models_for_16gb(self, tmp_path):
        """get_recommended_models returns medium models for <=16GB."""
        manager = ModelManager(model_dir=tmp_path / "models")
        manager._detect_available_memory = lambda: 16.0

        models = manager.get_recommended_models()

        assert len(models) == 2

    def test_returns_large_models_for_high_memory(self, tmp_path):
        """get_recommended_models returns large models for >16GB."""
        manager = ModelManager(model_dir=tmp_path / "models")
        manager._detect_available_memory = lambda: 32.0

        models = manager.get_recommended_models()

        assert len(models) == 2

    def test_marks_downloaded_models(self, tmp_path):
        """get_recommended_models marks downloaded models."""
        manager = ModelManager(model_dir=tmp_path / "models")
        manager._detect_available_memory = lambda: 8.0

        models = manager.get_recommended_models()
        first_model = models[0]
        model_dir = manager.get_model_path(first_model.id)
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").write_bytes(b"GGUF")

        models = manager.get_recommended_models()

        downloaded = [m for m in models if m.downloaded]
        assert len(downloaded) == 1


class TestModelManagerListModels:
    """Tests for ModelManager.list_downloaded_models method."""

    def test_returns_empty_when_no_models(self, tmp_path):
        """list_downloaded_models returns empty list when none exist."""
        manager = ModelManager(model_dir=tmp_path / "models")

        models = manager.list_downloaded_models()

        assert models == []

    def test_returns_empty_when_dir_missing(self, tmp_path):
        """list_downloaded_models returns empty when model_dir missing."""
        manager = ModelManager(model_dir=tmp_path / "nonexistent")

        models = manager.list_downloaded_models()

        assert models == []

    def test_finds_downloaded_model(self, tmp_path):
        """list_downloaded_models finds models with GGUF files."""
        manager = ModelManager(model_dir=tmp_path / "models")
        model_dir = tmp_path / "models" / "org" / "model"
        model_dir.mkdir(parents=True)
        gguf = model_dir / "model.gguf"
        gguf.write_bytes(b"GGUF" * 256)

        models = manager.list_downloaded_models()

        assert len(models) == 1
        assert models[0].id == "org/model"
        assert models[0].downloaded is True
        assert models[0].size_bytes == 1024

    def test_ignores_temp_directories(self, tmp_path):
        """list_downloaded_models ignores temporary directories."""
        manager = ModelManager(model_dir=tmp_path / "models")
        temp_dir = tmp_path / "models" / ".downloading"
        temp_dir.mkdir(parents=True)
        (temp_dir / "model.gguf").write_bytes(b"GGUF")

        models = manager.list_downloaded_models()

        assert models == []

    def test_finds_multiple_models(self, tmp_path):
        """list_downloaded_models finds all downloaded models."""
        manager = ModelManager(model_dir=tmp_path / "models")

        for org, name in [("org1", "model1"), ("org2", "model2")]:
            model_dir = tmp_path / "models" / org / name
            model_dir.mkdir(parents=True)
            (model_dir / "model.gguf").write_bytes(b"GGUF")

        models = manager.list_downloaded_models()

        assert len(models) == 2
        ids = {m.id for m in models}
        assert ids == {"org1/model1", "org2/model2"}


class TestModelManagerRemove:
    """Tests for ModelManager.remove_model method."""

    def test_removes_existing_model(self, tmp_path):
        """remove_model deletes model directory."""
        manager = ModelManager(model_dir=tmp_path / "models")
        model_dir = tmp_path / "models" / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").write_bytes(b"GGUF")

        result = manager.remove_model("org/model")

        assert result is True
        assert not model_dir.exists()

    def test_returns_false_for_missing(self, tmp_path):
        """remove_model returns False for missing model."""
        manager = ModelManager(model_dir=tmp_path / "models")

        result = manager.remove_model("org/nonexistent")

        assert result is False

    def test_cleans_empty_parents(self, tmp_path):
        """remove_model cleans up empty parent directories."""
        manager = ModelManager(model_dir=tmp_path / "models")
        model_dir = tmp_path / "models" / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").write_bytes(b"GGUF")

        manager.remove_model("org/model")

        assert not (tmp_path / "models" / "org").exists()


class TestModelManagerDownloadProgress:
    """Tests for ModelManager download progress tracking."""

    def test_get_download_progress_initial(self, tmp_path):
        """get_download_progress returns idle state initially."""
        manager = ModelManager(model_dir=tmp_path / "models")

        progress = manager.get_download_progress()

        assert progress["status"] == "idle"

    def test_is_download_active_false_initially(self, tmp_path):
        """is_download_active returns False initially."""
        manager = ModelManager(model_dir=tmp_path / "models")

        assert manager.is_download_active() is False

    def test_prepare_download_sets_progress(self, tmp_path):
        """prepare_download sets up progress tracking."""
        manager = ModelManager(
            model_dir=tmp_path / "models",
            tmp_dir=tmp_path / "tmp",
        )

        staging = manager.prepare_download(
            "org/model",
            source=DownloadSource.HUGGINGFACE,
            total_bytes=1000,
        )

        assert staging.exists()
        progress = manager.get_download_progress()
        assert progress["status"] == "downloading"
        assert progress["model_name"] == "org/model"
        assert progress["total_bytes"] == 1000

    def test_prepare_download_raises_if_active(self, tmp_path):
        """prepare_download raises if download in progress."""
        manager = ModelManager(
            model_dir=tmp_path / "models",
            tmp_dir=tmp_path / "tmp",
        )
        manager.prepare_download("org/model1")

        with pytest.raises(RuntimeError, match="already in progress"):
            manager.prepare_download("org/model2")

    def test_update_progress_tracks_bytes(self, tmp_path):
        """update_progress updates downloaded bytes."""
        manager = ModelManager(
            model_dir=tmp_path / "models",
            tmp_dir=tmp_path / "tmp",
        )
        manager.prepare_download("org/model", total_bytes=1000)

        manager.update_progress(500)

        progress = manager.get_download_progress()
        assert progress["downloaded_bytes"] == 500


class TestModelManagerDownloadCompletion:
    """Tests for ModelManager download completion."""

    def test_complete_download_moves_files(self, tmp_path):
        """complete_download moves staged files to final location."""
        manager = ModelManager(
            model_dir=tmp_path / "models",
            tmp_dir=tmp_path / "tmp",
        )
        staging = manager.prepare_download("org/model")
        (staging / "model.gguf").write_bytes(b"GGUF" * 100)

        final_path = manager.complete_download("org/model", staging)

        assert final_path == tmp_path / "models" / "org" / "model"
        assert (final_path / "model.gguf").exists()
        assert not staging.exists()

    def test_complete_download_updates_progress(self, tmp_path):
        """complete_download marks progress as completed."""
        manager = ModelManager(
            model_dir=tmp_path / "models",
            tmp_dir=tmp_path / "tmp",
        )
        staging = manager.prepare_download("org/model")
        (staging / "model.gguf").write_bytes(b"GGUF")

        manager.complete_download("org/model", staging)

        progress = manager.get_download_progress()
        assert progress["status"] == "completed"

    def test_fail_download_sets_error(self, tmp_path):
        """fail_download marks progress as failed."""
        manager = ModelManager(
            model_dir=tmp_path / "models",
            tmp_dir=tmp_path / "tmp",
        )
        manager.prepare_download("org/model")

        manager.fail_download("Network error")

        progress = manager.get_download_progress()
        assert progress["status"] == "failed"
        assert progress["error"] == "Network error"


class TestModelManagerCancellation:
    """Tests for ModelManager download cancellation."""

    def test_cancel_returns_false_when_idle(self, tmp_path):
        """cancel_download returns False when not downloading."""
        manager = ModelManager(model_dir=tmp_path / "models")

        result = manager.cancel_download()

        assert result is False

    def test_cancel_sets_canceling_status(self, tmp_path):
        """cancel_download sets canceling status."""
        manager = ModelManager(
            model_dir=tmp_path / "models",
            tmp_dir=tmp_path / "tmp",
        )
        manager.prepare_download("org/model")

        result = manager.cancel_download()

        assert result is True
        assert manager._progress.get_status() == DownloadStatus.CANCELING

    def test_confirm_cancellation_cleans_staging(self, tmp_path):
        """confirm_cancellation cleans up staging directory."""
        manager = ModelManager(
            model_dir=tmp_path / "models",
            tmp_dir=tmp_path / "tmp",
        )
        staging = manager.prepare_download("org/model")
        (staging / "partial.gguf").write_bytes(b"partial")
        manager.cancel_download()

        manager.confirm_cancellation(staging)

        assert not staging.exists()
        progress = manager.get_download_progress()
        assert progress["status"] == "cancelled"


class TestModelManagerSourceProbing:
    """Tests for ModelManager source probing."""

    def test_probe_huggingface_success(self, tmp_path):
        """probe_huggingface returns True on success."""
        manager = ModelManager(model_dir=tmp_path / "models")

        with patch("copaw.local_models.model_manager.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            result = manager.probe_huggingface()

        assert result is True

    def test_probe_huggingface_failure(self, tmp_path):
        """probe_huggingface returns False on error."""
        manager = ModelManager(model_dir=tmp_path / "models")

        with patch("copaw.local_models.model_manager.httpx.get") as mock_get:
            import httpx
            mock_get.side_effect = httpx.HTTPError("Connection failed")
            result = manager.probe_huggingface()

        assert result is False

    def test_probe_modelscope_success(self, tmp_path):
        """probe_modelscope returns True on success."""
        manager = ModelManager(model_dir=tmp_path / "models")

        with patch("copaw.local_models.model_manager.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            result = manager.probe_modelscope()

        assert result is True


class TestModelManagerMemoryDetection:
    """Tests for ModelManager memory detection."""

    def test_detect_memory_returns_float(self, tmp_path):
        """_detect_available_memory returns a float."""
        manager = ModelManager(model_dir=tmp_path / "models")

        memory = manager._detect_available_memory()

        assert isinstance(memory, float)
        assert memory > 0

    def test_detect_memory_fallback(self, tmp_path):
        """_detect_available_memory has fallback value."""
        manager = ModelManager(model_dir=tmp_path / "models")

        with patch.object(manager, "_get_linux_memory", side_effect=Exception):
            with patch.object(manager, "_get_macos_memory", side_effect=Exception):
                with patch.object(manager, "_get_windows_memory", side_effect=Exception):
                    memory = manager._detect_available_memory()

        assert memory == 8.0
