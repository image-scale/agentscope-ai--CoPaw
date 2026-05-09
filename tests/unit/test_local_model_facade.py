# -*- coding: utf-8 -*-
"""Tests for copaw.local_models.manager module."""

from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from copaw.local_models.manager import LocalModelFacade
from copaw.local_models.model_manager import ModelManager, LocalModelInfo, DownloadSource
from copaw.local_models.llamacpp import LlamaCppBackend


@pytest.fixture
def mock_model_manager(tmp_path):
    """Create a mock ModelManager."""
    manager = MagicMock(spec=ModelManager)
    manager.get_model_path.return_value = tmp_path / "model"
    manager.is_downloaded.return_value = False
    manager.list_downloaded_models.return_value = []
    manager.get_download_progress.return_value = {"status": "idle"}
    manager.is_download_active.return_value = False
    manager.get_recommended_models.return_value = []
    return manager


@pytest.fixture
def mock_llamacpp_backend(tmp_path):
    """Create a mock LlamaCppBackend."""
    backend = MagicMock(spec=LlamaCppBackend)
    backend.check_installation.return_value = (False, "not installed")
    backend.check_installability.return_value = (True, "")
    backend.is_installed.return_value = False
    backend.is_downloading.return_value = False
    backend.is_server_running.return_value = False
    backend.get_download_progress.return_value = {"status": "idle"}
    backend.get_server_status.return_value = {
        "running": False,
        "port": None,
        "model": None,
        "pid": None,
    }
    backend.check_health = AsyncMock(return_value=False)
    backend.start_server = AsyncMock(return_value=8080)
    backend.stop_server = AsyncMock()
    return backend


class TestLocalModelFacadeInit:
    """Tests for LocalModelFacade initialization."""

    def test_creates_default_managers(self, tmp_path):
        """Creates default ModelManager and LlamaCppBackend."""
        with patch("copaw.local_models.manager.ModelManager") as mock_mm:
            with patch("copaw.local_models.manager.LlamaCppBackend") as mock_lb:
                facade = LocalModelFacade()

        mock_mm.assert_called_once()
        mock_lb.assert_called_once()

    def test_accepts_custom_managers(self, mock_model_manager, mock_llamacpp_backend):
        """Accepts custom manager instances."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        assert facade._model_manager is mock_model_manager
        assert facade._llamacpp_backend is mock_llamacpp_backend


class TestLocalModelFacadeSingleton:
    """Tests for singleton pattern."""

    def test_get_instance_returns_same_object(self):
        """get_instance returns the same object."""
        LocalModelFacade.reset_instance()

        with patch("copaw.local_models.manager.ModelManager"):
            with patch("copaw.local_models.manager.LlamaCppBackend"):
                instance1 = LocalModelFacade.get_instance()
                instance2 = LocalModelFacade.get_instance()

        assert instance1 is instance2
        LocalModelFacade.reset_instance()

    def test_reset_instance_clears_singleton(self):
        """reset_instance clears the singleton."""
        LocalModelFacade.reset_instance()

        with patch("copaw.local_models.manager.ModelManager"):
            with patch("copaw.local_models.manager.LlamaCppBackend"):
                instance1 = LocalModelFacade.get_instance()
                LocalModelFacade.reset_instance()
                instance2 = LocalModelFacade.get_instance()

        assert instance1 is not instance2
        LocalModelFacade.reset_instance()


class TestLocalModelFacadeLlamaCpp:
    """Tests for llama.cpp binary management."""

    def test_check_llamacpp_installation(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to backend."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.check_llamacpp_installation()

        mock_llamacpp_backend.check_installation.assert_called_once()
        assert result == (False, "not installed")

    def test_check_llamacpp_installability(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to backend."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.check_llamacpp_installability()

        mock_llamacpp_backend.check_installability.assert_called_once()
        assert result == (True, "")

    def test_start_llamacpp_download(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to backend."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        facade.start_llamacpp_download()

        mock_llamacpp_backend.download.assert_called_once()

    def test_get_llamacpp_download_progress(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to backend."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.get_llamacpp_download_progress()

        mock_llamacpp_backend.get_download_progress.assert_called_once()
        assert result["status"] == "idle"

    def test_cancel_llamacpp_download(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to backend."""
        mock_llamacpp_backend.cancel_download.return_value = True
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.cancel_llamacpp_download()

        mock_llamacpp_backend.cancel_download.assert_called_once()
        assert result is True


class TestLocalModelFacadeServer:
    """Tests for server management."""

    def test_get_server_status(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to backend."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.get_server_status()

        mock_llamacpp_backend.get_server_status.assert_called_once()
        assert result["running"] is False

    def test_is_server_running(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to backend."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.is_server_running()

        mock_llamacpp_backend.is_server_running.assert_called_once()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_server_health(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to backend."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = await facade.check_server_health()

        mock_llamacpp_backend.check_health.assert_called_once()
        assert result is False

    @pytest.mark.asyncio
    async def test_setup_server(self, mock_model_manager, mock_llamacpp_backend, tmp_path):
        """Starts server with model path."""
        model_path = tmp_path / "model"
        mock_model_manager.get_model_path.return_value = model_path
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = await facade.setup_server("org/model")

        mock_llamacpp_backend.start_server.assert_called_once_with(
            model_path=model_path,
            model_name="org/model",
        )
        assert result == 8080

    @pytest.mark.asyncio
    async def test_shutdown_server(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to backend."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        await facade.shutdown_server()

        mock_llamacpp_backend.stop_server.assert_called_once()

    def test_force_shutdown_server(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to backend."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        facade.force_shutdown_server()

        mock_llamacpp_backend.force_stop_server.assert_called_once()


class TestLocalModelFacadeModels:
    """Tests for model management."""

    def test_get_recommended_models(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to model manager."""
        mock_model_manager.get_recommended_models.return_value = [
            LocalModelInfo(id="org/model", name="Model", size_bytes=1000),
        ]
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.get_recommended_models()

        mock_model_manager.get_recommended_models.assert_called_once()
        assert len(result) == 1

    def test_is_model_downloaded(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to model manager."""
        mock_model_manager.is_downloaded.return_value = True
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.is_model_downloaded("org/model")

        mock_model_manager.is_downloaded.assert_called_once_with("org/model")
        assert result is True

    def test_list_downloaded_models(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to model manager."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.list_downloaded_models()

        mock_model_manager.list_downloaded_models.assert_called_once()
        assert result == []

    def test_get_model_path(self, mock_model_manager, mock_llamacpp_backend, tmp_path):
        """Delegates to model manager."""
        mock_model_manager.get_model_path.return_value = tmp_path / "model"
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.get_model_path("org/model")

        mock_model_manager.get_model_path.assert_called_with("org/model")
        assert result == tmp_path / "model"

    def test_remove_model(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to model manager."""
        mock_model_manager.remove_model.return_value = True
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.remove_model("org/model")

        mock_model_manager.remove_model.assert_called_once_with("org/model")
        assert result is True


class TestLocalModelFacadeModelDownload:
    """Tests for model download management."""

    def test_get_model_download_progress(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to model manager."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.get_model_download_progress()

        mock_model_manager.get_download_progress.assert_called_once()
        assert result["status"] == "idle"

    def test_is_model_downloading(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to model manager."""
        mock_model_manager.is_download_active.return_value = True
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.is_model_downloading()

        mock_model_manager.is_download_active.assert_called_once()
        assert result is True

    def test_prepare_model_download(self, mock_model_manager, mock_llamacpp_backend, tmp_path):
        """Delegates to model manager."""
        staging = tmp_path / "staging"
        mock_model_manager.prepare_download.return_value = staging
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.prepare_model_download(
            "org/model",
            source=DownloadSource.HUGGINGFACE,
            total_bytes=1000,
        )

        mock_model_manager.prepare_download.assert_called_once_with(
            "org/model",
            source=DownloadSource.HUGGINGFACE,
            total_bytes=1000,
        )
        assert result == staging

    def test_complete_model_download(self, mock_model_manager, mock_llamacpp_backend, tmp_path):
        """Delegates to model manager."""
        staging = tmp_path / "staging"
        final = tmp_path / "final"
        mock_model_manager.complete_download.return_value = final
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.complete_model_download("org/model", staging)

        mock_model_manager.complete_download.assert_called_once_with("org/model", staging)
        assert result == final

    def test_fail_model_download(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to model manager."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        facade.fail_model_download("Error")

        mock_model_manager.fail_download.assert_called_once_with("Error")

    def test_cancel_model_download(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to model manager."""
        mock_model_manager.cancel_download.return_value = True
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.cancel_model_download()

        mock_model_manager.cancel_download.assert_called_once()
        assert result is True

    def test_update_model_download_progress(self, mock_model_manager, mock_llamacpp_backend):
        """Delegates to model manager."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        facade.update_model_download_progress(500)

        mock_model_manager.update_progress.assert_called_once_with(500)


class TestLocalModelFacadeCombinedOperations:
    """Tests for combined operations."""

    @pytest.mark.asyncio
    async def test_ensure_server_ready_model_not_downloaded(
        self, mock_model_manager, mock_llamacpp_backend
    ):
        """Raises if model not downloaded."""
        mock_model_manager.is_downloaded.return_value = False
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        with pytest.raises(RuntimeError, match="not downloaded"):
            await facade.ensure_server_ready("org/model")

    @pytest.mark.asyncio
    async def test_ensure_server_ready_success(
        self, mock_model_manager, mock_llamacpp_backend, tmp_path
    ):
        """Starts server when model is downloaded."""
        mock_model_manager.is_downloaded.return_value = True
        mock_model_manager.get_model_path.return_value = tmp_path / "model"
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = await facade.ensure_server_ready("org/model")

        assert result == 8080

    def test_get_full_status(self, mock_model_manager, mock_llamacpp_backend):
        """Returns combined status."""
        facade = LocalModelFacade(
            model_manager=mock_model_manager,
            llamacpp_backend=mock_llamacpp_backend,
        )

        result = facade.get_full_status()

        assert "llamacpp" in result
        assert "server" in result
        assert "models" in result
        assert result["llamacpp"]["installed"] is False
        assert result["server"]["running"] is False
