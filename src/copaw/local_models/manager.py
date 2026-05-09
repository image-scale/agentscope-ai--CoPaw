# -*- coding: utf-8 -*-
"""Unified facade for local model management.

Combines ModelManager and LlamaCppBackend into a single interface for
managing local LLM models and the llama.cpp inference server.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

from .llamacpp import LlamaCppBackend, LLAMA_CPP_RELEASE_URL, LLAMA_CPP_DEFAULT_TAG
from .model_manager import ModelManager, LocalModelInfo, DownloadSource


class LocalModelFacade:
    """Unified interface for local model management and inference.

    Provides a single entry point for:
    - Downloading and managing local LLM models
    - Installing and running llama.cpp server
    - Coordinating model loading and server lifecycle
    """

    _instance: ClassVar[LocalModelFacade | None] = None

    def __init__(
        self,
        *,
        model_manager: ModelManager | None = None,
        llamacpp_backend: LlamaCppBackend | None = None,
        llama_cpp_base_url: str | None = None,
        llama_cpp_release_tag: str | None = None,
    ) -> None:
        """Initialize the facade.

        Args:
            model_manager: Custom ModelManager instance.
            llamacpp_backend: Custom LlamaCppBackend instance.
            llama_cpp_base_url: Base URL for llama.cpp releases.
            llama_cpp_release_tag: Release tag for llama.cpp.
        """
        self._model_manager = model_manager or ModelManager()
        self._llamacpp_backend = llamacpp_backend or LlamaCppBackend(
            base_url=llama_cpp_base_url or LLAMA_CPP_RELEASE_URL,
            release_tag=llama_cpp_release_tag or LLAMA_CPP_DEFAULT_TAG,
        )
        self._server_lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> LocalModelFacade:
        """Get the singleton instance.

        Returns:
            The shared LocalModelFacade instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    # -------------------------------------------------------------------------
    # llama.cpp Binary Management
    # -------------------------------------------------------------------------

    def check_llamacpp_installation(self) -> tuple[bool, str]:
        """Check if llama.cpp is installed.

        Returns:
            Tuple of (installed, message).
        """
        return self._llamacpp_backend.check_installation()

    def check_llamacpp_installability(self) -> tuple[bool, str]:
        """Check if llama.cpp can be installed on this system.

        Returns:
            Tuple of (installable, error_message).
        """
        return self._llamacpp_backend.check_installability()

    def start_llamacpp_download(self) -> None:
        """Start downloading llama.cpp binaries.

        Raises:
            RuntimeError: If download already in progress or not installable.
        """
        self._llamacpp_backend.download()

    def get_llamacpp_download_progress(self) -> dict[str, Any]:
        """Get llama.cpp download progress.

        Returns:
            Dictionary with download status and progress.
        """
        return self._llamacpp_backend.get_download_progress()

    def cancel_llamacpp_download(self) -> bool:
        """Cancel llama.cpp download.

        Returns:
            True if cancellation was initiated.
        """
        return self._llamacpp_backend.cancel_download()

    def is_llamacpp_downloading(self) -> bool:
        """Check if llama.cpp is being downloaded.

        Returns:
            True if download is in progress.
        """
        return self._llamacpp_backend.is_downloading()

    # -------------------------------------------------------------------------
    # llama.cpp Server Management
    # -------------------------------------------------------------------------

    def get_server_status(self) -> dict[str, Any]:
        """Get llama.cpp server status.

        Returns:
            Dictionary with server state.
        """
        return self._llamacpp_backend.get_server_status()

    def is_server_running(self) -> bool:
        """Check if llama.cpp server is running.

        Returns:
            True if server is running.
        """
        return self._llamacpp_backend.is_server_running()

    async def check_server_health(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if server is healthy.
        """
        return await self._llamacpp_backend.check_health()

    async def setup_server(self, model_id: str) -> int:
        """Start llama.cpp server with a model.

        Args:
            model_id: Model repository ID to load.

        Returns:
            Port number the server is running on.

        Raises:
            RuntimeError: If llama.cpp not installed or model not found.
        """
        async with self._server_lock:
            model_path = self._model_manager.get_model_path(model_id)
            return await self._llamacpp_backend.start_server(
                model_path=model_path,
                model_name=model_id,
            )

    async def shutdown_server(self) -> None:
        """Stop the llama.cpp server."""
        async with self._server_lock:
            await self._llamacpp_backend.stop_server()

    def force_shutdown_server(self) -> None:
        """Synchronously stop the server (for cleanup)."""
        self._llamacpp_backend.force_stop_server()

    # -------------------------------------------------------------------------
    # Model Management
    # -------------------------------------------------------------------------

    def get_recommended_models(self) -> list[LocalModelInfo]:
        """Get recommended models for this system.

        Returns:
            List of recommended models based on system memory.
        """
        return self._model_manager.get_recommended_models()

    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if a model is downloaded.

        Args:
            model_id: Model repository ID.

        Returns:
            True if model is downloaded and ready.
        """
        return self._model_manager.is_downloaded(model_id)

    def list_downloaded_models(self) -> list[LocalModelInfo]:
        """List all downloaded models.

        Returns:
            List of downloaded model information.
        """
        return self._model_manager.list_downloaded_models()

    def get_model_path(self, model_id: str) -> Path:
        """Get the local path for a model.

        Args:
            model_id: Model repository ID.

        Returns:
            Path to the model directory.
        """
        return self._model_manager.get_model_path(model_id)

    def remove_model(self, model_id: str) -> bool:
        """Remove a downloaded model.

        Args:
            model_id: Model repository ID to remove.

        Returns:
            True if model was removed.
        """
        return self._model_manager.remove_model(model_id)

    # -------------------------------------------------------------------------
    # Model Download Management
    # -------------------------------------------------------------------------

    def get_model_download_progress(self) -> dict[str, Any]:
        """Get model download progress.

        Returns:
            Dictionary with download status and progress.
        """
        return self._model_manager.get_download_progress()

    def is_model_downloading(self) -> bool:
        """Check if a model download is in progress.

        Returns:
            True if downloading.
        """
        return self._model_manager.is_download_active()

    def prepare_model_download(
        self,
        model_id: str,
        source: DownloadSource | None = None,
        total_bytes: int | None = None,
    ) -> Path:
        """Prepare to download a model.

        Args:
            model_id: Model repository ID.
            source: Download source preference.
            total_bytes: Expected total size.

        Returns:
            Staging directory path.

        Raises:
            RuntimeError: If download already in progress.
        """
        return self._model_manager.prepare_download(
            model_id,
            source=source,
            total_bytes=total_bytes,
        )

    def complete_model_download(self, model_id: str, staging_dir: Path) -> Path:
        """Complete a model download.

        Args:
            model_id: Model repository ID.
            staging_dir: Staging directory with downloaded files.

        Returns:
            Final model directory path.
        """
        return self._model_manager.complete_download(model_id, staging_dir)

    def fail_model_download(self, error: str) -> None:
        """Mark model download as failed.

        Args:
            error: Error message.
        """
        self._model_manager.fail_download(error)

    def cancel_model_download(self) -> bool:
        """Cancel model download.

        Returns:
            True if cancellation was initiated.
        """
        return self._model_manager.cancel_download()

    def update_model_download_progress(self, downloaded_bytes: int) -> None:
        """Update model download progress.

        Args:
            downloaded_bytes: Bytes downloaded so far.
        """
        self._model_manager.update_progress(downloaded_bytes)

    # -------------------------------------------------------------------------
    # Combined Operations
    # -------------------------------------------------------------------------

    async def ensure_server_ready(
        self,
        model_id: str,
        timeout: float = 120.0,
    ) -> int:
        """Ensure server is running with the specified model.

        This is a convenience method that:
        1. Checks if the model is downloaded
        2. Starts the server if needed
        3. Waits for the server to become healthy

        Args:
            model_id: Model repository ID.
            timeout: Timeout for server readiness.

        Returns:
            Port number the server is running on.

        Raises:
            RuntimeError: If model not downloaded or server fails to start.
        """
        if not self.is_model_downloaded(model_id):
            raise RuntimeError(f"Model not downloaded: {model_id}")

        port = await self.setup_server(model_id)

        # Server readiness is already checked in setup_server
        return port

    def get_full_status(self) -> dict[str, Any]:
        """Get complete status of local model system.

        Returns:
            Dictionary with all status information.
        """
        return {
            "llamacpp": {
                "installed": self._llamacpp_backend.is_installed(),
                "installable": self._llamacpp_backend.check_installability()[0],
                "downloading": self._llamacpp_backend.is_downloading(),
                "download_progress": self._llamacpp_backend.get_download_progress(),
            },
            "server": self._llamacpp_backend.get_server_status(),
            "models": {
                "downloading": self._model_manager.is_download_active(),
                "download_progress": self._model_manager.get_download_progress(),
                "downloaded": [m.id for m in self._model_manager.list_downloaded_models()],
            },
        }
