# -*- coding: utf-8 -*-
"""Tests for copaw.local_models.llamacpp module."""

import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from copaw.local_models.llamacpp import (
    LlamaCppBackend,
    DownloadCancelled,
    LLAMA_CPP_RELEASE_URL,
    LLAMA_CPP_DEFAULT_TAG,
)
from copaw.local_models.download import DownloadStatus


class TestLlamaCppBackendInit:
    """Tests for LlamaCppBackend initialization."""

    def test_default_values(self, tmp_path):
        """Backend uses default values when none provided."""
        with patch("copaw.local_models.llamacpp.SECRET_DIR", tmp_path):
            backend = LlamaCppBackend()

        assert backend.base_url == LLAMA_CPP_RELEASE_URL
        assert backend.release_tag == LLAMA_CPP_DEFAULT_TAG
        assert backend.target_dir == tmp_path / "local_models" / "bin"

    def test_custom_values(self, tmp_path):
        """Backend accepts custom values."""
        backend = LlamaCppBackend(
            base_url="https://custom.url",
            release_tag="v1.0.0",
            target_dir=tmp_path / "custom",
        )

        assert backend.base_url == "https://custom.url"
        assert backend.release_tag == "v1.0.0"
        assert backend.target_dir == tmp_path / "custom"


class TestLlamaCppBackendDetection:
    """Tests for system detection."""

    def test_detect_os_linux(self, tmp_path):
        """Detects Linux OS."""
        with patch("platform.system", return_value="Linux"):
            backend = LlamaCppBackend(target_dir=tmp_path)

        assert backend.os_name == "linux"

    def test_detect_os_macos(self, tmp_path):
        """Detects macOS."""
        with patch("platform.system", return_value="Darwin"):
            backend = LlamaCppBackend(target_dir=tmp_path)

        assert backend.os_name == "macos"

    def test_detect_os_windows(self, tmp_path):
        """Detects Windows."""
        with patch("platform.system", return_value="Windows"):
            backend = LlamaCppBackend(target_dir=tmp_path)

        assert backend.os_name == "windows"

    def test_detect_arch_x64(self, tmp_path):
        """Detects x64 architecture."""
        with patch("platform.machine", return_value="x86_64"):
            backend = LlamaCppBackend(target_dir=tmp_path)

        assert backend.arch == "x64"

    def test_detect_arch_arm64(self, tmp_path):
        """Detects arm64 architecture."""
        with patch("platform.machine", return_value="arm64"):
            backend = LlamaCppBackend(target_dir=tmp_path)

        assert backend.arch == "arm64"

    def test_detect_backend_cpu_default(self, tmp_path):
        """Defaults to CPU backend."""
        with patch("platform.system", return_value="Linux"):
            backend = LlamaCppBackend(target_dir=tmp_path)

        assert backend.backend == "cpu"


class TestLlamaCppBackendExecutable:
    """Tests for executable path."""

    def test_executable_unix(self, tmp_path):
        """Returns correct path on Unix systems."""
        with patch("platform.system", return_value="Linux"):
            backend = LlamaCppBackend(target_dir=tmp_path / "bin")

        assert backend.executable == tmp_path / "bin" / "llama-server"

    def test_executable_windows(self, tmp_path):
        """Returns correct path on Windows."""
        with patch("platform.system", return_value="Windows"):
            backend = LlamaCppBackend(target_dir=tmp_path / "bin")

        assert backend.executable == tmp_path / "bin" / "llama-server.exe"


class TestLlamaCppBackendDownloadUrl:
    """Tests for download URL building."""

    def test_download_url_linux_x64(self, tmp_path):
        """Builds correct URL for Linux x64."""
        with patch("platform.system", return_value="Linux"):
            with patch("platform.machine", return_value="x86_64"):
                backend = LlamaCppBackend(
                    target_dir=tmp_path,
                    release_tag="b5270",
                )

        assert "llama-b5270-bin-ubuntu-x64.tar.gz" in backend.download_url

    def test_download_url_macos_arm64(self, tmp_path):
        """Builds correct URL for macOS arm64."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.machine", return_value="arm64"):
                backend = LlamaCppBackend(
                    target_dir=tmp_path,
                    release_tag="b5270",
                )

        assert "llama-b5270-bin-macos-arm64.tar.gz" in backend.download_url

    def test_download_url_windows_cpu(self, tmp_path):
        """Builds correct URL for Windows CPU."""
        with patch("platform.system", return_value="Windows"):
            with patch("platform.machine", return_value="x86_64"):
                backend = LlamaCppBackend(target_dir=tmp_path)
                backend._backend = "cpu"

        assert "win-cpu-x64.zip" in backend.download_url


class TestLlamaCppBackendInstallation:
    """Tests for installation checking."""

    def test_is_installed_false(self, tmp_path):
        """is_installed returns False when not installed."""
        backend = LlamaCppBackend(target_dir=tmp_path / "bin")

        assert backend.is_installed() is False

    def test_is_installed_true(self, tmp_path):
        """is_installed returns True when installed."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "llama-server").touch()

        with patch("platform.system", return_value="Linux"):
            backend = LlamaCppBackend(target_dir=bin_dir)

        assert backend.is_installed() is True

    def test_check_installation_not_installed(self, tmp_path):
        """check_installation returns message when not installed."""
        backend = LlamaCppBackend(target_dir=tmp_path / "bin")

        installed, message = backend.check_installation()

        assert installed is False
        assert "not installed" in message

    def test_check_installation_installed(self, tmp_path):
        """check_installation returns success when installed."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "llama-server").touch()

        with patch("platform.system", return_value="Linux"):
            backend = LlamaCppBackend(target_dir=bin_dir)

        installed, message = backend.check_installation()

        assert installed is True
        assert message == ""


class TestLlamaCppBackendInstallability:
    """Tests for installability checking."""

    def test_check_installability_linux(self, tmp_path):
        """Linux is installable."""
        with patch("platform.system", return_value="Linux"):
            with patch("platform.machine", return_value="x86_64"):
                backend = LlamaCppBackend(target_dir=tmp_path)

        installable, message = backend.check_installability()

        assert installable is True
        assert message == ""

    def test_check_installability_macos_arm64(self, tmp_path):
        """macOS arm64 is installable with right version."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.machine", return_value="arm64"):
                with patch("platform.mac_ver", return_value=("14.0", ("", "", ""), "")):
                    backend = LlamaCppBackend(target_dir=tmp_path)
                    installable, message = backend.check_installability()

        assert installable is True

    def test_check_installability_macos_x64_fails(self, tmp_path):
        """macOS x64 is not installable."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.machine", return_value="x86_64"):
                backend = LlamaCppBackend(target_dir=tmp_path)

        installable, message = backend.check_installability()

        assert installable is False
        assert "Apple Silicon" in message


class TestLlamaCppBackendDownloadProgress:
    """Tests for download progress tracking."""

    def test_get_download_progress_initial(self, tmp_path):
        """Initial progress is idle."""
        backend = LlamaCppBackend(target_dir=tmp_path)

        progress = backend.get_download_progress()

        assert progress["status"] == "idle"

    def test_is_downloading_false_initially(self, tmp_path):
        """is_downloading returns False initially."""
        backend = LlamaCppBackend(target_dir=tmp_path)

        assert backend.is_downloading() is False


class TestLlamaCppBackendServerStatus:
    """Tests for server status."""

    def test_get_server_status_not_running(self, tmp_path):
        """Server status when not running."""
        backend = LlamaCppBackend(target_dir=tmp_path)

        status = backend.get_server_status()

        assert status["running"] is False
        assert status["port"] is None
        assert status["model"] is None
        assert status["pid"] is None

    def test_is_server_running_false(self, tmp_path):
        """is_server_running returns False when stopped."""
        backend = LlamaCppBackend(target_dir=tmp_path)

        assert backend.is_server_running() is False


class TestLlamaCppBackendServerLifecycle:
    """Tests for server lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_server_not_installed_raises(self, tmp_path):
        """start_server raises when not installed."""
        backend = LlamaCppBackend(target_dir=tmp_path / "bin")
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        with pytest.raises(RuntimeError, match="not installed"):
            await backend.start_server(model_path, "test-model")

    @pytest.mark.asyncio
    async def test_start_server_model_not_found_raises(self, tmp_path):
        """start_server raises when model not found."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "llama-server").touch()

        with patch("platform.system", return_value="Linux"):
            backend = LlamaCppBackend(target_dir=bin_dir)

        with pytest.raises(FileNotFoundError, match="not found"):
            await backend.start_server(tmp_path / "nonexistent", "test")

    @pytest.mark.asyncio
    async def test_stop_server_when_not_running(self, tmp_path):
        """stop_server does nothing when not running."""
        backend = LlamaCppBackend(target_dir=tmp_path)

        await backend.stop_server()

        assert backend.is_server_running() is False

    def test_force_stop_server_when_not_running(self, tmp_path):
        """force_stop_server does nothing when not running."""
        backend = LlamaCppBackend(target_dir=tmp_path)

        backend.force_stop_server()

        assert backend.is_server_running() is False


class TestLlamaCppBackendHealth:
    """Tests for health checking."""

    @pytest.mark.asyncio
    async def test_check_health_no_port(self, tmp_path):
        """check_health returns False with no port."""
        backend = LlamaCppBackend(target_dir=tmp_path)

        result = await backend.check_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_check_health_connection_error(self, tmp_path):
        """check_health returns False on connection error."""
        backend = LlamaCppBackend(target_dir=tmp_path)

        result = await backend.check_health(port=9999)

        assert result is False


class TestLlamaCppBackendGgufResolution:
    """Tests for GGUF file resolution."""

    def test_resolve_gguf_file_direct(self, tmp_path):
        """Resolves direct GGUF file path."""
        gguf_file = tmp_path / "model.gguf"
        gguf_file.write_bytes(b"GGUF")

        backend = LlamaCppBackend(target_dir=tmp_path)
        result = backend._resolve_gguf_file(gguf_file)

        assert result == gguf_file.resolve()

    def test_resolve_gguf_file_wrong_extension_raises(self, tmp_path):
        """Raises for non-GGUF file."""
        other_file = tmp_path / "model.bin"
        other_file.write_bytes(b"data")

        backend = LlamaCppBackend(target_dir=tmp_path)

        with pytest.raises(RuntimeError, match=".gguf"):
            backend._resolve_gguf_file(other_file)

    def test_resolve_gguf_file_from_directory(self, tmp_path):
        """Finds GGUF file in directory."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        gguf_file = model_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF")

        backend = LlamaCppBackend(target_dir=tmp_path)
        result = backend._resolve_gguf_file(model_dir)

        assert result == gguf_file.resolve()

    def test_resolve_gguf_file_not_found_raises(self, tmp_path):
        """Raises when no GGUF found in directory."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        backend = LlamaCppBackend(target_dir=tmp_path)

        with pytest.raises(RuntimeError, match="No .gguf"):
            backend._resolve_gguf_file(model_dir)


class TestLlamaCppBackendFreePort:
    """Tests for free port finding."""

    def test_find_free_port(self, tmp_path):
        """find_free_port returns a valid port."""
        backend = LlamaCppBackend(target_dir=tmp_path)

        port = backend._find_free_port()

        assert isinstance(port, int)
        assert 1024 <= port <= 65535


class TestLlamaCppBackendFilename:
    """Tests for filename building."""

    def test_build_filename_linux_x64(self, tmp_path):
        """Builds correct filename for Linux x64."""
        with patch("platform.system", return_value="Linux"):
            with patch("platform.machine", return_value="x86_64"):
                backend = LlamaCppBackend(
                    target_dir=tmp_path,
                    release_tag="b5270",
                )

        filename = backend._build_filename()

        assert filename == "llama-b5270-bin-ubuntu-x64.tar.gz"

    def test_build_filename_linux_arm64(self, tmp_path):
        """Builds correct filename for Linux arm64."""
        with patch("platform.system", return_value="Linux"):
            with patch("platform.machine", return_value="aarch64"):
                backend = LlamaCppBackend(
                    target_dir=tmp_path,
                    release_tag="b5270",
                )

        filename = backend._build_filename()

        assert filename == "llama-b5270-bin-ubuntu-arm64.tar.gz"

    def test_build_filename_macos_arm64(self, tmp_path):
        """Builds correct filename for macOS arm64."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.machine", return_value="arm64"):
                backend = LlamaCppBackend(
                    target_dir=tmp_path,
                    release_tag="b5270",
                )

        filename = backend._build_filename()

        assert filename == "llama-b5270-bin-macos-arm64.tar.gz"

    def test_build_filename_macos_x64_raises(self, tmp_path):
        """Raises for macOS x64."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.machine", return_value="x86_64"):
                backend = LlamaCppBackend(target_dir=tmp_path)

        with pytest.raises(RuntimeError, match="arm64"):
            backend._build_filename()

    def test_build_filename_windows_cpu(self, tmp_path):
        """Builds correct filename for Windows CPU."""
        with patch("platform.system", return_value="Windows"):
            with patch("platform.machine", return_value="x86_64"):
                backend = LlamaCppBackend(
                    target_dir=tmp_path,
                    release_tag="b5270",
                )
                backend._backend = "cpu"

        filename = backend._build_filename()

        assert filename == "llama-b5270-bin-win-cpu-x64.zip"


class TestDownloadCancelled:
    """Tests for DownloadCancelled exception."""

    def test_is_exception(self):
        """DownloadCancelled is an exception."""
        assert issubclass(DownloadCancelled, Exception)

    def test_can_raise(self):
        """Can raise DownloadCancelled."""
        with pytest.raises(DownloadCancelled):
            raise DownloadCancelled("test")
