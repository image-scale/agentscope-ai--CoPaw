# -*- coding: utf-8 -*-
"""Llama.cpp backend for running local LLM models.

Manages llama.cpp server installation, lifecycle, and model loading.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

from ..settings import SECRET_DIR
from .download import (
    DownloadStatus,
    DownloadResult,
    ProgressTracker,
    start_download,
    apply_result,
)

logger = logging.getLogger(__name__)

LLAMA_CPP_RELEASE_URL = "https://github.com/ggerganov/llama.cpp/releases/download"
LLAMA_CPP_DEFAULT_TAG = "b5270"


class DownloadCancelled(Exception):
    """Raised when download is cancelled by user."""

    pass


class LlamaCppBackend:
    """Backend for managing llama.cpp server installation and lifecycle.

    Handles:
    - Downloading and extracting llama.cpp binaries
    - Starting and stopping the llama-server process
    - Health checking the server
    - Managing model loading
    """

    MIN_MACOS_VERSION = (13, 3)

    def __init__(
        self,
        base_url: str | None = None,
        release_tag: str | None = None,
        target_dir: Path | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            base_url: Base URL for llama.cpp releases.
            release_tag: Release tag to download.
            target_dir: Directory for binaries.
        """
        self.base_url = (base_url or LLAMA_CPP_RELEASE_URL).rstrip("/")
        self.release_tag = release_tag or LLAMA_CPP_DEFAULT_TAG
        self.target_dir = target_dir or SECRET_DIR / "local_models" / "bin"

        self._os_name = self._detect_os()
        self._arch = self._detect_arch()
        self._backend = self._detect_backend()

        self._server_process: subprocess.Popen[bytes] | None = None
        self._server_port: int | None = None
        self._server_model: str | None = None
        self._server_transitioning = False
        self._lock = threading.RLock()

        self._progress = ProgressTracker()
        self._download_thread: threading.Thread | None = None
        self._download_cancelled = threading.Event()

        atexit.register(self._cleanup_at_exit)

    @property
    def os_name(self) -> str:
        """Get the detected operating system."""
        return self._os_name

    @property
    def arch(self) -> str:
        """Get the detected architecture."""
        return self._arch

    @property
    def backend(self) -> str:
        """Get the detected backend (cpu or cuda)."""
        return self._backend

    @property
    def executable(self) -> Path:
        """Get the path to the llama-server executable."""
        if self._os_name == "windows":
            return self.target_dir / "llama-server.exe"
        return self.target_dir / "llama-server"

    @property
    def download_url(self) -> str:
        """Get the download URL for the current environment."""
        filename = self._build_filename()
        return f"{self.base_url}/{self.release_tag}/{filename}"

    def is_installed(self) -> bool:
        """Check if llama.cpp is installed."""
        return self.executable.exists()

    def check_installation(self) -> tuple[bool, str]:
        """Check if llama.cpp is installed with message.

        Returns:
            Tuple of (installed, message).
        """
        if self.is_installed():
            return True, ""
        return False, "llama.cpp server is not installed"

    def check_installability(self) -> tuple[bool, str]:
        """Check if llama.cpp can be installed on this system.

        Returns:
            Tuple of (installable, error_message).
        """
        if self._os_name == "macos":
            if self._arch != "arm64":
                return False, "Only Apple Silicon (M series) chips are supported"
            if not self._check_macos_version():
                return False, f"macOS {'.'.join(map(str, self.MIN_MACOS_VERSION))} or later required"

        try:
            self._build_filename()
        except RuntimeError as exc:
            return False, str(exc)

        return True, ""

    def get_download_progress(self) -> dict[str, Any]:
        """Get the current download progress."""
        return self._progress.snapshot()

    def is_downloading(self) -> bool:
        """Check if a download is in progress."""
        with self._lock:
            return (
                self._download_thread is not None
                and self._download_thread.is_alive()
            )

    def download(self, chunk_size: int = 1024 * 1024, timeout: int = 30) -> None:
        """Start downloading llama.cpp in the background.

        Args:
            chunk_size: Size of download chunks.
            timeout: Network timeout in seconds.

        Raises:
            RuntimeError: If download already in progress or not installable.
        """
        installable, message = self.check_installability()
        if not installable:
            raise RuntimeError(message)

        with self._lock:
            if self.is_downloading():
                raise RuntimeError("Download already in progress")

            self._download_cancelled.clear()
            start_download(
                self._progress,
                source=self.download_url,
                model_name="llama.cpp",
            )

            self._download_thread = threading.Thread(
                target=self._download_worker,
                args=(chunk_size, timeout),
                name="llama-cpp-download",
                daemon=True,
            )
            self._download_thread.start()

    def cancel_download(self) -> bool:
        """Cancel the current download.

        Returns:
            True if cancellation was initiated.
        """
        with self._lock:
            if not self.is_downloading():
                return False
            self._download_cancelled.set()
            self._progress.mark_canceling()
            return True

    def get_server_status(self) -> dict[str, Any]:
        """Get the current server status.

        Returns:
            Dictionary with server state.
        """
        with self._lock:
            running = self._is_server_running()
            return {
                "running": running,
                "port": self._server_port if running else None,
                "model": self._server_model if running else None,
                "pid": self._server_process.pid if running else None,
                "transitioning": self._server_transitioning,
            }

    def is_server_running(self) -> bool:
        """Check if the server is currently running."""
        with self._lock:
            return self._is_server_running()

    async def start_server(self, model_path: Path, model_name: str) -> int:
        """Start the llama.cpp server with a model.

        Args:
            model_path: Path to model directory or GGUF file.
            model_name: Name alias for the model.

        Returns:
            Port number the server is running on.

        Raises:
            RuntimeError: If not installed or model not found.
        """
        if not self.is_installed():
            raise RuntimeError("llama.cpp is not installed")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with self._lock:
            if self._server_model == model_name and self._is_server_running():
                return self._server_port  # type: ignore

            if self._is_server_running():
                await self.stop_server()

            self._server_transitioning = True

        try:
            gguf_path = self._resolve_gguf_file(model_path)
            port = self._find_free_port()

            process = subprocess.Popen(
                [
                    str(self.executable),
                    "--host", "127.0.0.1",
                    "--port", str(port),
                    "--model", str(gguf_path),
                    "--alias", model_name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            with self._lock:
                self._server_process = process
                self._server_port = port
                self._server_model = model_name

            await self._wait_for_server_ready(port)

            logger.info("llama.cpp server started on port %d for %s", port, model_name)
            return port

        except Exception:
            with self._lock:
                self._reset_server_state()
            raise
        finally:
            with self._lock:
                self._server_transitioning = False

    async def stop_server(self) -> None:
        """Stop the llama.cpp server if running."""
        with self._lock:
            process = self._server_process
            if not self._is_server_running():
                return
            self._server_transitioning = True

        try:
            process.terminate()  # type: ignore
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(process.wait),  # type: ignore
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                process.kill()  # type: ignore
                await asyncio.to_thread(process.wait)  # type: ignore
        finally:
            with self._lock:
                self._reset_server_state()

    def force_stop_server(self) -> None:
        """Synchronously stop the server (for cleanup)."""
        with self._lock:
            process = self._server_process
            if not self._is_server_running():
                return

        try:
            process.terminate()  # type: ignore
            process.wait(timeout=5)  # type: ignore
        except subprocess.TimeoutExpired:
            process.kill()  # type: ignore
            process.wait()  # type: ignore
        finally:
            with self._lock:
                self._reset_server_state()

    async def check_health(self, port: int | None = None) -> bool:
        """Check if the server is healthy.

        Args:
            port: Port to check, defaults to current server port.

        Returns:
            True if server is healthy.
        """
        check_port = port or self._server_port
        if check_port is None:
            return False

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"http://127.0.0.1:{check_port}/health")
                return response.status_code < 500
        except httpx.HTTPError:
            return False

    def _is_server_running(self) -> bool:
        """Check if server process is alive (must hold lock)."""
        return (
            self._server_process is not None
            and self._server_process.poll() is None
        )

    def _reset_server_state(self) -> None:
        """Reset server state (must hold lock)."""
        self._server_process = None
        self._server_port = None
        self._server_model = None
        self._server_transitioning = False

    async def _wait_for_server_ready(self, port: int, timeout: float = 120.0) -> None:
        """Wait for server to become ready."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            with self._lock:
                if not self._is_server_running():
                    raise RuntimeError("Server exited before becoming ready")

            if await self.check_health(port):
                return

            await asyncio.sleep(1.0)

        raise RuntimeError("Timed out waiting for server to start")

    def _download_worker(self, chunk_size: int, timeout: int) -> None:
        """Background download worker."""
        result: DownloadResult
        try:
            local_path = self._download_sync(chunk_size, timeout)
            result = DownloadResult(
                status=DownloadStatus.COMPLETED,
                local_path=str(local_path),
            )
        except DownloadCancelled:
            result = DownloadResult(
                status=DownloadStatus.CANCELLED,
                error="Download cancelled",
            )
        except Exception as exc:
            result = DownloadResult(
                status=DownloadStatus.FAILED,
                error=str(exc),
            )

        with self._lock:
            self._download_thread = None
            self._download_cancelled.clear()

        apply_result(self._progress, result)

    def _download_sync(self, chunk_size: int, timeout: int) -> Path:
        """Synchronous download and extraction."""
        self.target_dir.mkdir(parents=True, exist_ok=True)

        url = self.download_url
        filename = url.rsplit("/", 1)[-1]

        fd, temp_path = tempfile.mkstemp(
            prefix="llama-cpp-",
            suffix=f"-{filename}",
            dir=str(self.target_dir),
        )
        os.close(fd)
        temp_file = Path(temp_path)

        try:
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()

                    total = response.headers.get("content-length")
                    total_bytes = int(total) if total and total.isdigit() else None
                    downloaded = 0

                    with open(temp_file, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=chunk_size):
                            if self._download_cancelled.is_set():
                                raise DownloadCancelled()
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                self._progress.update_progress(
                                    downloaded,
                                    total_bytes=total_bytes,
                                )

            if self._download_cancelled.is_set():
                raise DownloadCancelled()

            self._extract_archive(temp_file)
            return self.target_dir

        finally:
            temp_file.unlink(missing_ok=True)

    def _extract_archive(self, archive_path: Path) -> None:
        """Extract archive to target directory."""
        staging = Path(tempfile.mkdtemp(
            prefix="llama-cpp-extract-",
            dir=str(self.target_dir.parent),
        ))

        try:
            shutil.unpack_archive(str(archive_path), str(staging))

            entries = list(staging.iterdir())
            source = staging
            if len(entries) == 1 and entries[0].is_dir():
                source = entries[0]

            for item in source.iterdir():
                dest = self.target_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))

        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _resolve_gguf_file(self, model_path: Path) -> Path:
        """Resolve path to a GGUF file."""
        if model_path.is_file():
            if model_path.suffix.lower() != ".gguf":
                raise RuntimeError(f"Model must be a .gguf file: {model_path}")
            return model_path.resolve()

        gguf_files = sorted(model_path.rglob("*.gguf"))
        if not gguf_files:
            raise RuntimeError(f"No .gguf files found in {model_path}")
        return gguf_files[0].resolve()

    @staticmethod
    def _find_free_port() -> int:
        """Find an available port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            return s.getsockname()[1]

    def _detect_os(self) -> str:
        """Detect the operating system."""
        system = platform.system().lower()
        mapping = {
            "windows": "windows",
            "darwin": "macos",
            "linux": "linux",
        }
        return mapping.get(system, system)

    def _detect_arch(self) -> str:
        """Detect the CPU architecture."""
        machine = platform.machine().lower()
        mapping = {
            "x86_64": "x64",
            "amd64": "x64",
            "arm64": "arm64",
            "aarch64": "arm64",
        }
        return mapping.get(machine, machine)

    def _detect_backend(self) -> str:
        """Detect the compute backend."""
        if self._os_name == "windows":
            if self._detect_cuda():
                return "cuda"
        return "cpu"

    def _detect_cuda(self) -> str | None:
        """Detect CUDA version on Windows."""
        if self._os_name != "windows":
            return None

        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                import re
                match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
                if match:
                    return match.group(1)
        except FileNotFoundError:
            pass
        return None

    def _check_macos_version(self) -> bool:
        """Check if macOS version is sufficient."""
        if self._os_name != "macos":
            return True

        mac_ver = platform.mac_ver()[0]
        if not mac_ver:
            return False

        parts = tuple(int(p) for p in mac_ver.split(".")[:2])
        return parts >= self.MIN_MACOS_VERSION

    def _build_filename(self) -> str:
        """Build the download filename for current environment."""
        tag = self.release_tag

        if self._os_name == "macos":
            if self._arch != "arm64":
                raise RuntimeError("Only arm64 is supported on macOS")
            return f"llama-{tag}-bin-macos-arm64.tar.gz"

        if self._os_name == "linux":
            return f"llama-{tag}-bin-ubuntu-{self._arch}.tar.gz"

        if self._os_name == "windows":
            if self._backend == "cuda":
                cuda_ver = self._detect_cuda() or "12.4"
                return f"llama-{tag}-bin-win-cuda-{cuda_ver}-x64.zip"
            return f"llama-{tag}-bin-win-cpu-{self._arch}.zip"

        raise RuntimeError(f"Unsupported OS: {self._os_name}")

    def _cleanup_at_exit(self) -> None:
        """Clean up server at process exit."""
        with suppress(Exception):
            self.force_stop_server()
