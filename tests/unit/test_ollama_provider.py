# -*- coding: utf-8 -*-
"""Tests for copaw.providers.ollama module - Ollama provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from copaw.providers.ollama import OllamaProvider, OLLAMA_DEFAULT_URL


class TestOllamaProviderInit:
    """Tests for OllamaProvider initialization."""

    def test_default_initialization(self):
        """Provider initializes with default Ollama settings."""
        provider = OllamaProvider()
        assert provider.id == "ollama"
        assert provider.name == "Ollama"
        assert provider.base_url == OLLAMA_DEFAULT_URL
        assert provider.is_local is True
        assert provider.require_api_key is False
        assert len(provider.models) == 0

    def test_custom_initialization(self):
        """Provider accepts custom configuration."""
        provider = OllamaProvider(
            base_url="http://custom:11434",
            api_key="custom-key",
        )
        assert provider.base_url == "http://custom:11434"
        assert provider.api_key == "custom-key"

    def test_is_local_by_default(self):
        """Ollama provider is marked as local by default."""
        provider = OllamaProvider()
        assert provider.is_local is True

    def test_no_api_key_required(self):
        """Ollama does not require API key by default."""
        provider = OllamaProvider()
        assert provider.require_api_key is False


class TestOllamaProviderCheckConnection:
    """Tests for OllamaProvider.check_connection method."""

    @pytest.mark.asyncio
    async def test_check_connection_success(self):
        """check_connection returns (True, 'OK') when Ollama is running."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            success, message = await provider.check_connection()

            assert success is True
            assert message == "OK"

    @pytest.mark.asyncio
    async def test_check_connection_ollama_not_running(self):
        """check_connection returns error when Ollama is not running."""
        provider = OllamaProvider()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            success, message = await provider.check_connection()

            assert success is False
            assert "Cannot connect" in message
            assert "Ollama running" in message

    @pytest.mark.asyncio
    async def test_check_connection_timeout(self):
        """check_connection handles timeout."""
        provider = OllamaProvider()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            success, message = await provider.check_connection()

            assert success is False
            assert "timeout" in message.lower()


class TestOllamaProviderFetchModels:
    """Tests for OllamaProvider.fetch_models method."""

    @pytest.mark.asyncio
    async def test_fetch_models_success(self):
        """fetch_models returns list of local models."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:latest"},
                {"name": "mistral:7b"},
                {"name": "phi3:mini"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            models = await provider.fetch_models()

            assert len(models) == 3
            assert models[0].id == "llama3:latest"
            assert models[1].id == "mistral:7b"
            assert models[2].id == "phi3:mini"

    @pytest.mark.asyncio
    async def test_fetch_models_formats_name(self):
        """fetch_models creates human-readable names."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3:latest"}]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            models = await provider.fetch_models()

            assert models[0].name == "Llama3 Latest"

    @pytest.mark.asyncio
    async def test_fetch_models_empty_when_none(self):
        """fetch_models returns empty list when no models."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            models = await provider.fetch_models()

            assert models == []

    @pytest.mark.asyncio
    async def test_fetch_models_failure(self):
        """fetch_models returns empty list on failure."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            models = await provider.fetch_models()

            assert models == []


class TestOllamaProviderCheckModelConnection:
    """Tests for OllamaProvider.check_model_connection method."""

    @pytest.mark.asyncio
    async def test_check_model_connection_success(self):
        """check_model_connection returns success for existing model."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            success, message = await provider.check_model_connection("llama3:latest")

            assert success is True
            assert message == "OK"

    @pytest.mark.asyncio
    async def test_check_model_connection_not_found(self):
        """check_model_connection returns error for missing model."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            success, message = await provider.check_model_connection("unknown")

            assert success is False
            assert "not found" in message.lower()

    @pytest.mark.asyncio
    async def test_check_model_connection_empty_id(self):
        """check_model_connection fails for empty model ID."""
        provider = OllamaProvider()

        success, message = await provider.check_model_connection("")

        assert success is False
        assert "Empty model ID" in message


class TestOllamaProviderGetChatModelInstance:
    """Tests for OllamaProvider.get_chat_model_instance."""

    def test_get_chat_model_instance_returns_config(self):
        """get_chat_model_instance returns configuration."""
        provider = OllamaProvider(base_url="http://localhost:11434")

        instance = provider.get_chat_model_instance("llama3:latest")

        assert instance["model_id"] == "llama3:latest"
        assert instance["provider_id"] == "ollama"
        assert instance["base_url"] == "http://localhost:11434/v1"

    def test_get_chat_model_instance_default_api_key(self):
        """get_chat_model_instance uses 'ollama' as default api_key."""
        provider = OllamaProvider()

        instance = provider.get_chat_model_instance("llama3")

        assert instance["api_key"] == "ollama"

    def test_get_openai_compatible_url(self):
        """get_openai_compatible_url returns /v1 endpoint."""
        provider = OllamaProvider(base_url="http://localhost:11434")

        url = provider.get_openai_compatible_url()

        assert url == "http://localhost:11434/v1"
