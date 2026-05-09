# -*- coding: utf-8 -*-
"""Tests for copaw.providers.openai_compat module - OpenAI-compatible provider."""

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import httpx

from copaw.providers.openai_compat import (
    OpenAICompatibleProvider,
    OPENAI_DEFAULT_URL,
    BUILTIN_OPENAI_MODELS,
)
from copaw.providers.base import ModelInfo


class TestOpenAIProviderInit:
    """Tests for OpenAICompatibleProvider initialization."""

    def test_default_initialization(self):
        """Provider initializes with default OpenAI settings."""
        provider = OpenAICompatibleProvider()
        assert provider.id == "openai"
        assert provider.name == "OpenAI"
        assert provider.base_url == OPENAI_DEFAULT_URL
        assert provider.api_key_prefix == "sk-"
        assert len(provider.models) == len(BUILTIN_OPENAI_MODELS)

    def test_custom_initialization(self):
        """Provider accepts custom configuration."""
        provider = OpenAICompatibleProvider(
            id="custom-openai",
            name="Custom OpenAI",
            base_url="https://custom.api.com/v1",
            api_key="sk-test123",
            is_local=True,
        )
        assert provider.id == "custom-openai"
        assert provider.name == "Custom OpenAI"
        assert provider.base_url == "https://custom.api.com/v1"
        assert provider.api_key == "sk-test123"
        assert provider.is_local is True

    def test_builtin_models_include_gpt4(self):
        """Built-in models include GPT-4 variants."""
        provider = OpenAICompatibleProvider()
        model_ids = [m.id for m in provider.models]
        assert "gpt-4o" in model_ids
        assert "gpt-4" in model_ids
        assert "gpt-3.5-turbo" in model_ids

    def test_custom_base_url_supported(self):
        """Provider supports custom base URLs for compatible APIs."""
        provider = OpenAICompatibleProvider(
            base_url="https://api.dashscope.com/v1",
        )
        assert provider.base_url == "https://api.dashscope.com/v1"


class TestOpenAIProviderCheckConnection:
    """Tests for OpenAICompatibleProvider.check_connection method."""

    @pytest.mark.asyncio
    async def test_check_connection_success(self):
        """check_connection returns (True, 'OK') on successful connection."""
        provider = OpenAICompatibleProvider(api_key="sk-test")

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
    async def test_check_connection_auth_failure(self):
        """check_connection returns (False, error) on authentication failure."""
        provider = OpenAICompatibleProvider(api_key="invalid-key")

        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            success, message = await provider.check_connection()

            assert success is False
            assert "Authentication failed" in message

    @pytest.mark.asyncio
    async def test_check_connection_timeout(self):
        """check_connection returns (False, error) on timeout."""
        provider = OpenAICompatibleProvider(api_key="sk-test")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            success, message = await provider.check_connection()

            assert success is False
            assert "timeout" in message.lower()

    @pytest.mark.asyncio
    async def test_check_connection_connect_error(self):
        """check_connection returns (False, error) on connection error."""
        provider = OpenAICompatibleProvider(
            base_url="https://invalid.example.com/v1",
            api_key="sk-test",
        )

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


class TestOpenAIProviderFetchModels:
    """Tests for OpenAICompatibleProvider.fetch_models method."""

    @pytest.mark.asyncio
    async def test_fetch_models_success(self):
        """fetch_models returns list of models from API."""
        provider = OpenAICompatibleProvider(api_key="sk-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4", "name": "GPT-4"},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            models = await provider.fetch_models()

            assert len(models) == 2
            assert models[0].id == "gpt-4"
            assert models[1].id == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_fetch_models_deduplicates(self):
        """fetch_models removes duplicate model IDs."""
        provider = OpenAICompatibleProvider(api_key="sk-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4", "name": "GPT-4"},
                {"id": "gpt-4", "name": "GPT-4 Duplicate"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            models = await provider.fetch_models()

            assert len(models) == 1
            assert models[0].id == "gpt-4"

    @pytest.mark.asyncio
    async def test_fetch_models_failure_returns_empty(self):
        """fetch_models returns empty list on API failure."""
        provider = OpenAICompatibleProvider(api_key="sk-test")

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


class TestOpenAIProviderCheckModelConnection:
    """Tests for OpenAICompatibleProvider.check_model_connection method."""

    @pytest.mark.asyncio
    async def test_check_model_connection_success(self):
        """check_model_connection returns (True, 'OK') for valid model."""
        provider = OpenAICompatibleProvider(api_key="sk-test")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            success, message = await provider.check_model_connection("gpt-4")

            assert success is True
            assert message == "OK"

    @pytest.mark.asyncio
    async def test_check_model_connection_not_found(self):
        """check_model_connection returns (False, error) for unknown model."""
        provider = OpenAICompatibleProvider(api_key="sk-test")

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            success, message = await provider.check_model_connection("unknown-model")

            assert success is False
            assert "not found" in message.lower()

    @pytest.mark.asyncio
    async def test_check_model_connection_empty_id(self):
        """check_model_connection returns (False, error) for empty model ID."""
        provider = OpenAICompatibleProvider(api_key="sk-test")

        success, message = await provider.check_model_connection("")

        assert success is False
        assert "Empty model ID" in message

    @pytest.mark.asyncio
    async def test_check_model_connection_whitespace_id(self):
        """check_model_connection handles whitespace-only model ID."""
        provider = OpenAICompatibleProvider(api_key="sk-test")

        success, message = await provider.check_model_connection("   ")

        assert success is False
        assert "Empty model ID" in message


class TestOpenAIProviderGetChatModelInstance:
    """Tests for OpenAICompatibleProvider.get_chat_model_instance method."""

    def test_get_chat_model_instance_returns_config(self):
        """get_chat_model_instance returns model configuration."""
        provider = OpenAICompatibleProvider(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            generate_kwargs={"temperature": 0.7},
        )

        instance = provider.get_chat_model_instance("gpt-4")

        assert instance["model_id"] == "gpt-4"
        assert instance["provider_id"] == "openai"
        assert instance["base_url"] == "https://api.openai.com/v1"
        assert instance["api_key"] == "sk-test"
        assert instance["generate_kwargs"]["temperature"] == 0.7

    def test_get_chat_model_instance_uses_provider_settings(self):
        """get_chat_model_instance uses provider's base_url and api_key."""
        provider = OpenAICompatibleProvider(
            id="custom",
            api_key="custom-key",
            base_url="https://custom.api.com/v1",
        )

        instance = provider.get_chat_model_instance("custom-model")

        assert instance["base_url"] == "https://custom.api.com/v1"
        assert instance["api_key"] == "custom-key"
        assert instance["provider_id"] == "custom"
