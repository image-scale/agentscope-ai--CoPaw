# -*- coding: utf-8 -*-
"""Tests for copaw.providers.anthropic_compat module - Anthropic provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from copaw.providers.anthropic_compat import (
    AnthropicCompatibleProvider,
    ANTHROPIC_DEFAULT_URL,
    BUILTIN_ANTHROPIC_MODELS,
)
from copaw.providers.base import ModelInfo


class TestAnthropicProviderInit:
    """Tests for AnthropicCompatibleProvider initialization."""

    def test_default_initialization(self):
        """Provider initializes with default Anthropic settings."""
        provider = AnthropicCompatibleProvider()
        assert provider.id == "anthropic"
        assert provider.name == "Anthropic"
        assert provider.base_url == ANTHROPIC_DEFAULT_URL
        assert provider.api_key_prefix == "sk-ant-"
        assert len(provider.models) == len(BUILTIN_ANTHROPIC_MODELS)

    def test_custom_initialization(self):
        """Provider accepts custom configuration."""
        provider = AnthropicCompatibleProvider(
            id="custom-anthropic",
            name="Custom Anthropic",
            base_url="https://custom.api.com",
            api_key="sk-ant-test123",
        )
        assert provider.id == "custom-anthropic"
        assert provider.name == "Custom Anthropic"
        assert provider.api_key == "sk-ant-test123"

    def test_builtin_models_include_claude(self):
        """Built-in models include Claude variants."""
        provider = AnthropicCompatibleProvider()
        model_ids = [m.id for m in provider.models]
        assert any("claude-3-5-sonnet" in m for m in model_ids)
        assert any("claude-3-opus" in m for m in model_ids)
        assert any("claude-3-haiku" in m for m in model_ids)

    def test_freeze_url_by_default(self):
        """Anthropic provider has frozen URL by default."""
        provider = AnthropicCompatibleProvider()
        assert provider.freeze_url is True


class TestAnthropicProviderCheckConnection:
    """Tests for AnthropicCompatibleProvider.check_connection method."""

    @pytest.mark.asyncio
    async def test_check_connection_success(self):
        """check_connection returns (True, 'OK') on successful connection."""
        provider = AnthropicCompatibleProvider(api_key="sk-ant-test")

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
        """check_connection returns (False, error) on auth failure."""
        provider = AnthropicCompatibleProvider(api_key="invalid-key")

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
        """check_connection handles timeout."""
        provider = AnthropicCompatibleProvider(api_key="sk-ant-test")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            success, message = await provider.check_connection()

            assert success is False
            assert "timeout" in message.lower()


class TestAnthropicProviderFetchModels:
    """Tests for AnthropicCompatibleProvider.fetch_models method."""

    @pytest.mark.asyncio
    async def test_fetch_models_success(self):
        """fetch_models returns list of models from API."""
        provider = AnthropicCompatibleProvider(api_key="sk-ant-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "claude-3-opus", "display_name": "Claude 3 Opus"},
                {"id": "claude-3-sonnet", "display_name": "Claude 3 Sonnet"},
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
            assert models[0].id == "claude-3-opus"
            assert models[0].name == "Claude 3 Opus"

    @pytest.mark.asyncio
    async def test_fetch_models_uses_display_name(self):
        """fetch_models uses display_name for model name."""
        provider = AnthropicCompatibleProvider(api_key="sk-ant-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "claude-3-opus-20240229", "display_name": "Claude 3 Opus"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            models = await provider.fetch_models()

            assert models[0].name == "Claude 3 Opus"

    @pytest.mark.asyncio
    async def test_fetch_models_failure_returns_empty(self):
        """fetch_models returns empty list on failure."""
        provider = AnthropicCompatibleProvider(api_key="sk-ant-test")

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


class TestAnthropicProviderCheckModelConnection:
    """Tests for AnthropicCompatibleProvider.check_model_connection method."""

    @pytest.mark.asyncio
    async def test_check_model_connection_success(self):
        """check_model_connection returns (True, 'OK') for valid model."""
        provider = AnthropicCompatibleProvider(api_key="sk-ant-test")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            success, message = await provider.check_model_connection("claude-3-opus")

            assert success is True
            assert message == "OK"

    @pytest.mark.asyncio
    async def test_check_model_connection_empty_id(self):
        """check_model_connection fails for empty model ID."""
        provider = AnthropicCompatibleProvider(api_key="sk-ant-test")

        success, message = await provider.check_model_connection("")

        assert success is False
        assert "Empty model ID" in message

    @pytest.mark.asyncio
    async def test_check_model_connection_not_found(self):
        """check_model_connection handles model not found."""
        provider = AnthropicCompatibleProvider(api_key="sk-ant-test")

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


class TestAnthropicProviderGetChatModelInstance:
    """Tests for AnthropicCompatibleProvider.get_chat_model_instance."""

    def test_get_chat_model_instance_returns_config(self):
        """get_chat_model_instance returns configuration dict."""
        provider = AnthropicCompatibleProvider(
            api_key="sk-ant-test",
            generate_kwargs={"max_tokens": 1000},
        )

        instance = provider.get_chat_model_instance("claude-3-opus")

        assert instance["model_id"] == "claude-3-opus"
        assert instance["provider_id"] == "anthropic"
        assert instance["api_key"] == "sk-ant-test"
        assert instance["generate_kwargs"]["max_tokens"] == 1000
