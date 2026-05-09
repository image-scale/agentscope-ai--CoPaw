# -*- coding: utf-8 -*-
"""Tests for copaw.providers.manager module - provider coordinator."""

import json
from pathlib import Path

import pytest

from copaw.providers.manager import ProviderCoordinator, ActiveModelConfig
from copaw.providers.openai_compat import OpenAICompatibleProvider
from copaw.providers.base import ModelInfo


@pytest.fixture
def temp_secret_dir(tmp_path):
    """Create a temporary secret directory for testing."""
    secret_dir = tmp_path / ".copaw.secret"
    secret_dir.mkdir(parents=True, exist_ok=True)
    return secret_dir


class TestProviderCoordinatorInit:
    """Tests for ProviderCoordinator initialization."""

    def test_creates_storage_directories(self, temp_secret_dir):
        """Coordinator creates storage directories on init."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        assert (temp_secret_dir / "providers").exists()
        assert (temp_secret_dir / "providers" / "builtin").exists()
        assert (temp_secret_dir / "providers" / "custom").exists()

    def test_registers_builtin_providers(self, temp_secret_dir):
        """Coordinator registers built-in providers."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        assert coordinator.get_provider("openai") is not None
        assert coordinator.get_provider("anthropic") is not None
        assert coordinator.get_provider("ollama") is not None

    def test_no_active_model_by_default(self, temp_secret_dir):
        """Coordinator has no active model by default."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        assert coordinator.active_model is None


class TestProviderCoordinatorGetProvider:
    """Tests for ProviderCoordinator.get_provider method."""

    def test_get_builtin_provider(self, temp_secret_dir):
        """get_provider returns built-in providers."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        provider = coordinator.get_provider("openai")

        assert provider is not None
        assert provider.id == "openai"
        assert provider.name == "OpenAI"

    def test_get_nonexistent_provider(self, temp_secret_dir):
        """get_provider returns None for unknown providers."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        provider = coordinator.get_provider("nonexistent")

        assert provider is None


class TestProviderCoordinatorListProviders:
    """Tests for ProviderCoordinator.list_providers method."""

    def test_list_providers_returns_all_builtin(self, temp_secret_dir):
        """list_providers returns all built-in providers."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        providers = coordinator.list_providers()

        provider_ids = [p.id for p in providers]
        assert "openai" in provider_ids
        assert "anthropic" in provider_ids
        assert "ollama" in provider_ids

    @pytest.mark.asyncio
    async def test_list_provider_info(self, temp_secret_dir):
        """list_provider_info returns ProviderInfo for all providers."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        infos = await coordinator.list_provider_info()

        assert len(infos) > 0
        assert all(hasattr(info, "id") for info in infos)


class TestProviderCoordinatorUpdateProvider:
    """Tests for ProviderCoordinator.update_provider method."""

    def test_update_provider_api_key(self, temp_secret_dir):
        """update_provider updates API key."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        result = coordinator.update_provider("openai", {"api_key": "sk-new-key"})

        assert result is True
        provider = coordinator.get_provider("openai")
        assert provider.api_key == "sk-new-key"

    def test_update_provider_persists_to_disk(self, temp_secret_dir):
        """update_provider persists changes to disk."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        coordinator.update_provider("openai", {"api_key": "sk-persisted"})

        config_file = temp_secret_dir / "providers" / "builtin" / "openai.json"
        assert config_file.exists()

        with open(config_file, "r") as f:
            data = json.load(f)
        assert data["api_key"] == "sk-persisted"

    def test_update_provider_unknown_returns_false(self, temp_secret_dir):
        """update_provider returns False for unknown providers."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        result = coordinator.update_provider("unknown", {"api_key": "key"})

        assert result is False


class TestProviderCoordinatorActivateModel:
    """Tests for ProviderCoordinator.activate_model method."""

    @pytest.mark.asyncio
    async def test_activate_model_success(self, temp_secret_dir):
        """activate_model sets active model configuration."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        success, message = await coordinator.activate_model("openai", "gpt-4o")

        assert success is True
        assert coordinator.active_model is not None
        assert coordinator.active_model.provider_id == "openai"
        assert coordinator.active_model.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_activate_model_persists(self, temp_secret_dir):
        """activate_model persists to disk."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        await coordinator.activate_model("openai", "gpt-4o")

        active_file = temp_secret_dir / "providers" / "active_model.json"
        assert active_file.exists()

        with open(active_file, "r") as f:
            data = json.load(f)
        assert data["provider_id"] == "openai"
        assert data["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_activate_model_invalid_provider(self, temp_secret_dir):
        """activate_model raises for invalid provider."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        with pytest.raises(ValueError, match="not found"):
            await coordinator.activate_model("invalid", "model")

    @pytest.mark.asyncio
    async def test_activate_model_invalid_model(self, temp_secret_dir):
        """activate_model raises for invalid model."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        with pytest.raises(ValueError, match="not found"):
            await coordinator.activate_model("openai", "nonexistent-model")


class TestProviderCoordinatorCustomProviders:
    """Tests for custom provider management."""

    @pytest.mark.asyncio
    async def test_add_custom_provider(self, temp_secret_dir):
        """add_custom_provider adds a new provider."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)
        custom = OpenAICompatibleProvider(
            id="custom-provider",
            name="Custom",
            base_url="https://custom.api.com/v1",
            api_key="sk-custom",
        )

        result = await coordinator.add_custom_provider(custom)

        assert result.id == "custom-provider"
        assert coordinator.get_provider("custom-provider") is not None

    @pytest.mark.asyncio
    async def test_add_custom_provider_resolves_conflicts(self, temp_secret_dir):
        """add_custom_provider resolves ID conflicts with built-ins."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)
        conflicting = OpenAICompatibleProvider(
            id="openai",
            name="Conflicting",
        )

        result = await coordinator.add_custom_provider(conflicting)

        assert result.id == "openai-custom"
        assert coordinator.get_provider("openai-custom") is not None

    @pytest.mark.asyncio
    async def test_add_custom_provider_persists(self, temp_secret_dir):
        """add_custom_provider persists to disk."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)
        custom = OpenAICompatibleProvider(
            id="persisted-custom",
            name="Persisted",
            api_key="sk-persisted",
        )

        await coordinator.add_custom_provider(custom)

        config_file = temp_secret_dir / "providers" / "custom" / "persisted-custom.json"
        assert config_file.exists()

    def test_remove_custom_provider(self, temp_secret_dir):
        """remove_custom_provider removes the provider."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)
        custom_path = temp_secret_dir / "providers" / "custom" / "to-remove.json"
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        custom_path.write_text(json.dumps({
            "id": "to-remove",
            "name": "To Remove",
            "base_url": "https://example.com/v1",
        }))

        coordinator2 = ProviderCoordinator(secret_dir=temp_secret_dir)
        result = coordinator2.remove_custom_provider("to-remove")

        assert result is True
        assert coordinator2.get_provider("to-remove") is None

    def test_remove_nonexistent_provider(self, temp_secret_dir):
        """remove_custom_provider returns False for unknown providers."""
        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        result = coordinator.remove_custom_provider("nonexistent")

        assert result is False


class TestProviderCoordinatorPersistence:
    """Tests for provider state persistence."""

    def test_loads_persisted_active_model(self, temp_secret_dir):
        """Coordinator loads persisted active model on init."""
        providers_dir = temp_secret_dir / "providers"
        providers_dir.mkdir(parents=True, exist_ok=True)
        active_file = providers_dir / "active_model.json"
        active_file.write_text(json.dumps({
            "provider_id": "openai",
            "model": "gpt-4",
        }))

        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        assert coordinator.active_model is not None
        assert coordinator.active_model.provider_id == "openai"
        assert coordinator.active_model.model == "gpt-4"

    def test_loads_persisted_provider_config(self, temp_secret_dir):
        """Coordinator loads persisted provider configs."""
        builtin_dir = temp_secret_dir / "providers" / "builtin"
        builtin_dir.mkdir(parents=True, exist_ok=True)
        openai_config = builtin_dir / "openai.json"
        openai_config.write_text(json.dumps({
            "api_key": "sk-persisted-key",
        }))

        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        provider = coordinator.get_provider("openai")
        assert provider.api_key == "sk-persisted-key"

    def test_loads_custom_providers_from_disk(self, temp_secret_dir):
        """Coordinator loads custom providers from disk."""
        custom_dir = temp_secret_dir / "providers" / "custom"
        custom_dir.mkdir(parents=True, exist_ok=True)
        custom_config = custom_dir / "my-custom.json"
        custom_config.write_text(json.dumps({
            "id": "my-custom",
            "name": "My Custom Provider",
            "base_url": "https://my.api.com/v1",
            "api_key": "sk-my-key",
            "chat_model": "OpenAIChatModel",
        }))

        coordinator = ProviderCoordinator(secret_dir=temp_secret_dir)

        provider = coordinator.get_provider("my-custom")
        assert provider is not None
        assert provider.name == "My Custom Provider"
        assert provider.api_key == "sk-my-key"


class TestActiveModelConfig:
    """Tests for ActiveModelConfig model."""

    def test_create_config(self):
        """ActiveModelConfig stores provider and model IDs."""
        config = ActiveModelConfig(provider_id="openai", model="gpt-4")

        assert config.provider_id == "openai"
        assert config.model == "gpt-4"

    def test_serialize_deserialize(self):
        """ActiveModelConfig can be serialized and deserialized."""
        config = ActiveModelConfig(provider_id="anthropic", model="claude-3")
        data = config.model_dump()
        restored = ActiveModelConfig.model_validate(data)

        assert restored == config
