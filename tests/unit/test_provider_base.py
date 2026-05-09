# -*- coding: utf-8 -*-
"""Tests for copaw.providers.base module - provider abstraction layer."""

from typing import Any, List

import pytest

from copaw.providers.base import ModelInfo, ProviderInfo, Provider


class ConcreteProvider(Provider):
    """Concrete implementation for testing the abstract Provider class."""

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        return True, "OK"

    async def fetch_models(self, timeout: float = 5) -> List[ModelInfo]:
        return self.models + self.extra_models

    async def check_model_connection(
        self,
        model_id: str,
        timeout: float = 5,
    ) -> tuple[bool, str]:
        if self.has_model(model_id):
            return True, "OK"
        return False, f"Model {model_id} not found"

    def get_chat_model_instance(self, model_id: str) -> Any:
        return {"model_id": model_id, "provider_id": self.id}


class TestModelInfo:
    """Tests for ModelInfo model."""

    def test_model_info_minimal(self):
        """ModelInfo can be created with just id and name."""
        model = ModelInfo(id="gpt-4", name="GPT-4")
        assert model.id == "gpt-4"
        assert model.name == "GPT-4"
        assert model.supports_multimodal is None
        assert model.supports_image is None
        assert model.supports_video is None
        assert model.probe_source is None

    def test_model_info_with_multimodal_flags(self):
        """ModelInfo stores multimodal capability flags."""
        model = ModelInfo(
            id="gpt-4-vision",
            name="GPT-4 Vision",
            supports_multimodal=True,
            supports_image=True,
            supports_video=False,
            probe_source="documentation",
        )
        assert model.supports_multimodal is True
        assert model.supports_image is True
        assert model.supports_video is False
        assert model.probe_source == "documentation"

    def test_model_info_serialization(self):
        """ModelInfo can be serialized and deserialized."""
        model = ModelInfo(id="claude-3", name="Claude 3", supports_image=True)
        data = model.model_dump()
        restored = ModelInfo.model_validate(data)
        assert restored == model


class TestProviderInfo:
    """Tests for ProviderInfo model."""

    def test_provider_info_minimal(self):
        """ProviderInfo can be created with just id and name."""
        info = ProviderInfo(id="openai", name="OpenAI")
        assert info.id == "openai"
        assert info.name == "OpenAI"
        assert info.base_url == ""
        assert info.api_key == ""
        assert info.models == []
        assert info.extra_models == []

    def test_provider_info_full(self):
        """ProviderInfo stores all provider configuration."""
        models = [ModelInfo(id="gpt-4", name="GPT-4")]
        extra_models = [ModelInfo(id="custom-model", name="Custom")]

        info = ProviderInfo(
            id="openai",
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key="sk-test123",
            chat_model="OpenAIChatModel",
            models=models,
            extra_models=extra_models,
            api_key_prefix="sk-",
            is_local=False,
            freeze_url=True,
            require_api_key=True,
            is_custom=False,
            support_model_discovery=True,
            generate_kwargs={"temperature": 0.7},
        )

        assert info.id == "openai"
        assert info.base_url == "https://api.openai.com/v1"
        assert info.api_key == "sk-test123"
        assert len(info.models) == 1
        assert len(info.extra_models) == 1
        assert info.freeze_url is True
        assert info.generate_kwargs == {"temperature": 0.7}


class TestProviderHasModel:
    """Tests for Provider.has_model method."""

    def test_has_model_in_models_list(self):
        """has_model returns True if model is in models list."""
        provider = ConcreteProvider(
            id="test",
            name="Test Provider",
            models=[ModelInfo(id="model-1", name="Model 1")],
        )
        assert provider.has_model("model-1") is True
        assert provider.has_model("nonexistent") is False

    def test_has_model_in_extra_models_list(self):
        """has_model returns True if model is in extra_models list."""
        provider = ConcreteProvider(
            id="test",
            name="Test Provider",
            extra_models=[ModelInfo(id="extra-model", name="Extra Model")],
        )
        assert provider.has_model("extra-model") is True

    def test_has_model_checks_both_lists(self):
        """has_model checks both models and extra_models."""
        provider = ConcreteProvider(
            id="test",
            name="Test Provider",
            models=[ModelInfo(id="model-1", name="Model 1")],
            extra_models=[ModelInfo(id="extra-1", name="Extra 1")],
        )
        assert provider.has_model("model-1") is True
        assert provider.has_model("extra-1") is True


class TestProviderUpdateConfig:
    """Tests for Provider.update_config method."""

    def test_update_config_updates_name(self):
        """update_config updates name from config dict."""
        provider = ConcreteProvider(id="test", name="Original")
        provider.update_config({"name": "Updated"})
        assert provider.name == "Updated"

    def test_update_config_updates_base_url(self):
        """update_config updates base_url when not frozen."""
        provider = ConcreteProvider(
            id="test",
            name="Test",
            base_url="https://old.url/v1",
            freeze_url=False,
        )
        provider.update_config({"base_url": "https://new.url/v1"})
        assert provider.base_url == "https://new.url/v1"

    def test_update_config_ignores_frozen_url(self):
        """update_config ignores base_url when freeze_url is True."""
        provider = ConcreteProvider(
            id="test",
            name="Test",
            base_url="https://frozen.url/v1",
            freeze_url=True,
        )
        provider.update_config({"base_url": "https://ignored.url/v1"})
        assert provider.base_url == "https://frozen.url/v1"

    def test_update_config_updates_api_key(self):
        """update_config updates api_key."""
        provider = ConcreteProvider(id="test", name="Test", api_key="old-key")
        provider.update_config({"api_key": "new-key"})
        assert provider.api_key == "new-key"

    def test_update_config_updates_extra_models(self):
        """update_config updates extra_models from dict data."""
        provider = ConcreteProvider(id="test", name="Test")
        provider.update_config({
            "extra_models": [
                {"id": "new-model", "name": "New Model"},
            ]
        })
        assert len(provider.extra_models) == 1
        assert provider.extra_models[0].id == "new-model"

    def test_update_config_ignores_none_values(self):
        """update_config ignores None values."""
        provider = ConcreteProvider(id="test", name="Original", api_key="key")
        provider.update_config({"name": None, "api_key": None})
        assert provider.name == "Original"
        assert provider.api_key == "key"


class TestProviderAddModel:
    """Tests for Provider.add_model method."""

    @pytest.mark.asyncio
    async def test_add_model_success(self):
        """add_model adds model to extra_models and returns success."""
        provider = ConcreteProvider(id="test", name="Test")
        new_model = ModelInfo(id="new-model", name="New Model")

        success, message = await provider.add_model(new_model)

        assert success is True
        assert message == ""
        assert len(provider.extra_models) == 1
        assert provider.extra_models[0].id == "new-model"

    @pytest.mark.asyncio
    async def test_add_model_duplicate_fails(self):
        """add_model returns failure if model already exists."""
        existing = ModelInfo(id="existing", name="Existing")
        provider = ConcreteProvider(
            id="test",
            name="Test",
            extra_models=[existing],
        )

        success, message = await provider.add_model(existing)

        assert success is False
        assert "already exists" in message
        assert len(provider.extra_models) == 1

    @pytest.mark.asyncio
    async def test_add_model_to_models_list(self):
        """add_model can add to models list with target parameter."""
        provider = ConcreteProvider(id="test", name="Test")
        new_model = ModelInfo(id="new-model", name="New Model")

        success, _ = await provider.add_model(new_model, target="models")

        assert success is True
        assert len(provider.models) == 1
        assert len(provider.extra_models) == 0

    @pytest.mark.asyncio
    async def test_add_model_invalid_target(self):
        """add_model returns failure for invalid target."""
        provider = ConcreteProvider(id="test", name="Test")
        new_model = ModelInfo(id="new-model", name="New Model")

        success, message = await provider.add_model(new_model, target="invalid")

        assert success is False
        assert "Invalid target" in message


class TestProviderDeleteModel:
    """Tests for Provider.delete_model method."""

    @pytest.mark.asyncio
    async def test_delete_model_removes_from_extra_models(self):
        """delete_model removes model from extra_models."""
        models = [
            ModelInfo(id="model-1", name="Model 1"),
            ModelInfo(id="model-2", name="Model 2"),
        ]
        provider = ConcreteProvider(id="test", name="Test", extra_models=models)

        success, _ = await provider.delete_model("model-1")

        assert success is True
        assert len(provider.extra_models) == 1
        assert provider.extra_models[0].id == "model-2"

    @pytest.mark.asyncio
    async def test_delete_model_nonexistent_is_safe(self):
        """delete_model succeeds even for nonexistent model."""
        provider = ConcreteProvider(id="test", name="Test")

        success, _ = await provider.delete_model("nonexistent")

        assert success is True


class TestProviderGetInfo:
    """Tests for Provider.get_info method."""

    @pytest.mark.asyncio
    async def test_get_info_masks_api_key_by_default(self):
        """get_info masks api_key when mock_secret=True."""
        provider = ConcreteProvider(
            id="test",
            name="Test",
            api_key="sk-supersecret",
            api_key_prefix="sk-",
        )

        info = await provider.get_info(mock_secret=True)

        assert info.api_key == "sk-******"
        assert "supersecret" not in info.api_key

    @pytest.mark.asyncio
    async def test_get_info_shows_api_key_when_not_mocked(self):
        """get_info shows full api_key when mock_secret=False."""
        provider = ConcreteProvider(
            id="test",
            name="Test",
            api_key="sk-supersecret",
        )

        info = await provider.get_info(mock_secret=False)

        assert info.api_key == "sk-supersecret"

    @pytest.mark.asyncio
    async def test_get_info_empty_api_key_stays_empty(self):
        """get_info handles empty api_key correctly."""
        provider = ConcreteProvider(id="test", name="Test", api_key="")

        info = await provider.get_info(mock_secret=True)

        assert info.api_key == ""

    @pytest.mark.asyncio
    async def test_get_info_returns_provider_info(self):
        """get_info returns ProviderInfo with all fields."""
        models = [ModelInfo(id="m1", name="M1")]
        provider = ConcreteProvider(
            id="test",
            name="Test Provider",
            base_url="https://api.test.com/v1",
            models=models,
            is_local=True,
            support_model_discovery=True,
        )

        info = await provider.get_info()

        assert isinstance(info, ProviderInfo)
        assert info.id == "test"
        assert info.name == "Test Provider"
        assert info.base_url == "https://api.test.com/v1"
        assert len(info.models) == 1
        assert info.is_local is True
