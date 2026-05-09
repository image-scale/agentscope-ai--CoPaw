# -*- coding: utf-8 -*-
"""Provider manager for coordinating multiple LLM providers.

This module provides a centralized manager for handling multiple LLM providers,
including built-in providers (OpenAI, Anthropic, Ollama) and custom user-defined
providers. It handles provider registration, activation, and persistence.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .base import ModelInfo, Provider, ProviderInfo
from .openai_compat import OpenAICompatibleProvider
from .anthropic_compat import AnthropicCompatibleProvider
from .ollama import OllamaProvider
from ..settings import SECRET_DIR

logger = logging.getLogger(__name__)


class ActiveModelConfig(BaseModel):
    """Configuration for the currently active model."""

    provider_id: str = Field(..., description="Provider ID")
    model: str = Field(..., description="Model ID")


class ProviderCoordinator:
    """Manager for coordinating multiple LLM providers.

    This class provides:
    - Registration of built-in providers
    - Support for custom providers
    - Active model management and persistence
    - Provider configuration updates
    """

    def __init__(self, secret_dir: Path | None = None) -> None:
        """Initialize the provider coordinator.

        Args:
            secret_dir: Directory for storing provider configurations.
                Defaults to SECRET_DIR from settings.
        """
        self._secret_dir = secret_dir or SECRET_DIR
        self._builtin_providers: Dict[str, Provider] = {}
        self._custom_providers: Dict[str, Provider] = {}
        self._active_model: ActiveModelConfig | None = None

        self._root_path = self._secret_dir / "providers"
        self._builtin_path = self._root_path / "builtin"
        self._custom_path = self._root_path / "custom"

        self._setup_storage()
        self._register_builtin_providers()
        self._load_persisted_state()

    def _setup_storage(self) -> None:
        """Create directory structure for provider storage."""
        for path in [self._root_path, self._builtin_path, self._custom_path]:
            path.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(path, 0o700)
            except Exception:
                pass

    def _register_builtin_providers(self) -> None:
        """Register all built-in providers."""
        self._add_builtin(OpenAICompatibleProvider(
            id="openai",
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key_prefix="sk-",
            freeze_url=True,
        ))

        self._add_builtin(AnthropicCompatibleProvider(
            id="anthropic",
            name="Anthropic",
            api_key_prefix="sk-ant-",
        ))

        self._add_builtin(OllamaProvider(
            id="ollama",
            name="Ollama",
        ))

        self._add_builtin(OpenAICompatibleProvider(
            id="copaw-local",
            name="CoPaw Local",
            is_local=True,
            require_api_key=False,
            models=[],
        ))

        self._add_builtin(OpenAICompatibleProvider(
            id="lmstudio",
            name="LM Studio",
            is_local=True,
            base_url="http://localhost:1234/v1",
            require_api_key=False,
            models=[],
        ))

    def _add_builtin(self, provider: Provider) -> None:
        """Add a built-in provider."""
        self._builtin_providers[provider.id] = provider

    def _load_persisted_state(self) -> None:
        """Load persisted provider configurations and active model."""
        for provider_id in self._builtin_providers:
            self._load_provider_config(provider_id, is_builtin=True)

        self._load_custom_providers()
        self._load_active_model()

    def _load_provider_config(self, provider_id: str, is_builtin: bool) -> None:
        """Load persisted configuration for a provider."""
        path = self._builtin_path if is_builtin else self._custom_path
        config_file = path / f"{provider_id}.json"

        if not config_file.exists():
            return

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            provider = self.get_provider(provider_id)
            if provider:
                provider.update_config(data)
        except Exception as e:
            logger.warning(f"Failed to load config for {provider_id}: {e}")

    def _load_custom_providers(self) -> None:
        """Load custom providers from storage."""
        if not self._custom_path.exists():
            return

        for config_file in self._custom_path.glob("*.json"):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                provider = self._create_provider_from_data(data)
                if provider:
                    self._custom_providers[provider.id] = provider
            except Exception as e:
                logger.warning(f"Failed to load custom provider from {config_file}: {e}")

    def _create_provider_from_data(self, data: Dict[str, Any]) -> Provider | None:
        """Create a provider instance from configuration data."""
        chat_model = data.get("chat_model", "OpenAIChatModel")
        provider_id = data.get("id", "")

        if not provider_id:
            return None

        if chat_model == "AnthropicChatModel":
            return AnthropicCompatibleProvider(**data)
        else:
            return OpenAICompatibleProvider(**data)

    def _load_active_model(self) -> None:
        """Load the active model configuration."""
        active_model_file = self._root_path / "active_model.json"

        if not active_model_file.exists():
            return

        try:
            with open(active_model_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._active_model = ActiveModelConfig.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load active model: {e}")

    def _save_provider(self, provider: Provider, is_builtin: bool) -> None:
        """Save provider configuration to disk."""
        path = self._builtin_path if is_builtin else self._custom_path
        config_file = path / f"{provider.id}.json"

        try:
            data = provider.model_dump(exclude={"models"} if is_builtin else set())
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save provider {provider.id}: {e}")

    def _save_active_model(self) -> None:
        """Save active model configuration to disk."""
        active_model_file = self._root_path / "active_model.json"

        try:
            if self._active_model:
                data = self._active_model.model_dump()
            else:
                data = {}

            with open(active_model_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save active model: {e}")

    def get_provider(self, provider_id: str) -> Provider | None:
        """Get a provider by ID.

        Args:
            provider_id: The provider ID to look up.

        Returns:
            The provider instance, or None if not found.
        """
        if provider_id in self._builtin_providers:
            return self._builtin_providers[provider_id]
        if provider_id in self._custom_providers:
            return self._custom_providers[provider_id]
        return None

    def list_providers(self) -> List[Provider]:
        """List all available providers.

        Returns:
            List of all providers (built-in and custom).
        """
        providers = list(self._builtin_providers.values())
        providers.extend(self._custom_providers.values())
        return providers

    async def list_provider_info(self) -> List[ProviderInfo]:
        """Get information for all providers.

        Returns:
            List of ProviderInfo for all providers.
        """
        infos = []
        for provider in self.list_providers():
            info = await provider.get_info()
            infos.append(info)
        return infos

    @property
    def active_model(self) -> ActiveModelConfig | None:
        """Get the currently active model configuration."""
        return self._active_model

    @active_model.setter
    def active_model(self, value: ActiveModelConfig | None) -> None:
        """Set the active model configuration."""
        self._active_model = value

    def update_provider(self, provider_id: str, config: Dict[str, Any]) -> bool:
        """Update a provider's configuration.

        Args:
            provider_id: The provider ID to update.
            config: Configuration values to update.

        Returns:
            True if updated successfully, False if provider not found.
        """
        provider = self.get_provider(provider_id)
        if not provider:
            return False

        provider.update_config(config)
        is_builtin = provider_id in self._builtin_providers
        self._save_provider(provider, is_builtin)
        return True

    async def activate_model(
        self,
        provider_id: str,
        model_id: str,
    ) -> tuple[bool, str]:
        """Activate a specific model.

        Args:
            provider_id: The provider ID.
            model_id: The model ID to activate.

        Returns:
            Tuple of (success, message).
        """
        provider = self.get_provider(provider_id)
        if not provider:
            raise ValueError(f"Provider '{provider_id}' not found")

        if not provider.has_model(model_id):
            raise ValueError(f"Model '{model_id}' not found in provider '{provider_id}'")

        self._active_model = ActiveModelConfig(
            provider_id=provider_id,
            model=model_id,
        )
        self._save_active_model()
        return True, "OK"

    def save_active_model(self, config: ActiveModelConfig | None) -> None:
        """Save the active model configuration.

        Args:
            config: The active model configuration to save.
        """
        self._active_model = config
        self._save_active_model()

    async def add_custom_provider(self, provider: Provider) -> Provider:
        """Add a custom provider.

        Args:
            provider: The provider to add.

        Returns:
            The added provider (with potentially modified ID if there was a conflict).
        """
        resolved_id = self._resolve_provider_id(provider.id)
        provider.id = resolved_id
        provider.is_custom = True

        self._custom_providers[resolved_id] = provider
        self._save_provider(provider, is_builtin=False)
        return provider

    def _resolve_provider_id(self, provider_id: str) -> str:
        """Resolve provider ID conflicts.

        Args:
            provider_id: The requested provider ID.

        Returns:
            A unique provider ID.
        """
        base_id = provider_id
        if base_id in self._builtin_providers:
            base_id = f"{base_id}-custom"

        resolved_id = base_id
        while resolved_id in self._builtin_providers or resolved_id in self._custom_providers:
            resolved_id = f"{resolved_id}-new"

        return resolved_id

    def remove_custom_provider(self, provider_id: str) -> bool:
        """Remove a custom provider.

        Args:
            provider_id: The provider ID to remove.

        Returns:
            True if removed, False if not found.
        """
        if provider_id not in self._custom_providers:
            return False

        del self._custom_providers[provider_id]

        config_file = self._custom_path / f"{provider_id}.json"
        if config_file.exists():
            try:
                os.remove(config_file)
            except Exception as e:
                logger.warning(f"Failed to remove provider file: {e}")

        return True

    def load_provider(
        self,
        provider_id: str,
        is_builtin: bool,
    ) -> Provider | None:
        """Load a provider configuration from disk.

        Args:
            provider_id: The provider ID to load.
            is_builtin: Whether this is a built-in provider.

        Returns:
            The loaded provider, or None if not found or invalid.
        """
        path = self._builtin_path if is_builtin else self._custom_path
        config_file = path / f"{provider_id}.json"

        if not config_file.exists():
            return None

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._create_provider_from_data(data)
        except Exception as e:
            logger.warning(f"Failed to load provider {provider_id}: {e}")
            return None
