# -*- coding: utf-8 -*-
"""Base classes and models for LLM provider abstraction.

This module defines the core abstractions for LLM providers, including:
- ModelInfo: Information about a specific model
- ProviderInfo: Configuration and metadata for a provider
- Provider: Abstract base class for all LLM providers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about a specific LLM model.

    Attributes:
        id: Model identifier used in API calls (e.g., "gpt-4", "claude-3").
        name: Human-readable display name for the model.
        supports_multimodal: Whether this model supports multimodal input.
        supports_image: Whether this model supports image input.
        supports_video: Whether this model supports video input.
        probe_source: How multimodal support was determined.
    """

    id: str = Field(..., description="Model identifier used in API calls")
    name: str = Field(..., description="Human-readable model name")
    supports_multimodal: bool | None = Field(
        default=None,
        description="Whether this model supports multimodal input",
    )
    supports_image: bool | None = Field(
        default=None,
        description="Whether this model supports image input",
    )
    supports_video: bool | None = Field(
        default=None,
        description="Whether this model supports video input",
    )
    probe_source: str | None = Field(
        default=None,
        description="Probe result source: 'documentation' or 'probed'",
    )


class ProviderInfo(BaseModel):
    """Configuration and metadata for an LLM provider.

    Attributes:
        id: Unique provider identifier.
        name: Human-readable provider name.
        base_url: API base URL for the provider.
        api_key: Authentication API key.
        chat_model: Name of the chat model class to use.
        models: List of pre-defined models for this provider.
        extra_models: List of user-added models.
        api_key_prefix: Expected prefix for the API key.
        is_local: Whether this is a local provider.
        freeze_url: Whether the base_url is frozen.
        require_api_key: Whether an API key is required.
        is_custom: Whether this is a user-created provider.
        support_model_discovery: Whether the provider supports model discovery.
        support_connection_check: Whether the provider supports connection checks.
        generate_kwargs: Additional generation parameters.
    """

    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Human-readable provider name")
    base_url: str = Field(default="", description="API base URL")
    api_key: str = Field(default="", description="API key for authentication")
    chat_model: str = Field(
        default="OpenAIChatModel",
        description="Chat model class name",
    )
    models: List[ModelInfo] = Field(
        default_factory=list,
        description="List of pre-defined models",
    )
    extra_models: List[ModelInfo] = Field(
        default_factory=list,
        description="List of user-added models",
    )
    api_key_prefix: str = Field(
        default="",
        description="Expected prefix for the API key",
    )
    is_local: bool = Field(
        default=False,
        description="Whether this is a local provider",
    )
    freeze_url: bool = Field(
        default=False,
        description="Whether the base_url should be frozen",
    )
    require_api_key: bool = Field(
        default=True,
        description="Whether an API key is required",
    )
    is_custom: bool = Field(
        default=False,
        description="Whether this is a user-created provider",
    )
    support_model_discovery: bool = Field(
        default=False,
        description="Whether model discovery is supported",
    )
    support_connection_check: bool = Field(
        default=True,
        description="Whether connection checking is supported",
    )
    generate_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Generation parameters",
    )


class Provider(ProviderInfo, ABC):
    """Abstract base class for LLM providers.

    Providers must implement methods for:
    - Connection checking
    - Model fetching
    - Model-specific connection checking
    - Chat model instantiation
    """

    @abstractmethod
    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        """Check if the provider is reachable.

        Args:
            timeout: Connection timeout in seconds.

        Returns:
            Tuple of (success, message). If success is True, the provider
            is reachable. Otherwise, message contains the error.
        """

    @abstractmethod
    async def fetch_models(self, timeout: float = 5) -> List[ModelInfo]:
        """Fetch available models from the provider.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            List of available models.
        """

    @abstractmethod
    async def check_model_connection(
        self,
        model_id: str,
        timeout: float = 5,
    ) -> tuple[bool, str]:
        """Check if a specific model is accessible.

        Args:
            model_id: ID of the model to check.
            timeout: Request timeout in seconds.

        Returns:
            Tuple of (success, message).
        """

    @abstractmethod
    def get_chat_model_instance(self, model_id: str) -> Any:
        """Create a chat model instance for the specified model.

        Args:
            model_id: ID of the model to use.

        Returns:
            A chat model instance configured for this provider and model.
        """

    def has_model(self, model_id: str) -> bool:
        """Check if the provider has a model with the given ID.

        Args:
            model_id: Model identifier to check.

        Returns:
            True if the model exists in models or extra_models.
        """
        all_models = self.models + self.extra_models
        return any(model.id == model_id for model in all_models)

    def update_config(self, config: dict[str, Any]) -> None:
        """Update provider configuration from a dictionary.

        Args:
            config: Dictionary containing configuration values to update.
                Supported keys: name, base_url, api_key, chat_model,
                api_key_prefix, generate_kwargs, extra_models.
        """
        if "name" in config and config["name"] is not None:
            self.name = str(config["name"])

        if (
            not self.freeze_url
            and "base_url" in config
            and config["base_url"] is not None
        ):
            self.base_url = str(config["base_url"])

        if "api_key" in config and config["api_key"] is not None:
            self.api_key = str(config["api_key"])

        if (
            self.is_custom
            and "chat_model" in config
            and config["chat_model"] is not None
        ):
            self.chat_model = str(config["chat_model"])

        if "api_key_prefix" in config and config["api_key_prefix"] is not None:
            self.api_key_prefix = str(config["api_key_prefix"])

        if (
            "generate_kwargs" in config
            and config["generate_kwargs"] is not None
            and isinstance(config["generate_kwargs"], dict)
        ):
            self.generate_kwargs = config["generate_kwargs"]

        if "extra_models" in config and config["extra_models"] is not None:
            self.extra_models = [
                model
                if isinstance(model, ModelInfo)
                else ModelInfo.model_validate(model)
                for model in config["extra_models"]
            ]

    async def add_model(
        self,
        model_info: ModelInfo,
        target: str = "extra_models",
        timeout: float = 10,
    ) -> tuple[bool, str]:
        """Add a model to the provider's model list.

        Args:
            model_info: Information about the model to add.
            target: Which list to add to ("extra_models" or "models").
            timeout: Operation timeout in seconds.

        Returns:
            Tuple of (success, message). If success is False, message
            contains the error reason.
        """
        existing_ids = {model.id for model in self.models + self.extra_models}
        if model_info.id in existing_ids:
            return False, f"Model '{model_info.id}' already exists"

        if target == "extra_models":
            self.extra_models.append(model_info)
        elif target == "models":
            self.models.append(model_info)
        else:
            return False, f"Invalid target '{target}' for adding model"

        return True, ""

    async def delete_model(
        self,
        model_id: str,
        timeout: float = 10,
    ) -> tuple[bool, str]:
        """Delete a model from the provider's extra_models list.

        Args:
            model_id: ID of the model to delete.
            timeout: Operation timeout in seconds.

        Returns:
            Tuple of (success, message).
        """
        self.extra_models = [
            model for model in self.extra_models if model.id != model_id
        ]
        return True, ""

    async def get_info(self, mock_secret: bool = True) -> ProviderInfo:
        """Get provider information with optionally masked API key.

        Args:
            mock_secret: If True, mask the API key in the output.

        Returns:
            ProviderInfo with the provider's configuration.
        """
        if mock_secret and self.api_key:
            api_key = self.api_key_prefix + "*" * 6
        else:
            api_key = self.api_key

        return ProviderInfo(
            id=self.id,
            name=self.name,
            base_url=self.base_url,
            api_key=api_key,
            chat_model=self.chat_model,
            models=self.models,
            extra_models=self.extra_models,
            api_key_prefix=self.api_key_prefix,
            is_local=self.is_local,
            is_custom=self.is_custom,
            support_model_discovery=self.support_model_discovery,
            support_connection_check=self.support_connection_check and not self.is_custom,
            freeze_url=self.freeze_url,
            require_api_key=self.require_api_key,
            generate_kwargs=self.generate_kwargs,
        )
