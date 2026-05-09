# -*- coding: utf-8 -*-
"""Anthropic LLM provider implementation.

This module provides a concrete Provider implementation for the Anthropic API.
"""

from __future__ import annotations

import logging
from typing import Any, List

import httpx

from .base import ModelInfo, Provider

logger = logging.getLogger(__name__)

ANTHROPIC_DEFAULT_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"

BUILTIN_ANTHROPIC_MODELS = [
    ModelInfo(id="claude-3-5-sonnet-20241022", name="Claude 3.5 Sonnet", supports_multimodal=True, supports_image=True),
    ModelInfo(id="claude-3-5-haiku-20241022", name="Claude 3.5 Haiku", supports_multimodal=True, supports_image=True),
    ModelInfo(id="claude-3-opus-20240229", name="Claude 3 Opus", supports_multimodal=True, supports_image=True),
    ModelInfo(id="claude-3-sonnet-20240229", name="Claude 3 Sonnet", supports_multimodal=True, supports_image=True),
    ModelInfo(id="claude-3-haiku-20240307", name="Claude 3 Haiku", supports_multimodal=True, supports_image=True),
]


class AnthropicCompatibleProvider(Provider):
    """Provider implementation for Anthropic API.

    This provider works with:
    - Anthropic API
    - Any Anthropic-compatible API
    """

    def __init__(
        self,
        id: str = "anthropic",
        name: str = "Anthropic",
        base_url: str = ANTHROPIC_DEFAULT_URL,
        api_key: str = "",
        models: List[ModelInfo] | None = None,
        extra_models: List[ModelInfo] | None = None,
        api_key_prefix: str = "sk-ant-",
        is_local: bool = False,
        freeze_url: bool = True,
        require_api_key: bool = True,
        is_custom: bool = False,
        support_model_discovery: bool = True,
        chat_model: str = "AnthropicChatModel",
        generate_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            id=id,
            name=name,
            base_url=base_url,
            api_key=api_key,
            models=models if models is not None else list(BUILTIN_ANTHROPIC_MODELS),
            extra_models=extra_models if extra_models is not None else [],
            api_key_prefix=api_key_prefix,
            is_local=is_local,
            freeze_url=freeze_url,
            require_api_key=require_api_key,
            is_custom=is_custom,
            support_model_discovery=support_model_discovery,
            chat_model=chat_model,
            generate_kwargs=generate_kwargs if generate_kwargs is not None else {},
        )

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for Anthropic API requests."""
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_API_VERSION,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        """Check if the provider is reachable.

        Args:
            timeout: Connection timeout in seconds.

        Returns:
            Tuple of (success, message).
        """
        url = f"{self.base_url.rstrip('/')}/v1/models"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=self._get_headers())
                if response.status_code == 200:
                    return True, "OK"
                elif response.status_code == 401:
                    return False, "Authentication failed - invalid API key"
                else:
                    return False, f"API error (status {response.status_code})"
        except httpx.TimeoutException:
            return False, f"Connection timeout to {self.base_url}"
        except httpx.ConnectError:
            return False, f"Cannot connect to {self.base_url}"
        except Exception as e:
            logger.exception("Unexpected error checking connection")
            return False, f"Unknown error: {str(e)}"

    async def fetch_models(self, timeout: float = 5) -> List[ModelInfo]:
        """Fetch available models from the provider.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            List of available models.
        """
        url = f"{self.base_url.rstrip('/')}/v1/models"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=self._get_headers())
                if response.status_code != 200:
                    return []

                data = response.json()
                return self._parse_models_response(data)
        except Exception:
            logger.exception("Failed to fetch models")
            return []

    def _parse_models_response(self, data: dict) -> List[ModelInfo]:
        """Parse models list from API response.

        Args:
            data: JSON response data from /models endpoint.

        Returns:
            List of ModelInfo objects.
        """
        models: List[ModelInfo] = []
        seen_ids: set[str] = set()

        model_list = data.get("data", [])
        for item in model_list:
            model_id = str(item.get("id", "")).strip()
            if not model_id or model_id in seen_ids:
                continue

            model_name = str(item.get("display_name", model_id)).strip() or model_id
            seen_ids.add(model_id)
            models.append(ModelInfo(id=model_id, name=model_name))

        return models

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
        model_id = (model_id or "").strip()
        if not model_id:
            return False, "Empty model ID"

        url = f"{self.base_url.rstrip('/')}/v1/messages"

        payload = {
            "model": model_id,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                )
                if response.status_code == 200:
                    return True, "OK"
                elif response.status_code == 404:
                    return False, f"Model '{model_id}' not found"
                elif response.status_code == 401:
                    return False, "Authentication failed"
                else:
                    return False, f"API error (status {response.status_code})"
        except httpx.TimeoutException:
            return False, f"Timeout connecting to model '{model_id}'"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def get_chat_model_instance(self, model_id: str) -> Any:
        """Create a chat model instance for the specified model.

        Args:
            model_id: ID of the model to use.

        Returns:
            A configuration dict for the chat model.
        """
        return {
            "model_id": model_id,
            "provider_id": self.id,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "generate_kwargs": self.generate_kwargs,
        }
