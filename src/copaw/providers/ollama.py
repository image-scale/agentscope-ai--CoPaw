# -*- coding: utf-8 -*-
"""Ollama LLM provider implementation.

This module provides a Provider implementation for local Ollama servers.
Ollama is a local model hosting solution that exposes an OpenAI-compatible
API for chat completions.
"""

from __future__ import annotations

import logging
from typing import Any, List

import httpx

from .base import ModelInfo, Provider

logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_URL = "http://localhost:11434"


class OllamaProvider(Provider):
    """Provider implementation for local Ollama server.

    Ollama runs locally and provides access to open-source models
    like Llama, Mistral, Phi, etc.
    """

    def __init__(
        self,
        id: str = "ollama",
        name: str = "Ollama",
        base_url: str = OLLAMA_DEFAULT_URL,
        api_key: str = "",
        models: List[ModelInfo] | None = None,
        extra_models: List[ModelInfo] | None = None,
        api_key_prefix: str = "",
        is_local: bool = True,
        freeze_url: bool = False,
        require_api_key: bool = False,
        is_custom: bool = False,
        support_model_discovery: bool = True,
        chat_model: str = "OpenAIChatModel",
        generate_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            id=id,
            name=name,
            base_url=base_url,
            api_key=api_key,
            models=models if models is not None else [],
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
        """Get HTTP headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        """Check if Ollama server is reachable.

        Args:
            timeout: Connection timeout in seconds.

        Returns:
            Tuple of (success, message).
        """
        url = f"{self.base_url.rstrip('/')}/api/tags"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return True, "OK"
                else:
                    return False, f"Ollama server error (status {response.status_code})"
        except httpx.TimeoutException:
            return False, f"Connection timeout to Ollama at {self.base_url}"
        except httpx.ConnectError:
            return False, f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
        except Exception as e:
            logger.exception("Unexpected error checking Ollama connection")
            return False, f"Unknown error: {str(e)}"

    async def fetch_models(self, timeout: float = 5) -> List[ModelInfo]:
        """Fetch available models from Ollama server.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            List of available models.
        """
        url = f"{self.base_url.rstrip('/')}/api/tags"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    return []

                data = response.json()
                return self._parse_models_response(data)
        except Exception:
            logger.exception("Failed to fetch Ollama models")
            return []

    def _parse_models_response(self, data: dict) -> List[ModelInfo]:
        """Parse models list from Ollama API response.

        Args:
            data: JSON response data from /api/tags endpoint.

        Returns:
            List of ModelInfo objects.
        """
        models: List[ModelInfo] = []
        seen_ids: set[str] = set()

        model_list = data.get("models", [])
        for item in model_list:
            model_name = str(item.get("name", "")).strip()
            if not model_name or model_name in seen_ids:
                continue

            seen_ids.add(model_name)
            display_name = model_name.replace(":", " ").title()
            models.append(ModelInfo(id=model_name, name=display_name))

        return models

    async def check_model_connection(
        self,
        model_id: str,
        timeout: float = 5,
    ) -> tuple[bool, str]:
        """Check if a specific model is accessible in Ollama.

        Args:
            model_id: ID (name) of the model to check.
            timeout: Request timeout in seconds.

        Returns:
            Tuple of (success, message).
        """
        model_id = (model_id or "").strip()
        if not model_id:
            return False, "Empty model ID"

        url = f"{self.base_url.rstrip('/')}/api/show"
        payload = {"name": model_id}

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    return True, "OK"
                elif response.status_code == 404:
                    return False, f"Model '{model_id}' not found in Ollama"
                else:
                    return False, f"Ollama error (status {response.status_code})"
        except httpx.TimeoutException:
            return False, f"Timeout checking model '{model_id}'"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def get_chat_model_instance(self, model_id: str) -> Any:
        """Create a chat model instance configuration.

        Args:
            model_id: ID of the model to use.

        Returns:
            A configuration dict for the chat model.
        """
        return {
            "model_id": model_id,
            "provider_id": self.id,
            "base_url": f"{self.base_url.rstrip('/')}/v1",
            "api_key": self.api_key or "ollama",
            "generate_kwargs": self.generate_kwargs,
        }

    def get_openai_compatible_url(self) -> str:
        """Get the OpenAI-compatible API URL.

        Returns:
            The /v1 endpoint URL for OpenAI-compatible requests.
        """
        return f"{self.base_url.rstrip('/')}/v1"
