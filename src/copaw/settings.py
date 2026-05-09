# -*- coding: utf-8 -*-
"""Environment variable loading and path configuration for CoPaw.

This module provides utilities for loading environment variables with type
safety and defaults, as well as defining all configuration paths and constants
used throughout the application.
"""
import os
from pathlib import Path


class EnvReader:
    """Utility to load and parse environment variables with type safety and defaults."""

    @staticmethod
    def get_bool(env_var: str, default: bool = False) -> bool:
        """Get a boolean environment variable, interpreting common truthy values.

        Args:
            env_var: Name of the environment variable to read.
            default: Default value if the variable is not set.

        Returns:
            True if the value is "true", "1", or "yes" (case-insensitive),
            otherwise False.
        """
        val = os.environ.get(env_var, str(default)).lower()
        return val in ("true", "1", "yes")

    @staticmethod
    def get_float(
        env_var: str,
        default: float = 0.0,
        min_value: float | None = None,
        max_value: float | None = None,
        allow_inf: bool = False,
    ) -> float:
        """Get a float environment variable with optional bounds and infinity handling.

        Args:
            env_var: Name of the environment variable to read.
            default: Default value if the variable is not set or invalid.
            min_value: Minimum allowed value (clamped if lower).
            max_value: Maximum allowed value (clamped if higher).
            allow_inf: Whether to allow infinity values.

        Returns:
            The parsed float value, clamped to bounds if specified.
        """
        try:
            value = float(os.environ.get(env_var, str(default)))
            if min_value is not None and value < min_value:
                return min_value
            if max_value is not None and value > max_value:
                return max_value
            if not allow_inf and (value == float("inf") or value == float("-inf")):
                return default
            return value
        except (TypeError, ValueError):
            return default

    @staticmethod
    def get_int(
        env_var: str,
        default: int = 0,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> int:
        """Get an integer environment variable with optional bounds.

        Args:
            env_var: Name of the environment variable to read.
            default: Default value if the variable is not set or invalid.
            min_value: Minimum allowed value (clamped if lower).
            max_value: Maximum allowed value (clamped if higher).

        Returns:
            The parsed integer value, clamped to bounds if specified.
        """
        try:
            value = int(os.environ.get(env_var, str(default)))
            if min_value is not None and value < min_value:
                return min_value
            if max_value is not None and value > max_value:
                return max_value
            return value
        except (TypeError, ValueError):
            return default

    @staticmethod
    def get_str(env_var: str, default: str = "") -> str:
        """Get a string environment variable with a default fallback.

        Args:
            env_var: Name of the environment variable to read.
            default: Default value if the variable is not set.

        Returns:
            The value of the environment variable, or the default.
        """
        return os.environ.get(env_var, default)


WORKING_DIR = (
    Path(EnvReader.get_str("COPAW_WORKING_DIR", "~/.copaw"))
    .expanduser()
    .resolve()
)

SECRET_DIR = (
    Path(EnvReader.get_str("COPAW_SECRET_DIR", f"{WORKING_DIR}.secret"))
    .expanduser()
    .resolve()
)

DEFAULT_MEDIA_DIR = WORKING_DIR / "media"
DEFAULT_LOCAL_PROVIDER_DIR = WORKING_DIR / "local_models"
MEMORY_DIR = WORKING_DIR / "memory"
MODELS_DIR = WORKING_DIR / "models"
CUSTOM_CHANNELS_DIR = WORKING_DIR / "custom_channels"

JOBS_FILE = EnvReader.get_str("COPAW_JOBS_FILE", "jobs.json")
CHATS_FILE = EnvReader.get_str("COPAW_CHATS_FILE", "chats.json")
CONFIG_FILE = EnvReader.get_str("COPAW_CONFIG_FILE", "config.json")
TOKEN_USAGE_FILE = EnvReader.get_str("COPAW_TOKEN_USAGE_FILE", "token_usage.json")
HEARTBEAT_FILE = EnvReader.get_str("COPAW_HEARTBEAT_FILE", "HEARTBEAT.md")
DEBUG_HISTORY_FILE = EnvReader.get_str("COPAW_DEBUG_HISTORY_FILE", "debug_history.jsonl")

HEARTBEAT_DEFAULT_EVERY = "6h"
HEARTBEAT_DEFAULT_TARGET = "main"
HEARTBEAT_TARGET_LAST = "last"
MAX_LOAD_HISTORY_COUNT = 10000

LOG_LEVEL_ENV = "COPAW_LOG_LEVEL"
RUNNING_IN_CONTAINER = EnvReader.get_bool("COPAW_RUNNING_IN_CONTAINER", False)

MODEL_PROVIDER_CHECK_TIMEOUT = EnvReader.get_float(
    "COPAW_MODEL_PROVIDER_CHECK_TIMEOUT",
    5.0,
    min_value=0,
    allow_inf=False,
)

PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH_ENV = "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"
DOCS_ENABLED = EnvReader.get_bool("COPAW_OPENAPI_DOCS", False)

MEMORY_COMPACT_KEEP_RECENT = EnvReader.get_int(
    "COPAW_MEMORY_COMPACT_KEEP_RECENT",
    3,
    min_value=0,
)

MEMORY_COMPACT_RATIO = EnvReader.get_float(
    "COPAW_MEMORY_COMPACT_RATIO",
    0.7,
    min_value=0,
    allow_inf=False,
)

DASHSCOPE_BASE_URL = EnvReader.get_str(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

CORS_ORIGINS = EnvReader.get_str("COPAW_CORS_ORIGINS", "").strip()

LLM_MAX_RETRIES = EnvReader.get_int("COPAW_LLM_MAX_RETRIES", 3, min_value=0)
LLM_BACKOFF_BASE = EnvReader.get_float("COPAW_LLM_BACKOFF_BASE", 1.0, min_value=0.1)
LLM_BACKOFF_CAP = EnvReader.get_float("COPAW_LLM_BACKOFF_CAP", 10.0, min_value=0.5)
LLM_MAX_CONCURRENT = EnvReader.get_int("COPAW_LLM_MAX_CONCURRENT", 10, min_value=1)
LLM_MAX_QPM = EnvReader.get_int("COPAW_LLM_MAX_QPM", 600, min_value=0)
LLM_RATE_LIMIT_PAUSE = EnvReader.get_float("COPAW_LLM_RATE_LIMIT_PAUSE", 5.0, min_value=1.0)
LLM_RATE_LIMIT_JITTER = EnvReader.get_float("COPAW_LLM_RATE_LIMIT_JITTER", 1.0, min_value=0.0)
LLM_ACQUIRE_TIMEOUT = EnvReader.get_float("COPAW_LLM_ACQUIRE_TIMEOUT", 300.0, min_value=10.0)

try:
    TOOL_GUARD_APPROVAL_TIMEOUT_SECONDS = max(
        float(os.environ.get("COPAW_TOOL_GUARD_APPROVAL_TIMEOUT_SECONDS", "600")),
        1.0,
    )
except (TypeError, ValueError):
    TOOL_GUARD_APPROVAL_TIMEOUT_SECONDS = 600.0

TRUNCATION_NOTICE_MARKER = "<<<TRUNCATED>>>"

BUILTIN_QA_AGENT_ID = "CoPaw_QA_Agent_0.1beta1"
BUILTIN_QA_AGENT_NAME = "QA Agent"
BUILTIN_QA_AGENT_SKILL_NAMES: tuple[str, ...] = ("guidance", "copaw_source_index")
