# -*- coding: utf-8 -*-
"""Pydantic configuration models for CoPaw.

Provides configuration management for agents, channels, and runtime settings
with validation, serialization, and default value handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .settings import (
    WORKING_DIR,
    SECRET_DIR,
    LLM_MAX_RETRIES,
    LLM_BACKOFF_BASE,
    LLM_MAX_CONCURRENT,
)


# =============================================================================
# Channel Configurations
# =============================================================================


class BaseChannelConfig(BaseModel):
    """Base configuration for messaging channels."""

    enabled: bool = Field(default=False, description="Whether channel is enabled")
    bot_prefix: str = Field(default="", description="Prefix for bot commands")
    filter_tool_messages: bool = Field(
        default=False,
        description="Filter out tool execution messages",
    )
    filter_thinking: bool = Field(
        default=False,
        description="Filter out thinking/reasoning blocks",
    )


class ConsoleChannelConfig(BaseChannelConfig):
    """Configuration for console/terminal channel."""

    enabled: bool = True
    media_dir: str | None = Field(
        default=None,
        description="Directory for media files",
    )


class DiscordChannelConfig(BaseChannelConfig):
    """Configuration for Discord bot channel."""

    bot_token: str = Field(default="", description="Discord bot token")
    accept_bot_messages: bool = Field(
        default=False,
        description="Accept messages from other bots",
    )


class TelegramChannelConfig(BaseChannelConfig):
    """Configuration for Telegram bot channel."""

    bot_token: str = Field(default="", description="Telegram bot token")
    http_proxy: str = Field(default="", description="HTTP proxy URL")
    show_typing: bool | None = Field(
        default=None,
        description="Show typing indicator",
    )


class ChannelConfig(BaseModel):
    """Container for all channel configurations."""

    model_config = ConfigDict(extra="allow")

    console: ConsoleChannelConfig = Field(
        default_factory=ConsoleChannelConfig,
    )
    discord: DiscordChannelConfig = Field(
        default_factory=DiscordChannelConfig,
    )
    telegram: TelegramChannelConfig = Field(
        default_factory=TelegramChannelConfig,
    )


# =============================================================================
# Runtime Configurations
# =============================================================================


class ContextCompactConfig(BaseModel):
    """Configuration for context window compaction."""

    enabled: bool = Field(
        default=True,
        description="Enable automatic context compaction",
    )
    compact_ratio: float = Field(
        default=0.75,
        ge=0.3,
        le=0.9,
        description="Trigger compaction when context reaches this ratio",
    )
    reserve_ratio: float = Field(
        default=0.1,
        ge=0.05,
        le=0.3,
        description="Preserve this ratio of recent context after compaction",
    )


class RuntimeConfig(BaseModel):
    """Agent runtime behavior configuration."""

    model_config = ConfigDict(extra="ignore")

    max_iterations: int = Field(
        default=100,
        ge=1,
        description="Maximum reasoning iterations per turn",
    )
    max_input_length: int = Field(
        default=128 * 1024,
        ge=1000,
        description="Maximum context window size in tokens",
    )
    llm_retry_enabled: bool = Field(
        default=LLM_MAX_RETRIES > 0,
        description="Enable retry on transient LLM errors",
    )
    llm_max_retries: int = Field(
        default=max(LLM_MAX_RETRIES, 1),
        ge=1,
        description="Maximum LLM retry attempts",
    )
    llm_backoff_base: float = Field(
        default=LLM_BACKOFF_BASE,
        ge=0.1,
        description="Base delay for exponential backoff",
    )
    llm_backoff_cap: float = Field(
        default=30.0,
        ge=0.5,
        description="Maximum backoff delay",
    )
    llm_max_concurrent: int = Field(
        default=LLM_MAX_CONCURRENT,
        ge=1,
        description="Maximum concurrent LLM calls",
    )
    context_compact: ContextCompactConfig = Field(
        default_factory=ContextCompactConfig,
    )

    @model_validator(mode="after")
    def validate_backoff(self) -> RuntimeConfig:
        """Validate backoff cap >= base."""
        if self.llm_backoff_cap < self.llm_backoff_base:
            raise ValueError("llm_backoff_cap must be >= llm_backoff_base")
        return self

    @property
    def compact_threshold(self) -> int:
        """Token count that triggers compaction."""
        return int(self.max_input_length * self.context_compact.compact_ratio)

    @property
    def compact_reserve(self) -> int:
        """Token count to preserve after compaction."""
        return int(self.max_input_length * self.context_compact.reserve_ratio)


# =============================================================================
# Tool Configurations
# =============================================================================


class BuiltinToolConfig(BaseModel):
    """Configuration for a single built-in tool."""

    name: str = Field(..., description="Tool function name")
    enabled: bool = Field(default=True, description="Whether tool is enabled")
    description: str = Field(default="", description="Tool description")


class ToolsConfig(BaseModel):
    """Built-in tools management configuration."""

    builtin_tools: Dict[str, BuiltinToolConfig] = Field(
        default_factory=lambda: {
            "execute_shell_command": BuiltinToolConfig(
                name="execute_shell_command",
                description="Execute shell commands",
            ),
            "read_file": BuiltinToolConfig(
                name="read_file",
                description="Read file contents",
            ),
            "write_file": BuiltinToolConfig(
                name="write_file",
                description="Write content to file",
            ),
            "edit_file": BuiltinToolConfig(
                name="edit_file",
                description="Edit file using find-and-replace",
            ),
        },
    )


# =============================================================================
# Agent Configurations
# =============================================================================


class HeartbeatConfig(BaseModel):
    """Heartbeat configuration for scheduled agent execution."""

    enabled: bool = Field(default=False, description="Enable heartbeat")
    every: str = Field(default="1h", description="Heartbeat interval (e.g., 1h, 30m)")
    target: str = Field(
        default="HEARTBEAT.md",
        description="File to use as heartbeat prompt",
    )


class AgentProfileRef(BaseModel):
    """Reference to an agent's workspace (stored in root config)."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique agent ID")
    workspace_dir: str = Field(..., description="Path to agent workspace")
    enabled: bool = Field(default=True, description="Whether agent is enabled")


class AgentConfig(BaseModel):
    """Complete agent configuration (stored in workspace/agent.json)."""

    id: str = Field(..., description="Unique agent ID")
    name: str = Field(..., description="Human-readable agent name")
    description: str = Field(default="", description="Agent description")
    workspace_dir: str = Field(default="", description="Path to workspace")
    language: str = Field(default="en", description="Agent language")
    system_prompt_files: List[str] = Field(
        default_factory=lambda: ["AGENTS.md", "PROFILE.md"],
        description="System prompt markdown files",
    )
    channels: ChannelConfig | None = Field(
        default=None,
        description="Channel configurations",
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Runtime configuration",
    )
    tools: ToolsConfig | None = Field(
        default=None,
        description="Tools configuration",
    )
    heartbeat: HeartbeatConfig | None = Field(
        default=None,
        description="Heartbeat configuration",
    )


class AgentsConfig(BaseModel):
    """Multi-agent management configuration."""

    active_agent: str = Field(
        default="default",
        description="Currently active agent ID",
    )
    agent_order: List[str] = Field(
        default_factory=lambda: ["default"],
        description="UI display order for agents",
    )
    profiles: Dict[str, AgentProfileRef] = Field(
        default_factory=lambda: {
            "default": AgentProfileRef(
                id="default",
                workspace_dir=f"{WORKING_DIR}/workspaces/default",
            ),
        },
        description="Agent profile references",
    )


# =============================================================================
# Root Configuration
# =============================================================================


class Config(BaseModel):
    """Root configuration model (config.json)."""

    channels: ChannelConfig = Field(default_factory=ChannelConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    user_timezone: str = Field(default="UTC", description="User timezone")
    show_tool_details: bool = Field(
        default=True,
        description="Show tool execution details in output",
    )

    @classmethod
    def from_file(cls, path: Path) -> Config:
        """Load configuration from a JSON file.

        Args:
            path: Path to config file.

        Returns:
            Parsed Config instance.
        """
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_file(self, path: Path) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path to save config.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.model_dump(exclude_none=True),
                f,
                ensure_ascii=False,
                indent=2,
            )


# =============================================================================
# Configuration Utilities
# =============================================================================


def get_config_path() -> Path:
    """Get the default configuration file path."""
    return SECRET_DIR / "config.json"


def load_config() -> Config:
    """Load configuration from the default path.

    Returns:
        Config instance (empty defaults if file doesn't exist).
    """
    return Config.from_file(get_config_path())


def save_config(config: Config) -> None:
    """Save configuration to the default path.

    Args:
        config: Configuration to save.
    """
    config.to_file(get_config_path())


def get_agent_workspace(agent_id: str) -> Path:
    """Get the workspace directory for an agent.

    Args:
        agent_id: Agent ID.

    Returns:
        Path to the agent's workspace.

    Raises:
        ValueError: If agent not found.
    """
    config = load_config()
    if agent_id not in config.agents.profiles:
        raise ValueError(f"Agent '{agent_id}' not found")

    profile = config.agents.profiles[agent_id]
    return Path(profile.workspace_dir).expanduser()


def load_agent_config(agent_id: str) -> AgentConfig:
    """Load an agent's full configuration.

    Args:
        agent_id: Agent ID.

    Returns:
        AgentConfig instance.

    Raises:
        ValueError: If agent not found.
    """
    workspace = get_agent_workspace(agent_id)
    config_path = workspace / "agent.json"

    if not config_path.exists():
        return AgentConfig(
            id=agent_id,
            name=agent_id.title(),
            workspace_dir=str(workspace),
        )

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AgentConfig.model_validate(data)


def save_agent_config(agent_id: str, config: AgentConfig) -> None:
    """Save an agent's configuration.

    Args:
        agent_id: Agent ID.
        config: Configuration to save.

    Raises:
        ValueError: If agent not found.
    """
    workspace = get_agent_workspace(agent_id)
    workspace.mkdir(parents=True, exist_ok=True)

    config_path = workspace / "agent.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            config.model_dump(exclude_none=True),
            f,
            ensure_ascii=False,
            indent=2,
        )
