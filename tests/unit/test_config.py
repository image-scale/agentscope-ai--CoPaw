# -*- coding: utf-8 -*-
"""Tests for copaw.config module."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from copaw.config import (
    BaseChannelConfig,
    ConsoleChannelConfig,
    DiscordChannelConfig,
    TelegramChannelConfig,
    ChannelConfig,
    ContextCompactConfig,
    RuntimeConfig,
    BuiltinToolConfig,
    ToolsConfig,
    HeartbeatConfig,
    AgentProfileRef,
    AgentConfig,
    AgentsConfig,
    Config,
    get_config_path,
    load_config,
    save_config,
    get_agent_workspace,
    load_agent_config,
    save_agent_config,
)


class TestBaseChannelConfig:
    """Tests for BaseChannelConfig."""

    def test_default_values(self):
        """Base channel has correct defaults."""
        config = BaseChannelConfig()

        assert config.enabled is False
        assert config.bot_prefix == ""
        assert config.filter_tool_messages is False
        assert config.filter_thinking is False

    def test_custom_values(self):
        """Base channel accepts custom values."""
        config = BaseChannelConfig(
            enabled=True,
            bot_prefix="!",
            filter_tool_messages=True,
        )

        assert config.enabled is True
        assert config.bot_prefix == "!"
        assert config.filter_tool_messages is True


class TestChannelConfigs:
    """Tests for specific channel configurations."""

    def test_console_enabled_by_default(self):
        """Console channel is enabled by default."""
        config = ConsoleChannelConfig()

        assert config.enabled is True

    def test_discord_config(self):
        """Discord config stores token."""
        config = DiscordChannelConfig(
            enabled=True,
            bot_token="test-token",
        )

        assert config.bot_token == "test-token"
        assert config.accept_bot_messages is False

    def test_telegram_config(self):
        """Telegram config stores token and proxy."""
        config = TelegramChannelConfig(
            enabled=True,
            bot_token="test-token",
            http_proxy="http://proxy:8080",
        )

        assert config.bot_token == "test-token"
        assert config.http_proxy == "http://proxy:8080"


class TestChannelConfigContainer:
    """Tests for ChannelConfig container."""

    def test_default_channels(self):
        """ChannelConfig has all default channels."""
        config = ChannelConfig()

        assert hasattr(config, "console")
        assert hasattr(config, "discord")
        assert hasattr(config, "telegram")

    def test_allows_extra_fields(self):
        """ChannelConfig allows extra channel types."""
        config = ChannelConfig(custom_channel={"enabled": True})

        assert hasattr(config, "custom_channel")


class TestContextCompactConfig:
    """Tests for ContextCompactConfig."""

    def test_default_values(self):
        """Context compact has correct defaults."""
        config = ContextCompactConfig()

        assert config.enabled is True
        assert config.compact_ratio == 0.75
        assert config.reserve_ratio == 0.1

    def test_validates_compact_ratio_bounds(self):
        """Validates compact_ratio bounds."""
        with pytest.raises(ValueError):
            ContextCompactConfig(compact_ratio=0.2)

        with pytest.raises(ValueError):
            ContextCompactConfig(compact_ratio=0.95)

    def test_validates_reserve_ratio_bounds(self):
        """Validates reserve_ratio bounds."""
        with pytest.raises(ValueError):
            ContextCompactConfig(reserve_ratio=0.01)

        with pytest.raises(ValueError):
            ContextCompactConfig(reserve_ratio=0.5)


class TestRuntimeConfig:
    """Tests for RuntimeConfig."""

    def test_default_values(self):
        """Runtime config has correct defaults."""
        config = RuntimeConfig()

        assert config.max_iterations == 100
        assert config.max_input_length == 128 * 1024
        assert config.llm_max_retries >= 1

    def test_validates_backoff_relationship(self):
        """Validates backoff_cap >= backoff_base."""
        with pytest.raises(ValueError, match="backoff_cap"):
            RuntimeConfig(
                llm_backoff_base=10.0,
                llm_backoff_cap=5.0,
            )

    def test_compact_threshold_property(self):
        """compact_threshold calculates correctly."""
        config = RuntimeConfig(
            max_input_length=10000,
            context_compact=ContextCompactConfig(compact_ratio=0.5),
        )

        assert config.compact_threshold == 5000

    def test_compact_reserve_property(self):
        """compact_reserve calculates correctly."""
        config = RuntimeConfig(
            max_input_length=10000,
            context_compact=ContextCompactConfig(reserve_ratio=0.2),
        )

        assert config.compact_reserve == 2000


class TestBuiltinToolConfig:
    """Tests for BuiltinToolConfig."""

    def test_requires_name(self):
        """BuiltinToolConfig requires name."""
        with pytest.raises(ValueError):
            BuiltinToolConfig()

    def test_default_enabled(self):
        """Tool is enabled by default."""
        config = BuiltinToolConfig(name="test_tool")

        assert config.enabled is True


class TestToolsConfig:
    """Tests for ToolsConfig."""

    def test_has_default_tools(self):
        """ToolsConfig has default builtin tools."""
        config = ToolsConfig()

        assert "execute_shell_command" in config.builtin_tools
        assert "read_file" in config.builtin_tools
        assert "write_file" in config.builtin_tools
        assert "edit_file" in config.builtin_tools


class TestHeartbeatConfig:
    """Tests for HeartbeatConfig."""

    def test_default_disabled(self):
        """Heartbeat is disabled by default."""
        config = HeartbeatConfig()

        assert config.enabled is False
        assert config.every == "1h"
        assert config.target == "HEARTBEAT.md"

    def test_custom_values(self):
        """Heartbeat accepts custom values."""
        config = HeartbeatConfig(
            enabled=True,
            every="30m",
            target="CUSTOM.md",
        )

        assert config.enabled is True
        assert config.every == "30m"


class TestAgentProfileRef:
    """Tests for AgentProfileRef."""

    def test_requires_id_and_workspace(self):
        """AgentProfileRef requires id and workspace_dir."""
        with pytest.raises(ValueError):
            AgentProfileRef()

    def test_stores_reference(self):
        """AgentProfileRef stores reference data."""
        ref = AgentProfileRef(
            id="test-agent",
            workspace_dir="/path/to/workspace",
        )

        assert ref.id == "test-agent"
        assert ref.workspace_dir == "/path/to/workspace"
        assert ref.enabled is True


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_requires_id_and_name(self):
        """AgentConfig requires id and name."""
        with pytest.raises(ValueError):
            AgentConfig()

    def test_default_values(self):
        """AgentConfig has correct defaults."""
        config = AgentConfig(id="test", name="Test Agent")

        assert config.id == "test"
        assert config.name == "Test Agent"
        assert config.language == "en"
        assert "AGENTS.md" in config.system_prompt_files

    def test_nested_configs(self):
        """AgentConfig supports nested configurations."""
        config = AgentConfig(
            id="test",
            name="Test Agent",
            runtime=RuntimeConfig(max_iterations=50),
            heartbeat=HeartbeatConfig(enabled=True),
        )

        assert config.runtime.max_iterations == 50
        assert config.heartbeat.enabled is True


class TestAgentsConfig:
    """Tests for AgentsConfig."""

    def test_default_has_default_agent(self):
        """AgentsConfig has default agent."""
        config = AgentsConfig()

        assert config.active_agent == "default"
        assert "default" in config.profiles
        assert "default" in config.agent_order


class TestConfig:
    """Tests for root Config."""

    def test_default_values(self):
        """Config has correct defaults."""
        config = Config()

        assert config.user_timezone == "UTC"
        assert config.show_tool_details is True
        assert config.channels is not None
        assert config.agents is not None

    def test_from_file_missing(self, tmp_path):
        """from_file returns defaults for missing file."""
        config = Config.from_file(tmp_path / "missing.json")

        assert config.user_timezone == "UTC"

    def test_from_file_existing(self, tmp_path):
        """from_file loads existing config."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "user_timezone": "America/New_York",
            "show_tool_details": False,
        }))

        config = Config.from_file(config_path)

        assert config.user_timezone == "America/New_York"
        assert config.show_tool_details is False

    def test_to_file(self, tmp_path):
        """to_file saves config."""
        config = Config(user_timezone="Europe/London")
        config_path = tmp_path / "config.json"

        config.to_file(config_path)

        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["user_timezone"] == "Europe/London"

    def test_roundtrip(self, tmp_path):
        """Config survives roundtrip serialization."""
        original = Config(
            user_timezone="Asia/Tokyo",
            channels=ChannelConfig(
                discord=DiscordChannelConfig(enabled=True, bot_token="token"),
            ),
            agents=AgentsConfig(
                active_agent="custom",
                profiles={
                    "custom": AgentProfileRef(id="custom", workspace_dir="/custom"),
                },
            ),
        )
        config_path = tmp_path / "config.json"

        original.to_file(config_path)
        restored = Config.from_file(config_path)

        assert restored.user_timezone == "Asia/Tokyo"
        assert restored.channels.discord.enabled is True
        assert restored.agents.active_agent == "custom"


class TestConfigUtilities:
    """Tests for configuration utility functions."""

    def test_get_config_path(self, tmp_path):
        """get_config_path returns correct path."""
        with patch("copaw.config.SECRET_DIR", tmp_path):
            path = get_config_path()

        assert path == tmp_path / "config.json"

    def test_load_config_missing(self, tmp_path):
        """load_config returns defaults when missing."""
        with patch("copaw.config.SECRET_DIR", tmp_path):
            config = load_config()

        assert config.user_timezone == "UTC"

    def test_load_config_existing(self, tmp_path):
        """load_config loads existing config."""
        (tmp_path / "config.json").write_text(json.dumps({
            "user_timezone": "UTC+8",
        }))

        with patch("copaw.config.SECRET_DIR", tmp_path):
            config = load_config()

        assert config.user_timezone == "UTC+8"

    def test_save_config(self, tmp_path):
        """save_config persists config."""
        with patch("copaw.config.SECRET_DIR", tmp_path):
            config = Config(user_timezone="Pacific/Auckland")
            save_config(config)

        assert (tmp_path / "config.json").exists()

    def test_get_agent_workspace(self, tmp_path):
        """get_agent_workspace returns correct path."""
        workspace = tmp_path / "workspaces" / "test"
        (tmp_path / "config.json").write_text(json.dumps({
            "agents": {
                "profiles": {
                    "test": {
                        "id": "test",
                        "workspace_dir": str(workspace),
                    },
                },
            },
        }))

        with patch("copaw.config.SECRET_DIR", tmp_path):
            result = get_agent_workspace("test")

        assert result == workspace

    def test_get_agent_workspace_not_found(self, tmp_path):
        """get_agent_workspace raises for missing agent."""
        (tmp_path / "config.json").write_text(json.dumps({}))

        with patch("copaw.config.SECRET_DIR", tmp_path):
            with pytest.raises(ValueError, match="not found"):
                get_agent_workspace("nonexistent")

    def test_load_agent_config_new(self, tmp_path):
        """load_agent_config creates new config."""
        workspace = tmp_path / "workspaces" / "test"
        workspace.mkdir(parents=True)
        (tmp_path / "config.json").write_text(json.dumps({
            "agents": {
                "profiles": {
                    "test": {
                        "id": "test",
                        "workspace_dir": str(workspace),
                    },
                },
            },
        }))

        with patch("copaw.config.SECRET_DIR", tmp_path):
            config = load_agent_config("test")

        assert config.id == "test"
        assert config.name == "Test"

    def test_load_agent_config_existing(self, tmp_path):
        """load_agent_config loads existing config."""
        workspace = tmp_path / "workspaces" / "test"
        workspace.mkdir(parents=True)
        (workspace / "agent.json").write_text(json.dumps({
            "id": "test",
            "name": "Custom Name",
            "language": "ja",
        }))
        (tmp_path / "config.json").write_text(json.dumps({
            "agents": {
                "profiles": {
                    "test": {
                        "id": "test",
                        "workspace_dir": str(workspace),
                    },
                },
            },
        }))

        with patch("copaw.config.SECRET_DIR", tmp_path):
            config = load_agent_config("test")

        assert config.id == "test"
        assert config.name == "Custom Name"
        assert config.language == "ja"

    def test_save_agent_config(self, tmp_path):
        """save_agent_config persists config."""
        workspace = tmp_path / "workspaces" / "test"
        (tmp_path / "config.json").write_text(json.dumps({
            "agents": {
                "profiles": {
                    "test": {
                        "id": "test",
                        "workspace_dir": str(workspace),
                    },
                },
            },
        }))

        with patch("copaw.config.SECRET_DIR", tmp_path):
            config = AgentConfig(id="test", name="Saved Agent")
            save_agent_config("test", config)

        assert (workspace / "agent.json").exists()
        data = json.loads((workspace / "agent.json").read_text())
        assert data["name"] == "Saved Agent"
