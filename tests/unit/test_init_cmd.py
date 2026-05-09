# -*- coding: utf-8 -*-
"""Tests for copaw.cli.init_cmd module."""

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
import pytest

from copaw.cli.init_cmd import init_cmd, create_default_workspace


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_paths(tmp_path):
    """Mock WORKING_DIR and SECRET_DIR."""
    working_dir = tmp_path / "working"
    secret_dir = tmp_path / "secret"
    with patch("copaw.cli.init_cmd.WORKING_DIR", str(working_dir)):
        with patch("copaw.cli.init_cmd.SECRET_DIR", str(secret_dir)):
            with patch("copaw.config.SECRET_DIR", secret_dir):
                yield working_dir, secret_dir


class TestInitCommand:
    """Tests for init CLI command."""

    def test_creates_directories(self, runner, mock_paths):
        """init creates working and config directories."""
        working_dir, secret_dir = mock_paths

        result = runner.invoke(init_cmd, ["--defaults"])

        assert result.exit_code == 0
        assert working_dir.exists()
        assert secret_dir.exists()

    def test_creates_config_file(self, runner, mock_paths):
        """init creates config.json."""
        working_dir, secret_dir = mock_paths

        result = runner.invoke(init_cmd, ["--defaults"])

        assert result.exit_code == 0
        config_path = secret_dir / "config.json"
        assert config_path.exists()

    def test_creates_agent_workspace(self, runner, mock_paths):
        """init creates agent workspace directory."""
        working_dir, secret_dir = mock_paths

        result = runner.invoke(init_cmd, ["--defaults"])

        assert result.exit_code == 0
        workspace = working_dir / "workspaces" / "default"
        assert workspace.exists()

    def test_creates_agent_config(self, runner, mock_paths):
        """init creates agent.json in workspace."""
        working_dir, secret_dir = mock_paths

        result = runner.invoke(init_cmd, ["--defaults"])

        assert result.exit_code == 0
        agent_config = working_dir / "workspaces" / "default" / "agent.json"
        assert agent_config.exists()

    def test_creates_system_prompt(self, runner, mock_paths):
        """init creates AGENTS.md system prompt."""
        working_dir, secret_dir = mock_paths

        result = runner.invoke(init_cmd, ["--defaults"])

        assert result.exit_code == 0
        prompt = working_dir / "workspaces" / "default" / "AGENTS.md"
        assert prompt.exists()
        assert "AI assistant" in prompt.read_text()

    def test_custom_agent_name(self, runner, mock_paths):
        """init accepts custom agent name."""
        working_dir, secret_dir = mock_paths

        result = runner.invoke(init_cmd, ["--defaults", "--agent-name", "custom"])

        assert result.exit_code == 0
        workspace = working_dir / "workspaces" / "custom"
        assert workspace.exists()

    def test_fails_without_force(self, runner, mock_paths):
        """init fails if config exists without --force."""
        working_dir, secret_dir = mock_paths
        secret_dir.mkdir(parents=True, exist_ok=True)
        (secret_dir / "config.json").write_text("{}")

        result = runner.invoke(init_cmd, ["--defaults"])

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_force_overwrites(self, runner, mock_paths):
        """init --force overwrites existing config."""
        working_dir, secret_dir = mock_paths
        secret_dir.mkdir(parents=True, exist_ok=True)
        (secret_dir / "config.json").write_text('{"old": true}')

        result = runner.invoke(init_cmd, ["--defaults", "--force"])

        assert result.exit_code == 0
        config = json.loads((secret_dir / "config.json").read_text())
        assert "old" not in config

    def test_shows_success_message(self, runner, mock_paths):
        """init shows success message."""
        working_dir, secret_dir = mock_paths

        result = runner.invoke(init_cmd, ["--defaults"])

        assert result.exit_code == 0
        assert "successfully" in result.output.lower()

    def test_interactive_prompt(self, runner, mock_paths):
        """init prompts for agent name without --defaults."""
        working_dir, secret_dir = mock_paths

        result = runner.invoke(init_cmd, input="my-agent\n")

        assert result.exit_code == 0
        workspace = working_dir / "workspaces" / "my-agent"
        assert workspace.exists()


class TestCreateDefaultWorkspace:
    """Tests for create_default_workspace function."""

    def test_creates_workspace(self, mock_paths):
        """Creates workspace and returns path."""
        working_dir, secret_dir = mock_paths

        workspace, config = create_default_workspace()

        assert workspace.exists()
        assert config.agents.active_agent == "default"

    def test_custom_agent_name(self, mock_paths):
        """Creates workspace with custom name."""
        working_dir, secret_dir = mock_paths

        workspace, config = create_default_workspace(agent_name="custom")

        assert "custom" in str(workspace)
        assert config.agents.active_agent == "custom"

    def test_raises_if_exists(self, mock_paths):
        """Raises FileExistsError if config exists."""
        working_dir, secret_dir = mock_paths
        secret_dir.mkdir(parents=True, exist_ok=True)
        (secret_dir / "config.json").write_text("{}")

        with pytest.raises(FileExistsError):
            create_default_workspace()

    def test_force_overwrites(self, mock_paths):
        """force=True overwrites existing config."""
        working_dir, secret_dir = mock_paths
        secret_dir.mkdir(parents=True, exist_ok=True)
        (secret_dir / "config.json").write_text('{"old": true}')

        workspace, config = create_default_workspace(force=True)

        assert workspace.exists()


class TestInitCommandConfig:
    """Tests for configuration created by init."""

    def test_config_has_agent_profile(self, runner, mock_paths):
        """Config contains agent profile reference."""
        working_dir, secret_dir = mock_paths

        runner.invoke(init_cmd, ["--defaults"])

        config = json.loads((secret_dir / "config.json").read_text())
        assert "agents" in config
        assert "profiles" in config["agents"]
        assert "default" in config["agents"]["profiles"]

    def test_config_has_active_agent(self, runner, mock_paths):
        """Config has active_agent set."""
        working_dir, secret_dir = mock_paths

        runner.invoke(init_cmd, ["--defaults"])

        config = json.loads((secret_dir / "config.json").read_text())
        assert config["agents"]["active_agent"] == "default"

    def test_agent_config_has_name(self, runner, mock_paths):
        """Agent config has human-readable name."""
        working_dir, secret_dir = mock_paths

        runner.invoke(init_cmd, ["--defaults", "--agent-name", "my-agent"])

        agent_path = working_dir / "workspaces" / "my-agent" / "agent.json"
        agent = json.loads(agent_path.read_text())
        assert agent["name"] == "My Agent"
