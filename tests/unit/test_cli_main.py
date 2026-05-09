# -*- coding: utf-8 -*-
"""Tests for copaw.cli.main module."""

from unittest.mock import patch, MagicMock

import click
from click.testing import CliRunner
import pytest

from copaw.cli.main import cli, LazyGroup, main


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


class TestLazyGroup:
    """Tests for LazyGroup."""

    def test_list_commands_includes_lazy(self):
        """list_commands includes both eager and lazy commands."""

        @click.command(name="eager")
        def eager_cmd():
            pass

        group = LazyGroup(
            lazy_subcommands={
                "lazy": ("some.module", "lazy_cmd"),
            },
        )
        group.add_command(eager_cmd)

        ctx = click.Context(group)
        commands = group.list_commands(ctx)

        assert "eager" in commands
        assert "lazy" in commands

    def test_list_commands_sorted(self):
        """list_commands returns sorted list."""
        group = LazyGroup(
            lazy_subcommands={
                "zebra": ("a.b", "cmd"),
                "apple": ("a.b", "cmd"),
            },
        )

        ctx = click.Context(group)
        commands = group.list_commands(ctx)

        assert commands == sorted(commands)

    def test_get_command_eager(self):
        """get_command returns eager commands."""

        @click.command(name="test")
        def test_cmd():
            pass

        group = LazyGroup()
        group.add_command(test_cmd)

        ctx = click.Context(group)
        cmd = group.get_command(ctx, "test")

        assert cmd is test_cmd

    def test_get_command_unknown(self):
        """get_command returns None for unknown commands."""
        group = LazyGroup()

        ctx = click.Context(group)
        cmd = group.get_command(ctx, "unknown")

        assert cmd is None

    def test_lazy_subcommands_stored(self):
        """Lazy subcommands are stored in the group."""
        group = LazyGroup(
            lazy_subcommands={
                "lazy": ("test.module", "lazy_cmd"),
            },
        )

        assert "lazy" in group.lazy_subcommands
        module_path, attr_name = group.lazy_subcommands["lazy"]
        assert module_path == "test.module"
        assert attr_name == "lazy_cmd"

    def test_get_command_caches_loaded(self):
        """get_command caches loaded lazy commands."""
        mock_cmd = click.Command("lazy_cmd", callback=lambda: None)
        mock_module = MagicMock()
        mock_module.lazy_cmd = mock_cmd

        group = LazyGroup(
            lazy_subcommands={
                "lazy": ("test.module", "lazy_cmd"),
            },
        )

        ctx = click.Context(group)

        with patch("builtins.__import__", return_value=mock_module) as mock_import:
            group.get_command(ctx, "lazy")
            group.get_command(ctx, "lazy")

            # Should only import once due to caching
            assert mock_import.call_count == 1


class TestCli:
    """Tests for main CLI group."""

    def test_help(self, runner):
        """CLI shows help."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "CoPaw" in result.output
        assert "--help" in result.output

    def test_version(self, runner):
        """CLI shows version."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "copaw" in result.output.lower()

    def test_verbose_flag(self, runner):
        """CLI accepts verbose flag."""
        with patch("copaw.cli.main.logging.basicConfig") as mock_config:
            result = runner.invoke(cli, ["--verbose", "version"])

            assert result.exit_code == 0
            mock_config.assert_called_once()


class TestVersionCommand:
    """Tests for version command."""

    def test_shows_version(self, runner):
        """version command shows version string."""
        result = runner.invoke(cli, ["version"])

        assert result.exit_code == 0
        assert "version" in result.output.lower()


class TestStatusCommand:
    """Tests for status command."""

    def test_shows_status(self, runner, tmp_path):
        """status command shows current status."""
        with patch("copaw.config.get_config_path") as mock_path:
            mock_path.return_value = tmp_path / "config.json"
            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Status" in result.output

    def test_shows_active_agent(self, runner, tmp_path):
        """status command shows active agent."""
        import json

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "agents": {
                "active_agent": "test-agent",
                "profiles": {},
            },
        }))

        with patch("copaw.config.get_config_path") as mock_path:
            mock_path.return_value = config_file
            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "test-agent" in result.output


class TestMainFunction:
    """Tests for main() entry point."""

    def test_keyboard_interrupt(self):
        """main() handles KeyboardInterrupt."""
        with patch("copaw.cli.main.cli") as mock_cli:
            mock_cli.side_effect = KeyboardInterrupt()

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 130

    def test_exception_handling(self):
        """main() handles unexpected exceptions."""
        with patch("copaw.cli.main.cli") as mock_cli:
            mock_cli.side_effect = RuntimeError("Test error")

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1


class TestLazySubcommands:
    """Tests for lazy subcommand registration."""

    def test_lazy_commands_defined(self):
        """Lazy subcommands are defined in the CLI group."""
        # Get the lazy_subcommands from the cli group
        assert hasattr(cli, "lazy_subcommands")
        assert "init" in cli.lazy_subcommands
        assert "providers" in cli.lazy_subcommands
        assert "models" in cli.lazy_subcommands

    def test_lazy_command_module_paths(self):
        """Lazy subcommands have correct module paths."""
        assert cli.lazy_subcommands["init"] == ("copaw.cli.init_cmd", "init_cmd")
        assert cli.lazy_subcommands["providers"] == ("copaw.cli.providers_cmd", "providers_group")
        assert cli.lazy_subcommands["models"] == ("copaw.cli.providers_cmd", "models_group")
