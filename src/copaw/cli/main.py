# -*- coding: utf-8 -*-
"""CLI main entry point for CoPaw.

Provides the main CLI group with lazy-loading subcommand support for
better startup performance.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import click

from .. import __version__

logger = logging.getLogger(__name__)


class LazyGroup(click.Group):
    """Click group with lazy-loading subcommand support.

    Defers importing subcommand modules until the command is actually
    invoked, improving CLI startup time.
    """

    def __init__(
        self,
        *args: Any,
        lazy_subcommands: dict[str, tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the lazy group.

        Args:
            lazy_subcommands: Mapping of command name to (module_path, attr_name).
            *args: Passed to click.Group.
            **kwargs: Passed to click.Group.
        """
        super().__init__(*args, **kwargs)
        self.lazy_subcommands = lazy_subcommands or {}

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List all available commands (eager and lazy).

        Args:
            ctx: Click context.

        Returns:
            Sorted list of command names.
        """
        eager = super().list_commands(ctx)
        lazy = list(self.lazy_subcommands.keys())
        return sorted(set(eager) | set(lazy))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Get a command by name, loading lazily if needed.

        Args:
            ctx: Click context.
            cmd_name: Command name to get.

        Returns:
            The command if found, None otherwise.
        """
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd

        if cmd_name in self.lazy_subcommands:
            module_path, attr_name = self.lazy_subcommands[cmd_name]
            try:
                module = __import__(module_path, fromlist=[attr_name])
                cmd = getattr(module, attr_name)
                self.add_command(cmd, cmd_name)
                return cmd
            except Exception as exc:
                logger.error("Failed to load command '%s': %s", cmd_name, exc)
                return None

        return None


@click.group(
    cls=LazyGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    lazy_subcommands={
        "init": ("copaw.cli.init_cmd", "init_cmd"),
        "providers": ("copaw.cli.providers_cmd", "providers_group"),
        "models": ("copaw.cli.providers_cmd", "models_group"),
    },
)
@click.version_option(version=__version__, prog_name="copaw")
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """CoPaw - AI Assistant Platform CLI.

    Manage providers, models, and workspaces for your AI assistant.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if verbose:
        logging.basicConfig(level=logging.DEBUG)


@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show the CoPaw version."""
    click.echo(f"CoPaw version {__version__}")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show the current CoPaw status."""
    from ..config import load_config

    config = load_config()
    click.echo("CoPaw Status")
    click.echo(f"  Active agent: {config.agents.active_agent}")
    click.echo(f"  Timezone: {config.user_timezone}")

    enabled_channels = [
        name for name, channel in vars(config.channels).items()
        if hasattr(channel, "enabled") and channel.enabled
    ]
    if enabled_channels:
        click.echo(f"  Enabled channels: {', '.join(enabled_channels)}")
    else:
        click.echo("  No channels enabled")


def main() -> None:
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
