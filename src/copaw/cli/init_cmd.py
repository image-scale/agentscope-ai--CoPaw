# -*- coding: utf-8 -*-
"""CLI init command for initializing CoPaw workspace."""

from __future__ import annotations

from pathlib import Path

import click

from ..config import (
    Config,
    AgentsConfig,
    AgentConfig,
    AgentProfileRef,
    get_config_path,
    save_config,
    save_agent_config,
)
from ..settings import WORKING_DIR, SECRET_DIR


@click.command("init")
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing configuration if present.",
)
@click.option(
    "--defaults",
    "use_defaults",
    is_flag=True,
    help="Use defaults only, no interactive prompts.",
)
@click.option(
    "--agent-name",
    default="default",
    help="Name for the initial agent.",
)
@click.pass_context
def init_cmd(
    ctx: click.Context,
    force: bool,
    use_defaults: bool,
    agent_name: str,
) -> None:
    """Initialize CoPaw workspace with default configuration.

    Creates the working directory structure, configuration files,
    and sets up the default agent workspace.
    """
    config_path = get_config_path()

    if config_path.exists() and not force:
        click.echo(f"Configuration already exists at {config_path}")
        click.echo("Use --force to overwrite.")
        ctx.exit(1)

    click.echo("Initializing CoPaw workspace...")

    working_dir = Path(WORKING_DIR).expanduser()
    secret_dir = Path(SECRET_DIR).expanduser()

    click.echo(f"  Working directory: {working_dir}")
    click.echo(f"  Config directory: {secret_dir}")

    working_dir.mkdir(parents=True, exist_ok=True)
    secret_dir.mkdir(parents=True, exist_ok=True)

    if not use_defaults:
        agent_name = click.prompt(
            "Agent name",
            default=agent_name,
            type=str,
        )

    workspace_dir = working_dir / "workspaces" / agent_name
    workspace_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        agents=AgentsConfig(
            active_agent=agent_name,
            agent_order=[agent_name],
            profiles={
                agent_name: AgentProfileRef(
                    id=agent_name,
                    workspace_dir=str(workspace_dir),
                ),
            },
        ),
    )

    save_config(config)
    click.echo(f"  Created config: {config_path}")

    agent_config = AgentConfig(
        id=agent_name,
        name=agent_name.title().replace("-", " ").replace("_", " "),
        workspace_dir=str(workspace_dir),
    )
    save_agent_config(agent_name, agent_config)
    click.echo(f"  Created agent config: {workspace_dir / 'agent.json'}")

    system_prompt = workspace_dir / "AGENTS.md"
    if not system_prompt.exists():
        system_prompt.write_text(
            f"# {agent_config.name}\n\n"
            "You are a helpful AI assistant.\n"
        )
        click.echo(f"  Created system prompt: {system_prompt}")

    click.echo("")
    click.echo("CoPaw initialized successfully!")
    click.echo(f"  Active agent: {agent_name}")
    click.echo(f"  Workspace: {workspace_dir}")


def create_default_workspace(
    agent_name: str = "default",
    force: bool = False,
) -> tuple[Path, Config]:
    """Programmatically create a default workspace.

    Args:
        agent_name: Name for the agent.
        force: Overwrite existing configuration.

    Returns:
        Tuple of (workspace_path, config).

    Raises:
        FileExistsError: If config exists and force is False.
    """
    config_path = get_config_path()

    if config_path.exists() and not force:
        raise FileExistsError(f"Configuration already exists: {config_path}")

    working_dir = Path(WORKING_DIR).expanduser()
    secret_dir = Path(SECRET_DIR).expanduser()

    working_dir.mkdir(parents=True, exist_ok=True)
    secret_dir.mkdir(parents=True, exist_ok=True)

    workspace_dir = working_dir / "workspaces" / agent_name
    workspace_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        agents=AgentsConfig(
            active_agent=agent_name,
            agent_order=[agent_name],
            profiles={
                agent_name: AgentProfileRef(
                    id=agent_name,
                    workspace_dir=str(workspace_dir),
                ),
            },
        ),
    )

    save_config(config)

    agent_config = AgentConfig(
        id=agent_name,
        name=agent_name.title(),
        workspace_dir=str(workspace_dir),
    )
    save_agent_config(agent_name, agent_config)

    return workspace_dir, config
