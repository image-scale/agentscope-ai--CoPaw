# -*- coding: utf-8 -*-
"""CLI commands for managing LLM providers and models."""

from __future__ import annotations

import asyncio
from typing import Any

import click

from ..providers.manager import ProviderCoordinator, ActiveModelConfig


def get_coordinator() -> ProviderCoordinator:
    """Get a provider coordinator instance."""
    return ProviderCoordinator()


@click.group("providers")
def providers_group() -> None:
    """Manage LLM providers."""
    pass


@providers_group.command("list")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information.")
def list_providers(verbose: bool) -> None:
    """List all available LLM providers."""
    coordinator = get_coordinator()
    providers = coordinator.list_providers()

    if not providers:
        click.echo("No providers available.")
        return

    click.echo("Available providers:")
    for provider in providers:
        status = "local" if provider.is_local else "remote"
        tag = " (custom)" if provider.is_custom else ""
        click.echo(f"  {provider.id}: {provider.name} [{status}]{tag}")

        if verbose:
            if provider.api_key:
                masked = provider.api_key[:8] + "..." if len(provider.api_key) > 8 else "***"
                click.echo(f"    API key: {masked}")
            if provider.base_url:
                click.echo(f"    URL: {provider.base_url}")
            model_count = len(provider.models)
            click.echo(f"    Models: {model_count}")


@providers_group.command("info")
@click.argument("provider_id")
def provider_info(provider_id: str) -> None:
    """Show detailed information about a provider."""
    coordinator = get_coordinator()
    provider = coordinator.get_provider(provider_id)

    if not provider:
        click.echo(f"Provider '{provider_id}' not found.", err=True)
        raise SystemExit(1)

    click.echo(f"Provider: {provider.name}")
    click.echo(f"  ID: {provider.id}")
    click.echo(f"  Type: {'local' if provider.is_local else 'remote'}")
    click.echo(f"  Custom: {'yes' if provider.is_custom else 'no'}")

    if provider.base_url:
        click.echo(f"  URL: {provider.base_url}")

    if provider.api_key:
        masked = provider.api_key[:8] + "..." if len(provider.api_key) > 8 else "***"
        click.echo(f"  API key: {masked}")
    elif provider.require_api_key:
        click.echo("  API key: not set (required)")
    else:
        click.echo("  API key: not required")

    click.echo(f"  Models: {len(provider.models)}")
    for model in provider.models[:5]:
        click.echo(f"    - {model.id}: {model.name}")
    if len(provider.models) > 5:
        click.echo(f"    ... and {len(provider.models) - 5} more")


@providers_group.command("configure")
@click.argument("provider_id")
@click.option("--api-key", help="Set the API key.")
@click.option("--base-url", help="Set the base URL.")
def configure_provider(
    provider_id: str,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Configure a provider's settings."""
    coordinator = get_coordinator()
    provider = coordinator.get_provider(provider_id)

    if not provider:
        click.echo(f"Provider '{provider_id}' not found.", err=True)
        raise SystemExit(1)

    config: dict[str, Any] = {}
    if api_key is not None:
        config["api_key"] = api_key
    if base_url is not None:
        if provider.freeze_url:
            click.echo(f"Provider '{provider_id}' has a fixed URL.", err=True)
            raise SystemExit(1)
        config["base_url"] = base_url

    if not config:
        click.echo("No configuration options provided.")
        return

    if coordinator.update_provider(provider_id, config):
        click.echo(f"Provider '{provider_id}' configured successfully.")
    else:
        click.echo(f"Failed to configure provider '{provider_id}'.", err=True)
        raise SystemExit(1)


@providers_group.command("check")
@click.argument("provider_id")
def check_provider(provider_id: str) -> None:
    """Check connectivity to a provider."""
    coordinator = get_coordinator()
    provider = coordinator.get_provider(provider_id)

    if not provider:
        click.echo(f"Provider '{provider_id}' not found.", err=True)
        raise SystemExit(1)

    click.echo(f"Checking connectivity to '{provider.name}'...")

    async def do_check() -> bool:
        return await provider.check_connection()

    try:
        connected = asyncio.run(do_check())
        if connected:
            click.echo(f"Connection successful!")
        else:
            click.echo(f"Connection failed.", err=True)
            raise SystemExit(1)
    except Exception as e:
        click.echo(f"Connection error: {e}", err=True)
        raise SystemExit(1)


@click.group("models")
def models_group() -> None:
    """Manage LLM models."""
    pass


@models_group.command("list")
@click.option("--provider", "-p", help="Filter by provider ID.")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information.")
def list_models(provider: str | None, verbose: bool) -> None:
    """List available models."""
    coordinator = get_coordinator()

    if provider:
        p = coordinator.get_provider(provider)
        if not p:
            click.echo(f"Provider '{provider}' not found.", err=True)
            raise SystemExit(1)
        providers_list = [p]
    else:
        providers_list = coordinator.list_providers()

    active = coordinator.active_model
    total_models = 0

    for p in providers_list:
        if not p.models:
            continue

        click.echo(f"\n{p.name} ({p.id}):")
        for model in p.models:
            is_active = (
                active
                and active.provider_id == p.id
                and active.model == model.id
            )
            marker = " *" if is_active else ""
            click.echo(f"  {model.id}: {model.name}{marker}")

            if verbose:
                if model.supports_multimodal is not None:
                    mm = "yes" if model.supports_multimodal else "no"
                    click.echo(f"    Multimodal: {mm}")

            total_models += 1

    if total_models == 0:
        click.echo("No models available.")
    else:
        click.echo(f"\n{total_models} model(s) total.")
        if active:
            click.echo(f"Active: {active.provider_id}/{active.model}")


@models_group.command("activate")
@click.argument("provider_id")
@click.argument("model_id")
def activate_model(provider_id: str, model_id: str) -> None:
    """Set the active model."""
    coordinator = get_coordinator()

    async def do_activate() -> tuple[bool, str]:
        return await coordinator.activate_model(provider_id, model_id)

    try:
        success, message = asyncio.run(do_activate())
        if success:
            click.echo(f"Activated model: {provider_id}/{model_id}")
        else:
            click.echo(f"Failed to activate: {message}", err=True)
            raise SystemExit(1)
    except ValueError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1)


@models_group.command("active")
def show_active_model() -> None:
    """Show the currently active model."""
    coordinator = get_coordinator()
    active = coordinator.active_model

    if not active:
        click.echo("No active model set.")
        return

    provider = coordinator.get_provider(active.provider_id)
    provider_name = provider.name if provider else active.provider_id

    click.echo(f"Active model: {active.model}")
    click.echo(f"Provider: {provider_name} ({active.provider_id})")


@models_group.command("discover")
@click.argument("provider_id")
def discover_models(provider_id: str) -> None:
    """Discover available models from a provider."""
    coordinator = get_coordinator()
    provider = coordinator.get_provider(provider_id)

    if not provider:
        click.echo(f"Provider '{provider_id}' not found.", err=True)
        raise SystemExit(1)

    click.echo(f"Discovering models from '{provider.name}'...")

    async def do_discover() -> list:
        return await provider.discover_models()

    try:
        models = asyncio.run(do_discover())
        if models:
            click.echo(f"Found {len(models)} model(s):")
            for model in models:
                click.echo(f"  - {model.id}: {model.name}")
        else:
            click.echo("No models discovered.")
    except Exception as e:
        click.echo(f"Discovery failed: {e}", err=True)
        raise SystemExit(1)
