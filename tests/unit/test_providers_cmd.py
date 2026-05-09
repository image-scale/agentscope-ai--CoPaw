# -*- coding: utf-8 -*-
"""Tests for copaw.cli.providers_cmd module."""

from unittest.mock import patch, MagicMock, AsyncMock

import click
from click.testing import CliRunner
import pytest

from copaw.cli.providers_cmd import (
    providers_group,
    models_group,
    get_coordinator,
)
from copaw.providers.base import ModelInfo
from copaw.providers.manager import ActiveModelConfig


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_coordinator():
    """Create a mock ProviderCoordinator."""
    coordinator = MagicMock()
    coordinator.list_providers.return_value = []
    coordinator.active_model = None
    return coordinator


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = MagicMock()
    provider.id = "test-provider"
    provider.name = "Test Provider"
    provider.is_local = False
    provider.is_custom = False
    provider.api_key = ""
    provider.base_url = "https://api.test.com"
    provider.require_api_key = True
    provider.freeze_url = False
    provider.models = []
    return provider


class TestGetCoordinator:
    """Tests for get_coordinator function."""

    def test_returns_coordinator(self):
        """Returns a ProviderCoordinator instance."""
        with patch("copaw.cli.providers_cmd.ProviderCoordinator") as mock:
            result = get_coordinator()
            mock.assert_called_once()


class TestProvidersGroup:
    """Tests for providers CLI group."""

    def test_providers_help(self, runner):
        """providers shows help."""
        result = runner.invoke(providers_group, ["--help"])

        assert result.exit_code == 0
        assert "providers" in result.output.lower()


class TestProvidersListCommand:
    """Tests for providers list command."""

    def test_list_empty(self, runner, mock_coordinator):
        """list shows message when no providers."""
        mock_coordinator.list_providers.return_value = []

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["list"])

        assert result.exit_code == 0
        assert "No providers" in result.output

    def test_list_providers(self, runner, mock_coordinator, mock_provider):
        """list shows available providers."""
        mock_coordinator.list_providers.return_value = [mock_provider]

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["list"])

        assert result.exit_code == 0
        assert "test-provider" in result.output
        assert "Test Provider" in result.output

    def test_list_shows_local_tag(self, runner, mock_coordinator, mock_provider):
        """list shows local tag for local providers."""
        mock_provider.is_local = True
        mock_coordinator.list_providers.return_value = [mock_provider]

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["list"])

        assert result.exit_code == 0
        assert "local" in result.output

    def test_list_shows_custom_tag(self, runner, mock_coordinator, mock_provider):
        """list shows custom tag for custom providers."""
        mock_provider.is_custom = True
        mock_coordinator.list_providers.return_value = [mock_provider]

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["list"])

        assert result.exit_code == 0
        assert "(custom)" in result.output

    def test_list_verbose(self, runner, mock_coordinator, mock_provider):
        """list -v shows detailed information."""
        mock_provider.api_key = "sk-test1234567890"
        mock_coordinator.list_providers.return_value = [mock_provider]

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["list", "-v"])

        assert result.exit_code == 0
        assert "sk-test1..." in result.output
        assert "api.test.com" in result.output


class TestProvidersInfoCommand:
    """Tests for providers info command."""

    def test_info_not_found(self, runner, mock_coordinator):
        """info fails for unknown provider."""
        mock_coordinator.get_provider.return_value = None

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["info", "unknown"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_info_shows_details(self, runner, mock_coordinator, mock_provider):
        """info shows provider details."""
        mock_coordinator.get_provider.return_value = mock_provider

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["info", "test-provider"])

        assert result.exit_code == 0
        assert "Test Provider" in result.output
        assert "test-provider" in result.output


class TestProvidersConfigureCommand:
    """Tests for providers configure command."""

    def test_configure_not_found(self, runner, mock_coordinator):
        """configure fails for unknown provider."""
        mock_coordinator.get_provider.return_value = None

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["configure", "unknown", "--api-key", "test"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_configure_no_options(self, runner, mock_coordinator, mock_provider):
        """configure with no options shows message."""
        mock_coordinator.get_provider.return_value = mock_provider

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["configure", "test-provider"])

        assert result.exit_code == 0
        assert "No configuration options" in result.output

    def test_configure_api_key(self, runner, mock_coordinator, mock_provider):
        """configure sets API key."""
        mock_coordinator.get_provider.return_value = mock_provider
        mock_coordinator.update_provider.return_value = True

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["configure", "test-provider", "--api-key", "new-key"])

        assert result.exit_code == 0
        assert "successfully" in result.output
        mock_coordinator.update_provider.assert_called_with("test-provider", {"api_key": "new-key"})

    def test_configure_frozen_url(self, runner, mock_coordinator, mock_provider):
        """configure fails for frozen URL."""
        mock_provider.freeze_url = True
        mock_coordinator.get_provider.return_value = mock_provider

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["configure", "test-provider", "--base-url", "http://new.url"])

        assert result.exit_code == 1
        assert "fixed URL" in result.output


class TestProvidersCheckCommand:
    """Tests for providers check command."""

    def test_check_not_found(self, runner, mock_coordinator):
        """check fails for unknown provider."""
        mock_coordinator.get_provider.return_value = None

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["check", "unknown"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_check_success(self, runner, mock_coordinator, mock_provider):
        """check shows success for connected provider."""
        mock_provider.check_connection = AsyncMock(return_value=True)
        mock_coordinator.get_provider.return_value = mock_provider

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["check", "test-provider"])

        assert result.exit_code == 0
        assert "successful" in result.output

    def test_check_failure(self, runner, mock_coordinator, mock_provider):
        """check shows failure for disconnected provider."""
        mock_provider.check_connection = AsyncMock(return_value=False)
        mock_coordinator.get_provider.return_value = mock_provider

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(providers_group, ["check", "test-provider"])

        assert result.exit_code == 1
        assert "failed" in result.output


class TestModelsGroup:
    """Tests for models CLI group."""

    def test_models_help(self, runner):
        """models shows help."""
        result = runner.invoke(models_group, ["--help"])

        assert result.exit_code == 0
        assert "models" in result.output.lower()


class TestModelsListCommand:
    """Tests for models list command."""

    def test_list_empty(self, runner, mock_coordinator):
        """list shows message when no models."""
        mock_coordinator.list_providers.return_value = []

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["list"])

        assert result.exit_code == 0
        assert "No models" in result.output

    def test_list_models(self, runner, mock_coordinator, mock_provider):
        """list shows available models."""
        mock_provider.models = [
            ModelInfo(id="model-1", name="Model One"),
            ModelInfo(id="model-2", name="Model Two"),
        ]
        mock_coordinator.list_providers.return_value = [mock_provider]

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["list"])

        assert result.exit_code == 0
        assert "model-1" in result.output
        assert "Model One" in result.output
        assert "2 model(s)" in result.output

    def test_list_filter_by_provider(self, runner, mock_coordinator, mock_provider):
        """list -p filters by provider."""
        mock_provider.models = [ModelInfo(id="test-model", name="Test Model")]
        mock_coordinator.get_provider.return_value = mock_provider

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["list", "-p", "test-provider"])

        assert result.exit_code == 0
        assert "test-model" in result.output

    def test_list_filter_unknown_provider(self, runner, mock_coordinator):
        """list -p fails for unknown provider."""
        mock_coordinator.get_provider.return_value = None

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["list", "-p", "unknown"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_list_shows_active_marker(self, runner, mock_coordinator, mock_provider):
        """list marks active model."""
        mock_provider.models = [ModelInfo(id="active-model", name="Active")]
        mock_coordinator.list_providers.return_value = [mock_provider]
        mock_coordinator.active_model = ActiveModelConfig(
            provider_id="test-provider",
            model="active-model",
        )

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["list"])

        assert result.exit_code == 0
        assert "*" in result.output


class TestModelsActivateCommand:
    """Tests for models activate command."""

    def test_activate_success(self, runner, mock_coordinator):
        """activate sets active model."""
        mock_coordinator.activate_model = AsyncMock(return_value=(True, "OK"))

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["activate", "provider", "model"])

        assert result.exit_code == 0
        assert "Activated" in result.output

    def test_activate_failure(self, runner, mock_coordinator):
        """activate shows error on failure."""
        mock_coordinator.activate_model = AsyncMock(return_value=(False, "Error"))

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["activate", "provider", "model"])

        assert result.exit_code == 1
        assert "Failed" in result.output

    def test_activate_provider_not_found(self, runner, mock_coordinator):
        """activate fails for unknown provider."""
        mock_coordinator.activate_model = AsyncMock(side_effect=ValueError("Provider not found"))

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["activate", "unknown", "model"])

        assert result.exit_code == 1
        assert "not found" in result.output


class TestModelsActiveCommand:
    """Tests for models active command."""

    def test_active_none(self, runner, mock_coordinator):
        """active shows message when no model active."""
        mock_coordinator.active_model = None

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["active"])

        assert result.exit_code == 0
        assert "No active model" in result.output

    def test_active_shows_model(self, runner, mock_coordinator, mock_provider):
        """active shows current model."""
        mock_coordinator.active_model = ActiveModelConfig(
            provider_id="test-provider",
            model="test-model",
        )
        mock_coordinator.get_provider.return_value = mock_provider

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["active"])

        assert result.exit_code == 0
        assert "test-model" in result.output
        assert "Test Provider" in result.output


class TestModelsDiscoverCommand:
    """Tests for models discover command."""

    def test_discover_not_found(self, runner, mock_coordinator):
        """discover fails for unknown provider."""
        mock_coordinator.get_provider.return_value = None

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["discover", "unknown"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_discover_success(self, runner, mock_coordinator, mock_provider):
        """discover shows found models."""
        mock_provider.discover_models = AsyncMock(return_value=[
            ModelInfo(id="discovered-1", name="Discovered One"),
        ])
        mock_coordinator.get_provider.return_value = mock_provider

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["discover", "test-provider"])

        assert result.exit_code == 0
        assert "discovered-1" in result.output
        assert "1 model(s)" in result.output

    def test_discover_none(self, runner, mock_coordinator, mock_provider):
        """discover shows message when no models."""
        mock_provider.discover_models = AsyncMock(return_value=[])
        mock_coordinator.get_provider.return_value = mock_provider

        with patch("copaw.cli.providers_cmd.get_coordinator", return_value=mock_coordinator):
            result = runner.invoke(models_group, ["discover", "test-provider"])

        assert result.exit_code == 0
        assert "No models discovered" in result.output
