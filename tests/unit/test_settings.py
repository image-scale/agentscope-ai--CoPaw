# -*- coding: utf-8 -*-
"""Tests for copaw.settings module - environment variable loading and paths."""

import os
from pathlib import Path

import pytest


class TestEnvReader:
    """Tests for EnvReader utility class."""

    def test_get_bool_returns_default_when_not_set(self, monkeypatch):
        """get_bool returns default when VAR is not set."""
        monkeypatch.delenv("TEST_BOOL_VAR", raising=False)

        from copaw.settings import EnvReader

        assert EnvReader.get_bool("TEST_BOOL_VAR", False) is False
        assert EnvReader.get_bool("TEST_BOOL_VAR", True) is True

    def test_get_bool_returns_true_for_truthy_values(self, monkeypatch):
        """get_bool returns True for 'true', '1', 'yes' (case-insensitive)."""
        from copaw.settings import EnvReader

        for truthy_val in ["true", "True", "TRUE", "1", "yes", "YES", "Yes"]:
            monkeypatch.setenv("TEST_BOOL_VAR", truthy_val)
            assert EnvReader.get_bool("TEST_BOOL_VAR", False) is True

    def test_get_bool_returns_false_for_falsy_values(self, monkeypatch):
        """get_bool returns False for other values."""
        from copaw.settings import EnvReader

        for falsy_val in ["false", "False", "0", "no", "NO", "random"]:
            monkeypatch.setenv("TEST_BOOL_VAR", falsy_val)
            assert EnvReader.get_bool("TEST_BOOL_VAR", True) is False

    def test_get_int_returns_default_when_not_set(self, monkeypatch):
        """get_int returns default when VAR is not set."""
        monkeypatch.delenv("TEST_INT_VAR", raising=False)

        from copaw.settings import EnvReader

        assert EnvReader.get_int("TEST_INT_VAR", 10) == 10
        assert EnvReader.get_int("TEST_INT_VAR", 42) == 42

    def test_get_int_parses_valid_int(self, monkeypatch):
        """get_int parses valid integer strings."""
        from copaw.settings import EnvReader

        monkeypatch.setenv("TEST_INT_VAR", "25")
        assert EnvReader.get_int("TEST_INT_VAR", 0) == 25

    def test_get_int_clamps_to_min_value(self, monkeypatch):
        """get_int clamps value to min_value."""
        from copaw.settings import EnvReader

        monkeypatch.setenv("TEST_INT_VAR", "-5")
        assert EnvReader.get_int("TEST_INT_VAR", 10, min_value=0) == 0

    def test_get_int_clamps_to_max_value(self, monkeypatch):
        """get_int clamps value to max_value."""
        from copaw.settings import EnvReader

        monkeypatch.setenv("TEST_INT_VAR", "150")
        assert EnvReader.get_int("TEST_INT_VAR", 10, max_value=100) == 100

    def test_get_int_returns_default_for_invalid(self, monkeypatch):
        """get_int returns default for invalid values."""
        from copaw.settings import EnvReader

        monkeypatch.setenv("TEST_INT_VAR", "not_an_int")
        assert EnvReader.get_int("TEST_INT_VAR", 99) == 99

    def test_get_float_returns_default_when_not_set(self, monkeypatch):
        """get_float returns default when VAR is not set."""
        monkeypatch.delenv("TEST_FLOAT_VAR", raising=False)

        from copaw.settings import EnvReader

        assert EnvReader.get_float("TEST_FLOAT_VAR", 1.5) == 1.5

    def test_get_float_parses_valid_float(self, monkeypatch):
        """get_float parses valid float strings."""
        from copaw.settings import EnvReader

        monkeypatch.setenv("TEST_FLOAT_VAR", "3.14")
        assert EnvReader.get_float("TEST_FLOAT_VAR", 0.0) == 3.14

    def test_get_float_clamps_to_bounds(self, monkeypatch):
        """get_float clamps value to min/max bounds."""
        from copaw.settings import EnvReader

        monkeypatch.setenv("TEST_FLOAT_VAR", "-0.5")
        assert EnvReader.get_float("TEST_FLOAT_VAR", 1.0, min_value=0.0) == 0.0

        monkeypatch.setenv("TEST_FLOAT_VAR", "15.0")
        assert EnvReader.get_float("TEST_FLOAT_VAR", 1.0, max_value=10.0) == 10.0

    def test_get_float_returns_default_for_infinity_when_disallowed(self, monkeypatch):
        """get_float returns default for infinity when allow_inf=False."""
        from copaw.settings import EnvReader

        monkeypatch.setenv("TEST_FLOAT_VAR", "inf")
        assert EnvReader.get_float("TEST_FLOAT_VAR", 5.0, allow_inf=False) == 5.0

        monkeypatch.setenv("TEST_FLOAT_VAR", "-inf")
        assert EnvReader.get_float("TEST_FLOAT_VAR", 5.0, allow_inf=False) == 5.0

    def test_get_float_allows_infinity_when_enabled(self, monkeypatch):
        """get_float allows infinity when allow_inf=True."""
        from copaw.settings import EnvReader

        monkeypatch.setenv("TEST_FLOAT_VAR", "inf")
        assert EnvReader.get_float("TEST_FLOAT_VAR", 5.0, allow_inf=True) == float("inf")

    def test_get_float_returns_default_for_invalid(self, monkeypatch):
        """get_float returns default for invalid values."""
        from copaw.settings import EnvReader

        monkeypatch.setenv("TEST_FLOAT_VAR", "not_a_float")
        assert EnvReader.get_float("TEST_FLOAT_VAR", 2.5) == 2.5

    def test_get_str_returns_default_when_not_set(self, monkeypatch):
        """get_str returns default when VAR is not set."""
        monkeypatch.delenv("TEST_STR_VAR", raising=False)

        from copaw.settings import EnvReader

        assert EnvReader.get_str("TEST_STR_VAR", "default") == "default"

    def test_get_str_returns_value_when_set(self, monkeypatch):
        """get_str returns the value when VAR is set."""
        from copaw.settings import EnvReader

        monkeypatch.setenv("TEST_STR_VAR", "custom_value")
        assert EnvReader.get_str("TEST_STR_VAR", "default") == "custom_value"


class TestPathConstants:
    """Tests for path constants."""

    def test_working_dir_from_env(self, monkeypatch, tmp_path):
        """WORKING_DIR can be set via COPAW_WORKING_DIR env var."""
        custom_path = tmp_path / "custom_copaw"
        monkeypatch.setenv("COPAW_WORKING_DIR", str(custom_path))

        import importlib
        import copaw.settings as settings

        importlib.reload(settings)

        assert settings.WORKING_DIR == custom_path.resolve()

    def test_secret_dir_from_env(self, monkeypatch, tmp_path):
        """SECRET_DIR can be set via COPAW_SECRET_DIR env var."""
        custom_secret = tmp_path / "custom_secret"
        monkeypatch.setenv("COPAW_SECRET_DIR", str(custom_secret))

        import importlib
        import copaw.settings as settings

        importlib.reload(settings)

        assert settings.SECRET_DIR == custom_secret.resolve()

    def test_default_working_dir_is_home_copaw(self, monkeypatch):
        """Default WORKING_DIR is ~/.copaw."""
        monkeypatch.delenv("COPAW_WORKING_DIR", raising=False)

        import importlib
        import copaw.settings as settings

        importlib.reload(settings)

        expected = Path("~/.copaw").expanduser().resolve()
        assert settings.WORKING_DIR == expected


class TestLLMConstants:
    """Tests for LLM configuration constants."""

    def test_llm_max_retries_default(self, monkeypatch):
        """LLM_MAX_RETRIES defaults to 3."""
        monkeypatch.delenv("COPAW_LLM_MAX_RETRIES", raising=False)

        import importlib
        import copaw.settings as settings

        importlib.reload(settings)

        assert settings.LLM_MAX_RETRIES == 3

    def test_llm_backoff_base_default(self, monkeypatch):
        """LLM_BACKOFF_BASE defaults to 1.0."""
        monkeypatch.delenv("COPAW_LLM_BACKOFF_BASE", raising=False)

        import importlib
        import copaw.settings as settings

        importlib.reload(settings)

        assert settings.LLM_BACKOFF_BASE == 1.0

    def test_llm_max_concurrent_default(self, monkeypatch):
        """LLM_MAX_CONCURRENT defaults to 10."""
        monkeypatch.delenv("COPAW_LLM_MAX_CONCURRENT", raising=False)

        import importlib
        import copaw.settings as settings

        importlib.reload(settings)

        assert settings.LLM_MAX_CONCURRENT == 10


class TestFileConstants:
    """Tests for file name constants."""

    def test_jobs_file_default(self, monkeypatch):
        """JOBS_FILE defaults to 'jobs.json'."""
        monkeypatch.delenv("COPAW_JOBS_FILE", raising=False)

        import importlib
        import copaw.settings as settings

        importlib.reload(settings)

        assert settings.JOBS_FILE == "jobs.json"

    def test_config_file_default(self, monkeypatch):
        """CONFIG_FILE defaults to 'config.json'."""
        monkeypatch.delenv("COPAW_CONFIG_FILE", raising=False)

        import importlib
        import copaw.settings as settings

        importlib.reload(settings)

        assert settings.CONFIG_FILE == "config.json"

    def test_chats_file_default(self, monkeypatch):
        """CHATS_FILE defaults to 'chats.json'."""
        monkeypatch.delenv("COPAW_CHATS_FILE", raising=False)

        import importlib
        import copaw.settings as settings

        importlib.reload(settings)

        assert settings.CHATS_FILE == "chats.json"
