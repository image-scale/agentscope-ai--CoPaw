# -*- coding: utf-8 -*-
"""Tests for copaw.workspace module."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from copaw.workspace import (
    WorkspaceState,
    ConversationMessage,
    Conversation,
    Workspace,
    WorkspaceManager,
)
from copaw.config import AgentConfig, Config, AgentsConfig, AgentProfileRef


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return workspace_dir


@pytest.fixture
def mock_config():
    """Create a mock root configuration."""
    return Config(
        agents=AgentsConfig(
            active_agent="test-agent",
            profiles={
                "test-agent": AgentProfileRef(
                    id="test-agent",
                    workspace_dir="/tmp/test-workspace",
                ),
            },
        ),
    )


@pytest.fixture
def mock_agent_config():
    """Create a mock agent configuration."""
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        workspace_dir="/tmp/test-workspace",
    )


class TestWorkspaceState:
    """Tests for WorkspaceState enum."""

    def test_states_exist(self):
        """All expected states exist."""
        assert WorkspaceState.STOPPED == "stopped"
        assert WorkspaceState.STARTING == "starting"
        assert WorkspaceState.RUNNING == "running"
        assert WorkspaceState.STOPPING == "stopping"
        assert WorkspaceState.ERROR == "error"


class TestConversationMessage:
    """Tests for ConversationMessage."""

    def test_create_message(self):
        """Creates message with required fields."""
        msg = ConversationMessage(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp == 0.0
        assert msg.metadata == {}

    def test_to_dict(self):
        """Converts to dictionary."""
        msg = ConversationMessage(
            role="assistant",
            content="Hi there",
            timestamp=1234567890.0,
            metadata={"key": "value"},
        )

        data = msg.to_dict()

        assert data["role"] == "assistant"
        assert data["content"] == "Hi there"
        assert data["timestamp"] == 1234567890.0
        assert data["metadata"] == {"key": "value"}

    def test_from_dict(self):
        """Creates from dictionary."""
        data = {
            "role": "user",
            "content": "Test message",
            "timestamp": 123.0,
            "metadata": {"foo": "bar"},
        }

        msg = ConversationMessage.from_dict(data)

        assert msg.role == "user"
        assert msg.content == "Test message"
        assert msg.timestamp == 123.0
        assert msg.metadata == {"foo": "bar"}

    def test_from_dict_defaults(self):
        """Uses defaults for missing fields."""
        msg = ConversationMessage.from_dict({})

        assert msg.role == "user"
        assert msg.content == ""
        assert msg.timestamp == 0.0
        assert msg.metadata == {}


class TestConversation:
    """Tests for Conversation."""

    def test_create_conversation(self):
        """Creates conversation with required fields."""
        conv = Conversation(id="conv-1", agent_id="test-agent")

        assert conv.id == "conv-1"
        assert conv.agent_id == "test-agent"
        assert conv.messages == []
        assert conv.created_at == 0.0

    def test_add_message(self):
        """Adds messages to conversation."""
        conv = Conversation(id="conv-1", agent_id="test-agent")

        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there")

        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[0].content == "Hello"
        assert conv.messages[1].role == "assistant"

    def test_add_message_with_metadata(self):
        """Adds message with metadata."""
        conv = Conversation(id="conv-1", agent_id="test-agent")

        conv.add_message("user", "Test", custom_field="value")

        assert conv.messages[0].metadata["custom_field"] == "value"

    def test_add_message_updates_timestamp(self):
        """Adding message updates conversation timestamp."""
        conv = Conversation(id="conv-1", agent_id="test-agent")

        conv.add_message("user", "Hello")

        assert conv.updated_at > 0

    def test_to_dict(self):
        """Converts to dictionary."""
        conv = Conversation(
            id="conv-1",
            agent_id="test-agent",
            created_at=100.0,
            updated_at=200.0,
        )
        conv.messages.append(ConversationMessage(role="user", content="Hi"))

        data = conv.to_dict()

        assert data["id"] == "conv-1"
        assert data["agent_id"] == "test-agent"
        assert len(data["messages"]) == 1
        assert data["created_at"] == 100.0
        assert data["updated_at"] == 200.0

    def test_from_dict(self):
        """Creates from dictionary."""
        data = {
            "id": "conv-2",
            "agent_id": "agent-2",
            "messages": [{"role": "user", "content": "Test"}],
            "created_at": 50.0,
            "updated_at": 60.0,
            "metadata": {"key": "val"},
        }

        conv = Conversation.from_dict(data)

        assert conv.id == "conv-2"
        assert conv.agent_id == "agent-2"
        assert len(conv.messages) == 1
        assert conv.created_at == 50.0
        assert conv.metadata == {"key": "val"}


class TestWorkspace:
    """Tests for Workspace."""

    def test_create_workspace(self, tmp_workspace):
        """Creates workspace with agent ID."""
        with patch("copaw.workspace.get_agent_workspace", return_value=tmp_workspace):
            ws = Workspace("test-agent")

        assert ws.agent_id == "test-agent"
        assert ws.state == WorkspaceState.STOPPED
        assert ws.is_running is False

    def test_explicit_workspace_dir(self, tmp_workspace):
        """Uses explicit workspace directory."""
        ws = Workspace("test-agent", workspace_dir=tmp_workspace)

        assert ws.workspace_dir == tmp_workspace

    def test_start_workspace(self, tmp_workspace, mock_config, mock_agent_config):
        """Starts workspace successfully."""
        with patch("copaw.workspace.load_config", return_value=mock_config):
            with patch("copaw.workspace.load_agent_config", return_value=mock_agent_config):
                with patch("copaw.workspace.ProviderCoordinator"):
                    ws = Workspace("test-agent", workspace_dir=tmp_workspace)
                    result = ws.start()

        assert result is True
        assert ws.state == WorkspaceState.RUNNING
        assert ws.is_running is True

    def test_start_workspace_loads_config(self, tmp_workspace, mock_config, mock_agent_config):
        """Start loads configuration."""
        with patch("copaw.workspace.load_config", return_value=mock_config) as load_cfg:
            with patch("copaw.workspace.load_agent_config", return_value=mock_agent_config) as load_agent:
                with patch("copaw.workspace.ProviderCoordinator"):
                    ws = Workspace("test-agent", workspace_dir=tmp_workspace)
                    ws.start()

        load_cfg.assert_called_once()
        load_agent.assert_called_once_with("test-agent")

    def test_start_workspace_error(self, tmp_workspace):
        """Start handles errors."""
        with patch("copaw.workspace.load_config", side_effect=Exception("Config error")):
            ws = Workspace("test-agent", workspace_dir=tmp_workspace)
            result = ws.start()

        assert result is False
        assert ws.state == WorkspaceState.ERROR
        assert "Config error" in ws.error

    def test_stop_workspace(self, tmp_workspace, mock_config, mock_agent_config):
        """Stops workspace successfully."""
        with patch("copaw.workspace.load_config", return_value=mock_config):
            with patch("copaw.workspace.load_agent_config", return_value=mock_agent_config):
                with patch("copaw.workspace.ProviderCoordinator"):
                    ws = Workspace("test-agent", workspace_dir=tmp_workspace)
                    ws.start()
                    result = ws.stop()

        assert result is True
        assert ws.state == WorkspaceState.STOPPED
        assert ws.is_running is False

    def test_stop_already_stopped(self, tmp_workspace):
        """Stop on stopped workspace returns True."""
        with patch("copaw.workspace.get_agent_workspace", return_value=tmp_workspace):
            ws = Workspace("test-agent")
            result = ws.stop()

        assert result is True

    def test_restart_workspace(self, tmp_workspace, mock_config, mock_agent_config):
        """Restarts workspace."""
        with patch("copaw.workspace.load_config", return_value=mock_config):
            with patch("copaw.workspace.load_agent_config", return_value=mock_agent_config):
                with patch("copaw.workspace.ProviderCoordinator"):
                    ws = Workspace("test-agent", workspace_dir=tmp_workspace)
                    ws.start()
                    result = ws.restart()

        assert result is True
        assert ws.is_running is True

    def test_get_system_prompt(self, tmp_workspace, mock_config, mock_agent_config):
        """Gets system prompt from files."""
        (tmp_workspace / "AGENTS.md").write_text("# Agent\nYou are helpful.")
        (tmp_workspace / "PROFILE.md").write_text("# Profile\nBe concise.")

        mock_agent_config.system_prompt_files = ["AGENTS.md", "PROFILE.md"]

        with patch("copaw.workspace.load_config", return_value=mock_config):
            with patch("copaw.workspace.load_agent_config", return_value=mock_agent_config):
                with patch("copaw.workspace.ProviderCoordinator"):
                    ws = Workspace("test-agent", workspace_dir=tmp_workspace)
                    ws.start()
                    prompt = ws.get_system_prompt()

        assert "You are helpful" in prompt
        assert "Be concise" in prompt

    def test_get_system_prompt_missing_file(self, tmp_workspace, mock_config, mock_agent_config):
        """Skips missing prompt files."""
        (tmp_workspace / "AGENTS.md").write_text("# Agent")

        mock_agent_config.system_prompt_files = ["AGENTS.md", "MISSING.md"]

        with patch("copaw.workspace.load_config", return_value=mock_config):
            with patch("copaw.workspace.load_agent_config", return_value=mock_agent_config):
                with patch("copaw.workspace.ProviderCoordinator"):
                    ws = Workspace("test-agent", workspace_dir=tmp_workspace)
                    ws.start()
                    prompt = ws.get_system_prompt()

        assert "Agent" in prompt
        assert "MISSING" not in prompt


class TestWorkspaceConversations:
    """Tests for Workspace conversation management."""

    @pytest.fixture
    def running_workspace(self, tmp_workspace, mock_config, mock_agent_config):
        """Create a running workspace."""
        with patch("copaw.workspace.load_config", return_value=mock_config):
            with patch("copaw.workspace.load_agent_config", return_value=mock_agent_config):
                with patch("copaw.workspace.ProviderCoordinator"):
                    ws = Workspace("test-agent", workspace_dir=tmp_workspace)
                    ws.start()
                    yield ws
                    ws.stop()

    def test_create_conversation(self, running_workspace):
        """Creates a new conversation."""
        conv = running_workspace.create_conversation("conv-1")

        assert conv.id == "conv-1"
        assert conv.agent_id == "test-agent"

    def test_create_conversation_auto_id(self, running_workspace):
        """Creates conversation with auto-generated ID."""
        conv = running_workspace.create_conversation()

        assert conv.id is not None
        assert len(conv.id) > 0

    def test_get_conversation(self, running_workspace):
        """Gets a conversation by ID."""
        running_workspace.create_conversation("conv-1")

        conv = running_workspace.get_conversation("conv-1")

        assert conv is not None
        assert conv.id == "conv-1"

    def test_get_conversation_not_found(self, running_workspace):
        """Returns None for unknown conversation."""
        conv = running_workspace.get_conversation("unknown")

        assert conv is None

    def test_get_active_conversation(self, running_workspace):
        """Gets the active conversation."""
        running_workspace.create_conversation("conv-1")

        conv = running_workspace.get_active_conversation()

        assert conv is not None
        assert conv.id == "conv-1"

    def test_set_active_conversation(self, running_workspace):
        """Sets the active conversation."""
        running_workspace.create_conversation("conv-1")
        running_workspace.create_conversation("conv-2")

        result = running_workspace.set_active_conversation("conv-1")

        assert result is True
        assert running_workspace.get_active_conversation().id == "conv-1"

    def test_set_active_conversation_not_found(self, running_workspace):
        """Returns False for unknown conversation."""
        result = running_workspace.set_active_conversation("unknown")

        assert result is False

    def test_list_conversations(self, running_workspace):
        """Lists all conversations."""
        running_workspace.create_conversation("conv-1")
        running_workspace.create_conversation("conv-2")

        convs = running_workspace.list_conversations()

        assert len(convs) == 2
        ids = [c.id for c in convs]
        assert "conv-1" in ids
        assert "conv-2" in ids

    def test_delete_conversation(self, running_workspace):
        """Deletes a conversation."""
        running_workspace.create_conversation("conv-1")

        result = running_workspace.delete_conversation("conv-1")

        assert result is True
        assert running_workspace.get_conversation("conv-1") is None

    def test_delete_conversation_not_found(self, running_workspace):
        """Returns False for unknown conversation."""
        result = running_workspace.delete_conversation("unknown")

        assert result is False

    def test_delete_active_conversation_clears_active(self, running_workspace):
        """Deleting active conversation clears active."""
        running_workspace.create_conversation("conv-1")

        running_workspace.delete_conversation("conv-1")

        assert running_workspace.get_active_conversation() is None

    def test_save_and_load_conversations(self, tmp_workspace, mock_config, mock_agent_config):
        """Conversations persist across restarts."""
        with patch("copaw.workspace.load_config", return_value=mock_config):
            with patch("copaw.workspace.load_agent_config", return_value=mock_agent_config):
                with patch("copaw.workspace.ProviderCoordinator"):
                    ws1 = Workspace("test-agent", workspace_dir=tmp_workspace)
                    ws1.start()
                    conv = ws1.create_conversation("conv-1")
                    conv.add_message("user", "Hello")
                    ws1.stop()

                    ws2 = Workspace("test-agent", workspace_dir=tmp_workspace)
                    ws2.start()

        loaded = ws2.get_conversation("conv-1")
        assert loaded is not None
        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "Hello"


class TestWorkspaceManager:
    """Tests for WorkspaceManager."""

    def test_create_manager(self):
        """Creates workspace manager."""
        manager = WorkspaceManager()

        assert manager.active_workspace is None
        assert manager.list_workspaces() == []

    def test_load_workspace(self, tmp_workspace):
        """Loads a workspace."""
        with patch("copaw.workspace.get_agent_workspace", return_value=tmp_workspace):
            manager = WorkspaceManager()
            ws = manager.load_workspace("test-agent")

        assert ws is not None
        assert ws.agent_id == "test-agent"

    def test_load_workspace_returns_existing(self, tmp_workspace):
        """Returns existing workspace."""
        with patch("copaw.workspace.get_agent_workspace", return_value=tmp_workspace):
            manager = WorkspaceManager()
            ws1 = manager.load_workspace("test-agent")
            ws2 = manager.load_workspace("test-agent")

        assert ws1 is ws2

    def test_get_workspace(self, tmp_workspace):
        """Gets a loaded workspace."""
        with patch("copaw.workspace.get_agent_workspace", return_value=tmp_workspace):
            manager = WorkspaceManager()
            manager.load_workspace("test-agent")

            ws = manager.get_workspace("test-agent")

        assert ws is not None

    def test_get_workspace_not_found(self):
        """Returns None for unloaded workspace."""
        manager = WorkspaceManager()

        ws = manager.get_workspace("unknown")

        assert ws is None

    def test_start_workspace(self, tmp_workspace, mock_config, mock_agent_config):
        """Starts a workspace."""
        with patch("copaw.workspace.get_agent_workspace", return_value=tmp_workspace):
            with patch("copaw.workspace.load_config", return_value=mock_config):
                with patch("copaw.workspace.load_agent_config", return_value=mock_agent_config):
                    with patch("copaw.workspace.ProviderCoordinator"):
                        manager = WorkspaceManager()
                        ws = manager.start_workspace("test-agent")

        assert ws.is_running is True

    def test_stop_workspace(self, tmp_workspace, mock_config, mock_agent_config):
        """Stops a workspace."""
        with patch("copaw.workspace.get_agent_workspace", return_value=tmp_workspace):
            with patch("copaw.workspace.load_config", return_value=mock_config):
                with patch("copaw.workspace.load_agent_config", return_value=mock_agent_config):
                    with patch("copaw.workspace.ProviderCoordinator"):
                        manager = WorkspaceManager()
                        manager.start_workspace("test-agent")
                        result = manager.stop_workspace("test-agent")

        assert result is True

    def test_stop_workspace_not_found(self):
        """Returns False for unknown workspace."""
        manager = WorkspaceManager()

        result = manager.stop_workspace("unknown")

        assert result is False

    def test_set_active_workspace(self, tmp_workspace):
        """Sets the active workspace."""
        with patch("copaw.workspace.get_agent_workspace", return_value=tmp_workspace):
            manager = WorkspaceManager()
            manager.load_workspace("test-agent")

            result = manager.set_active_workspace("test-agent")

        assert result is True
        assert manager.active_workspace.agent_id == "test-agent"

    def test_set_active_workspace_not_found(self):
        """Returns False for unknown workspace."""
        manager = WorkspaceManager()

        result = manager.set_active_workspace("unknown")

        assert result is False

    def test_list_workspaces(self, tmp_workspace):
        """Lists all loaded workspaces."""
        with patch("copaw.workspace.get_agent_workspace", return_value=tmp_workspace):
            manager = WorkspaceManager()
            manager.load_workspace("agent-1")
            manager.load_workspace("agent-2")

            workspaces = manager.list_workspaces()

        assert len(workspaces) == 2

    def test_stop_all(self, tmp_workspace, mock_config, mock_agent_config):
        """Stops all workspaces."""
        with patch("copaw.workspace.get_agent_workspace", return_value=tmp_workspace):
            with patch("copaw.workspace.load_config", return_value=mock_config):
                with patch("copaw.workspace.load_agent_config", return_value=mock_agent_config):
                    with patch("copaw.workspace.ProviderCoordinator"):
                        manager = WorkspaceManager()
                        manager.start_workspace("test-agent")
                        manager.stop_all()

        ws = manager.get_workspace("test-agent")
        assert ws.state == WorkspaceState.STOPPED

    def test_load_from_config(self, tmp_workspace, mock_config):
        """Loads workspaces from config."""
        with patch("copaw.workspace.get_agent_workspace", return_value=tmp_workspace):
            with patch("copaw.workspace.load_config", return_value=mock_config):
                manager = WorkspaceManager()
                manager.load_from_config()

        assert manager.get_workspace("test-agent") is not None
        assert manager.active_workspace.agent_id == "test-agent"
