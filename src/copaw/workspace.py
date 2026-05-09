# -*- coding: utf-8 -*-
"""Workspace system for managing agent runtime environments.

Provides lifecycle management, service coordination, and state persistence
for agent workspaces.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from .config import (
    AgentConfig,
    Config,
    load_agent_config,
    load_config,
    get_agent_workspace,
)
from .providers.manager import ProviderCoordinator

logger = logging.getLogger(__name__)


class WorkspaceState(str, Enum):
    """Workspace lifecycle states."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ConversationMessage:
    """A single message in a conversation."""

    role: str
    content: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationMessage:
        """Create from dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Conversation:
    """A conversation with an agent."""

    id: str
    agent_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to the conversation."""
        import time

        msg = ConversationMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata,
        )
        self.messages.append(msg)
        self.updated_at = msg.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Conversation:
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            agent_id=data.get("agent_id", ""),
            messages=[
                ConversationMessage.from_dict(m)
                for m in data.get("messages", [])
            ],
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            metadata=data.get("metadata", {}),
        )


class Workspace:
    """Manages an agent's runtime environment.

    Provides:
    - Lifecycle management (start/stop)
    - Configuration loading
    - System prompt assembly
    - Conversation management
    - Service coordination
    """

    def __init__(
        self,
        agent_id: str,
        workspace_dir: Path | None = None,
    ) -> None:
        """Initialize workspace for an agent.

        Args:
            agent_id: The agent ID to load.
            workspace_dir: Optional explicit workspace path.
        """
        self._agent_id = agent_id
        self._state = WorkspaceState.STOPPED
        self._error: str | None = None

        if workspace_dir:
            self._workspace_dir = Path(workspace_dir)
        else:
            self._workspace_dir = get_agent_workspace(agent_id)

        self._agent_config: AgentConfig | None = None
        self._root_config: Config | None = None
        self._provider_coordinator: ProviderCoordinator | None = None
        self._conversations: Dict[str, Conversation] = {}
        self._active_conversation_id: str | None = None

    @property
    def agent_id(self) -> str:
        """Get the agent ID."""
        return self._agent_id

    @property
    def workspace_dir(self) -> Path:
        """Get the workspace directory."""
        return self._workspace_dir

    @property
    def state(self) -> WorkspaceState:
        """Get the current workspace state."""
        return self._state

    @property
    def error(self) -> str | None:
        """Get the error message if in ERROR state."""
        return self._error

    @property
    def is_running(self) -> bool:
        """Check if the workspace is running."""
        return self._state == WorkspaceState.RUNNING

    @property
    def agent_config(self) -> AgentConfig | None:
        """Get the agent configuration."""
        return self._agent_config

    @property
    def root_config(self) -> Config | None:
        """Get the root configuration."""
        return self._root_config

    @property
    def provider_coordinator(self) -> ProviderCoordinator | None:
        """Get the provider coordinator."""
        return self._provider_coordinator

    def start(self) -> bool:
        """Start the workspace.

        Returns:
            True if started successfully, False otherwise.
        """
        if self._state == WorkspaceState.RUNNING:
            return True

        self._state = WorkspaceState.STARTING
        self._error = None

        try:
            self._root_config = load_config()
            self._agent_config = load_agent_config(self._agent_id)
            self._provider_coordinator = ProviderCoordinator()
            self._load_conversations()

            self._state = WorkspaceState.RUNNING
            logger.info("Workspace started: %s", self._agent_id)
            return True

        except Exception as e:
            self._state = WorkspaceState.ERROR
            self._error = str(e)
            logger.error("Failed to start workspace: %s", e)
            return False

    def stop(self) -> bool:
        """Stop the workspace.

        Returns:
            True if stopped successfully, False otherwise.
        """
        if self._state == WorkspaceState.STOPPED:
            return True

        self._state = WorkspaceState.STOPPING

        try:
            self._save_conversations()
            self._provider_coordinator = None
            self._state = WorkspaceState.STOPPED
            logger.info("Workspace stopped: %s", self._agent_id)
            return True

        except Exception as e:
            self._state = WorkspaceState.ERROR
            self._error = str(e)
            logger.error("Failed to stop workspace: %s", e)
            return False

    def restart(self) -> bool:
        """Restart the workspace.

        Returns:
            True if restarted successfully, False otherwise.
        """
        self.stop()
        return self.start()

    def get_system_prompt(self) -> str:
        """Load and assemble the system prompt.

        Returns:
            The assembled system prompt.
        """
        if not self._agent_config:
            return ""

        parts: List[str] = []

        for filename in self._agent_config.system_prompt_files:
            path = self._workspace_dir / filename
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8").strip()
                    if content:
                        parts.append(content)
                except Exception as e:
                    logger.warning("Failed to read %s: %s", path, e)

        return "\n\n".join(parts)

    def create_conversation(self, conversation_id: str | None = None) -> Conversation:
        """Create a new conversation.

        Args:
            conversation_id: Optional ID for the conversation.

        Returns:
            The new conversation.
        """
        import time
        import uuid

        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        now = time.time()
        conversation = Conversation(
            id=conversation_id,
            agent_id=self._agent_id,
            created_at=now,
            updated_at=now,
        )
        self._conversations[conversation_id] = conversation
        self._active_conversation_id = conversation_id
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The conversation or None if not found.
        """
        return self._conversations.get(conversation_id)

    def get_active_conversation(self) -> Conversation | None:
        """Get the active conversation.

        Returns:
            The active conversation or None.
        """
        if self._active_conversation_id:
            return self._conversations.get(self._active_conversation_id)
        return None

    def set_active_conversation(self, conversation_id: str) -> bool:
        """Set the active conversation.

        Args:
            conversation_id: The conversation ID.

        Returns:
            True if set successfully, False if not found.
        """
        if conversation_id in self._conversations:
            self._active_conversation_id = conversation_id
            return True
        return False

    def list_conversations(self) -> List[Conversation]:
        """List all conversations.

        Returns:
            List of all conversations.
        """
        return list(self._conversations.values())

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: The conversation ID.

        Returns:
            True if deleted, False if not found.
        """
        if conversation_id not in self._conversations:
            return False

        del self._conversations[conversation_id]

        if self._active_conversation_id == conversation_id:
            self._active_conversation_id = None

        return True

    def _get_conversations_path(self) -> Path:
        """Get the path to conversations file."""
        return self._workspace_dir / "conversations.json"

    def _load_conversations(self) -> None:
        """Load conversations from disk."""
        path = self._get_conversations_path()
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._conversations = {
                cid: Conversation.from_dict(cdata)
                for cid, cdata in data.get("conversations", {}).items()
            }
            self._active_conversation_id = data.get("active_conversation_id")

        except Exception as e:
            logger.warning("Failed to load conversations: %s", e)

    def _save_conversations(self) -> None:
        """Save conversations to disk."""
        path = self._get_conversations_path()

        try:
            data = {
                "conversations": {
                    cid: c.to_dict()
                    for cid, c in self._conversations.items()
                },
                "active_conversation_id": self._active_conversation_id,
            }

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning("Failed to save conversations: %s", e)


class WorkspaceManager:
    """Manages multiple agent workspaces.

    Provides:
    - Workspace creation and loading
    - Active workspace management
    - Workspace switching
    """

    def __init__(self) -> None:
        """Initialize the workspace manager."""
        self._workspaces: Dict[str, Workspace] = {}
        self._active_workspace_id: str | None = None

    @property
    def active_workspace(self) -> Workspace | None:
        """Get the active workspace."""
        if self._active_workspace_id:
            return self._workspaces.get(self._active_workspace_id)
        return None

    def get_workspace(self, agent_id: str) -> Workspace | None:
        """Get a workspace by agent ID.

        Args:
            agent_id: The agent ID.

        Returns:
            The workspace or None if not found.
        """
        return self._workspaces.get(agent_id)

    def load_workspace(self, agent_id: str) -> Workspace:
        """Load or create a workspace for an agent.

        Args:
            agent_id: The agent ID.

        Returns:
            The workspace.
        """
        if agent_id in self._workspaces:
            return self._workspaces[agent_id]

        workspace = Workspace(agent_id)
        self._workspaces[agent_id] = workspace
        return workspace

    def start_workspace(self, agent_id: str) -> Workspace:
        """Load and start a workspace.

        Args:
            agent_id: The agent ID.

        Returns:
            The started workspace.
        """
        workspace = self.load_workspace(agent_id)
        if not workspace.is_running:
            workspace.start()
        return workspace

    def stop_workspace(self, agent_id: str) -> bool:
        """Stop a workspace.

        Args:
            agent_id: The agent ID.

        Returns:
            True if stopped, False if not found.
        """
        workspace = self._workspaces.get(agent_id)
        if workspace:
            return workspace.stop()
        return False

    def set_active_workspace(self, agent_id: str) -> bool:
        """Set the active workspace.

        Args:
            agent_id: The agent ID.

        Returns:
            True if set, False if not found.
        """
        if agent_id not in self._workspaces:
            return False
        self._active_workspace_id = agent_id
        return True

    def list_workspaces(self) -> List[Workspace]:
        """List all loaded workspaces.

        Returns:
            List of workspaces.
        """
        return list(self._workspaces.values())

    def stop_all(self) -> None:
        """Stop all workspaces."""
        for workspace in self._workspaces.values():
            workspace.stop()

    def load_from_config(self) -> None:
        """Load workspaces from root configuration."""
        config = load_config()
        for agent_id in config.agents.profiles:
            self.load_workspace(agent_id)

        if config.agents.active_agent:
            self._active_workspace_id = config.agents.active_agent
