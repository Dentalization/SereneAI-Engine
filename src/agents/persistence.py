"""Conversation persistence and checkpointing system.

This module provides:
- JSON-based checkpoint storage for conversation state
- Session resumption capabilities
- State history tracking
- Cleanup utilities for old sessions
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from src.agents.state_models import AgentState, CheckpointState

logger = logging.getLogger(__name__)


class ConversationPersistence:
    """Manages conversation state persistence to disk."""

    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        """Initialize persistence manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"ConversationPersistence: Initialized with dir {self.checkpoint_dir}")

    def save_checkpoint(self, state: AgentState, version: int = 1) -> bool:
        """Save conversation state to checkpoint file.

        Args:
            state: Current agent state to persist
            version: Checkpoint version number

        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint = CheckpointState(
                conversation_id=state.conversation_id,
                state=state,
                timestamp=datetime.now(),
                version=version,
            )

            file_path = self._get_checkpoint_path(state.conversation_id)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"ConversationPersistence: Saved checkpoint for {state.conversation_id}")
            return True

        except Exception as e:
            logger.error(f"ConversationPersistence: Save failed - {e}")
            return False

    def load_checkpoint(self, conversation_id: str) -> Optional[AgentState]:
        """Load conversation state from checkpoint file.

        Args:
            conversation_id: ID of conversation to resume

        Returns:
            AgentState if found and valid, None otherwise
        """
        try:
            file_path = self._get_checkpoint_path(conversation_id)

            if not file_path.exists():
                logger.warning(f"ConversationPersistence: No checkpoint found for {conversation_id}")
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            checkpoint = CheckpointState.from_dict(data)
            logger.info(
                f"ConversationPersistence: Loaded checkpoint for {conversation_id} "
                f"(version {checkpoint.version}, saved {checkpoint.timestamp})"
            )

            return checkpoint.state

        except Exception as e:
            logger.error(f"ConversationPersistence: Load failed - {e}")
            return None

    def list_conversations(self, limit: int = 50) -> List[Dict[str, str]]:
        """List all stored conversations with metadata.

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversation metadata dicts
        """
        conversations = []

        try:
            checkpoint_files = sorted(
                self.checkpoint_dir.glob("conv_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            for file_path in checkpoint_files[:limit]:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    conversations.append({
                        "conversation_id": data["conversation_id"],
                        "timestamp": data["timestamp"],
                        "version": data.get("version", 1),
                        "file_path": str(file_path),
                    })
                except Exception as e:
                    logger.warning(f"ConversationPersistence: Skipping corrupt file {file_path} - {e}")

            logger.info(f"ConversationPersistence: Found {len(conversations)} conversations")
            return conversations

        except Exception as e:
            logger.error(f"ConversationPersistence: List failed - {e}")
            return []

    def delete_checkpoint(self, conversation_id: str) -> bool:
        """Delete a conversation checkpoint.

        Args:
            conversation_id: ID of conversation to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self._get_checkpoint_path(conversation_id)

            if file_path.exists():
                file_path.unlink()
                logger.info(f"ConversationPersistence: Deleted checkpoint for {conversation_id}")
                return True
            else:
                logger.warning(f"ConversationPersistence: No checkpoint to delete for {conversation_id}")
                return False

        except Exception as e:
            logger.error(f"ConversationPersistence: Delete failed - {e}")
            return False

    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """Remove checkpoint files older than specified days.

        Args:
            days: Delete checkpoints older than this many days

        Returns:
            Number of files deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted_count = 0

            for file_path in self.checkpoint_dir.glob("conv_*.json"):
                try:
                    # Check file modification time
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                    if file_mtime < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"ConversationPersistence: Cleaned up {file_path.name}")

                except Exception as e:
                    logger.warning(f"ConversationPersistence: Failed to clean {file_path} - {e}")

            logger.info(f"ConversationPersistence: Cleaned up {deleted_count} old checkpoints")
            return deleted_count

        except Exception as e:
            logger.error(f"ConversationPersistence: Cleanup failed - {e}")
            return 0

    def _get_checkpoint_path(self, conversation_id: str) -> Path:
        """Get file path for a conversation checkpoint.

        Args:
            conversation_id: Conversation ID

        Returns:
            Path to checkpoint file
        """
        # Sanitize conversation_id for filename
        safe_id = "".join(c for c in conversation_id if c.isalnum() or c in "._-")
        return self.checkpoint_dir / f"{safe_id}.json"


# Global persistence manager instance
_persistence: Optional[ConversationPersistence] = None


def get_persistence() -> ConversationPersistence:
    """Get or create global persistence manager instance."""
    global _persistence
    if _persistence is None:
        _persistence = ConversationPersistence()
    return _persistence


def save_state(state: AgentState) -> bool:
    """Convenience function to save state."""
    return get_persistence().save_checkpoint(state)


def load_state(conversation_id: str) -> Optional[AgentState]:
    """Convenience function to load state."""
    return get_persistence().load_checkpoint(conversation_id)


def resume_conversation(conversation_id: str) -> Optional[AgentState]:
    """Resume a conversation from checkpoint.

    Args:
        conversation_id: ID of conversation to resume

    Returns:
        Loaded state or None if not found
    """
    state = load_state(conversation_id)
    if state:
        logger.info(f"Resumed conversation {conversation_id} with {len(state.history)} messages")
    return state