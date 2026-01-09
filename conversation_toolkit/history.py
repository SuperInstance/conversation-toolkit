"""
Conversation history and summarization.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

from .conversation import Conversation, Message, Role, ConversationSummary


class SummarizationStrategy(str, Enum):
    """Strategies for summarizing conversations"""
    SIMPLE = "simple"  # Basic message count and topics
    RECENT_FOCUS = "recent_focus"  # Focus on recent messages
    KEY_POINTS = "key_points"  # Extract key points
    LLM_BASED = "llm_based"  # Use LLM to summarize


class HistoryManager:
    """
    Manage conversation history with compression and summarization.

    Example:
        history = HistoryManager()

        # Add conversation
        history.add_conversation(conversation)

        # Search conversations
        results = history.search("AI safety")

        # Get recent conversations
        recent = history.get_recent(limit=10)
    """

    def __init__(
        self,
        max_conversations: int = 1000,
        max_summaries_per_conversation: int = 10
    ):
        """
        Initialize the history manager.

        Args:
            max_conversations: Maximum conversations to store
            max_summaries_per_conversation: Max summaries per conversation
        """
        self.max_conversations = max_conversations
        self.max_summaries_per_conversation = max_summaries_per_conversation
        self._conversations: Dict[str, Conversation] = {}
        self._created_at = time.time()

    def add_conversation(self, conversation: Conversation) -> None:
        """
        Add or update a conversation.

        Args:
            conversation: Conversation to add
        """
        self._conversations[conversation.id] = conversation

        # Enforce max limit (remove oldest)
        if len(self._conversations) > self.max_conversations:
            oldest = sorted(self._conversations.items(), key=lambda x: x[1].created_at)[0]
            del self._conversations[oldest[0]]

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)

    def get_recent(self, limit: int = 10) -> List[Conversation]:
        """Get recent conversations."""
        sorted_conv = sorted(
            self._conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True
        )
        return sorted_conv[:limit]

    def search(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversations by content.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching conversations with snippets
        """
        query_lower = query.lower()
        results = []

        for conv in self._conversations.values():
            # Search in messages
            for msg in conv.messages:
                if query_lower in msg.content.lower():
                    results.append({
                        "conversation_id": conv.id,
                        "conversation_updated": conv.updated_at,
                        "message_id": msg.timestamp,
                        "snippet": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    })
                    break

            if len(results) >= limit:
                break

        return results

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns True if deleted."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored conversations."""
        total_messages = sum(c.message_count for c in self._conversations.values())
        total_tokens = sum(c.estimated_tokens for c in self._conversations.values())

        return {
            "total_conversations": len(self._conversations),
            "total_messages": total_messages,
            "total_estimated_tokens": total_tokens,
            "oldest_conversation": min(
                (c.created_at for c in self._conversations.values()),
                default=None
            ),
            "newest_conversation": max(
                (c.updated_at for c in self._conversations.values()),
                default=None
            )
        }


class ConversationHistory:
    """
    Track history within a single conversation.

    Useful for implementing conversation features like:
    - Branching conversations
    - Undo/redo
    - Forking conversations

    Example:
        history = ConversationHistory()

        # Save current state
        history.save_state(conversation)

        # Get previous state
        previous = history.get_previous_state()

        # Branch from a point
        branch = history.branch_from(state_id)
    """

    def __init__(self, max_states: int = 50):
        """
        Initialize conversation history.

        Args:
            max_states: Maximum states to keep
        """
        self.max_states = max_states
        self._states: Dict[str, List[Message]] = {}
        self._state_order: List[str] = []
        self._current_state: Optional[str] = None
        self._state_counter: int = 0  # Monotonically increasing counter

    def save_state(self, messages: List[Message]) -> str:
        """
        Save current conversation state.

        Args:
            messages: Current messages

        Returns:
            State ID
        """
        state_id = f"state_{self._state_counter}_{int(time.time())}"
        self._state_counter += 1

        # Deep copy messages
        state_messages = [m.copy() for m in messages]

        self._states[state_id] = state_messages
        self._state_order.append(state_id)
        self._current_state = state_id

        # Enforce max limit
        while len(self._state_order) > self.max_states:
            old_state = self._state_order.pop(0)
            del self._states[old_state]

        return state_id

    def get_state(self, state_id: str) -> Optional[List[Message]]:
        """Get a saved state."""
        return self._states.get(state_id)

    def get_current_state(self) -> Optional[List[Message]]:
        """Get current state."""
        if self._current_state:
            return self.get_state(self._current_state)
        return None

    def get_previous_state(self) -> Optional[List[Message]]:
        """Get previous state."""
        if not self._state_order:
            return None

        current_idx = self._state_order.index(self._current_state) if self._current_state else len(self._state_order)
        if current_idx > 0:
            prev_state_id = self._state_order[current_idx - 1]
            return self.get_state(prev_state_id)

        return None

    def restore_state(self, state_id: str) -> Optional[List[Message]]:
        """Restore to a specific state."""
        if state_id not in self._states:
            return None

        self._current_state = state_id
        return self.get_state(state_id)

    def branch_from(self, state_id: str) -> Optional[List[Message]]:
        """Create a new branch from a state."""
        state = self.get_state(state_id)
        if state is None:
            return None

        # Create new state from old (save_state returns state_id, but we want messages)
        self.save_state(state)
        # Return the messages (state) that were saved
        return state
