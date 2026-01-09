"""Tests for history module."""

import pytest
import time

from conversation_toolkit import (
    Conversation,
    Role,
    HistoryManager,
    ConversationHistory,
    SummarizationStrategy,
)


class TestHistoryManager:
    """Test HistoryManager class."""

    def test_create_manager(self):
        """Test creating history manager."""
        manager = HistoryManager(max_conversations=100)
        assert manager.max_conversations == 100

    def test_add_conversation(self):
        """Test adding a conversation."""
        manager = HistoryManager()
        conv = Conversation()
        manager.add_conversation(conv)
        assert manager.get_conversation(conv.id) is not None

    def test_get_conversation(self):
        """Test getting a conversation."""
        manager = HistoryManager()
        conv = Conversation()
        manager.add_conversation(conv)

        retrieved = manager.get_conversation(conv.id)
        assert retrieved is not None
        assert retrieved.id == conv.id

    def test_get_nonexistent_conversation(self):
        """Test getting nonexistent conversation."""
        manager = HistoryManager()
        assert manager.get_conversation("fake-id") is None

    def test_get_recent(self):
        """Test getting recent conversations."""
        manager = HistoryManager()

        # Create conversations with different timestamps
        conv1 = Conversation()
        conv1._updated_at = time.time() - 100  # Older
        conv2 = Conversation()
        conv2._updated_at = time.time() - 50   # Middle
        conv3 = Conversation()
        conv3._updated_at = time.time() - 10   # Newest

        manager.add_conversation(conv1)
        manager.add_conversation(conv2)
        manager.add_conversation(conv3)

        recent = manager.get_recent(limit=2)
        assert len(recent) == 2
        assert recent[0].id == conv3.id  # Newest first
        assert recent[1].id == conv2.id

    def test_search(self):
        """Test searching conversations."""
        manager = HistoryManager()

        conv1 = Conversation()
        conv1.add_user("The weather is sunny today.")
        manager.add_conversation(conv1)

        conv2 = Conversation()
        conv2.add_user("I love learning about AI.")
        manager.add_conversation(conv2)

        # Search for "weather"
        results = manager.search("weather", limit=10)
        assert len(results) == 1
        assert results[0]["conversation_id"] == conv1.id

        # Search for "AI"
        results = manager.search("AI", limit=10)
        assert len(results) == 1
        assert results[0]["conversation_id"] == conv2.id

    def test_search_case_insensitive(self):
        """Test that search is case insensitive."""
        manager = HistoryManager()
        conv = Conversation()
        conv.add_user("Hello WORLD")
        manager.add_conversation(conv)

        results = manager.search("world")
        assert len(results) == 1

    def test_delete_conversation(self):
        """Test deleting a conversation."""
        manager = HistoryManager()
        conv = Conversation()
        manager.add_conversation(conv)

        assert manager.get_conversation(conv.id) is not None
        assert manager.delete_conversation(conv.id) is True
        assert manager.get_conversation(conv.id) is None

    def test_delete_nonexistent_conversation(self):
        """Test deleting nonexistent conversation."""
        manager = HistoryManager()
        assert manager.delete_conversation("fake-id") is False

    def test_get_stats(self):
        """Test getting statistics."""
        manager = HistoryManager()

        conv1 = Conversation()
        conv1.add_user("Message 1")
        conv1.add_assistant("Response 1")

        conv2 = Conversation()
        conv2.add_user("Message 2")

        manager.add_conversation(conv1)
        manager.add_conversation(conv2)

        stats = manager.get_stats()
        assert stats["total_conversations"] == 2
        # conv1 has 2 messages (user + assistant), conv2 has 1 message (user)
        assert stats["total_messages"] == 3
        assert stats["total_estimated_tokens"] > 0

    def test_max_conversations_limit(self):
        """Test that max conversations limit is enforced."""
        manager = HistoryManager(max_conversations=3)

        # Add 5 conversations
        conversations = [Conversation() for _ in range(5)]
        for conv in conversations:
            manager.add_conversation(conv)

        # Should only keep 3 (newest)
        stats = manager.get_stats()
        assert stats["total_conversations"] == 3


class TestConversationHistory:
    """Test ConversationHistory class."""

    def test_create_history(self):
        """Test creating conversation history."""
        history = ConversationHistory(max_states=50)
        assert history.max_states == 50
        assert history._current_state is None

    def test_save_state(self):
        """Test saving a state."""
        history = ConversationHistory()
        conv = Conversation()
        conv.add_user("Message 1")
        conv.add_user("Message 2")

        state_id = history.save_state(conv.messages)
        assert state_id is not None
        assert state_id in history._states

    def test_get_state(self):
        """Test getting a state."""
        history = ConversationHistory()
        conv = Conversation()
        conv.add_user("Message 1")

        state_id = history.save_state(conv.messages)
        state = history.get_state(state_id)

        assert state is not None
        assert len(state) == 1

    def test_get_current_state(self):
        """Test getting current state."""
        history = ConversationHistory()
        conv = Conversation()
        conv.add_user("Message 1")

        history.save_state(conv.messages)
        current = history.get_current_state()

        assert current is not None
        assert len(current) == 1

    def test_get_previous_state(self):
        """Test getting previous state."""
        history = ConversationHistory()

        # Save first state
        conv1 = Conversation()
        conv1.add_user("Message 1")
        history.save_state(conv1.messages)

        # Save second state
        conv2 = Conversation()
        conv2.add_user("Message 2")
        history.save_state(conv2.messages)

        # Get previous should return first state
        previous = history.get_previous_state()
        assert previous is not None
        assert len(previous) == 1
        assert previous[0].content == "Message 1"

    def test_restore_state(self):
        """Test restoring a state."""
        history = ConversationHistory()

        conv1 = Conversation()
        conv1.add_user("Original")
        state_id = history.save_state(conv1.messages)

        # Modify
        conv2 = Conversation()
        conv2.add_user("Modified")

        # Restore
        restored = history.restore_state(state_id)
        assert restored is not None
        assert len(restored) == 1
        assert restored[0].content == "Original"

    def test_restore_nonexistent_state(self):
        """Test restoring nonexistent state."""
        history = ConversationHistory()
        assert history.restore_state("fake-id") is None

    def test_branch_from(self):
        """Test branching from a state."""
        history = ConversationHistory()

        conv = Conversation()
        conv.add_user("Original")
        state_id = history.save_state(conv.messages)

        # Create branch
        branch = history.branch_from(state_id)
        assert branch is not None
        assert len(branch) == 1
        assert branch[0].content == "Original"

    def test_max_states_limit(self):
        """Test that max states limit is enforced."""
        history = ConversationHistory(max_states=3)

        # Add 5 states
        for i in range(5):
            conv = Conversation()
            conv.add_user(f"Message {i}")
            history.save_state(conv.messages)

        # Should only keep 3 most recent
        assert len(history._states) == 3


class TestSummarizationStrategy:
    """Test SummarizationStrategy enum."""

    def test_strategies(self):
        """Test all strategy values."""
        assert SummarizationStrategy.SIMPLE == "simple"
        assert SummarizationStrategy.RECENT_FOCUS == "recent_focus"
        assert SummarizationStrategy.KEY_POINTS == "key_points"
        assert SummarizationStrategy.LLM_BASED == "llm_based"
