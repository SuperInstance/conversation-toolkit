"""Tests for context module."""

import pytest

from conversation_toolkit import (
    Message,
    Role,
    ContextManager,
    ContextStrategy,
    TokenEstimator,
    ContextWindowInfo,
)


class TestTokenEstimator:
    """Test TokenEstimator class."""

    def test_default_estimator(self):
        """Test default token estimator."""
        estimator = TokenEstimator()
        msg = Message(role=Role.USER, content="This is a test message")
        tokens = estimator.estimate_message(msg)
        assert tokens > 0

    def test_provider_specific(self):
        """Test provider-specific estimation."""
        openai_estimator = TokenEstimator(provider="openai")
        anthropic_estimator = TokenEstimator(provider="anthropic")

        msg = Message(role=Role.USER, content="Test message content here")

        # Anthropic typically has higher token count for same text
        openai_tokens = openai_estimator.estimate_message(msg)
        anthropic_tokens = anthropic_estimator.estimate_message(msg)

        assert anthropic_tokens >= openai_tokens

    def test_estimate_messages(self):
        """Test estimating multiple messages."""
        estimator = TokenEstimator()
        messages = [
            Message(role=Role.USER, content="First message"),
            Message(role=Role.ASSISTANT, content="Second message"),
            Message(role=Role.USER, content="Third message"),
        ]
        tokens = estimator.estimate_messages(messages)
        assert tokens > 0

    def test_estimate_string(self):
        """Test estimating string directly."""
        estimator = TokenEstimator()
        text = "This is some text to estimate"
        tokens = estimator.estimate_string(text)
        assert tokens > 0


class TestContextWindowInfo:
    """Test ContextWindowInfo class."""

    def test_create_info(self):
        """Test creating context window info."""
        info = ContextWindowInfo(
            total_tokens=100000,
            available_tokens=50000,
            used_tokens=50000,
            usage_percent=50.0,
            messages_count=10
        )
        assert info.total_tokens == 100000
        assert info.available_tokens == 50000
        assert info.usage_percent == 50.0

    def test_info_with_strategy(self):
        """Test info with strategy."""
        info = ContextWindowInfo(
            total_tokens=100000,
            available_tokens=50000,
            used_tokens=50000,
            usage_percent=50.0,
            messages_count=10,
            strategy_used=ContextStrategy.DROP_OLDEST
        )
        assert info.strategy_used == ContextStrategy.DROP_OLDEST


class TestContextManager:
    """Test ContextManager class."""

    def test_create_manager(self):
        """Test creating context manager."""
        manager = ContextManager(
            max_tokens=128000,
            strategy=ContextStrategy.DROP_OLDEST,
            reserve_tokens=1000
        )
        assert manager.max_tokens == 128000
        assert manager.reserve_tokens == 1000

    def test_check_context_under_limit(self):
        """Test checking context when under limit."""
        manager = ContextManager(max_tokens=10000)
        messages = [
            Message(role=Role.USER, content="Short message"),
        ]
        info = manager.check_context(messages)
        assert info.available_tokens > 0
        assert info.messages_count == 1

    def test_check_context_over_limit(self):
        """Test checking context when over limit."""
        manager = ContextManager(max_tokens=100)
        messages = [
            Message(role=Role.USER, content="x" * 500),
        ]
        info = manager.check_context(messages)
        # available_tokens is clamped at 0, but usage_percent shows overflow
        assert info.available_tokens == 0
        assert info.usage_percent > 100

    def test_trim_to_fit_no_trim_needed(self):
        """Test trimming when no trim needed."""
        manager = ContextManager(max_tokens=10000)
        messages = [
            Message(role=Role.USER, content="Short message"),
        ]
        trimmed = manager.trim_to_fit(messages)
        assert len(trimmed) == 1

    def test_trim_to_fit_drop_oldest(self):
        """Test trimming with DROP_OLDEST strategy."""
        manager = ContextManager(
            max_tokens=30,  # Very small limit
            strategy=ContextStrategy.DROP_OLDEST,
            reserve_tokens=5
        )
        messages = [
            Message(role=Role.USER, content="x" * 400),  # ~100 tokens
            Message(role=Role.USER, content="y" * 400),  # ~100 tokens
        ]
        trimmed = manager.trim_to_fit(messages)
        # Should drop at least one message
        assert len(trimmed) < 2

    def test_trim_to_fit_drop_system(self):
        """Test trimming with DROP_SYSTEM strategy."""
        manager = ContextManager(
            max_tokens=30,  # Very small to trigger trimming
            strategy=ContextStrategy.DROP_SYSTEM,
            reserve_tokens=5
        )
        messages = [
            Message(role=Role.SYSTEM, content="System 1"),
            Message(role=Role.SYSTEM, content="System 2"),
            Message(role=Role.USER, content="x" * 400),  # Large content to trigger trim
        ]
        trimmed = manager.trim_to_fit(messages)
        # Should keep only one system message
        system_msgs = [m for m in trimmed if m.role == Role.SYSTEM]
        assert len(system_msgs) == 1

    def test_drop_oldest_preserves_system(self):
        """Test that DROP_OLDEST preserves system messages."""
        manager = ContextManager(
            max_tokens=50,
            strategy=ContextStrategy.DROP_OLDEST
        )
        messages = [
            Message(role=Role.SYSTEM, content="System instruction"),
            Message(role=Role.USER, content="User message"),
        ]
        trimmed = manager._drop_oldest(messages)
        # System message should remain
        assert any(m.role == Role.SYSTEM for m in trimmed)

    def test_get_messages_for_api_openai(self):
        """Test getting messages for OpenAI API."""
        manager = ContextManager(max_tokens=10000)
        messages = [
            Message(role=Role.SYSTEM, content="System"),
            Message(role=Role.USER, content="User"),
        ]
        api_msgs = manager.get_messages_for_api(messages, format="openai")
        assert len(api_msgs) == 2
        assert api_msgs[0]["role"] == "system"

    def test_get_messages_for_api_anthropic(self):
        """Test getting messages for Anthropic API."""
        manager = ContextManager(max_tokens=10000)
        messages = [
            Message(role=Role.SYSTEM, content="System 1"),
            Message(role=Role.SYSTEM, content="System 2"),
            Message(role=Role.USER, content="User"),
        ]
        api_msgs = manager.get_messages_for_api(messages, format="anthropic")
        # Anthropic combines system messages
        assert api_msgs[0]["role"] == "system"
        assert "System 1" in api_msgs[0]["content"]
        assert "System 2" in api_msgs[0]["content"]


class TestContextStrategy:
    """Test ContextStrategy enum."""

    def test_strategies(self):
        """Test all strategy values."""
        assert ContextStrategy.DROP_OLDEST == "drop_oldest"
        assert ContextStrategy.DROP_SYSTEM == "drop_system"
        assert ContextStrategy.SUMMARIZE == "summarize"
        assert ContextStrategy.TRUNCATE == "truncate"
        assert ContextStrategy.COMPRESS == "compress"
