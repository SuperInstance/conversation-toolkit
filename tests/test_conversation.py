"""Tests for conversation module."""

import pytest
import time

from conversation_toolkit import (
    Conversation,
    Message,
    Role,
    ConversationSummary,
    ConversationOptions,
    MessageTemplate,
)


class TestMessage:
    """Test Message class."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role=Role.USER, content="Hello, world!")
        assert msg.role == Role.USER
        assert msg.content == "Hello, world!"
        assert msg.is_user
        assert not msg.is_assistant
        assert not msg.is_system

    def test_message_to_dict(self):
        """Test converting message to dict."""
        msg = Message(role=Role.USER, content="Test")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Test"}

    def test_message_with_name(self):
        """Test message with name."""
        msg = Message(role=Role.USER, content="Test", name="John")
        d = msg.to_dict()
        assert "name" in d
        assert d["name"] == "John"

    def test_message_copy(self):
        """Test copying a message."""
        msg = Message(
            role=Role.USER,
            content="Test",
            metadata={"key": "value"}
        )
        copy = msg.copy()
        assert copy.role == msg.role
        assert copy.content == msg.content
        assert copy.metadata == msg.metadata
        assert copy is not msg

    def test_openai_format(self):
        """Test OpenAI format conversion."""
        msg = Message(role=Role.ASSISTANT, content="Response")
        assert msg.to_openai_format() == {
            "role": "assistant",
            "content": "Response"
        }

    def test_anthropic_format(self):
        """Test Anthropic format conversion."""
        msg = Message(role=Role.USER, content="Question")
        assert msg.to_anthropic_format() == {
            "role": "user",
            "content": "Question"
        }


class TestConversation:
    """Test Conversation class."""

    def test_create_conversation(self):
        """Test creating a conversation."""
        conv = Conversation()
        assert conv.message_count == 0
        assert conv.id is not None

    def test_create_with_system_message(self):
        """Test creating conversation with system message."""
        conv = Conversation(system_message="You are helpful.")
        assert conv.message_count == 1
        assert conv.messages[0].role == Role.SYSTEM

    def test_add_messages(self):
        """Test adding messages."""
        conv = Conversation()
        conv.add_user("Hello")
        conv.add_assistant("Hi there!")
        assert conv.message_count == 2
        assert conv.user_message_count == 1
        assert conv.assistant_message_count == 1

    def test_add_function_message(self):
        """Test adding function result message."""
        conv = Conversation()
        msg = conv.add_function("Result", name="get_weather")
        assert msg.role == Role.FUNCTION
        assert msg.name == "get_weather"

    def test_add_tool_message(self):
        """Test adding tool result message."""
        conv = Conversation()
        msg = conv.add_tool("Tool result", name="search")
        assert msg.role == Role.TOOL
        assert msg.name == "search"

    def test_add_existing_message(self):
        """Test adding existing message."""
        conv = Conversation()
        msg = Message(role=Role.USER, content="Test")
        conv.add_message(msg)
        assert conv.message_count == 1

    def test_get_messages_filtered(self):
        """Test filtering messages by role."""
        conv = Conversation()
        conv.add_system("System")
        conv.add_user("User 1")
        conv.add_user("User 2")
        conv.add_assistant("Assistant")

        user_msgs = conv.get_messages(role=Role.USER)
        assert len(user_msgs) == 2

    def test_get_messages_last_n(self):
        """Test getting last N messages."""
        conv = Conversation()
        for i in range(10):
            conv.add_user(f"Message {i}")

        last_3 = conv.get_messages(last_n=3)
        assert len(last_3) == 3
        assert last_3[0].content == "Message 7"

    def test_estimated_tokens(self):
        """Test token estimation."""
        conv = Conversation()
        conv.add_user("This is a test message with some content")
        assert conv.estimated_tokens > 0

    def test_get_info(self):
        """Test getting conversation info."""
        conv = Conversation(system_message="System")
        conv.add_user("User")
        conv.add_assistant("Assistant")

        info = conv.get_info()
        assert info["message_count"] == 3
        assert info["user_messages"] == 1
        assert info["assistant_messages"] == 1
        assert info["has_system_message"] is True

    def test_clear(self):
        """Test clearing conversation."""
        conv = Conversation(system_message="System")
        conv.add_user("User")
        conv.clear()
        assert conv.message_count == 0
        assert len(conv.summaries) == 0

    def test_reset_to_summary(self):
        """Test resetting to system messages only."""
        conv = Conversation(system_message="System")
        conv.add_user("User")
        conv.add_assistant("Assistant")
        conv.reset_to_summary()
        assert conv.message_count == 1
        assert conv.messages[0].role == Role.SYSTEM

    def test_get_messages_for_api_openai(self):
        """Test getting messages in OpenAI format."""
        conv = Conversation(system_message="System")
        conv.add_user("User")

        messages = conv.get_messages_for_api(format="openai")
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "System"}
        assert messages[1] == {"role": "user", "content": "User"}

    def test_get_messages_for_api_anthropic(self):
        """Test getting messages in Anthropic format."""
        conv = Conversation(system_message="System")
        conv.add_user("User")

        messages = conv.get_messages_for_api(format="anthropic")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"

    def test_summarize_old_messages(self):
        """Test summarizing old messages."""
        conv = Conversation()
        for i in range(15):
            conv.add_user(f"Message {i}")
            conv.add_assistant(f"Response {i}")

        summary = conv.summarize_old_messages(keep_last=10)
        assert summary is not None
        assert len(conv.summaries) > 0

    def test_serialize_deserialize(self):
        """Test serialization and deserialization."""
        conv = Conversation(system_message="System")
        conv.add_user("User")
        conv.add_assistant("Assistant")

        data = conv.to_dict()
        assert "id" in data
        assert "messages" in data
        assert len(data["messages"]) == 3

        restored = Conversation.from_dict(data)
        assert restored.message_count == 3
        assert restored.id == conv.id

    def test_auto_summarize(self):
        """Test auto-summarization."""
        options = ConversationOptions(
            auto_summarize=True,
            summarize_threshold=10
        )
        conv = Conversation(options=options)

        for i in range(12):
            conv.add_user(f"Message {i}")
            conv.add_assistant(f"Response {i}")

        # Should have triggered summarization
        assert len(conv.summaries) > 0


class TestMessageTemplate:
    """Test MessageTemplate class."""

    def test_render_template(self):
        """Test rendering a template."""
        template = MessageTemplate(
            role=Role.USER,
            template="Hello, {name}!"
        )
        msg = template.render(name="World")
        assert msg.content == "Hello, World!"
        assert msg.role == Role.USER

    def test_render_to_dict(self):
        """Test rendering template to dict."""
        template = MessageTemplate(
            role=Role.USER,
            template="Tell me about {topic}"
        )
        result = template.render_to_dict(topic="AI")
        assert result == {"role": "user", "content": "Tell me about AI"}

    def test_render_missing_param(self):
        """Test rendering with missing parameter."""
        template = MessageTemplate(
            role=Role.USER,
            template="Hello, {name}!"
        )
        with pytest.raises(KeyError):
            template.render()


class TestConversationOptions:
    """Test ConversationOptions class."""

    def test_default_options(self):
        """Test default options."""
        options = ConversationOptions()
        assert options.max_messages == 100
        assert options.max_tokens == 128000
        assert options.auto_summarize is True
        assert options.summarize_threshold == 50

    def test_custom_options(self):
        """Test custom options."""
        options = ConversationOptions(
            max_messages=200,
            auto_summarize=False
        )
        assert options.max_messages == 200
        assert options.auto_summarize is False
