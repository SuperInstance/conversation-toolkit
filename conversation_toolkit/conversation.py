"""
Core conversation management classes.
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import json


class Role(str, Enum):
    """Message roles in a conversation"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class Message:
    """
    A message in a conversation.

    Example:
        msg = Message(
            role=Role.USER,
            content="Hello, how are you?"
        )
        print(msg.to_dict())
    """
    role: Role
    content: str
    name: Optional[str] = None
    token_count: Optional[int] = None
    timestamp: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for LLM APIs."""
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d

    def to_openai_format(self) -> Dict[str, str]:
        """Convert to OpenAI format."""
        return {"role": self.role.value, "content": self.content}

    def to_anthropic_format(self) -> Dict[str, str]:
        """Convert to Anthropic format."""
        return {"role": self.role.value, "content": self.content}

    def copy(self) -> 'Message':
        """Create a copy of this message."""
        return Message(
            role=self.role,
            content=self.content,
            name=self.name,
            token_count=self.token_count,
            timestamp=self.timestamp,
            metadata=self.metadata.copy()
        )

    @property
    def is_user(self) -> bool:
        """Check if this is a user message."""
        return self.role == Role.USER

    @property
    def is_assistant(self) -> bool:
        """Check if this is an assistant message."""
        return self.role == Role.ASSISTANT

    @property
    def is_system(self) -> bool:
        """Check if this is a system message."""
        return self.role == Role.SYSTEM


@dataclass
class ConversationSummary:
    """Summary of a conversation or part of a conversation."""
    summary: str
    message_count: int
    total_tokens: int
    key_topics: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class ConversationOptions:
    """Options for conversation behavior."""
    max_messages: int = 100
    max_tokens: int = 128000
    auto_summarize: bool = True
    summarize_threshold: int = 50  # Summarize after N messages
    include_timestamps: bool = True
    track_tokens: bool = True
    compress_messages: bool = False
    system_message_behavior: str = "keep_first"  # "keep_first", "keep_all", "drop"


class Conversation:
    """
    Manage a multi-turn conversation with an LLM.

    Example:
        conv = Conversation()

        # Add system message
        conv.add_system("You are a helpful assistant.")

        # Add user message
        conv.add_user("What is the capital of France?")

        # Simulate assistant response
        conv.add_assistant("The capital of France is Paris.")

        # Get messages for API
        messages = conv.get_messages()
        print(messages)

        # Get conversation info
        print(f"Messages: {conv.message_count}")
        print(f"Estimated tokens: {conv.estimated_tokens}")
    """

    def __init__(
        self,
        system_message: Optional[str] = None,
        options: Optional[ConversationOptions] = None
    ):
        """
        Initialize a new conversation.

        Args:
            system_message: Optional system message to start with
            options: Conversation options
        """
        self.options = options or ConversationOptions()
        self.messages: List[Message] = []
        self.summaries: List[ConversationSummary] = []
        self._id = str(uuid.uuid4())
        self._created_at = time.time()
        self._updated_at = time.time()

        if system_message:
            self.add_system(system_message)

    def add_system(self, content: str, **metadata) -> Message:
        """Add a system message."""
        return self._add_message(Role.SYSTEM, content, **metadata)

    def add_user(self, content: str, **metadata) -> Message:
        """Add a user message."""
        return self._add_message(Role.USER, content, **metadata)

    def add_assistant(self, content: str, **metadata) -> Message:
        """Add an assistant message."""
        return self._add_message(Role.ASSISTANT, content, **metadata)

    def add_function(self, content: str, name: Optional[str] = None, **metadata) -> Message:
        """Add a function result message."""
        return self._add_message(Role.FUNCTION, content, name=name, **metadata)

    def add_tool(self, content: str, name: Optional[str] = None, **metadata) -> Message:
        """Add a tool result message."""
        return self._add_message(Role.TOOL, content, name=name, **metadata)

    def add_message(self, message: Message) -> None:
        """Add an existing message to the conversation."""
        self.messages.append(message)
        self._updated_at = time.time()
        self._check_summarize()

    def _add_message(self, role: Role, content: str, name: Optional[str] = None, **metadata) -> Message:
        """Add a message and return it."""
        msg = Message(role=role, content=content, name=name, metadata=metadata)
        self.add_message(msg)
        return msg

    def _check_summarize(self) -> None:
        """Check if we should summarize old messages."""
        if (self.options.auto_summarize and
            self.message_count >= self.options.summarize_threshold):
            self.summarize_old_messages()

    def get_messages(
        self,
        role: Optional[Role] = None,
        last_n: Optional[int] = None
    ) -> List[Message]:
        """
        Get messages from the conversation.

        Args:
            role: Optional filter by role
            last_n: Optional get only last N messages

        Returns:
            List of messages
        """
        messages = self.messages

        if role:
            messages = [m for m in messages if m.role == role]

        if last_n:
            messages = messages[-last_n:]

        return messages

    def get_messages_for_api(
        self,
        format: str = "openai"  # "openai" or "anthropic"
    ) -> List[Dict[str, str]]:
        """
        Get messages formatted for an LLM API.

        Args:
            format: API format ("openai" or "anthropic")

        Returns:
            List of message dictionaries
        """
        messages = self.get_messages()

        if format == "openai":
            return [m.to_openai_format() for m in messages]
        elif format == "anthropic":
            # Anthropic expects system message first
            system_msgs = [m for m in messages if m.role == Role.SYSTEM]
            user_assistant = [m for m in messages if m.role != Role.SYSTEM]
            result = []
            if system_msgs:
                # Combine system messages
                system_content = "\n\n".join(m.content for m in system_msgs)
                result.append({"role": "system", "content": system_content})
            result.extend([m.to_anthropic_format() for m in user_assistant])
            return result

        return [m.to_dict() for m in messages]

    def summarize_old_messages(self, keep_last: int = 10) -> Optional[ConversationSummary]:
        """
        Summarize old messages, keeping recent ones.

        Args:
            keep_last: Number of recent messages to keep

        Returns:
            Summary object
        """
        if self.message_count <= keep_last:
            return None

        # Messages to summarize
        old_messages = self.messages[:-keep_last]
        recent_messages = self.messages[-keep_last:]

        # Create simple summary
        topics = set()
        for msg in old_messages:
            # Extract key words (very basic)
            words = msg.content.lower().split()
            topics.update(words[:5])  # First 5 words as potential topics

        summary_text = f"Conversation contained {len(old_messages)} messages about {', '.join(list(topics)[:10])}"

        summary = ConversationSummary(
            summary=summary_text,
            message_count=len(old_messages),
            total_tokens=sum(m.token_count or 0 for m in old_messages) or self.estimated_tokens,
            key_topics=list(topics)[:10]
        )

        self.summaries.append(summary)

        # Replace old messages with summary
        # In production, you'd use an LLM to generate a proper summary
        summary_msg = Message(
            role=Role.SYSTEM,
            content=f"[Previous conversation summary: {summary_text}]"
        )

        self.messages = [summary_msg] + recent_messages

        return summary

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.summaries.clear()
        self._updated_at = time.time()

    def reset_to_summary(self) -> None:
        """Reset to just the system message and summaries."""
        system_msgs = [m for m in self.messages if m.role == Role.SYSTEM]
        self.messages = system_msgs.copy()

    @property
    def message_count(self) -> int:
        """Get total number of messages."""
        return len(self.messages)

    @property
    def user_message_count(self) -> int:
        """Get number of user messages."""
        return sum(1 for m in self.messages if m.is_user)

    @property
    def assistant_message_count(self) -> int:
        """Get number of assistant messages."""
        return sum(1 for m in self.messages if m.is_assistant)

    @property
    def estimated_tokens(self) -> int:
        """Get estimated token count (rough approximation)."""
        return sum(len(m.content) // 4 for m in self.messages)

    @property
    def id(self) -> str:
        """Get conversation ID."""
        return self._id

    @property
    def created_at(self) -> float:
        """Get creation timestamp."""
        return self._created_at

    @property
    def updated_at(self) -> float:
        """Get last update timestamp."""
        return self._updated_at

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive conversation information."""
        return {
            "id": self._id,
            "message_count": self.message_count,
            "user_messages": self.user_message_count,
            "assistant_messages": self.assistant_message_count,
            "estimated_tokens": self.estimated_tokens,
            "summaries": len(self.summaries),
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "has_system_message": any(m.is_system for m in self.messages)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation to dictionary."""
        return {
            "id": self._id,
            "messages": [m.to_dict() for m in self.messages],
            "summaries": [
                {
                    "summary": s.summary,
                    "message_count": s.message_count,
                    "total_tokens": s.total_tokens,
                    "key_topics": s.key_topics,
                    "timestamp": s.timestamp
                }
                for s in self.summaries
            ],
            "created_at": self._created_at,
            "updated_at": self._updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Load conversation from dictionary."""
        conv = cls()
        conv._id = data.get("id", str(uuid.uuid4()))
        conv._created_at = data.get("created_at", time.time())
        conv._updated_at = data.get("updated_at", time.time())

        for msg_data in data.get("messages", []):
            msg = Message(
                role=Role(msg_data["role"]),
                content=msg_data["content"],
                name=msg_data.get("name"),
                token_count=msg_data.get("token_count"),
                timestamp=msg_data.get("timestamp", time.time()),
                metadata=msg_data.get("metadata", {})
            )
            conv.messages.append(msg)

        for summary_data in data.get("summaries", []):
            conv.summaries.append(ConversationSummary(
                summary=summary_data["summary"],
                message_count=summary_data["message_count"],
                total_tokens=summary_data["total_tokens"],
                key_topics=summary_data.get("key_topics", []),
                timestamp=summary_data["timestamp"]
            ))

        return conv


@dataclass
class MessageTemplate:
    """
    A template for generating messages.

    Example:
        template = MessageTemplate(
            role=Role.USER,
            template="Tell me about {topic}"
        )
        msg = template.render(topic="AI safety")
    """
    role: Role
    template: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def render(self, **kwargs) -> Message:
        """
        Render the template with parameters.

        Args:
            **kwargs: Template parameters

        Returns:
            Rendered Message
        """
        content = self.template.format(**kwargs)
        return Message(role=self.role, content=content)

    def render_to_dict(self, **kwargs) -> Dict[str, str]:
        """Render and return as dict."""
        msg = self.render(**kwargs)
        return msg.to_dict()
