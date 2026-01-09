"""
Context window management for conversations.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .conversation import Message, Role


class ContextStrategy(str, Enum):
    """Strategies for managing context window"""
    DROP_OLDEST = "drop_oldest"  # Drop oldest messages first
    DROP_SYSTEM = "drop_system"  # Keep only one system message
    SUMMARIZE = "summarize"  # Summarize old messages
    TRUNCATE = "truncate"  # Truncate messages
    COMPRESS = "compress"  # Compress multiple messages


@dataclass
class ContextWindowInfo:
    """Information about context window usage"""
    total_tokens: int
    available_tokens: int
    used_tokens: int
    usage_percent: float
    messages_count: int
    strategy_used: Optional[ContextStrategy] = None


class TokenEstimator:
    """
    Estimate token counts for messages.

    Uses character-based estimation with provider-specific adjustments.
    """

    # Approximate characters per token by provider
    CHAR_RATIO = {
        "openai": 4.0,
        "anthropic": 3.7,
        "google": 4.0,
        "meta": 4.0,
        "default": 4.0
    }

    def __init__(self, provider: str = "openai"):
        """
        Initialize token estimator.

        Args:
            provider: LLM provider for adjustment
        """
        self.provider = provider
        self.ratio = self.CHAR_RATIO.get(provider, self.CHAR_RATIO["default"])

    def estimate_message(self, message: Message) -> int:
        """Estimate tokens in a message."""
        # Base estimation from characters
        tokens = len(message.content) // self.ratio

        # Add overhead for role and formatting
        overhead = 3  # role + formatting
        if message.name:
            overhead += 2

        return tokens + overhead

    def estimate_messages(self, messages: List[Message]) -> int:
        """Estimate tokens in a list of messages."""
        return sum(self.estimate_message(m) for m in messages)

    def estimate_string(self, text: str) -> int:
        """Estimate tokens in a string."""
        return len(text) // self.ratio


class ContextManager:
    """
    Manage context window for conversations.

    Ensures conversations fit within model context windows
    by applying various strategies.

    Example:
        manager = ContextManager(
            max_tokens=128000,
            strategy=ContextStrategy.DROP_OLDEST
        )

        # Check if messages fit
        info = manager.check_context(messages)

        # Trim if needed
        trimmed = manager.trim_to_fit(messages)

        # Get messages for API
        api_messages = manager.get_messages_for_api(messages)
    """

    def __init__(
        self,
        max_tokens: int = 128000,
        strategy: ContextStrategy = ContextStrategy.DROP_OLDEST,
        reserve_tokens: int = 1000  # Reserve space for response
    ):
        """
        Initialize the context manager.

        Args:
            max_tokens: Maximum context window size
            strategy: Strategy to use when over limit
            reserve_tokens: Tokens to reserve for output
        """
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.reserve_tokens = reserve_tokens
        self.estimator = TokenEstimator()

    def check_context(self, messages: List[Message]) -> ContextWindowInfo:
        """
        Check if messages fit in context window.

        Args:
            messages: List of messages

        Returns:
            ContextWindowInfo with usage details
        """
        used_tokens = self.estimator.estimate_messages(messages)
        available = self.max_tokens - used_tokens - self.reserve_tokens

        return ContextWindowInfo(
            total_tokens=self.max_tokens,
            available_tokens=max(0, available),
            used_tokens=used_tokens,
            usage_percent=(used_tokens / self.max_tokens) * 100,
            messages_count=len(messages)
        )

    def trim_to_fit(
        self,
        messages: List[Message]
    ) -> List[Message]:
        """
        Trim messages to fit within context window.

        Args:
            messages: List of messages

        Returns:
            Trimmed list of messages
        """
        info = self.check_context(messages)

        # Check if we need to trim (usage > 100% means we're over limit)
        if info.usage_percent <= 100:
            return messages

        # Need to trim
        if self.strategy == ContextStrategy.DROP_OLDEST:
            return self._drop_oldest(messages)
        elif self.strategy == ContextStrategy.DROP_SYSTEM:
            return self._drop_system(messages)
        elif self.strategy == ContextStrategy.SUMMARIZE:
            return self._summarize(messages)
        else:
            return self._drop_oldest(messages)

    def _drop_oldest(self, messages: List[Message]) -> List[Message]:
        """Drop oldest messages until fits."""
        result = messages.copy()
        while result and self.check_context(result).usage_percent > 100:
            # Remove first non-system message
            for i, msg in enumerate(result):
                if msg.role != Role.SYSTEM:
                    result.pop(i)
                    break
        return result

    def _drop_system(self, messages: List[Message]) -> List[Message]:
        """Drop system messages, keeping only one."""
        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        other_msgs = [m for m in messages if m.role != Role.SYSTEM]

        # Keep at most one system message (the first)
        if system_msgs:
            other_msgs = [system_msgs[0]] + other_msgs

        return other_msgs

    def _summarize(self, messages: List[Message]) -> List[Message]:
        """Summarize old messages (placeholder)."""
        # In production, you'd use an LLM to generate summaries
        # For now, just drop oldest as fallback
        return self._drop_oldest(messages)

    def get_messages_for_api(
        self,
        messages: List[Message],
        format: str = "openai"
    ) -> List[Dict[str, str]]:
        """
        Get messages formatted for API, ensuring they fit in context.

        Args:
            messages: List of messages
            format: Format ("openai" or "anthropic")

        Returns:
            Formatted message dictionaries
        """
        trimmed = self.trim_to_fit(messages)

        if format == "openai":
            return [m.to_openai_format() for m in trimmed]
        else:
            # Anthropic format (system first)
            system_msgs = [m for m in trimmed if m.role == Role.SYSTEM]
            other_msgs = [m for m in trimmed if m.role != Role.SYSTEM]
            result = []
            if system_msgs:
                content = "\n\n".join(m.content for m in system_msgs)
                result.append({"role": "system", "content": content})
            result.extend([m.to_anthropic_format() for m in other_msgs])
            return result
