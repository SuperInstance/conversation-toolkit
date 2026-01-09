"""
Conversation Toolkit - Manage multi-turn LLM conversations.

Features:
- Message management with role tracking
- Conversation history and summarization
- Token counting and budget management
- Context window optimization
- Message deduplication and compression
- Template-based message generation
"""

from .conversation import (
    Conversation,
    Message,
    Role,
    ConversationSummary,
    MessageTemplate,
    ConversationOptions,
)
from .history import (
    ConversationHistory,
    HistoryManager,
    SummarizationStrategy,
)
from .context import (
    ContextManager,
    ContextStrategy,
    TokenEstimator,
    ContextWindowInfo,
)
from .templates import (
    TemplateEngine,
    Template,
    render_template,
    load_common_templates,
)

__version__ = "1.0.0"
__all__ = [
    "Conversation",
    "Message",
    "Role",
    "ConversationSummary",
    "MessageTemplate",
    "ConversationOptions",
    "ConversationHistory",
    "HistoryManager",
    "SummarizationStrategy",
    "ContextManager",
    "ContextStrategy",
    "TokenEstimator",
    "ContextWindowInfo",
    "TemplateEngine",
    "Template",
    "render_template",
    "load_common_templates",
]
