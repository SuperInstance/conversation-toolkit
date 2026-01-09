# Conversation Toolkit

A Python library for managing multi-turn LLM conversations with context window optimization, message templates, and conversation history tracking.

## Features

- **Message Management**: Create, format, and track messages with role-based organization
- **Conversation Tracking**: Full conversation lifecycle with UUID identification and timestamps
- **Context Window Management**: Token estimation and automatic context trimming strategies
- **Message Templates**: Reusable template engine for consistent prompt generation
- **History Management**: Search, summarize, and analyze conversation history
- **Multi-Format Support**: OpenAI and Anthropic API message formats

## Installation

```bash
pip install conversation-toolkit
```

## Quick Start

```python
from conversation_toolkit import Conversation, Role

# Create a conversation with a system message
conv = Conversation(system_message="You are a helpful assistant.")

# Add messages
conv.add_user("What is the capital of France?")
conv.add_assistant("The capital of France is Paris.")

# Get messages for OpenAI API
messages = conv.get_messages_for_api(format="openai")
print(messages)
# Output: [
#   {'role': 'system', 'content': 'You are a helpful assistant.'},
#   {'role': 'user', 'content': 'What is the capital of France?'},
#   {'role': 'assistant', 'content': 'The capital of France is Paris.'}
# ]

# Get conversation info
print(f"Messages: {conv.message_count}")
print(f"Estimated tokens: {conv.estimated_tokens}")
```

## Core Concepts

### Message

A `Message` represents a single message in a conversation:

```python
from conversation_toolkit import Message, Role

msg = Message(
    role=Role.USER,
    content="Explain quantum computing",
    metadata={"category": "science"}
)
```

### Conversation

A `Conversation` manages multiple messages:

```python
from conversation_toolkit import Conversation, ConversationOptions

# Create with custom options
options = ConversationOptions(
    max_messages=100,
    max_tokens=128000,
    auto_summarize=True,
    summarize_threshold=50
)

conv = Conversation(system_message="You are an expert tutor.", options=options)

# Add messages using convenience methods
conv.add_user("Teach me about recursion.")
conv.add_assistant("Recursion is when a function calls itself...")

# Filter messages
user_messages = conv.get_messages(role=Role.USER)
last_5 = conv.get_messages(last_n=5)
```

### Context Manager

Manage context window limits:

```python
from conversation_toolkit import ContextManager, ContextStrategy

manager = ContextManager(
    max_tokens=128000,
    strategy=ContextStrategy.DROP_OLDEST,
    reserve_tokens=1000
)

# Check if messages fit
info = manager.check_context(conv.messages)
print(f"Usage: {info.usage_percent:.1f}%")
print(f"Available: {info.available_tokens} tokens")

# Trim to fit if needed
trimmed = manager.trim_to_fit(conv.messages)

# Get formatted for API
api_messages = manager.get_messages_for_api(conv.messages, format="openai")
```

### Template Engine

Use templates for consistent prompts:

```python
from conversation_toolkit import TemplateEngine, Role

engine = TemplateEngine()

# Register a template
engine.register(
    "explain",
    Role.USER,
    "Explain {topic} in {style} tone",
    parameters=["topic", "style"]
)

# Render the template
msg = engine.render("explain", topic="machine learning", style="simple")
print(msg.content)  # "Explain machine learning in simple tone"

# Set global variables
engine.set_var("name", "Student")
engine.register("greeting", Role.USER, "Hello, {name}!")
```

### History Manager

Manage and search multiple conversations:

```python
from conversation_toolkit import HistoryManager

history = HistoryManager(max_conversations=1000)

# Add conversations
history.add_conversation(conv)

# Search conversations
results = history.search("machine learning", limit=10)

# Get recent conversations
recent = history.get_recent(limit=5)

# Get statistics
stats = history.get_stats()
print(f"Total conversations: {stats['total_conversations']}")
print(f"Total messages: {stats['total_messages']}")
```

## Context Strategies

The `ContextStrategy` enum defines how to handle overflow:

| Strategy | Description |
|----------|-------------|
| `DROP_OLDEST` | Drop oldest non-system messages first |
| `DROP_SYSTEM` | Keep only one system message |
| `SUMMARIZE` | Summarize old messages (placeholder) |
| `TRUNCATE` | Truncate message content |
| `COMPRESS` | Compress multiple messages together |

## API Formats

### OpenAI Format

```python
messages = conv.get_messages_for_api(format="openai")
# [{'role': 'system', 'content': '...'}, ...]
```

### Anthropic Format

```python
messages = conv.get_messages_for_api(format="anthropic")
# System message first, then user/assistant messages
```

## Message Roles

- `Role.SYSTEM` - System instructions
- `Role.USER` - User messages
- `Role.ASSISTANT` - Assistant responses
- `Role.FUNCTION` - Function call results
- `Role.TOOL` - Tool execution results

## Examples

### Conversation with Auto-Summarization

```python
from conversation_toolkit import Conversation, ConversationOptions

options = ConversationOptions(
    auto_summarize=True,
    summarize_threshold=10  # Summarize after 10 messages
)

conv = Conversation(
    system_message="You are a helpful assistant.",
    options=options
)

# Add many messages...
for i in range(15):
    conv.add_user(f"Question {i}")
    conv.add_assistant(f"Answer {i}")

# Old messages are automatically summarized
print(len(conv.summaries))  # 1 or more
```

### Conversation History with Branching

```python
from conversation_toolkit import ConversationHistory

history = ConversationHistory(max_states=50)

# Save current state
state_id = history.save_state(conv.messages)

# Modify conversation
conv.add_user("New question")

# Restore previous state
previous = history.restore_state(state_id)

# Create a branch
branch_id = history.branch_from(state_id)
```

## License

MIT License - see LICENSE file for details.
