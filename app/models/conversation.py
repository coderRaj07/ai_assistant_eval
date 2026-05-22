"""
Conversation data models.

Defines the core data structures for conversation messages, model responses,
safety checks, guardrail events, and context statistics.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """
    A single message in a conversation thread.

    Attributes:
        role: Who sent the message (system, user, assistant)
        content: The message text
        metadata: Optional metadata (latency, guardrail info, errors)
        timestamp: When the message was created
    """
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        d = {"role": self.role.value, "content": self.content}
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialize from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            metadata=data.get("metadata"),
        )


@dataclass
class ConversationHistory:
    """
    Manages a list of messages in a conversation session.
    Handles pruning and serialization.
    """
    messages: List[Message] = field(default_factory=list)
    max_messages: int = 20

    def add(self, message: Message) -> None:
        """Add a message and prune if needed (preserves system prompt)."""
        self.messages.append(message)
        self._prune()

    def _prune(self) -> None:
        """Remove oldest messages beyond max_messages, preserving system prompt."""
        if len(self.messages) > self.max_messages + 1:  # +1 for system prompt
            system = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            non_system = [m for m in self.messages if m.role != MessageRole.SYSTEM]
            self.messages = system + non_system[-(self.max_messages):]

    def clear(self, system_prompt: str = "") -> None:
        """Clear all messages, optionally setting a system prompt."""
        self.messages = []
        if system_prompt:
            self.messages.append(
                Message(role=MessageRole.SYSTEM, content=system_prompt)
            )

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Serialize all messages to list of dicts."""
        return [m.to_dict() for m in self.messages]

    def get_user_messages(self) -> List[Message]:
        """Return only user messages."""
        return [m for m in self.messages if m.role == MessageRole.USER]

    def get_last_user_content(self) -> str:
        """Get the content of the last user message."""
        user_msgs = self.get_user_messages()
        return user_msgs[-1].content if user_msgs else ""

    def __len__(self) -> int:
        return len(self.messages)


@dataclass
class ModelResponse:
    """
    Standardized response from any model (OSS or Frontier).

    Attributes:
        response: The generated text
        latency_s: Response time in seconds
        guardrail_triggered: Whether guardrails were activated
        safety_category: Category of safety issue if triggered
        error: Error message if request failed
    """
    response: str = ""
    latency_s: float = 0.0
    guardrail_triggered: bool = False
    safety_category: str = "safe"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "latency_s": self.latency_s,
            "guardrail_triggered": self.guardrail_triggered,
            "safety_category": self.safety_category,
            "error": self.error,
        }


@dataclass
class SafetyCheckResult:
    """Result of a safety/guardrail check."""
    is_safe: bool
    refusal: Optional[str] = None
    category: str = "safe"


@dataclass
class GuardrailEvent:
    """A logged guardrail event for observability."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    event_type: str = ""
    category: str = ""
    user_input: str = ""
    interaction_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "type": self.event_type,
            "category": self.category,
            "user_input": self.user_input[:200],
            "interaction_number": self.interaction_number,
        }


@dataclass
class ContextStats:
    """Statistics about the current conversation context."""
    total_messages: int = 0
    total_characters: int = 0
    interaction_count: int = 0
    guardrail_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_messages": self.total_messages,
            "total_characters": self.total_characters,
            "interaction_count": self.interaction_count,
            "guardrail_events": self.guardrail_events,
        }