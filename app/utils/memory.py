"""
Memory Manager - Adapter between app components and utils/memory.py functions.

Provides an object-oriented MemoryManager class that wraps the functional
session-state-based memory utilities in utils/memory.py for use by UI components.
"""

from typing import Dict, List, Optional, Any

from app.models.conversation import (
    ConversationHistory,
    Message,
    MessageRole,
    ContextStats,
    GuardrailEvent,
)
from app.config import ModelConfig, MAX_CONTEXT_MESSAGES


class MemoryManager:
    """
    Manages conversation memory and guardrail logs via Streamlit session state.

    Wraps the functional utils/memory.py module behind a clean class interface
    that UI components can depend on.
    """

    def __init__(self, st_session):
        """
        Initialize with a Streamlit session object.

        Args:
            st_session: A Streamlit-like session object (st.session_state)
        """
        self._st = st_session
        self._init_session()

    # ------------------------------------------------------------------
    # Session Initialization
    # ------------------------------------------------------------------
    def _init_session(self) -> None:
        """Ensure session state keys exist."""
        if "messages" not in self._st:
            self._st.messages = []
        if "interaction_count" not in self._st:
            self._st.interaction_count = 0
        if "guardrail_log" not in self._st:
            self._st.guardrail_log = []
        # Ensure a system prompt is always the first message
        if not self._st.messages:
            self._st.messages.append({
                "role": "system",
                "content": ModelConfig.SYSTEM_PROMPT,
            })

    # ------------------------------------------------------------------
    # Message Management
    # ------------------------------------------------------------------
    @property
    def interaction_count(self) -> int:
        return self._st.interaction_count

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all conversation messages as dicts."""
        return self._st.messages

    def get_messages_dict(self) -> List[Dict[str, str]]:
        """Get messages suitable for model APIs (role + content only)."""
        return [{"role": m["role"], "content": m["content"]} for m in self._st.messages]

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation and increment interaction count."""
        self._st.messages.append({
            "role": "user",
            "content": content,
        })
        self._st.interaction_count += 1
        self._prune()

    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add an assistant message with optional metadata."""
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if metadata:
            msg["metadata"] = metadata
        self._st.messages.append(msg)
        self._prune()

    def _prune(self) -> None:
        """Remove oldest messages beyond limit, preserving system prompt."""
        if len(self._st.messages) > MAX_CONTEXT_MESSAGES + 1:  # +1 for system
            self._st.messages = (
                [self._st.messages[0]]
                + self._st.messages[-(MAX_CONTEXT_MESSAGES):]
            )

    def clear(self) -> None:
        """Clear conversation history but keep system prompt."""
        self._st.messages = [{
            "role": "system",
            "content": ModelConfig.SYSTEM_PROMPT,
        }]
        self._st.interaction_count = 0

    # ------------------------------------------------------------------
    # Guardrail Logging
    # ------------------------------------------------------------------
    def get_guardrail_log(self) -> List[Dict[str, Any]]:
        return self._st.guardrail_log

    def log_guardrail_event(self, event: GuardrailEvent) -> None:
        """Log a guardrail event."""
        self._st.guardrail_log.append(event.to_dict())

    # ------------------------------------------------------------------
    # Context Statistics
    # ------------------------------------------------------------------
    def get_context_stats(self) -> ContextStats:
        """Return current context statistics."""
        messages = self.get_messages()
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return ContextStats(
            total_messages=len(messages),
            total_characters=total_chars,
            interaction_count=self._st.interaction_count,
            guardrail_events=len(self._st.guardrail_log),
        )