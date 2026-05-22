"""
Conversational Memory System.

Provides short-term memory for multi-turn conversations with:
- Session-based storage via Streamlit session state
- Automatic conversation pruning (keeps context window manageable)
- Optional system prompt injection
"""

from typing import List, Dict, Optional


# Maximum number of messages to keep in context (to avoid exceeding context windows)
MAX_CONTEXT_MESSAGES = 20
SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant. You provide accurate, well-reasoned responses to user queries. When you don't know something, you say so. You avoid making harmful, biased, or discriminatory statements. You refuse to assist with illegal activities, violence, or harmful actions."""


def init_memory(st, system_prompt: Optional[str] = None) -> None:
    """Initialize conversation memory in Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0
    
    if "guardrail_log" not in st.session_state:
        st.session_state.guardrail_log = []
    
    # Always ensure system prompt is first message
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "system",
            "content": system_prompt or SYSTEM_PROMPT,
        })


def add_message(role: str, content: str, st, metadata: Optional[Dict] = None) -> None:
    """
    Add a message to the conversation history.
    
    Args:
        role: 'user', 'assistant', or 'system'
        content: Message text
        st: Streamlit session
        metadata: Optional dict with guardrail info, latency, etc.
    """
    message = {
        "role": role,
        "content": content,
    }
    
    if metadata:
        message["metadata"] = metadata
    
    st.session_state.messages.append(message)
    st.session_state.interaction_count += 1
    
    # Prune old messages if context gets too long (preserve system prompt)
    if len(st.session_state.messages) > MAX_CONTEXT_MESSAGES + 1:  # +1 for system prompt
        # Keep system prompt + last N messages
        st.session_state.messages = (
            [st.session_state.messages[0]] +  # Keep system prompt
            st.session_state.messages[-(MAX_CONTEXT_MESSAGES):]  # Keep last N
        )


def get_messages(st) -> List[Dict[str, str]]:
    """Get all conversation messages."""
    return st.session_state.messages


def log_guardrail_event(st, event_type: str, category: str, user_input: str) -> None:
    """Log a guardrail event for observability."""
    import datetime
    
    event = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": event_type,
        "category": category,
        "user_input": user_input[:200],  # Truncate for privacy
        "interaction_number": st.session_state.interaction_count,
    }
    st.session_state.guardrail_log.append(event)


def get_guardrail_log(st) -> List[Dict]:
    """Get the guardrail event log."""
    return st.session_state.guardrail_log


def clear_memory(st) -> None:
    """Clear conversation history."""
    st.session_state.messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT,
    }]
    st.session_state.interaction_count = 0


def get_context_stats(st) -> Dict:
    """Get statistics about the current conversation context."""
    messages = get_messages(st)
    total_chars = sum(len(m["content"]) for m in messages)
    
    return {
        "total_messages": len(messages),
        "total_characters": total_chars,
        "interaction_count": st.session_state.interaction_count,
        "guardrail_events": len(st.session_state.guardrail_log),
    }