"""
Sidebar Component - Controls and statistics panel.

Provides model selection, settings toggles, conversation controls,
and context statistics in a collapsible sidebar layout.
"""

import streamlit as st
from typing import Optional, Callable

from app.utils.memory import MemoryManager
from app.config import MODEL_LABELS


class Sidebar:
    """
    Sidebar UI component for the Streamlit app.

    Handles:
    - Model selection radio
    - Advanced settings (metadata display, guardrails toggle)
    - Clear conversation button
    - Context statistics display
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager

    def render(self) -> dict:
        """
        Render the sidebar and return selected configuration.

        Returns:
            dict with:
                - assistant_type: str ('OSS' or 'Frontier')
                - show_metadata: bool
                - enable_guardrails: bool
        """
        st.markdown("### Controls")

        # Model selection
        assistant_type = st.radio(
            "Select Assistant",
            ["OSS Assistant (Qwen2.5)", "Frontier Assistant (OpenRouter)"],
            index=0 if st.session_state.get("chat_tab", "OSS Assistant") == "OSS Assistant" else 1,
            key="assistant_selector",
        )

        st.markdown("---")

        # Advanced settings
        with st.expander("⚙️ Settings"):
            show_metadata = st.checkbox("Show response metadata", value=True)
            enable_guardrails = st.checkbox("Enable guardrails", value=True)

        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True, type="secondary"):
            self.memory.clear()
            st.rerun()

        st.markdown("---")

        # Context stats
        stats = self.memory.get_context_stats()
        st.markdown("### Context Stats")
        st.metric("Messages", stats.total_messages)
        st.metric("Interactions", stats.interaction_count)
        st.metric("Guardrail Events", stats.guardrail_events)

        return {
            "assistant_type": "OSS" if "OSS" in assistant_type else "Frontier",
            "show_metadata": show_metadata,
            "enable_guardrails": enable_guardrails,
        }