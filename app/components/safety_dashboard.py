"""
Safety Dashboard Component - Guardrail monitoring interface.

Displays guardrail event metrics, detection pattern documentation,
and a real-time guardrail event log table.
"""

import streamlit as st
import pandas as pd
from typing import Optional

from app.utils.memory import MemoryManager


class SafetyDashboard:
    """
    Safety & Guardrail monitoring dashboard.

    Handles:
    - Key guardrail metrics display
    - Input/Output detection pattern documentation
    - Real-time guardrail event log table
    """

    INPUT_CATEGORIES = {
        "Malware/Cyber": ["malware", "virus", "ransomware", "hacking"],
        "Violence": ["kill", "bomb", "weapons", "torture"],
        "Self-harm": ["suicide", "self-harm", "cutting"],
        "Illegal Activities": ["steal", "drugs", "fraud", "money laundering"],
        "Discrimination": ["gender inferiority", "racial superiority", "religious bias"],
        "Jailbreak Attempts": [
            "Ignore instructions", "DAN mode", "Roleplay bypass",
            "Prompt leak", "Hypothetical harmful scenarios",
        ],
        "Stereotypes": ["Racial stereotypes", "Gender stereotypes", "Professional stereotypes"],
    }

    OUTPUT_CHECKS = {
        "Bias Detection": "Checks for gender/racial generalizations",
        "Violence Detection": "Detects harmful instructions",
        "Misinformation": "Flags known false claims",
        "Hate Speech": "Monitors discriminatory content",
    }

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager

    def render(self) -> None:
        """Render the safety dashboard."""
        st.markdown("### 🛡️ Safety & Guardrail Dashboard")

        guardrail_log = self.memory.get_guardrail_log()

        self._render_metrics(guardrail_log)
        st.markdown("---")
        self._render_coverage()
        self._render_event_log(guardrail_log)

    def _render_metrics(self, guardrail_log: list) -> None:
        """Render the top-level guardrail metrics."""
        dash_col1, dash_col2, dash_col3, dash_col4 = st.columns(4)

        with dash_col1:
            st.metric("Total Guardrail Events", len(guardrail_log))
        with dash_col2:
            st.metric("Active Interactions", self.memory.interaction_count)
        with dash_col3:
            st.metric("Messages in Context", len(self.memory.get_messages()))
        with dash_col4:
            categories = set(e.get("category", "unknown") for e in guardrail_log)
            st.metric("Guardrail Categories", len(categories))

    def _render_coverage(self) -> None:
        """Render guardrail coverage documentation."""
        st.markdown("### 🔍 Guardrail Coverage")

        guardrail_col1, guardrail_col2 = st.columns(2)

        with guardrail_col1:
            st.markdown("#### Input Detection Patterns")
            for category, patterns in self.INPUT_CATEGORIES.items():
                st.markdown(f"**{category}**: {', '.join(patterns)}")

        with guardrail_col2:
            st.markdown("#### Output Safety Checks")
            for check, desc in self.OUTPUT_CHECKS.items():
                st.markdown(f"**{check}**: {desc}")

            st.markdown("#### Refusal Responses")
            st.markdown("Model provides contextual, category-specific refusal messages.")
            st.markdown("Harmless refusals maintain helpfulness while declining harmful requests.")

    def _render_event_log(self, guardrail_log: list) -> None:
        """Render the guardrail event log table."""
        if guardrail_log:
            st.markdown("---")
            st.markdown("### 📋 Guardrail Event Log")

            log_df = pd.DataFrame(guardrail_log)
            if not log_df.empty:
                display_cols = ["timestamp", "type", "category", "interaction_number"]
                display_log = (
                    log_df[display_cols].copy()
                    if all(c in log_df.columns for c in display_cols)
                    else log_df
                )
                st.dataframe(display_log, use_container_width=True)
        else:
            st.info(
                "No guardrail events recorded in this session. "
                "Try prompts from the jailbreak or safety categories."
            )