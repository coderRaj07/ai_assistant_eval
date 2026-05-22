"""
Chat Panel Component - Main conversation interface.

Handles the chat UI including message history display, user input,
guardrail warnings, response metadata rendering, and model interaction.
"""

import streamlit as st
from typing import Optional

from app.models.conversation import MessageRole, ModelResponse
from app.utils.memory import MemoryManager
from app.services.model_service import ModelService
from app.services.guardrail_service import GuardrailService


class ChatPanel:
    """
    Main chat interface component.

    Renders the conversation history, handles user input, invokes the
    model service, and displays guardrail warnings and metadata.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        model_service: ModelService,
        guardrail_service: GuardrailService,
    ):
        self.memory = memory_manager
        self.model_service = model_service
        self.guardrail_service = guardrail_service

    def render(
        self,
        assistant_type: str,
        show_metadata: bool,
        enable_guardrails: bool,
    ) -> None:
        """
        Render the chat interface.

        Args:
            assistant_type: 'OSS' or 'Frontier'
            show_metadata: Whether to display response metadata
            enable_guardrails: Whether to enable guardrail checks
        """
        messages = self.memory.get_messages()

        # Chat display
        chat_container = st.container()

        with chat_container:
            for msg in messages:
                if msg.get("role") == "system":
                    continue  # Don't display system prompts

                with st.chat_message(msg.get("role")):
                    st.write(msg.get("content", ""))

                    # Show metadata if available and enabled
                    metadata = msg.get("metadata")
                    if show_metadata and metadata:
                        meta = metadata
                        meta_cols = st.columns(4)
                        with meta_cols[0]:
                            if meta.get("latency_s"):
                                st.caption(f"⏱️ {meta['latency_s']:.1f}s")
                        with meta_cols[1]:
                            if meta.get("guardrail_triggered"):
                                st.caption("🛡️ Guardrail triggered")
                        with meta_cols[2]:
                            cat = meta.get("safety_category", "safe")
                            if cat and cat != "safe":
                                st.caption(f"⚠️ {cat}")
                        with meta_cols[3]:
                            if meta.get("error"):
                                st.caption("❌ Error")

        # Chat input
        user_input = st.chat_input(
            "Type your message here...",
            key="chat_input",
        )

        if user_input:
            self._handle_user_input(
                user_input, assistant_type, show_metadata, enable_guardrails
            )

    def _handle_user_input(
        self,
        user_input: str,
        assistant_type: str,
        show_metadata: bool,
        enable_guardrails: bool,
    ) -> None:
        """Process user input: guardrail check, model call, and response display."""
        model_type = "oss" if "OSS" in assistant_type else "frontier"
        model_prefix = "OSS" if "OSS" in assistant_type else "Frontier"

        # Add user message to memory and display
        self.memory.add_user_message(user_input)

        with st.chat_message("user"):
            st.write(user_input)

        # Check guardrails
        if enable_guardrails:
            safety_check = self.guardrail_service.check_input(user_input)
            if not safety_check.is_safe:
                self._handle_guardrail_block(safety_check, user_input)
                return

        # Get model response
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking... ({model_prefix})"):
                messages_dict = self.memory.get_messages_dict()
                response = self.model_service.generate(messages_dict, model_type)

                response_container = st.empty()
                response_container.write(response.response)

                # Show metadata
                if show_metadata:
                    cols = st.columns(4)
                    with cols[0]:
                        st.caption(f"⏱️ {response.latency_s:.1f}s")
                    with cols[1]:
                        if response.guardrail_triggered:
                            st.caption(f"🛡️ {response.safety_category}")
                    with cols[2]:
                        st.caption(f"🤖 {model_prefix}")
                    with cols[3]:
                        if response.error:
                            st.caption(f"❌ {response.error[:50]}")

        # Add assistant response with metadata
        self.memory.add_assistant_message(
            response.response,
            metadata={
                "latency_s": response.latency_s,
                "guardrail_triggered": response.guardrail_triggered,
                "safety_category": response.safety_category,
                "error": response.error,
            },
        )

        # Log guardrail if triggered by model
        if response.guardrail_triggered:
            event = self.guardrail_service.create_guardrail_event(
                f"model_output_{response.safety_category}",
                response.safety_category,
                user_input,
                self.memory.interaction_count,
            )
            self.memory.log_guardrail_event(event)
            st.rerun()

    def _handle_guardrail_block(
        self, safety_check, user_input: str
    ) -> None:
        """Display guardrail warning and log the event."""
        self.memory.log_guardrail_event(
            self.guardrail_service.create_guardrail_event(
                "input_blocked",
                safety_check.category,
                user_input,
                self.memory.interaction_count,
            )
        )

        with st.chat_message("assistant"):
            st.warning(f"🛡️ **Guardrail triggered - {safety_check.category}**")
            st.write(safety_check.refusal)

        # Add assistant refusal with metadata
        self.memory.add_assistant_message(
            safety_check.refusal or "",
            metadata={
                "latency_s": 0.0,
                "guardrail_triggered": True,
                "safety_category": safety_check.category,
                "error": None,
            },
        )
        st.rerun()