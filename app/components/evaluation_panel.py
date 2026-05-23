"""
Evaluation Panel Component - Live evaluation dashboard.

Provides quick test functionality to compare both models on a given prompt,
displays evaluation categories, and shows links to past results and full
evaluation runner.
"""

import streamlit as st
from pathlib import Path
from typing import Optional

from app.config import EVALUATION_CATEGORIES_INFO, QUICK_TEST_CATEGORIES
from app.services.model_service import ModelService
from app.services.guardrail_service import GuardrailService


class EvaluationPanel:
    """
    Evaluation dashboard UI component.

    Handles:
    - Test category overview display
    - Quick test prompt submission
    - Model comparison results display
    - Past results browser
    - Full evaluation runner link
    """

    def __init__(
        self,
        model_service: ModelService,
        guardrail_service: GuardrailService,
    ):
        self.model_service = model_service
        self.guardrail_service = guardrail_service

    def render(self) -> None:
        """Render the evaluation dashboard."""
        st.markdown("### 📊 Evaluation Dashboard")
        st.markdown("Run the evaluation framework to compare both models across categories.")

        eval_col1, eval_col2, eval_col3 = st.columns(3)

        with eval_col1:
            self._render_categories()

        with eval_col2:
            self._render_quick_test()

        with eval_col3:
            self._render_past_results()

    def _render_categories(self) -> None:
        """Render the test categories overview."""
        st.markdown("#### 🎯 Test Categories")
        for cat, desc in EVALUATION_CATEGORIES_INFO.items():
            st.markdown(f"- **{cat}**: {desc}")

    def _render_quick_test(self) -> None:
        """Render the quick test form."""
        st.markdown("#### 📋 Quick Test")
        st.markdown("Test a specific prompt against both models:")

        test_prompt = st.text_area(
            "Enter a test prompt:",
            value="What is the capital of France?",
            height=100,
        )

        test_category = st.selectbox(
            "Category:",
            QUICK_TEST_CATEGORIES,
        )

        if st.button("🚀 Test Both Models", use_container_width=True, type="primary"):
            self._run_quick_test(test_prompt, test_category)

    def _run_quick_test(self, prompt: str, category: str) -> None:
        """Run a quick comparison test on both models."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        with st.spinner("Testing OSS model..."):
            oss_response = self.model_service.generate(messages, "oss")

        with st.spinner("Testing Frontier model..."):
            frontier_response = self.model_service.generate(messages, "frontier")

        st.markdown("---")
        st.markdown("### Results")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("#### OSS (Qwen2.5-0.5B)")
            st.info(oss_response.response)
            st.caption(
                f"Latency: {oss_response.latency_s:.1f}s | "
                f"Guardrail: {oss_response.guardrail_triggered}"
            )

        with res_col2:
            st.markdown("#### Frontier (OpenRouter)")
            st.info(frontier_response.response)
            st.caption(
                f"Latency: {frontier_response.latency_s:.1f}s | "
                f"Guardrail: {frontier_response.guardrail_triggered}"
            )

        # Safety analysis of outputs
        st.markdown("#### Quick Analysis")
        oss_safe = self.guardrail_service.check_input(oss_response.response).is_safe
        frontier_safe = self.guardrail_service.check_input(frontier_response.response).is_safe

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "OSS Output Safe",
                "✅" if oss_safe else "⚠️",
                delta=None if oss_safe else "Flagged",
            )
        with col_b:
            st.metric(
                "Frontier Output Safe",
                "✅" if frontier_safe else "⚠️",
                delta=None if frontier_safe else "Flagged",
            )

    def _render_past_results(self) -> None:
        """Render past evaluation results browser."""
        st.markdown("#### 📁 Past Results")

        from app.config import EVALUATION_OUTPUT_DIR

        if EVALUATION_OUTPUT_DIR.exists():
            result_files = sorted(
                EVALUATION_OUTPUT_DIR.glob("evaluation_results_*.json"),
                reverse=True,
            )
            if result_files:
                st.markdown(f"Found {len(result_files)} evaluation result files")
                for f in result_files[:5]:
                    st.markdown(f"- {f.name}")
            else:
                st.info("No evaluation results yet. Run `python evaluation/evaluator.py`")
        else:
            st.info("No evaluation results found")

        st.markdown("---")
        st.markdown("#### 🚀 Run Full Evaluation")
        st.markdown("Execute in terminal:")
        st.code(
            "cd /media/rajendra/HDD5/Assignments/Olive.ai/ai_assistant_eval "
            "&& python evaluation/evaluator.py",
            language="bash",
        )
        st.markdown("This will test all 170+ prompts and generate comparison visualizations.")