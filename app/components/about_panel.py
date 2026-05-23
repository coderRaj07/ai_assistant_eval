"""
About Panel Component - Project information display.

Provides an overview of the project, architecture diagram, model
comparison, cost estimates, and relevant links.
"""

import streamlit as st


class AboutPanel:
    """
    About/Information page component.

    Renders the project description, architecture overview, model specs,
    cost estimates, and links to external resources.
    """

    def render(self) -> None:
        """Render the about page."""
        st.markdown("### ℹ️ About This Project")

        about_col1, about_col2 = st.columns([2, 1])

        with about_col1:
            self._render_description()

        with about_col2:
            self._render_model_info()
            self._render_cost_estimate()
            self._render_links()

    def _render_description(self) -> None:
        """Render project description and architecture."""
        st.markdown("""
        #### AI Assistant Evaluation Platform

        This platform compares two AI assistants:

        1. **Open Source Assistant**: Qwen2.5-0.5B-Instruct (via Hugging Face Inference API)
        2. **Frontier Assistant**: OpenRouter (Gemini/Llama/Phi/Mistral with free fallback)

        #### Key Features

        - **Multi-turn conversations** with context memory
        - **Guardrails & Safety Layer**: Input/output filtering with category-specific refusal responses
        - **Observability**: Latency tracking, guardrail event logging, context statistics
        - **Evaluation Framework**: 170+ prompts across 17 categories
        - **LLM-as-Judge** evaluation with visual comparison reports

        #### Architecture

        ```
        app/
        ├── main.py                 # Entry point
        ├── config.py               # Configuration & constants
        ├── models/                 # Data models (dataclasses)
        ├── services/               # Business logic layer
        ├── components/             # Streamlit UI components
        └── utils/                  # Model wrappers & memory

        evaluation/
        ├── evaluator.py            # Orchestrator
        ├── judge.py                # LLM-as-judge evaluation
        ├── metrics.py              # Metrics calculations
        ├── visualizations.py       # Chart generation
        └── prompts.json            # 170+ evaluation prompts
        ```
        """)

    def _render_model_info(self) -> None:
        """Render model comparison table."""
        st.markdown("#### Models")
        st.markdown("""
        | Model | Type | Params |
        |-------|------|--------|
        | Qwen2.5-0.5B | OSS | 500M |
        | OpenRouter (Gemini/Llama/Phi/Mistral) | Frontier | varies |
        """)

    def _render_cost_estimate(self) -> None:
        """Render cost estimation information."""
        st.markdown("#### Cost Estimate")
        st.markdown("""
        **OSS (HF Inference API)**:
        - Free tier available
        - ~$0.01/1000 inferences

        **Frontier (OpenRouter Free Models)**:
        - Gemini 2.0 Flash: 10 req/min free
        - Llama 3.2 3B: 20 req/min free
        - Phi-3 Mini: 20 req/min free
        - Mistral 7B: 10 req/min free
        """)

    def _render_links(self) -> None:
        """Render external links."""
        st.markdown("#### Links")
        st.markdown("- [GitHub Repository](https://github.com/coderraj07/ai_assistant_eval)")
        st.markdown("- [Hugging Face Model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)")
        st.markdown("- [OpenRouter](https://openrouter.ai/)")