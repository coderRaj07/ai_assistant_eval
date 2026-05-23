"""
AI Assistant Evaluation Platform - Entry Point
===============================================
Compare Open Source (Qwen2.5-0.5B-Instruct) vs Frontier (OpenRouter) AI Assistants.

Features:
- Multi-turn conversation with both assistants
- Guardrails & safety layer with real-time indicators
- Evaluation dashboard with comparison metrics
- Observability: latency tracking, guardrail logs
- Context-aware memory with pruning
"""

import streamlit as st
from dotenv import load_dotenv

from app.config import PAGE_TITLE, PAGE_ICON, LAYOUT
from app.models.conversation import MessageRole
from app.utils.memory import MemoryManager
from app.services.model_service import ModelService
from app.services.guardrail_service import GuardrailService
from app.components.sidebar import Sidebar
from app.components.chat_panel import ChatPanel
from app.components.evaluation_panel import EvaluationPanel
from app.components.safety_dashboard import SafetyDashboard
from app.components.about_panel import AboutPanel

load_dotenv()

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inject Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .main-header {
        color: #1E3A5F;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #4A6FA5;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }
    .safety-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .safe { background-color: #d4edda; color: #155724; }
    .warning { background-color: #fff3cd; color: #856404; }
    .danger { background-color: #f8d7da; color: #721c24; }
    .info { background-color: #d1ecf1; color: #0c5460; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Initialize Services & Components
# ---------------------------------------------------------------------------

# Session-based state for chat_tab
if "chat_tab" not in st.session_state:
    st.session_state.chat_tab = "OSS Assistant"

# Core services (singletons per session)
memory_manager = MemoryManager(st.session_state)
model_service = ModelService()
guardrail_service = GuardrailService()

# UI Components
sidebar = Sidebar(memory_manager)
chat_panel = ChatPanel(memory_manager, model_service, guardrail_service)
evaluation_panel = EvaluationPanel(model_service, guardrail_service)
safety_dashboard = SafetyDashboard(memory_manager)
about_panel = AboutPanel()

# ---------------------------------------------------------------------------
# Render Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">🤖 AI Assistant Evaluation Platform</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Comparing Open Source (Qwen2.5-0.5B) vs Frontier (OpenRouter Free Models) AI Assistants</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Render Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat Interface",
    "📊 Live Evaluation",
    "🛡️ Safety Dashboard",
    "ℹ️ About",
])

# --- Tab 1: Chat Interface ---
with tab1:
    col1, col2 = st.columns([1, 3])

    with col1:
        sidebar_config = sidebar.render()

    with col2:
        chat_panel.render(
            assistant_type=sidebar_config["assistant_type"],
            show_metadata=sidebar_config["show_metadata"],
            enable_guardrails=sidebar_config["enable_guardrails"],
        )

# --- Tab 2: Evaluation Dashboard ---
with tab2:
    evaluation_panel.render()

# --- Tab 3: Safety Dashboard ---
with tab3:
    safety_dashboard.render()

# --- Tab 4: About ---
with tab4:
    about_panel.render()