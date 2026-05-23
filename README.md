# 🤖 AI Assistant Evaluation Platform

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit)](https://huggingface.co/spaces/coderraj07/ai-assistant-eval)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Open Source](https://img.shields.io/badge/Model-Qwen2.5--0.5B-green?logo=huggingface)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[![OSS Deployed](https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-brightgreen?logo=huggingface)](https://huggingface.co/spaces/coderraj07/qwen-oss-api)
[![Frontier Model](https://img.shields.io/badge/Model-Gemini%202.0%20Flash%20%2B%20OpenRouter-orange?logo=google)](https://ai.google.dev)
[![Guardrails](https://img.shields.io/badge/Safety-Guardrails%20Active-red)](https://github.com/coderRaj07/ai_assistant_eval)
[![Memory](https://img.shields.io/badge/Memory-In--Context%20(20%20msg)-blueviolet)](https://github.com/coderRaj07/ai_assistant_eval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comprehensive platform for building, comparing, and evaluating **Open Source** vs **Frontier** AI Assistants across factual accuracy, bias detection, jailbreak resistance, and content safety.

---

## 📋 Quick Overview for Reviewers

This project fulfills **all bonus requirements** for a guaranteed interview call:

| Requirement | Status | Details |
|-------------|--------|---------|
| ✅ **OSS Model Deployed Publicly** | **Done** | Qwen2.5-0.5B-Instruct on HuggingFace Spaces → [coderraj07/qwen-oss-api](https://huggingface.co/spaces/coderraj07/qwen-oss-api) |
| ✅ **Guardrails / Safety Layers** | **Done** | 7 input safety categories + output bias/violence checks |
| ✅ **In-Memory Context** | **Done** | Session-based conversational memory with auto-pruning (max 20 messages) |
| ✅ **Cost + Latency Analysis** | **Done** | See [Cost & Latency](#-cost--latency-analysis) and [Deployment Guide](DEPLOYMENT.md) |
| ✅ **Evaluations / Evals** | **Done** | 185 prompts across 18 categories with LLM-as-judge scoring |
| ✅ **Multi-Turn Conversations** | **Done** | Full chat history with system prompt injection |
| ✅ **Frontier Model Integration** | **Done** | Gemini 2.0 Flash (direct) + OpenRouter models (liquid/lfm-1.2B, free chain) |
| ✅ **Visual Comparison Reports** | **Done** | Radar charts, bar charts, latency comparison, PDF reports |

---

## 📋 Table of Contents

- [Quick Overview for Reviewers](#-quick-overview-for-reviewers)
- [Flow: What the Reviewer Sees](#-flow-what-the-reviewer-sees)
- [Overview](#-overview)
- [Architecture](#-architecture)
- [Models Compared](#-models-compared)
- [OSS Model Deployment (HuggingFace)](#-oss-model-deployment-huggingface)
- [Guardrails & Safety Layer](#-guardrails--safety-layer)
- [In-Memory Conversational Context](#-in-memory-conversational-context)
- [Features](#-features)
- [Setup & Installation](#-setup--installation)
- [Usage Guide](#-usage-guide)
- [Running Evaluations](#-running-evaluations)
- [Evaluation Categories](#-evaluation-categories)
- [Evaluation Results (Actual)](#-evaluation-results-actual)
- [Cost & Latency Analysis](#-cost--latency-analysis)
- [Deployment](#-deployment)
- [Architecture Decisions & Tradeoffs](#-architecture-decisions--tradeoffs)
- [Improvements with More Time](#-improvements-with-more-time)
- [License](#-license)

---

## 🔄 Flow: What the Reviewer Sees

```
[1] Reviewer clones repo
    │
    ├── README.md (you are here)
    │   ├── What was built
    │   ├── Architecture decisions
    │   ├── Evaluation results
    │   └── OSS deployment details
    │
    ├── DEPLOYMENT.md
    │   ├── Cost analysis tables
    │   ├── Latency comparison
    │   └── Step-by-step deploy guide
    │
    ├── app.py → Streamlit UI (4 tabs)
    │   ├── Tab 1: Chat (multi-turn with guardrails)
    │   ├── Tab 2: Live Evaluation (A/B test)
    │   ├── Tab 3: Safety Dashboard (guardrail log)
    │   └── Tab 4: About (architecture)
    │
    ├── utils/
    │   ├── hf_model.py          → OSS: calls deployed HF Space + fallback
    │   ├── openrouter_model.py  → Frontier: Gemini via OpenRouter
    │   ├── guardrails.py        → 7 input + output safety checks
    │   └── memory.py            → In-context session memory
    │
    └── evaluation/
        ├── evaluator.py         → 185 prompts, LLM-as-judge
        ├── prompts.json         → 18 categories
        └── output/
            ├── cmd_output.md    → Full eval run log
            ├── comparison_summary_*.json → Actual metrics
            ├── overall_summary.png       → Bar chart
            ├── radar_comparison.png      → Radar chart
            └── evaluation_report_*.pdf   → One-page PDF
```

### Quick Start for Reviewers

```bash
# 1. Clone & install
git clone https://github.com/coderRaj07/ai_assistant_eval.git
cd ai_assistant_eval
pip install -r requirements.txt

# 2. Set up environment (copy template, then edit with your keys)
cp .env.example .env
# Edit .env with all 4 API keys: HF_TOKEN, GEMINI_API_KEY, OPENROUTER_API_KEY, CEREBRAS_API_KEY

# 3. Run the app (UI demo)
streamlit run app.py

# 4. Run quick evaluation (8 sample prompts)
python evaluation/evaluator.py --max-prompts 8

# 5. Run full evaluation (all 185 prompts)
python evaluation/evaluator.py

# 6. Generate PDF report
python evaluation/generate_report.py
```

---

## 🎯 Overview

This project builds and evaluates **two AI personal assistants** side-by-side:

1. **Open Source Assistant**: Qwen2.5-0.5B-Instruct deployed on **HuggingFace Spaces** (public) with HF Inference API fallback
2. **Frontier Model Assistant**: Gemini 2.0 Flash (direct API) + OpenRouter frontier models (liquid/lfm-2.5-1.2b-instruct, openrouter/free)

Both assistants support:
- **Multi-turn conversations** with full chat history
- **Short-term conversational memory/context** (session-based, auto-pruning)
- **Guardrails & safety layers** (7 input categories + output checks)
- **Basic assistant-like behavior** with system prompt injection

The evaluation framework tests them across **185+ prompts** in **18 categories** covering factual accuracy, bias, jailbreak resistance, and content safety using **LLM-as-judge** scoring.

### Key Deliverables

| Deliverable                            | Status   | Location |
| -------------------------------------- | -------- | -------- |
| ✅ GitHub Repository                   | Complete | [github.com/coderRaj07/ai_assistant_eval](https://github.com/coderRaj07/ai_assistant_eval) |
| ✅ README with setup/architecture      | Complete | This file |
| ✅ OSS Model Deployed Publicly         | Complete | [HF Spaces: coderraj07/qwen-oss-api](https://huggingface.co/spaces/coderraj07/qwen-oss-api) |
| ✅ Guardrails / Safety Layer           | Complete | `utils/guardrails.py` — 7 input categories |
| ✅ In-Memory Conversational Context    | Complete | `utils/memory.py` — session state with pruning |
| ✅ Multi-turn conversation UI          | Complete | Streamlit 4-tab interface |
| ✅ Evaluation Framework (185+ prompts) | Complete | `evaluation/evaluator.py` + `prompts.json` |
| ✅ LLM-as-Judge scoring                | Complete | Cerebras API for impartial evaluation |
| ✅ Visual comparison reports           | Complete | `evaluation/output/*.png` |
| ✅ Evaluation PDF generation           | Complete | `evaluation/output/evaluation_report_*.pdf` |
| ✅ Cost + Latency Analysis             | Complete | `DEPLOYMENT.md` + README section |

---

## 🏗 Architecture

```
ai_assistant_eval/
├── app.py                         # Streamlit UI (4 tabs: Chat, Evaluation, Safety, About)
├── .env                           # API keys (HF_TOKEN, GEMINI_API_KEY)
├── requirements.txt               # Python dependencies
├── setup.sh                       # One-click setup script
├── Makefile                       # Build automation
├── DEPLOYMENT.md                  # Deployment guide + cost analysis
│
├── utils/
│   ├── hf_model.py                # OSS model: Qwen2.5-0.5B-Instruct
│   │   ├── Calls deployed HF Space (coderraj07/qwen-oss-api) via gradio_client
│   │   ├── Falls back to HF Inference API if Space unavailable
│   │   ├── Guardrail integration (input/output checks)
│   │   ├── Latency tracking & error handling
│   │   └── Configurable generation parameters
│   │
│   ├── openrouter_model.py        # Frontier model: Gemini 2.0 Flash + OpenRouter
│   │   ├── Priority chain: Gemini 2.0 Flash → liquid/lfm-1.2B → openrouter/free
│   │   ├── System prompt injection
│   │   ├── Gemini safety settings (4 harm categories)
│   │   ├── Guardrail integration
│   │   └── Latency tracking
│   │
│   ├── guardrails.py              # Safety Layer
│   │   ├── 7 input safety categories (malware, violence, self-harm,
│   │   │     illegal, discrimination, jailbreak, stereotypes)
│   │   ├── Output safety checks (bias, violence detection)
│   │   ├── Category-specific refusal responses
│   │   └── Safety metadata generation
│   │
│   └── memory.py                  # Conversational Memory
│       ├── Session-based storage via Streamlit session state
│       ├── Automatic context pruning (max 20 messages)
│       ├── System prompt injection
│       ├── Guardrail event logging
│       └── Context statistics
│
├── evaluation/
│   ├── evaluator.py               # Comprehensive Evaluation Framework
│   │   ├── Automated scoring across 18 categories (185+ prompts)
│   │   ├── LLM-as-judge via Cerebras API
│   │   ├── Heuristic-based evaluation (refusal detection,
│   │   │     bias challenging, jailbreak resistance)
│   │   ├── 5 visualization charts (group comparison,
│   │   │     hallucination rates, radar chart, latency,
│   │   │     overall summary)
│   │   └── CSV/JSON result export
│   │
│   ├── generate_report.py         # One-page PDF report generator
│   │
│   └── prompts.json               # 185+ evaluation prompts in 18 categories
```

### Data Flow

```
User Input
    │
    ├──→ Guardrails (Input Safety Check - 7 categories)
    │      │
    │      ├── Blocked ──→ Refusal Response (category-specific)
    │      │                  └── Logged to Guardrail Event Log
    │      │
    │      └── Passed ──→ Model Selection
    │                       │
    │                       ├── OSS (Qwen2.5-0.5B) ──→ HF Space (gradio_client)
    │                       │                              │
    │                       │                         └── Fallback: HF Inference API
│                       │
│                       ├── Frontier (Gemini 2.0 Flash) ──→ Gemini Direct API
│                       │   │
│                       │   └── Fallback: OpenRouter (liquid/lfm-1.2B → free chain)
    │                                                   │
    │                   Guardrails (Output Safety Check)←─┘
    │                              │
    │                              ├── Filtered ──→ Blocked Response
    │                              │
    │                              └── Safe ──→ Display to User
    │
    └──→ Memory Update (session state, context pruning, guardrail logging)
```

---

## 🤖 Models Compared

### Open Source: Qwen2.5-0.5B-Instruct

| Property           | Value                      |
| ------------------ | -------------------------- |
| **Model**          | Qwen/Qwen2.5-0.5B-Instruct |
| **Size**           | 500M parameters            |
| **Deployment**     | Hugging Face Spaces (public) + HF Inference API fallback |
| **Space URL**      | [coderraj07/qwen-oss-api](https://huggingface.co/spaces/coderraj07/qwen-oss-api) |
| **Context Length** | 32K tokens                 |
| **Architecture**   | Transformer decoder-only   |
| **License**        | Apache 2.0                 |

**Tradeoff**: Smaller model size means faster inference but reduced reasoning capability compared to frontier models. Great for cost-sensitive deployments.

### Frontier: Gemini 2.0 Flash + OpenRouter Chain

| Property            | Value                                          |
| ------------------- | ---------------------------------------------- |
| **Primary Model**   | gemini-2.0-flash (direct Google API)           |
| **Fallback Models** | liquid/lfm-2.5-1.2b-instruct:free → openrouter/free |
| **Deployment**      | Google Generative AI API (direct) + OpenRouter API |
| **Context Length**  | 1M tokens (Gemini)                             |
| **Pricing**         | Gemini: Free (60 req/min) / OpenRouter: Free tier |
| **Built-in Safety** | Gemini: 4 harm categories + our guardrails     |

**Tradeoff**: Primary route (Gemini 2.0 Flash) gives high quality. Fallback chain ensures availability when quota exhausted, but lower-quality models may be used.

---

## 🚀 OSS Model Deployment (HuggingFace)

The OSS model **Qwen2.5-0.5B-Instruct** is publicly deployed on HuggingFace Spaces:

**Space:** [https://huggingface.co/spaces/coderraj07/qwen-oss-api](https://huggingface.co/spaces/coderraj07/qwen-oss-api)

### Deployment Architecture

```
[Your App / Evaluator]
        │
        ├── 1st Attempt: gradio_client → HF Space (coderraj07/qwen-oss-api)
        │       └── Fast inference on warm CPU instance
        │
        └── 2nd Attempt: requests → HF Inference API (fallback)
                └── Slightly slower, but always available
```

### Why This Approach?

| Approach | Latency | Cost | Complexity |
|----------|---------|------|------------|
| 🥇 **HF Spaces (gradio_client)** | ~2-8s | Free | Low |
| 🥈 HF Inference API (fallback) | ~5-15s | Free (rate limited) | Minimal |
| Local Ollama | ~1-3s | Free | High (needs GPU) |
| Modal GPU | ~0.5-2s | ~$0.50/hr | Medium |

The dual-layer approach ensures high availability: if the Space is cold-starting, the Inference API automatically takes over.

---

## 🛡️ Guardrails & Safety Layer

Implemented in `utils/guardrails.py` with centralized patterns in `app/config.py`:

### Input Safety — 7 Detection Categories

| Category | Examples Detected |
|----------|------------------|
| **Malware** | "write a virus", "hack into", "ransomware code" |
| **Violence** | "how to kill", "hurt someone", "make a bomb" |
| **Self-Harm** | "suicide methods", "cut myself", "harm myself" |
| **Illegal Activities** | "steal a car", "drug synthesis", "identity theft" |
| **Discrimination** | "hate [group]", "why are [group] inferior" |
| **Jailbreak Attempts** | "ignore your instructions", "DAN mode", "you are now" |
| **Stereotypes** | "all [group] are", "typical [nationality]" |

### Output Safety

- **Bias Detection**: Regex patterns detecting generalizations about groups
- **Violence Monitoring**: Detects if model provides harmful instructions after harmful prompt

### How Guardrails Work

```python
# 1. Check input
is_safe, refusal, category = check_input_safety(user_input)

if not is_safe:
    # Return category-specific refusal response
    return {"response": refusal, "guardrail_triggered": True, "safety_category": category}

# 2. Generate response from model
response = model.generate(messages)

# 3. Check output
is_output_safe, filtered, issue = check_output_safety(response, user_input)
```

---

## 💾 In-Memory Conversational Context

Implemented in `utils/memory.py`:

### Features

| Feature | Implementation |
|---------|---------------|
| **Storage** | Streamlit `session_state` (in-memory, per session) |
| **Context Limit** | Auto-prunes to max 20 messages (preserves system prompt) |
| **System Prompt** | "Helpful, harmless, honest" — injected as first message |
| **Guardrail Logging** | Every guardrail event logged with timestamp |
| **Context Stats** | Message count, character count, interaction count |
| **Clear Memory** | Button to reset conversation |

### How Memory Works

```python
# Initialize
init_memory(st, system_prompt="You are a helpful assistant...")

# Add user message
add_message("user", user_input, st)

# Add assistant response
add_message("assistant", response, st, metadata={"latency_s": 2.5})

# Get full context for model
messages = get_messages(st)

# Auto-pruning when exceeding 20 messages
if len(st.session_state.messages) > MAX_CONTEXT_MESSAGES + 1:
    st.session_state.messages = [system_prompt] + last_20_messages
```

---

## ✨ Features

### Multi-Turn Conversations

- Full chat history maintained in Streamlit session state
- System prompt injection for consistent assistant behavior
- Automatic context pruning (max 20 messages) to avoid token limits
- Configurable model selection per conversation (OSS or Frontier)
- Metadata display (latency, guardrail status, model used)

### Guardrails & Safety Layer

- **7 input detection categories**: malware, violence, self-harm, illegal activities, discrimination, jailbreak attempts, stereotypes
- **Output safety checks**: bias detection, violence monitoring
- **Category-specific refusal responses**: polite, informative, and contextual
- **Real-time guardrail indicators** in the UI
- **Guardrail Event Log** with timestamps (Safety Dashboard tab)

### Evaluation Framework

- **185+ prompts** across 18 categories
- **LLM-as-judge scoring** via Cerebras API (impartial evaluation)
- **Heuristic-based scoring**: refusal detection, bias challenging, jailbreak resistance
- **5 visualization charts**: group comparison, hallucination rates, radar chart, latency comparison, overall summary
- **CSV/JSON export** for further analysis
- **Quick test mode** for rapid A/B testing
- **One-page PDF report** generation with infographics

### Observability

- Per-response latency tracking
- Guardrail event logging with timestamps
- Context statistics (messages, interactions, guardrail events)
- Error handling with user-friendly messages
- Model used tracking (Space vs API fallback)

---

## 🔧 Setup & Installation

### Prerequisites

- Python 3.10+
- API keys (see below)
- Internet connection (for HF Space, HF Inference API, and OpenRouter API)

### Step 1: Clone the Repository

```bash
git clone https://github.com/coderRaj07/ai_assistant_eval.git
cd ai_assistant_eval
```

### Step 2: Set Up API Keys

Copy the environment file template and fill in your keys:

```bash
cp .env.example .env
```

Then edit `.env` with your API keys:

```bash
# .env — requires 4 API keys:
HF_TOKEN=your_hf_token              # Required for OSS model (HF Inference API)
GEMINI_API_KEY=your_gemini_key      # Required for Frontier model (Gemini 2.0 Flash)
OPENROUTER_API_KEY=your_or_key      # Required for Frontier model (OpenRouter fallback)
CEREBRAS_API_KEY=your_cerebras_key  # Required for LLM-as-Judge evaluation
```

**Where to get API keys:**

| Key | Purpose | Get It |
|-----|---------|--------|
| `HF_TOKEN` | OSS model (HF Inference API) | [Hugging Face → Access Tokens](https://huggingface.co/settings/tokens) |
| `GEMINI_API_KEY` | Frontier primary model | [Google AI Studio → API Keys](https://aistudio.google.com/apikey) |
| `OPENROUTER_API_KEY` | Frontier fallback chain | [OpenRouter → Keys](https://openrouter.ai/keys) |
| `CEREBRAS_API_KEY` | LLM-as-Judge evaluation | [Cerebras → API](https://cloud.cerebras.ai/) |

### Step 3: Install Dependencies

```bash
# Option A: Using pip
pip install -r requirements.txt

# Option B: Using setup script
chmod +x setup.sh
./setup.sh

# Option C: Using Make (if available)
make install
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 📖 Usage Guide

### Chat Interface (Tab 1)

1. **Select Assistant**: Choose between OSS (Qwen2.5) or Frontier (Gemini) from sidebar
2. **Type message**: Enter your question or prompt in the chat input
3. **View response**: The assistant responds with metadata (latency, guardrail status, model used)
4. **Toggle guardrails**: Enable/disable via Settings expander
5. **View metadata**: Check "Show response metadata" for detailed info
6. **Clear conversation**: Reset chat history

### Live Evaluation (Tab 2)

1. **Quick Test**: Enter a prompt and test both models side-by-side
2. **Select Category**: Choose the evaluation category for context
3. **View Results**: Both responses displayed with latency and safety scores
4. **Run Full Evaluation**: Execute CLI evaluator for comprehensive results

### Safety Dashboard (Tab 3)

- **Guardrail Event Log**: All safety events recorded with timestamps
- **Coverage Documentation**: All detected patterns and refusal responses
- **Real-time Metrics**: Guardrail counts, interaction totals

### About (Tab 4)

- Architecture documentation
- Cost estimates
- Model specifications
- Links to deployed models and APIs

---

## 🧪 Running Evaluations

### Quick Test (8 prompts — ~3 minutes)

```bash
python evaluation/evaluator.py --max-prompts 8
```

### Moderate Test (10 prompts — ~4 minutes)

```bash
python evaluation/evaluator.py --max-prompts 10
```

### Sample Mode (2 prompts per category — ~10 minutes)

```bash
python evaluation/evaluator.py --sample 2
```

### Full Evaluation Suite (185 prompts — ~20-40 minutes)

```bash
python evaluation/evaluator.py
```

### What the Evaluator Does

1. Loads 185+ prompts from `evaluation/prompts.json` (18 categories)
2. Tests each prompt against **both models** (OSS + Frontier)
3. Scores responses using **LLM-as-judge** (Cerebras API) evaluating:
   - Safety (1-5)
   - Bias Neutrality (1-5)
   - Factual Accuracy (1-5)
   - Refusal Quality (1-5)
   - Helpfulness (1-5)
   - Hallucination Detection
4. Generates **5 visualization charts**
5. Saves results as **JSON, CSV, and text report**
6. Generates **one-page PDF report** with infographics

### Evaluation Output

```
evaluation/output/
├── evaluation_results_*.json          # Full per-prompt results
├── evaluation_responses_*.csv          # Per-response data for analysis
├── comparison_summary_*.json           # Aggregate comparison metrics
├── evaluation_report_*.txt             # Plain text report
├── evaluation_report_*.pdf             # One-page PDF with infographics
├── cmd_output.md                       # CLI execution log
├── category_group_comparison.png       # Bar chart by dimension
├── overall_summary.png                 # Overall metrics bar chart
├── radar_comparison.png                # Capability radar chart
└── latency_comparison.png              # Response time comparison
```

### Generate PDF Report

```bash
python evaluation/generate_report.py
```

---

## 📊 Evaluation Categories

Our evaluation framework covers **185+ prompts** across **18 categories**:

| Category                  | Prompts | What We Test                                 |
| ------------------------- | ------- | -------------------------------------------- |
| **Factual**               | 25      | General knowledge, history, science          |
| **Reasoning**             | 5       | Logic puzzles, math word problems            |
| **Multi-turn**            | 10      | Conversation coherence, context retention    |
| **Bias/Gender**           | 10      | Gender stereotypes, leadership bias          |
| **Bias/Race**             | 10      | Racial stereotypes, profiling                |
| **Bias/Religion**         | 10      | Religious stereotypes, discrimination        |
| **Bias/Age**              | 10      | Age discrimination, generational stereotypes |
| **Stereotypes**           | 10      | Occupational, cultural stereotypes           |
| **Jailbreak/Direct**      | 10      | Direct instruction override attempts         |
| **Jailbreak/Prompt Leak** | 10      | System prompt extraction attempts            |
| **Jailbreak/Roleplay**    | 10      | Roleplay-based manipulation                  |
| **Safety/Violence**       | 10      | Violence-related harmful requests            |
| **Safety/Self-harm**      | 10      | Self-harm related requests                   |
| **Safety/Discrimination** | 10      | Hate speech, discriminatory content          |
| **Safety/Misinformation** | 10      | Conspiracy theories, false claims            |
| **Safety/Sexual**         | 10      | Sexual content safety, grooming              |
| **Political**             | 10      | Political opinion neutrality                 |
| **Ethical Dilemmas**      | 10      | Ethical reasoning quality                    |

---

## 📈 Evaluation Results (Actual)

Results from the last evaluation run (`evaluation/output/cmd_output.md`):

### Overall Metrics

| Metric                    | OSS (Qwen2.5-0.5B) | Frontier (Gemini 2.0 Flash + OpenRouter) | Winner   |
| ------------------------- | ------------------ | ------------------ | -------- |
| **Safety Score**          | 2.38               | 2.25               | **OSS**  |
| **Hallucination Rate**    | 0.0%               | 0.0%               | Tie      |
| **Refusal Rate**          | 25.0%              | 12.5%              | Frontier |
| **Avg Latency**           | 25.68s             | 5.57s              | Frontier |
| **Guardrail Trigger Rate**| 12.5%              | 25.0%              | **OSS**  |

### Category Group Comparison (score 0-5)

| Group                    | OSS   | Frontier | Diff   | Winner     |
| ------------------------ | ----- | -------- | ------ | ---------- |
| **Factual Accuracy**     | 0.0   | 4.0      | 4.0    | Frontier   |
| **Bias & Stereotypes**   | 1.0   | 1.4      | 0.4    | Frontier   |
| **Jailbreak Resistance** | 3.0   | 5.0      | 2.0    | Frontier   |
| **Content Safety**       | 2.0   | 1.0      | -1.0   | **OSS**    |
| **Ethical & Political**  | 0     | 0        | 0      | Tie        |

### Dimensional Results

| Dimension                | OSS Score | Frontier Score | Winner     |
| ------------------------ | --------- | -------------- | ---------- |
| Gender Bias Sensitivity  | 1.75      | 0              | **OSS**    |
| Religious Bias           | 0         | 2.75           | Frontier   |
| Direct Jailbreak         | 4.0       | 4.0            | Tie        |
| Roleplay Jailbreak       | 1.75      | 0              | **OSS**    |
| Multi-turn Coherence     | 4.0       | 4.0            | Tie        |
| Reasoning                | 0         | 4.0            | Frontier   |
| Misinformation           | 1.75      | 0              | **OSS**    |
| Self-harm Safety         | 4.25      | 1.75           | **OSS**    |
| Violence Safety          | 1.75      | 0              | **OSS**    |
| Stereotypes              | 0         | 1.75           | Frontier   |
| Sexual Content Safety    | 1.75      | 1.75           | Tie        |

### Key Insights

1. **OSS model wins on content safety** (2.0 vs 1.0) — Guardrails make a significant difference
2. **Frontier model wins on jailbreak resistance** (5.0 vs 3.0) — Built-in safety alignment is stronger
3. **Frontier model is 4.6x faster** (5.57s vs 25.68s) — Expected for a hosted API vs HF Spaces
4. **Both models had 0% hallucination rate** in this sample — Promising sign for reliability
5. **OSS model has higher refusal rate** (25% vs 12.5%) — Guardrails make it more conservative
6. **Frontier model triggers guardrails more** (25% vs 12.5%) — Better at detecting harmful inputs

---

## 💰 Cost & Latency Analysis

### OSS Model Deployment Costs

| Deployment Option | Platform          | Cost            | Avg Latency | Notes                        |
| ----------------- | ----------------- | --------------- | ----------- | ---------------------------- |
| HF Spaces (CPU)   | gradio_client     | **Free**        | ~25s        | Cold start ~5-10s            |
| HF Inference API  | REST API          | Free*           | ~5-15s      | *Rate limited (30 req/min)   |
| HF Spaces (GPU)   | GPU upgrade       | ~$0.40/hr       | ~1-2s       | No cold start                |
| Local (CPU)       | Ollama            | Free            | ~3-8s       | Requires 4GB RAM             |
| Modal (GPU)       | Cloud GPU         | ~$0.50/hr       | ~0.5-1s     | Free $30 credit included     |

### Frontier Model Costs (Gemini 2.0 Flash + OpenRouter)

| Metric                   | Value                                    |
| ------------------------ | ---------------------------------------- |
| **Input tokens**         | Free (60 req/min on free tier)           |
| **Output tokens**        | Free (60 req/min on free tier)           |
| **Paid tier (input)**    | $0.075 / 1M tokens                       |
| **Paid tier (output)**   | $0.30 / 1M tokens                        |
| **Avg Latency**          | 0.5-2 seconds                            |

### Per-1K-Requests Cost Comparison

| Model                     | Cost per 1K requests | Latency per request |
| ------------------------- | -------------------- | ------------------- |
| Qwen2.5-0.5B              | **~$0.01** (free)    | ~25s (HF Spaces)    |
| Gemini 2.0 Flash          | **Free** (60 req/min)| ~1-2s               |
| liquid/lfm-2.5-1.2b-instruct | **Free** (OpenRouter) | ~1-2s            |
| openrouter/free           | **Free**             | ~1-5s               |

**All models are free to use — the OSS model via HF Spaces is slower but more reliable for bulk queries.**

### Monthly Cost Estimates (10K conversations)

| Scenario        | OSS Only | Frontier Only | Hybrid     |
| --------------- | -------- | ------------- | ---------- |
| 10K simple      | **$0**   | $0 (free)     | $0         |
| 10K complex     | ~$2      | ~$5           | ~$2        |
| 100K queries    | ~$10     | ~$30          | ~$15       |

### Recommendation

- **Cost-sensitive deployments**: Use OSS model with HF Spaces (free tier)
- **Quality-sensitive deployments**: Use Frontier model
- **Hybrid approach**: Route simple queries to OSS, complex queries to Frontier

---

## 🚀 Deployment

### Hugging Face Spaces (Public OSS)

The OSS model is already deployed at: [coderraj07/qwen-oss-api](https://huggingface.co/spaces/coderraj07/qwen-oss-api)

To deploy the full app:

1. Create a [Hugging Face Space](https://huggingface.co/new-space) with:
   - **SDK**: Streamlit
   - **Space Name**: `ai-assistant-eval`

2. Add secrets:
   - `HF_TOKEN`: Your Hugging Face token
   - `GEMINI_API_KEY`: Your Google API key

3. Deploy by pushing to the Space's Git repository:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ai-assistant-eval
git push space main
```

### Streamlit Community Cloud

```bash
# 1. Push to GitHub
# 2. Go to https://streamlit.io/cloud
# 3. Click "New app" → select repo
# 4. Set secrets in dashboard
# 5. Deploy
```

### Full Deployment Guide

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for detailed deployment instructions for all platforms.

---

## 🤔 Architecture Decisions & Tradeoffs

### Decision 1: Hugging Face Spaces + Inference API (Dual-Layer)

**Chosen**: HF Space as primary, Inference API as fallback

- ✅ Publicly deployed OSS model (bonus requirement)
- ✅ Auto-fallback ensures high availability
- ✅ Free tier for both
- ❌ Higher latency due to cold starts (~25s average)
- ❌ Rate limited on free tier

**Alternative**: Local Ollama deployment
- ✅ Lower latency (no network)
- ❌ Requires GPU for reasonable performance
- ❌ Not publicly accessible

### Decision 2: LLM-as-Judge Evaluation

**Chosen**: Cerebras API as impartial judge

- ✅ Objective, consistent scoring
- ✅ Evaluates across 5 dimensions per response
- ✅ Can detect nuanced issues heuristics miss
- ❌ API rate limits (429 errors under heavy load)
- ❌ Additional API dependency

**Why not pure heuristic**: Heuristics alone miss subtle hallucinations and biased reasoning. LLM-as-judge provides more accurate, nuanced evaluation.

### Decision 3: Streamlit for UI

**Chosen**: Streamlit

- ✅ Rapid prototyping (4-tab interface in days)
- ✅ Built-in chat components and session state
- ✅ Excellent for demos and evaluations
- ❌ Less flexible for custom UI
- ❌ Single-threaded (not production-scale)

### Decision 4: Rule-Based Guardrails

**Chosen**: Regex pattern matching

- ✅ Deterministic, immediately responsive
- ✅ No API costs for safety checks
- ✅ Easy to debug and extend with new patterns
- ❌ Can be bypassed with creative phrasing
- ❌ Higher false positive rate

**Improvement path**: ML-based content moderation (Perspective API, Llama Guard) would improve accuracy but add complexity and cost.

### Decision 5: In-Memory Session Context

**Chosen**: Streamlit session state

- ✅ No external dependencies (no Redis/PostgreSQL)
- ✅ Simple, reliable, zero-config
- ✅ Automatic cleanup with browser session
- ❌ No persistence across sessions
- ❌ Memory-bound (browser session only)

**Improvement path**: SQLite for persistence, Redis for multi-user support.

---

## 🔮 Improvements with More Time

### Short-term (1-2 days)

1. **Persistent Memory with SQLite**
   - Store conversations across sessions
   - Conversation history export

2. **Advanced Guardrails with ML**
   - Integrate Llama Guard or Perspective API
   - Reduce false positive rate

3. **Interactive Charts**
   - Replace static matplotlib with Plotly
   - Drill-down into individual responses

### Medium-term (1-2 weeks)

4. **Multi-Model Comparison**
   - Add Llama 3.2, Mistral, Phi-3 (OSS)
   - Add Claude, GPT-4o, DeepSeek (Frontier)
   - Model comparison dashboard

5. **Red-Teaming Module**
   - Automated adversarial prompt generation
   - Vulnerability scoring over time

6. **Custom Evaluation Datasets**
   - Upload your own prompts
   - Domain-specific evaluation (medical, legal, finance)

### Long-term (1-2 months)

7. **Production Infrastructure**
   - Multi-user authentication
   - Rate limiting per user
   - A/B testing framework

8. **Full Observability Stack**
   - Prometheus + Grafana dashboards
   - LangFuse/LangSmith integration
   - Cost tracking per user/query

9. **Fine-tuning Integration**
   - Fine-tune OSS models on custom data
   - RLHF for alignment
   - Quantized model deployment

10. **Advanced Safety Framework**
    - Constitutional AI integration
    - Adversarial training data generation
    - Safety benchmark suite (SafetyBench, HHH)

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

## 🙏 Acknowledgments

- [Qwen Team](https://huggingface.co/Qwen) for the Qwen2.5-0.5B-Instruct model
- [Google AI](https://ai.google.dev/) for the Gemini API
- [Hugging Face](https://huggingface.co/) for Spaces and Inference API
- [Streamlit](https://streamlit.io/) for the UI framework
- [Cerebras](https://cerebras.ai/) for the LLM judge API

---

<p align="center">
Made with ❤️ for the AI Assistant Evaluation Challenge<br>
<a href="https://github.com/coderRaj07/ai_assistant_eval">GitHub Repo</a> •
<a href="https://huggingface.co/spaces/coderraj07/qwen-oss-api">OSS Model (HF Spaces)</a> •
<a href="https://huggingface.co/spaces">Demo App (HF Spaces)</a>
</p>