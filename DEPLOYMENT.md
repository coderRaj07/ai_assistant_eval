# Deployment Guide & Cost Analysis

## 🚀 Quick Deployment Options

### Option 1: Hugging Face Spaces (FREE - Recommended)
The easiest way to deploy publicly at zero cost.

**Steps:**
1. Go to https://huggingface.co/new-space
2. Set:
   - **Space Name**: `ai-assistant-eval`
   - **SDK**: Streamlit
   - **Hardware**: CPU Free (enough for this app)
3. Add these **Secrets** (Settings → Repository Secrets):
   ```
   OPENROUTER_API_KEY = sk-or-v1-...
   HF_TOKEN = hf_...
   GEMINI_API_KEY = ...
   ```
4. Deploy via Git:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ai-assistant-eval
   git push space main
   ```

**Cost**: \$0/month (free tier: ~1000 requests/day on CPU)

### Option 2: Streamlit Community Cloud (FREE)
https://streamlit.io/cloud

1. Push to GitHub
2. Login at https://streamlit.io/cloud
3. Click "New app" → select your repo
4. Set secrets in Streamlit Cloud dashboard
5. Deploy

**Cost**: \$0/month (free tier: 1 app, public only)

### Option 3: Modal (for GPU)
Deploy the OSS model with GPU for lower latency.

```python
# modal_deploy.py
import modal
app = modal.App("ai-assistant-eval")
```

**Cost**: \$0/month (free tier: \$30 credit) → GPU: ~\$0.50/hr (A10G)

---

## 💰 Cost & Latency Analysis

### OpenRouter Free Models (Frontier Assistant)

| Model | Type | Latency | Cost | Reliability |
|-------|------|---------|------|-------------|
| liquid/lfm-2.5-1.2b-instruct:free | OSS (1.2B) | ~1-2s | Free | High |
| openai/gpt-4o-mini:free | Frontier | ~0.5-1s | Free | High |
| meta-llama/llama-3.2-3b-instruct:free | OSS (3B) | ~1-2s | Free | High |
| nvidia/nemotron-nano-9b-v2:free | OSS (9B) | ~2-3s | Free | Medium |

### HF Inference API (OSS Assistant)

| Model | Platform | Latency | Cost | Notes |
|-------|----------|---------|------|-------|
| Qwen2.5-0.5B | HF API | 2-8s | Free* | *Rate limited without PRO |
| Liquid LFM 1.2B | OpenRouter | 1-2s | Free | Primary model |

### Direct API Costs (if quota exhausted)

| Provider | Model | Input/1M tokens | Output/1M tokens |
|----------|-------|-----------------|------------------|
| Google Gemini | Gemini 2.0 Flash | Free (60 req/min) | Free |
| OpenRouter | GPT-4o-mini | $0.15 | $0.60 |
| OpenRouter | Liquid LFM 1.2B | Free | Free |

### Monthly Cost Estimates (10K conversations)

| Scenario | OSS Only | Frontier Only | Hybrid |
|----------|----------|---------------|--------|
| 10K simple queries | $0 | $0 (free tier) | $0 |
| 10K complex queries | ~$2 (HF PRO) | ~$5 | ~$2 |
| 100K queries | ~$10 | ~$30 | ~$15 |

---

## 🧪 How to Test Locally

```bash
# 1. Run the Streamlit app
cd ai_assistant_eval
streamlit run app.py

# 2. Quick API test
python3 -c "
from dotenv import load_dotenv
load_dotenv('.env')
from utils.openrouter_model import generate_response
r = generate_response([{'role':'user','content':'Hello!'}])
print(f'Response: {r[\"response\"][:100]}')
print(f'Model: {r[\"model_used\"]}')
print(f'Latency: {r[\"latency_s\"]}s')
"

# 3. Run evaluation (185 prompts, takes ~10-20 min)
python3 evaluation/evaluator.py

# 4. Generate PDF report
python3 evaluation/generate_report.py
```

---

## 📁 Deliverables Checklist

| Deliverable | Status | Notes |
|-------------|--------|-------|
| ✅ GitHub Repo | Complete | https://github.com/coderRaj07/ai_assistant_eval |
| ✅ README with setup | Complete | See README.md |
| ✅ Evaluation PDF | Ready | `python3 evaluation/generate_report.py` |
| ✅ Cost analysis | Complete | See this file |
| ✅ Architecture decisions | Complete | In README.md |
| ✅ OSS model (Qwen2.5) | Complete | HF API + OpenRouter fallback |
| ✅ Frontier model (OpenRouter) | Complete | Free model chain + Gemini fallback |
| ✅ Multi-turn conversations | Complete | With memory & pruning |
| ✅ Guardrails & safety | Complete | 7 categories + output checks |
| ✅ 185 evaluation prompts | Complete | 18 categories |
| ✅ Hugging Face Spaces | Ready | Deploy with git push |
| 🔄 Demo screenshots | Pending | Run app, take screenshots |