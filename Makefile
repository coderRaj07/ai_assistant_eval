.PHONY: install run test evaluate report report-pdf clean deploy help

# ============================================
# AI Assistant Evaluation Platform - Makefile
# ============================================

# Colors for output
GREEN := \033[0;32m
BLUE := \033[0;34m
NC := \033[0m # No Color

help:
	@echo "Usage:"
	@echo "  make install     - Install dependencies"
	@echo "  make run         - Run the Streamlit app"
	@echo "  make evaluate    - Run full evaluation suite (185 prompts)"
	@echo "  make test        - Quick API test for both models"
	@echo "  make report      - Generate evaluation PDF report"
	@echo "  make clean       - Clean cache and temporary files"
	@echo "  make deploy      - Deploy to Hugging Face Spaces"
	@echo ""
	@echo "Quick start: make install && make test"

install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

run:
	@echo "$(BLUE)Starting Streamlit app...$(NC)"
	streamlit run app.py

evaluate:
	@echo "$(BLUE)Running full evaluation (185 prompts across 18 categories)...$(NC)"
	@echo "$(BLUE)This may take 10-20 minutes depending on API speeds...$(NC)"
	time python3 evaluation/evaluator.py
	@echo "$(GREEN)✓ Evaluation complete$(NC)"

test-quick:
	@echo "$(BLUE)Running quick API connectivity test...$(NC)"
	python3 -c "
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')
from utils.openrouter_model import generate_response as or_gen
print('1/2 Testing Frontier (OpenRouter)...', end=' ')
r = or_gen([{'role': 'user', 'content': 'What is 2+2? Answer briefly.'}])
print(f'{r[\"model_used\"]} | {r[\"latency_s\"]}s | {r[\"response\"][:80]}')
print('2/2 Testing OSS (Qwen2.5)...', end=' ')
from utils.hf_model import generate_response as hf_gen
r = hf_gen([{'role': 'user', 'content': 'What is 2+2? Answer briefly.'}])
print(f'{r[\"latency_s\"]}s | {r[\"response\"][:80]}')
print('$(GREEN)✓ Quick test passed$(NC)')
	"

test: test-quick

report:
	@echo "$(BLUE)Generating evaluation PDF report...$(NC)"
	python3 evaluation/generate_report.py
	@echo "$(GREEN)✓ Report generated$(NC)"

report-pdf:
	@echo "$(BLUE)Generating PDF report from latest evaluation results...$(NC)"
	python3 evaluation/generate_report.py
	@echo "$(GREEN)✓ PDF report saved to evaluation/output/$(NC)"

clean:
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	rm -rf .pytest_cache build/ dist/ *.egg-info/
	@echo "$(GREEN)✓ Clean complete$(NC)"

deploy:
	@echo "$(BLUE)Preparing for Hugging Face Spaces deployment...$(NC)"
	@echo ""
	@echo "=== STEPS ==="
	@echo "1. Go to https://huggingface.co/new-space"
	@echo "2. Space Name: ai-assistant-eval"
	@echo "3. SDK: Streamlit"
	@echo "4. Hardware: CPU (free)"
	@echo "5. Add secrets (Settings > Repository Secrets):"
	@echo "   OPENROUTER_API_KEY = your_key"
	@echo "   HF_TOKEN = your_token"
	@echo "   GEMINI_API_KEY = your_key"
	@echo "6. Deploy:"
	@echo "   git remote add space https://huggingface.co/spaces/YOUR_USER/ai-assistant-eval"
	@echo "   git push space main"
	@echo ""
	@echo "== OR use Streamlit Cloud: https://streamlit.io/cloud =="
