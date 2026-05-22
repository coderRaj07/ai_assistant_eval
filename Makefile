.PHONY: install run test evaluate clean deploy help

# ============================================
# AI Assistant Evaluation Platform - Makefile
# ============================================

# Colors for output
GREEN := \033[0;32m
BLUE := \033[0;34m
NC := \033[0m # No Color

help:
	@echo "Usage:"
	@echo "  make install    - Install dependencies"
	@echo "  make run        - Run the Streamlit app"
	@echo "  make evaluate   - Run full evaluation suite"
	@echo "  make test       - Run quick test"
	@echo "  make clean      - Clean cache and temporary files"
	@echo "  make deploy     - Deploy to Hugging Face Spaces"

install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

run:
	@echo "$(BLUE)Starting Streamlit app...$(NC)"
	streamlit run app.py

evaluate:
	@echo "$(BLUE)Running evaluation suite...$(NC)"
	python evaluation/evaluator.py
	@echo "$(GREEN)✓ Evaluation complete$(NC)"

test:
	@echo "$(BLUE)Running quick test...$(NC)"
	python -c "
from utils.hf_model import generate_response as hf
from utils.gemini_model import generate_response as gemini
print('Testing OSS model...')
result = hf([{'role': 'user', 'content': 'What is 2+2?'}])
print(f'OSS: {result[\"response\"][:100]}... ({result[\"latency_s\"]}s)')
print('Testing Frontier model...')
result = gemini([{'role': 'user', 'content': 'What is 2+2?'}])
print(f'Frontier: {result[\"response\"][:100]}... ({result[\"latency_s\"]}s)')
print('$(GREEN)✓ Quick test passed$(NC)')
"

clean:
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	rm -rf .pytest_cache
	rm -rf build/ dist/ *.egg-info/
	@echo "$(GREEN)✓ Clean complete$(NC)"

deploy:
	@echo "$(BLUE)Preparing for Hugging Face Spaces deployment...$(NC)"
	@echo "Make sure you have:"
	@echo "  1. Created a Space at https://huggingface.co/new-space"
	@echo "  2. Added HF_TOKEN and GEMINI_API_KEY as secrets"
	@echo ""
	@echo "To deploy:"
	@echo "  git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ai-assistant-eval"
	@echo "  git push space main"