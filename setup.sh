#!/bin/bash
# ============================================
# AI Assistant Evaluation Platform - Setup Script
# ============================================

set -e

echo "============================================"
echo "AI Assistant Evaluation Platform Setup"
echo "============================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  WARNING: No .env file found."
    echo "   Creating .env template..."
    echo "# Hugging Face token (get from https://huggingface.co/settings/tokens)" > .env
    echo "HF_TOKEN=your_token_here" >> .env
    echo "# OpenRouter API key (get from https://openrouter.ai/keys)" >> .env
    echo "OPENROUTER_API_KEY=your_key_here" >> .env
    echo "# Google Gemini API key (optional fallback, get from https://aistudio.google.com/apikey)" >> .env
    echo "GEMINI_API_KEY=your_key_here" >> .env
    echo "   Please edit .env with your actual API keys."
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Create evaluation output directory
echo ""
echo "Creating directories..."
mkdir -p evaluation/output

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Activate environment: source venv/bin/activate"
echo "  3. Run app: streamlit run app.py"
echo "  4. Run evaluation: python evaluation/evaluator.py"
echo ""