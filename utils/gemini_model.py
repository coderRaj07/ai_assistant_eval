"""
Frontier Assistant - Gemini 2.0 Flash via Google Generative AI API.
Delegates to openrouter_model which tries Gemini first, then
falls back to OpenRouter frontier models (GPT-4o-mini, DeepSeek, etc.).
"""

from utils.openrouter_model import generate_response as generate_response
