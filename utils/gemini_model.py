"""
Frontier Assistant - Gemini via OpenRouter (primary) with direct Gemini fallback.

Uses OpenRouter to access Gemini 2.0 Flash (free tier). Falls back to
direct Gemini API if OpenRouter is unavailable, then to other free
OpenRouter models.
"""

from utils.openrouter_model import generate_response as or_generate

FREE_MODEL_CHAIN = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]


def generate_response(messages):
    """
    Generate response using Gemini (via OpenRouter free tier).
    Falls back through free model chain on quota exhaustion.

    Delegates to openrouter_model.generate_response which handles
    the full fallback chain, guardrails, and latency tracking.
    """
    return or_generate(messages)
