"""
Frontier Assistant - OpenRouter API with automatic free model fallback.

Uses OpenRouter as the gateway to frontier/advanced models. Automatically
falls back through a chain of free models when quota/rate limits are hit.
As a last resort, falls back to direct Gemini API call.

MODEL CHAIN (Frontier):
  1. liquid/lfm-2.5-1.2b-instruct:free — fast small model (OpenRouter)
  2. openai/gpt-4o-mini:free — frontier-class (OpenRouter)
  3. nvidia/nemotron-nano-9b-v2:free — larger OSS (OpenRouter)
  4. Gemini 2.0 Flash — direct API (last resort)

NOTE: These models are DIFFERENT from the OSS fallback chain (Llama 3.2, Phi-3, Mistral).
"""

import os
import time
import json
import requests
from typing import List, Dict, Any, Optional, Tuple

from utils.guardrails import check_input_safety, check_output_safety

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Frontier model chain — DIFFERENT from OSS fallback chain
# These are frontier/larger models accessed through OpenRouter free tier
FREE_MODEL_CHAIN = [
    "liquid/lfm-2.5-1.2b-instruct:free",
    "openai/gpt-4o-mini:free",
    "nvidia/nemotron-nano-9b-v2:free",
]

SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant. You provide accurate, well-reasoned responses to user queries. When you don't know something, you say so. You avoid making harmful, biased, or discriminatory statements. You refuse to assist with illegal activities, violence, or harmful actions."""


def build_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build OpenRouter-compatible messages with system prompt."""
    has_system = any(m.get("role") == "system" for m in messages)
    result = []
    if not has_system:
        result.append({"role": "system", "content": SYSTEM_PROMPT})
    for msg in messages:
        if msg.get("role") == "system" and has_system:
            continue
        result.append({"role": msg["role"], "content": msg["content"]})
    return result


def extract_text(data) -> Optional[str]:
    """Extract response text from various OpenRouter response formats."""
    if isinstance(data, dict):
        choices = data.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            msg = choices[0].get("message", {})
            if isinstance(msg, dict):
                return msg.get("content")
    return None


def call_openrouter(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> Tuple[Optional[str], Optional[str], float]:
    """Call OpenRouter API with a specific model."""
    if not OPENROUTER_API_KEY:
        return None, "No OPENROUTER_API_KEY", 0.0

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/coderraj07/ai_assistant_eval",
        "X-Title": "AI Assistant Evaluation (Frontier)",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start_time = time.time()
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        latency = round(time.time() - start_time, 2)
        data = response.json()

        if response.status_code == 200:
            text = extract_text(data)
            if text:
                return text, None, latency

        error_msg = ""
        if isinstance(data, dict):
            error_msg = data.get("error", {}).get("message", str(data))
        else:
            error_msg = str(data)

        if response.status_code == 429:
            return None, f"rate_limited:{model}", latency
        elif response.status_code == 402:
            return None, f"quota_exceeded:{model}", latency
        elif response.status_code == 404:
            return None, f"not_found:{model}", latency
        else:
            return None, f"error_{response.status_code}:{error_msg[:100]}", latency

    except requests.exceptions.Timeout:
        return None, f"timeout:{model}", round(time.time() - start_time, 2)
    except requests.exceptions.RequestException as e:
        return None, f"connection_error:{str(e)[:100]}", round(time.time() - start_time, 2)
    except Exception as e:
        return None, f"unexpected:{str(e)[:100]}", round(time.time() - start_time, 2)


def call_gemini_direct(
    messages: List[Dict[str, str]],
) -> Tuple[Optional[str], Optional[str], float]:
    """Call Gemini 2.0 Flash directly as last resort fallback."""
    if not GEMINI_API_KEY:
        return None, "No GEMINI_API_KEY", 0.0
    start = time.time()
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        or_messages = build_messages(messages)
        gemini_contents = []
        for m in or_messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            gemini_contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=m["content"])])
            )
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                max_output_tokens=1024, temperature=0.7,
            ),
        )
        latency = round(time.time() - start, 2)
        return response.text, None, latency
    except Exception as e:
        return None, f"gemini_error:{str(e)[:100]}", round(time.time() - start, 2)


def generate_response(
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Generate a response via Frontier model chain.
    
    PRIORITY (Frontier chain):
      1. liquid/lfm-2.5-1.2b-instruct:free (OpenRouter)
      2. openai/gpt-4o-mini:free (OpenRouter) 
      3. nvidia/nemotron-nano-9b-v2:free (OpenRouter)
      4. Gemini 2.0 Flash (direct API, last resort)
    
    NOTE: These models are DIFFERENT from the OSS chain (Llama 3.2, Phi-3, Mistral).
    The "model_used" field always indicates exactly which model responded.
    
    Returns:
        Dict with keys: response, latency_s, guardrail_triggered, 
                        safety_category, error, model_used
    """
    result = {
        "response": "",
        "latency_s": 0.0,
        "guardrail_triggered": False,
        "safety_category": "safe",
        "error": None,
        "model_used": None,
    }

    # Input guardrail check (duplicated from model_service.py for safety)
    user_messages = [m for m in messages if m["role"] == "user"]
    last_user_input = user_messages[-1]["content"] if user_messages else ""
    is_safe, refusal, category = check_input_safety(last_user_input)
    if not is_safe:
        result["response"] = refusal
        result["guardrail_triggered"] = True
        result["safety_category"] = category
        return result

    openrouter_messages = build_messages(messages)
    total_start = time.time()
    last_error = None

    # Try each model in the frontier chain
    for model in FREE_MODEL_CHAIN:
        text, error, latency = call_openrouter(openrouter_messages, model)

        if text is not None:
            result["response"] = text.strip()
            result["latency_s"] = round(time.time() - total_start, 2)
            result["model_used"] = model
            is_output_safe, filtered_response, issue = check_output_safety(
                result["response"], last_user_input
            )
            if not is_output_safe:
                result["response"] = filtered_response
                result["guardrail_triggered"] = True
                result["safety_category"] = issue
            return result

        last_error = error
        is_retryable = error and any(x in error for x in ["rate_limited", "quota_exceeded", "not_found"])
        if is_retryable:
            time.sleep(0.5)
            continue
        if error and any(x in error for x in ["auth_error", "connection_error", "unexpected"]):
            break

    # Last resort: direct Gemini API
    text, error, latency = call_gemini_direct(messages)
    if text is not None:
        result["response"] = text.strip()
        result["latency_s"] = round(time.time() - total_start, 2)
        result["model_used"] = "gemini-2.0-flash (direct)"
        is_output_safe, filtered_response, issue = check_output_safety(
            result["response"], last_user_input
        )
        if not is_output_safe:
            result["response"] = filtered_response
            result["guardrail_triggered"] = True
            result["safety_category"] = issue
        return result

    result["error"] = last_error or "all models failed"
    result["model_used"] = "none"
    result["response"] = "⚠️ All AI models are currently rate-limited. Please try again later."
    return result