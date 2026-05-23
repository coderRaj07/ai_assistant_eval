"""
Open Source Assistant - Qwen2.5-0.5B-Instruct via Hugging Face Inference API.
With OpenRouter OSS fallback when HF API is unavailable.

Uses proper chat template formatting for Qwen2.5 instruction-tuned models.
Includes guardrails integration, latency tracking, and automatic fallback.
"""

import os
import time
import requests
from typing import List, Dict, Any, Optional, Tuple

from utils.guardrails import check_input_safety, check_output_safety

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-0.5B-Instruct"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# OpenRouter OSS fallback models (free, open-source)
OR_FALLBACK_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OR_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# Qwen2.5 uses a specific chat template format
def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Format messages using Qwen2.5 instruction template.
    
    Qwen2.5 format:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # Add assistant prefix so the model generates a response
    prompt += "<|im_start|>assistant\n"
    return prompt


def call_openrouter_oss(
    messages: List[Dict[str, str]],
    model: str,
) -> Tuple[Optional[str], Optional[str], float]:
    """Call OpenRouter with an OSS model as fallback."""
    if not OPENROUTER_API_KEY:
        return None, "No OPENROUTER_API_KEY", 0.0
    
    or_headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/coderraj07/ai_assistant_eval",
        "X-Title": "AI Assistant Evaluation",
    }
    
    # Build messages for OpenRouter format
    or_messages = []
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        or_messages.append({
            "role": "system",
            "content": "You are a helpful, harmless, and honest AI assistant."
        })
    for m in messages:
        if m.get("role") == "system" and has_system:
            continue
        or_messages.append({"role": m["role"], "content": m["content"]})
    
    payload = {
        "model": model,
        "messages": or_messages,
        "max_tokens": 512,
        "temperature": 0.7,
    }
    
    start_time = time.time()
    try:
        response = requests.post(OR_API_URL, headers=or_headers, json=payload, timeout=30)
        latency = round(time.time() - start_time, 2)
        data = response.json()
        
        if response.status_code == 200:
            choices = data.get("choices", [])
            if choices and len(choices) > 0:
                text = choices[0].get("message", {}).get("content", "")
                if text:
                    return text.strip(), None, latency
        
        return None, f"or_error:{model}", latency
    except Exception as e:
        return None, f"or_exception:{str(e)[:100]}", round(time.time() - start_time, 2)


def generate_response(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Dict[str, Any]:
    """
    Generate a response with guardrails and latency tracking.
    
    Tries Hugging Face Inference API first, falls back to OpenRouter
    OSS models if HF is unavailable.
    
    Returns:
        Dict with keys: response, latency_s, guardrail_triggered, safety_category, error
    """
    result = {
        "response": "",
        "latency_s": 0.0,
        "guardrail_triggered": False,
        "safety_category": "safe",
        "error": None,
    }
    
    # Get the last user message for safety check
    user_messages = [m for m in messages if m["role"] == "user"]
    last_user_input = user_messages[-1]["content"] if user_messages else ""
    
    # Check input safety
    is_safe, refusal, category = check_input_safety(last_user_input)
    if not is_safe:
        result["response"] = refusal
        result["guardrail_triggered"] = True
        result["safety_category"] = category
        return result
    
    # ── Try Hugging Face Inference API first ──
    prompt = format_chat_prompt(messages)
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "return_full_text": False,
        }
    }
    
    start_time = time.time()
    hf_failed = False
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            raw_response = data[0].get("generated_text", "")
            result["response"] = raw_response.strip()
            result["latency_s"] = round(time.time() - start_time, 2)
        elif isinstance(data, dict) and "error" in data:
            hf_failed = True
        else:
            hf_failed = True
            
    except requests.exceptions.RequestException:
        hf_failed = True
    
    # ── Fallback to OpenRouter OSS models ──
    if hf_failed:
        total_start = time.time()
        for model in OR_FALLBACK_MODELS:
            text, error, latency = call_openrouter_oss(messages, model)
            if text is not None:
                result["response"] = text
                result["latency_s"] = round(time.time() - total_start, 2)
                result["error"] = f"hf_fallback:using_{model}"
                break
        
        if not result["response"]:
            result["error"] = "HF API unavailable and no OpenRouter fallback worked"
            result["response"] = "⚠️ OSS model temporarily unavailable. Please try again later."
    
    # ── Check output safety ──
    if result["response"]:
        is_output_safe, filtered_response, issue = check_output_safety(
            result["response"], last_user_input
        )
        if not is_output_safe:
            result["response"] = filtered_response
            result["guardrail_triggered"] = True
            result["safety_category"] = issue
    
    return result