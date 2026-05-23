"""
Open Source Assistant - Qwen2.5-0.5B-Instruct via Hugging Face Inference API.

This is the PURE OSS assistant. It ONLY calls Qwen2.5-0.5B-Instruct.
NO fallback to OpenRouter or any other model.

MODEL: Qwen/Qwen2.5-0.5B-Instruct (500M params, Apache 2.0 license)
PLATFORM: Hugging Face Inference API
"""

import os
import time
import requests
from typing import List, Dict, Any

from utils.guardrails import check_input_safety, check_output_safety

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-0.5B-Instruct"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """Format messages using Qwen2.5 instruction template."""
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def generate_response(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """
    Generate a response from Qwen2.5-0.5B-Instruct via Hugging Face Inference API.
    
    PURE OSS: Only calls Qwen2.5-0.5B-Instruct. No fallback.
    
    Returns:
        Dict with response, latency_s, guardrail_triggered, safety_category, error, model_used
    """
    result = {
        "response": "", "latency_s": 0.0, "guardrail_triggered": False,
        "safety_category": "safe", "error": None,
        "model_used": "Qwen2.5-0.5B-Instruct (HF Inference API)",
    }

    user_messages = [m for m in messages if m["role"] == "user"]
    last_user_input = user_messages[-1]["content"] if user_messages else ""

    # Input guardrail
    is_safe, refusal, category = check_input_safety(last_user_input)
    if not is_safe:
        result["response"] = refusal
        result["guardrail_triggered"] = True
        result["safety_category"] = category
        return result

    # Format prompt for Qwen2.5 chat template
    prompt = format_chat_prompt(messages)
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens, "temperature": temperature,
            "top_p": top_p, "do_sample": True, "return_full_text": False,
        },
    }

    start_time = time.time()
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            raw_response = data[0].get("generated_text", "")
            result["response"] = raw_response.strip()
            result["latency_s"] = round(time.time() - start_time, 2)
        elif isinstance(data, dict) and "error" in data:
            error_msg = data["error"]
            result["error"] = f"hf_api_error: {error_msg[:200]}"
            # Provide helpful message instead of silent fallback
            result["response"] = (
                "⚠️ **Qwen2.5-0.5B OSS Model unavailable via Hugging Face Inference API.**\n\n"
                f"Error: {error_msg[:150]}\n\n"
                "**Suggestions:**\n"
                "1. Check your HF_TOKEN in .env file\n"
                "2. HF Inference API requires a PRO subscription for this model\n"
                "3. Deploy locally: `ollama run qwen2.5:0.5b`\n"
                "4. Or try again later when the API queue is clear"
            )
        else:
            result["error"] = f"unexpected_response: {str(data)[:200]}"
            result["response"] = "⚠️ Unexpected response from Hugging Face API."

    except requests.exceptions.Timeout:
        result["error"] = "timeout"
        result["response"] = "⚠️ Request timed out (60s). Hugging Face API is slow. Please try again."
    except requests.exceptions.RequestException as e:
        result["error"] = f"connection_error: {str(e)[:200]}"
        result["response"] = (
            "⚠️ **Qwen2.5-0.5B OSS Model unreachable.**\n\n"
            "Cannot connect to Hugging Face Inference API.\n\n"
            "**To run the OSS assistant locally:**\n"
            "1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`\n"
            "2. Pull Qwen2.5: `ollama pull qwen2.5:0.5b`\n"
            "3. Run: `ollama run qwen2.5:0.5b`\n"
            "4. Then select 'Frontier Assistant' which uses OpenRouter API"
        )

    # Output guardrail
    if result["response"] and not result["error"]:
        is_output_safe, filtered_response, issue = check_output_safety(
            result["response"], last_user_input
        )
        if not is_output_safe:
            result["response"] = filtered_response
            result["guardrail_triggered"] = True
            result["safety_category"] = issue

    return result