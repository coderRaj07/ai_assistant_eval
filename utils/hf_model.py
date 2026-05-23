"""
Open Source Assistant - Qwen2.5-0.5B-Instruct via your deployed HF Space (gradio_client).
With Hugging Face Inference API fallback when Space is unavailable.

FEATURES:
  - Calls your deployed Space: coderraj07/qwen-oss-api via gradio_client
  - Falls back to HF Inference API if Space fails
  - Guardrails (input/output safety checks)
  - Latency tracking
  - Multi-turn conversation support (Qwen2.5 chat template)
  - Proper error messages
"""

import os
import time
import requests
from typing import List, Dict, Any

from utils.guardrails import check_input_safety, check_output_safety

HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-0.5B-Instruct"

# Your deployed HF Space
SPACE_URL = "coderraj07/qwen-oss-api"

# gradio_client for your Space
try:
    from gradio_client import Client as GradioClient
    client = GradioClient(SPACE_URL)
    HAS_GRADIO = True
except Exception as e:
    client = None
    HAS_GRADIO = False
    print(f"[HF Model] gradio_client unavailable: {e}")


def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """Format messages using Qwen2.5 instruction template."""
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def call_your_space(message: str) -> Dict[str, Any]:
    """
    Call your deployed HF Space via gradio_client.
    
    Args:
        message: The user's input text
    
    Returns:
        Dict with response, latency_s, error
    """
    result = {"response": "", "latency_s": 0.0, "error": None}
    
    if not HAS_GRADIO or client is None:
        result["error"] = "gradio_client not available"
        return result
    
    start_time = time.time()
    try:
        response = client.predict(
            message,
            api_name="/generate"
        )
        result["latency_s"] = round(time.time() - start_time, 2)
        
        if response and str(response).strip():
            result["response"] = str(response).strip()
        else:
            result["error"] = "empty_response"
            
    except Exception as e:
        result["error"] = f"space_error: {str(e)[:200]}"
        result["latency_s"] = round(time.time() - start_time, 2)
    
    return result


def call_hf_inference_api(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """
    Fallback: Call Hugging Face Inference API directly.
    """
    result = {"response": "", "latency_s": 0.0, "error": None}
    prompt = format_chat_prompt(messages)
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens, "temperature": temperature,
            "top_p": top_p, "do_sample": True, "return_full_text": False,
        },
    }
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    start_time = time.time()
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            raw_response = data[0].get("generated_text", "")
            result["response"] = raw_response.strip()
            result["latency_s"] = round(time.time() - start_time, 2)
        else:
            result["error"] = f"hf_api_error: {str(data)[:200]}"
            
    except requests.exceptions.RequestException as e:
        result["error"] = f"hf_connection_error: {str(e)[:200]}"
    
    return result


def generate_response(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """
    Generate a response from Qwen2.5-0.5B-Instruct.
    
    PRIORITY:
      1. Your deployed HF Space (gradio_client)
      2. Hugging Face Inference API (fallback)
    
    Features:
      - Guardrails (input/output)
      - Multi-turn context via Qwen2.5 chat template
      - Latency tracking
      - Clear error messages
    
    Returns:
        Dict with response, latency_s, guardrail_triggered, safety_category, error, model_used
    """
    result = {
        "response": "", "latency_s": 0.0, "guardrail_triggered": False,
        "safety_category": "safe", "error": None,
        "model_used": "Qwen2.5-0.5B-Instruct (HF Space)",
    }

    # Get last user message for guardrail checks
    user_messages = [m for m in messages if m["role"] == "user"]
    last_user_input = user_messages[-1]["content"] if user_messages else ""

    # ── Input Guardrail Check ──
    is_safe, refusal, category = check_input_safety(last_user_input)
    if not is_safe:
        result["response"] = refusal
        result["guardrail_triggered"] = True
        result["safety_category"] = category
        return result

    total_start = time.time()

    # ── Attempt 1: Your deployed HF Space (gradio_client) ──
    space_result = call_your_space(last_user_input)
    if space_result["response"]:
        result["response"] = space_result["response"]
        result["latency_s"] = round(time.time() - total_start, 2)
        result["model_used"] = "Qwen2.5-0.5B-Instruct (HF Space)"
        if space_result.get("error"):
            result["error"] = space_result["error"]
    else:
        # ── Attempt 2: HF Inference API fallback ──
        hf_result = call_hf_inference_api(messages, max_new_tokens, temperature, top_p)
        if hf_result["response"]:
            result["response"] = hf_result["response"]
            result["latency_s"] = round(time.time() - total_start, 2)
            result["model_used"] = "Qwen2.5-0.5B-Instruct (HF API)"
            if hf_result.get("error"):
                result["error"] = hf_result["error"]
        else:
            # Both failed — give a helpful error
            space_err = space_result.get("error", "unknown")
            hf_err = hf_result.get("error", "unknown")
            result["error"] = f"space: {space_err}, hf: {hf_err}"
            result["model_used"] = "none"
            result["response"] = (
                "⚠️ **Qwen2.5-0.5B OSS Model unavailable.**\n\n"
                "Neither your deployed HF Space nor the HF Inference API responded.\n\n"
                "**Suggestions:**\n"
                "1. Wake your Space: https://huggingface.co/spaces/coderraj07/qwen-oss-api\n"
                "2. Check HF_TOKEN in .env\n"
                "3. Use Frontier Assistant (OpenRouter) instead"
            )

    # ── Output Guardrail Check ──
    if result["response"] and not result.get("guardrail_triggered", False):
        is_output_safe, filtered_response, issue = check_output_safety(
            result["response"], last_user_input
        )
        if not is_output_safe:
            result["response"] = filtered_response
            result["guardrail_triggered"] = True
            result["safety_category"] = issue

    return result