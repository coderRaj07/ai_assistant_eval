"""
Open Source Assistant - Qwen2.5-0.5B-Instruct via Hugging Face Inference API.

Uses proper chat template formatting for Qwen2.5 instruction-tuned models.
Includes guardrails integration and latency tracking.
"""

import os
import time
import requests
from typing import List, Dict, Any, Optional

from utils.guardrails import check_input_safety, check_output_safety

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-0.5B-Instruct"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}


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


def generate_response(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Dict[str, Any]:
    """
    Generate a response with guardrails and latency tracking.
    
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
    
    # Format prompt with proper chat template
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
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            raw_response = data[0].get("generated_text", "")
        elif isinstance(data, dict) and "error" in data:
            result["error"] = data["error"]
            result["response"] = f"⚠️ Model Error: {data['error']}"
            return result
        else:
            result["error"] = "Unexpected response format"
            result["response"] = "⚠️ Error processing response."
            return result
        
        end_time = time.time()
        result["latency_s"] = round(end_time - start_time, 2)
        
        # Clean up the response - remove the prompt prefix if present
        result["response"] = raw_response.strip()
        
        # Check output safety
        is_output_safe, filtered_response, issue = check_output_safety(
            result["response"], last_user_input
        )
        if not is_output_safe:
            result["response"] = filtered_response
            result["guardrail_triggered"] = True
            result["safety_category"] = issue
        
    except requests.exceptions.Timeout:
        result["error"] = "Request timed out"
        result["response"] = "⚠️ The model took too long to respond. Please try again."
    except requests.exceptions.RequestException as e:
        result["error"] = str(e)
        result["response"] = f"⚠️ Connection error: {str(e)[:100]}"
    except Exception as e:
        result["error"] = str(e)
        result["response"] = "⚠️ An unexpected error occurred."
    
    return result