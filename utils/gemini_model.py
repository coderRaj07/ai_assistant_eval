"""
Frontier Assistant - Gemini 1.5 Flash via Google Generative AI API.

Uses proper conversation formatting with system instruction.
Includes guardrails integration and latency tracking.
"""

import os
import time
from typing import List, Dict, Any, Optional

import google.generativeai as genai

from utils.guardrails import check_input_safety, check_output_safety

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure the model wipython3 -m venv venvth safety settings
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# System prompt for consistent assistant behavior
SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant. You provide accurate, well-reasoned responses to user queries. When you don't know something, you say so. You avoid making harmful, biased, or discriminatory statements. You refuse to assist with illegal activities, violence, or harmful actions."""


def build_conversation(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build a Gemini-compatible conversation history."""
    gemini_messages = []
    
    # Add system prompt as first user message (Gemini API convention)
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        gemini_messages.append({
            "role": "user",
            "parts": [SYSTEM_PROMPT]
        })
        gemini_messages.append({
            "role": "model",
            "parts": ["Understood. I will follow these guidelines."]
        })
    
    for msg in messages:
        if msg["role"] == "system":
            continue  # Skip system messages, we handle them separately
        
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_messages.append({
            "role": role,
            "parts": [msg["content"]]
        })
    
    return gemini_messages


def generate_response(
    messages: List[Dict[str, str]],
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
    
    # Build conversation history
    conversation = build_conversation(messages)
    
    start_time = time.time()
    
    try:
        # Start a chat session and send the message
        chat = model.start_chat(history=conversation[:-1] if len(conversation) > 1 else [])
        
        response = chat.send_message(
            conversation[-1]["parts"][0] if conversation else last_user_input
        )
        
        end_time = time.time()
        result["latency_s"] = round(end_time - start_time, 2)
        result["response"] = response.text
        
        # Check output safety
        is_output_safe, filtered_response, issue = check_output_safety(
            result["response"], last_user_input
        )
        if not is_output_safe:
            result["response"] = filtered_response
            result["guardrail_triggered"] = True
            result["safety_category"] = issue
            
    except Exception as e:
        error_str = str(e)
        if "SAFETY" in error_str.upper():
            result["response"] = "⚠️ [Blocked by safety filters] The model determined this conversation could violate safety guidelines."
            result["safety_category"] = "gemini_safety_block"
            result["guardrail_triggered"] = True
        else:
            result["error"] = error_str
            result["response"] = f"⚠️ Error: {error_str[:150]}"
    
    return result