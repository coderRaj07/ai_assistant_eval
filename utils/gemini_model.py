"""
Frontier Assistant - Gemini 1.5 Flash via Google Gen AI API.

Uses proper conversation formatting with system instruction.
Includes guardrails integration and latency tracking.
"""

import os
import time
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types

from utils.guardrails import check_input_safety, check_output_safety

# Configure the model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
}

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_MEDIUM_AND_ABOVE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_MEDIUM_AND_ABOVE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_MEDIUM_AND_ABOVE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_MEDIUM_AND_ABOVE",
    ),
]

SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant. You provide accurate, well-reasoned responses to user queries. When you don't know something, you say so. You avoid making harmful, biased, or discriminatory statements. You refuse to assist with illegal activities, violence, or harmful actions."""


def build_conversation(
    messages: List[Dict[str, str]],
) -> List[types.Content]:
    """Build a Gemini-compatible conversation history."""
    gemini_parts = []

    # Add system prompt as first user message (Gemini API convention)
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        gemini_parts.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=SYSTEM_PROMPT)],
            )
        )
        gemini_parts.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text="Understood. I will follow these guidelines.")],
            )
        )

    for msg in messages:
        if msg["role"] == "system":
            continue

        role = "model" if msg["role"] == "assistant" else "user"
        gemini_parts.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])],
            )
        )

    return gemini_parts


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
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=conversation,
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024,
                safety_settings=safety_settings,
            ),
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
            result["response"] = (
                "⚠️ [Blocked by safety filters] The model determined this "
                "conversation could violate safety guidelines."
            )
            result["safety_category"] = "gemini_safety_block"
            result["guardrail_triggered"] = True
        else:
            result["error"] = error_str
            result["response"] = f"⚠️ Error: {error_str[:150]}"

    return result