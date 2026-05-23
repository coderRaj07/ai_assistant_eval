"""
Model Service - Orchestrates model calls.

Provides a unified interface for interacting with both OSS (Qwen2.5) and
Frontier (OpenRouter) models. Handles guardrail integration, latency
tracking, and standardized response formatting.
"""

from typing import Dict, List, Optional, Any

from app.models.conversation import ModelResponse, SafetyCheckResult
from app.services.guardrail_service import GuardrailService
from app.config import ModelConfig


class ModelService:
    """
    Unified service for generating responses from AI models.

    Provides:
    - Guardrail checks before/after model invocation
    - Standardized ModelResponse objects
    - Latency tracking
    - Error handling
    """

    def __init__(self):
        self.guardrail_service = GuardrailService()

    def generate(
        self,
        messages: List[Dict[str, str]],
        model_type: str,
    ) -> ModelResponse:
        """
        Generate a response from the specified model with guardrails.

        Args:
            messages: List of message dicts (role, content)
            model_type: 'oss' or 'frontier'

        Returns:
            ModelResponse with response text, latency, guardrail info, errors
        """
        from utils.hf_model import generate_response as hf_generate
        from utils.openrouter_model import generate_response as or_generate

        result = ModelResponse()

        # Get the last user message for safety check
        user_messages = [m for m in messages if m.get("role") == "user"]
        last_user_input = user_messages[-1]["content"] if user_messages else ""

        # --- Input Guardrail Check ---
        safety_check = self.guardrail_service.check_input(last_user_input)
        if not safety_check.is_safe:
            result.response = safety_check.refusal or ""
            result.guardrail_triggered = True
            result.safety_category = safety_check.category
            return result

        if model_type == "oss":
            response_dict = hf_generate(messages)
        else:
            response_dict = or_generate(messages)

        # Map legacy dict response to ModelResponse
        result.response = response_dict.get("response", "")
        result.latency_s = response_dict.get("latency_s", 0.0)
        result.guardrail_triggered = response_dict.get("guardrail_triggered", False)
        result.safety_category = response_dict.get("safety_category", "safe")
        result.error = response_dict.get("error")

        # --- Output Guardrail Check ---
        if result.response:
            output_check = self.guardrail_service.check_output(
                result.response, last_user_input
            )
            if not output_check.is_safe:
                result.response = output_check.refusal or result.response
                result.guardrail_triggered = True
                result.safety_category = output_check.category

        return result

    @staticmethod
    def get_model_name(model_type: str) -> str:
        names = {
            "oss": "Qwen2.5-0.5B (OSS via HF API)",
            "frontier": "OpenRouter (Frontier)",
        }
        return names.get(model_type, model_type)

    @staticmethod
    def get_model_label(model_type: str) -> str:
        labels = {
            "oss": ModelConfig.OSS_MODEL_ID,
            "frontier": " | ".join(ModelConfig.FRONTIER_MODEL_CHAIN),
        }
        return labels.get(model_type, model_type)
