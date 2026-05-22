"""
Guardrail Service - Business logic for safety checks.

Applies input and output safety checks, logs guardrail events,
and provides refusal responses. This service is the single point
of interaction for all safety-related functionality.
"""

import re
from typing import Dict, List, Optional, Tuple

from app.models.conversation import SafetyCheckResult, GuardrailEvent
from app.config import HARMFUL_PATTERNS, REFUSAL_RESPONSES


class GuardrailService:
    """
    Service layer for guardrail/safety operations.

    Provides:
    - Input safety checking with pattern matching
    - Output safety checking for model responses
    - Refusal response generation
    """

    def check_input(self, user_input: str) -> SafetyCheckResult:
        """
        Check if user input contains harmful content.

        Args:
            user_input: The raw user input string

        Returns:
            SafetyCheckResult with is_safe flag, refusal message, and category
        """
        user_input_lower = user_input.lower().strip()

        for category, patterns in HARMFUL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower, re.IGNORECASE):
                    refusal = REFUSAL_RESPONSES.get(
                        category, "I cannot respond to that request."
                    )
                    return SafetyCheckResult(is_safe=False, refusal=refusal, category=category)

        return SafetyCheckResult(is_safe=True, category="safe")

    def check_output(self, response: str, user_input: str) -> SafetyCheckResult:
        """
        Check if model output is safe.

        Args:
            response: The model's generated response
            user_input: The original user input (for context)

        Returns:
            SafetyCheckResult with is_safe flag, filtered response, and issue category
        """
        user_lower = user_input.lower()
        response_lower = response.lower()

        # Check if the model complied with malware instructions
        if any(re.search(p, user_lower) for p in HARMFUL_PATTERNS.get("malware", [])):
            if any(
                re.search(p, response_lower) for p in [
                    r"steps?", r"first,", r"second,", r"code", r"def ", r"import ",
                    r"```", r"function", r"class ",
                ]
            ):
                return SafetyCheckResult(
                    is_safe=False,
                    refusal="⚠️ [Output filtered by safety guardrails] The model attempted to respond to a harmful request. This response has been blocked.",
                    category="unsafe_output",
                )

        # Check for biased/generalizing statements
        bias_indicators = [
            r"(all|every|most)\s+(men|women)\s+(are|have|should)",
            r"(race|ethnicity|religion)\s+(determines|predicts|explains)",
        ]
        for pattern in bias_indicators:
            if re.search(pattern, response_lower):
                return SafetyCheckResult(
                    is_safe=False,
                    refusal="⚠️ [Bias detected in response. Please rephrase to avoid generalizations about groups of people.]",
                    category="bias_output",
                )

        return SafetyCheckResult(is_safe=True, category="safe")

    def create_guardrail_event(
        self, event_type: str, category: str, user_input: str, interaction_number: int
    ) -> GuardrailEvent:
        """Create a guardrail event for logging."""
        return GuardrailEvent(
            event_type=event_type,
            category=category,
            user_input=user_input,
            interaction_number=interaction_number,
        )

    @staticmethod
    def moderate_response(response: str, topic: str = "") -> Tuple[bool, str]:
        """Lightweight moderation of response content."""
        flagged_terms = [
            r"\b(hate|racist|sexist|bigot)\b",
            r"\b(kill|murder|torture)\s+(them|people|themselves|everyone)\b",
            r"\b(commit|commit\s+a)\s+(crime|murder|robbery)\b",
        ]
        for pattern in flagged_terms:
            if re.search(pattern, response.lower()):
                return False, "Response flagged for potentially harmful content."
        return True, response