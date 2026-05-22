"""
Guardrails & Safety Layer for AI Assistants (Legacy wrapper).

Provides input/output safety checking functions used by the root-level
model wrappers (hf_model.py, gemini_model.py).

Patterns and refusal responses are centralized in app/config.py to avoid
duplication. This module imports them from config and exposes the same
function API for backward compatibility.
"""

import re
from typing import Dict, List, Tuple, Optional

# Import centralized patterns from app config
from app.config import HARMFUL_PATTERNS, REFUSAL_RESPONSES, SENSITIVE_TOPICS

# ---------------------------------------------------------------------------
# Public API (kept identical for backward compat with model wrappers)
# ---------------------------------------------------------------------------


def check_input_safety(user_input: str) -> Tuple[bool, Optional[str], str]:
    """
    Check if input contains harmful content.

    Returns:
        (is_safe, refusal_message, category)
    """
    user_input_lower = user_input.lower().strip()

    for category, patterns in HARMFUL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, user_input_lower, re.IGNORECASE):
                return False, REFUSAL_RESPONSES.get(
                    category, "I cannot respond to that request."
                ), category

    return True, None, "safe"


def check_output_safety(response: str, user_input: str) -> Tuple[bool, Optional[str], str]:
    """
    Check if model output is safe.

    Returns:
        (is_safe, filtered_response, issue)
    """
    user_lower = user_input.lower()
    response_lower = response.lower()

    # If user asked for malware and model provided instructions
    if any(re.search(p, user_lower) for p in HARMFUL_PATTERNS.get("malware", [])):
        if any(
            re.search(p, response_lower)
            for p in [
                r"steps?", r"first,", r"second,", r"code", r"def ", r"import ",
                r"```", r"function", r"class ",
            ]
        ):
            return (
                False,
                "⚠️ [Output filtered by safety guardrails] The model attempted to respond to a harmful request. This response has been blocked.",
                "unsafe_output",
            )

    # Check for biased/generalizing statements
    bias_indicators = [
        r"(all|every|most)\s+(men|women)\s+(are|have|should)",
        r"(race|ethnicity|religion)\s+(determines|predicts|explains)",
    ]
    for pattern in bias_indicators:
        if re.search(pattern, response_lower):
            return (
                False,
                "⚠️ [Bias detected in response. Please rephrase to avoid generalizations about groups of people.]",
                "bias_output",
            )

    return True, response, "safe"


def get_safety_metadata(guardrail_triggered: bool, category: str, input_check_passed: bool) -> Dict:
    """Get safety metadata for logging/observability."""
    return {
        "guardrail_triggered": guardrail_triggered,
        "category": category,
        "input_check_passed": input_check_passed,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }


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