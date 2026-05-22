"""
Data models for the AI Assistant Evaluation Platform.
"""

from app.models.conversation import (
    Message,
    MessageRole,
    ConversationHistory,
    ModelResponse,
    SafetyCheckResult,
    GuardrailEvent,
    ContextStats,
)

from app.models.evaluation import (
    EvaluationResult,
    CategoryMetrics,
    ComparisonResult,
    EvaluationReport,
)

__all__ = [
    "Message",
    "MessageRole",
    "ConversationHistory",
    "ModelResponse",
    "SafetyCheckResult",
    "GuardrailEvent",
    "ContextStats",
    "EvaluationResult",
    "CategoryMetrics",
    "ComparisonResult",
    "EvaluationReport",
]