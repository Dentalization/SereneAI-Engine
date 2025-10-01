"""Custom exceptions for LLM-driven system with fail-fast philosophy.

This module defines exceptions for pure LLM operation without fallbacks.
All exceptions provide actionable context for users and developers.
"""
from __future__ import annotations

from typing import Optional


class LLMError(Exception):
    """Base exception for all LLM-related errors.

    Philosophy: Fail fast with clear context, NO silent fallbacks.
    """

    def __init__(self, message: str, user_action: Optional[str] = None):
        """Initialize LLM error.

        Args:
            message: Technical error message for developers
            user_action: Actionable suggestion for end users
        """
        self.user_action = user_action or "Please try again or rephrase your input."
        super().__init__(message)


class LLMValidationError(LLMError):
    """LLM output failed schema validation after retries.

    This indicates:
    - LLM returned malformed JSON
    - Response missing required fields
    - Response values violate constraints

    DO NOT fall back to rules. Instead: improve prompts or model.
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_output: Optional[str] = None,
        user_action: Optional[str] = None,
    ):
        """Initialize validation error.

        Args:
            message: Error description
            attempts: Number of retry attempts made
            last_output: Last LLM output that failed validation
            user_action: Suggestion for user
        """
        self.attempts = attempts
        self.last_output = last_output
        super().__init__(
            f"{message} (after {attempts} attempts)",
            user_action=user_action
            or "Please provide more specific details or rephrase your question.",
        )


class LLMAPIError(LLMError):
    """LLM API call failed (network, auth, quota, etc).

    This indicates infrastructure issues, not user error.
    """

    def __init__(
        self,
        message: str,
        api_provider: str = "unknown",
        status_code: Optional[int] = None,
    ):
        """Initialize API error.

        Args:
            message: Error description
            api_provider: Which API failed (e.g., "Gemini", "Cohere")
            status_code: HTTP status code if applicable
        """
        self.api_provider = api_provider
        self.status_code = status_code

        user_msg = (
            f"Our AI service ({api_provider}) is temporarily unavailable. "
            "Please wait a moment and try again."
        )

        super().__init__(f"API Error [{api_provider}]: {message}", user_action=user_msg)


class LLMTimeoutError(LLMError):
    """LLM did not respond within timeout period.

    This usually indicates:
    - Very slow API response
    - Network congestion
    - Large prompt processing time
    """

    def __init__(self, timeout_seconds: float, provider: str = "unknown"):
        """Initialize timeout error.

        Args:
            timeout_seconds: How long we waited
            provider: Which LLM provider timed out
        """
        self.timeout_seconds = timeout_seconds
        self.provider = provider

        super().__init__(
            f"LLM timeout after {timeout_seconds}s ({provider})",
            user_action="The AI is taking too long. Please try with a simpler question.",
        )


# Agent-specific exceptions


class TriageError(LLMError):
    """Triage agent failed to classify user query.

    User should provide clearer input.
    """

    pass


class AnamnesisError(LLMError):
    """Anamnesis agent failed to extract symptoms.

    User should provide more medical details.
    """

    pass


class VisionError(LLMError):
    """Vision agent failed to process image.

    User should provide better quality image.
    """

    pass


class VisionQualityError(VisionError):
    """Image quality assessment failed.

    Cannot determine if image is acceptable for analysis.
    """

    pass


class RAGError(LLMError):
    """RAG agent failed to retrieve or generate response.

    System should check knowledge base or prompt quality.
    """

    pass


class ClaimValidationError(RAGError):
    """Claim validator failed to validate response against sources.

    System should check if sources are adequate.
    """

    pass


class OrchestratorError(LLMError):
    """Orchestrator failed to route or coordinate agents.

    This indicates a fundamental workflow issue.
    """

    pass
