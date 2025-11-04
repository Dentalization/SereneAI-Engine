"""
Guardrails Middleware for content safety and jailbreak detection.
Follows LangChain 1.0 middleware patterns.
"""

import re
from typing import List

from langchain.agents.middleware import AgentMiddleware, before_model

from src.config import get_settings


class GuardrailsMiddleware(AgentMiddleware):
    """
    Content safety and jailbreak detection middleware.

    Detects:
    - Prompt injection attempts
    - Jailbreak attempts
    - Inappropriate content
    - Medical misinformation patterns

    Example:
        >>> middleware = GuardrailsMiddleware()
        >>> agent = create_agent(..., middleware=[middleware])
    """

    # Jailbreak detection patterns
    JAILBREAK_PATTERNS = [
        r"ignore (previous|all|your) (instructions?|prompts?)",
        r"disregard (previous|all|your) (instructions?|prompts?)",
        r"forget (previous|all|your) (instructions?|prompts?)",
        r"you are (now|a) (DAN|jailbreak|unrestricted)",
        r"pretend (you are|to be) (DAN|unrestricted)",
        r"act as (if )?you (are|were) (not )?a ",
        r"roleplay as",
        r"simulate (being|a) ",
    ]

    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"system:",
        r"assistant:",
        r"\[INST\]",
        r"\[/INST\]",
    ]

    # Medical misinformation patterns (basic)
    MISINFORMATION_PATTERNS = [
        r"(dentist|doctor)s? are (useless|unnecessary|scam)",
        r"cure (all |every )?dental (problem|issue|disease) (with|using)",
        r"(miracle|instant) (cure|treatment|remedy)",
    ]

    def __init__(self):
        super().__init__()
        self.violations: List[str] = []

    @before_model
    def check_guardrails(self, state):
        """
        Check content before LLM invocation.

        Raises:
            ValueError: If content violates guardrails
        """
        messages = state.get("messages", [])
        if not messages:
            return {}

        # Get last message
        last_message = messages[-1]
        content = (
            last_message.content if hasattr(last_message, "content") else str(last_message)
        ).lower()

        violations = []

        # Check jailbreak attempts
        for pattern in self.JAILBREAK_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append("jailbreak_attempt")
                break

        # Check prompt injection
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append("prompt_injection")
                break

        # Check medical misinformation
        for pattern in self.MISINFORMATION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append("medical_misinformation")
                break

        # Check length (DoS prevention)
        if len(content) > 10000:
            violations.append("excessive_length")

        if violations:
            self.violations.extend(violations)

            # Log to audit
            config = get_settings()
            if config.enable_audit_logging:
                self._audit_log_violation(state, violations)

            # Block request
            raise ValueError(
                f"Content violates guardrails: {', '.join(set(violations))}. "
                "Request blocked for safety."
            )

        return {}

    def _audit_log_violation(self, state, violations: List[str]) -> None:
        """Log guardrail violations to audit log."""
        import json
        from datetime import datetime
        from pathlib import Path

        config = get_settings()
        audit_log_path = Path(config.audit_log_path)
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "guardrail_violation",
            "violations": violations,
            "conversation_id": state.get("conversation_id"),
            "user_id": (
                state.get("user_profile", {}).get("user_id")
                if isinstance(state.get("user_profile"), dict)
                else None
            ),
        }

        with open(audit_log_path, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
