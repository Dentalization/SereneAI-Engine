"""
PII Protection Middleware following LangChain 1.0 patterns.
Detects and redacts PII before sending to LLM, logs for audit.
"""

import hashlib
import re
from typing import Dict, List, Literal

from langchain.agents.middleware import AgentMiddleware, after_model, before_model

from src.config import get_settings


class PIIProtectionMiddleware(AgentMiddleware):
    """
    Detect and redact PII before sending to LLM.
    Follows LangChain 1.0 middleware hooks pattern.

    Strategies:
    - redact: Replace with [REDACTED_TYPE]
    - mask: Partially obscure (e.g., ***-***-1234)
    - hash: Replace with hash (e.g., [HASH_abc123])
    - block: Raise exception when PII detected

    Example:
        >>> middleware = PIIProtectionMiddleware(strategy="mask")
        >>> agent = create_agent(..., middleware=[middleware])
    """

    # PII detection patterns
    PII_PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "nik": re.compile(r"\b\d{16}\b"),  # Indonesian NIK
        "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
        "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        "mac_address": re.compile(r"\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b"),
        "url": re.compile(r"https?://[^\s]+"),
    }

    def __init__(self, strategy: Literal["redact", "mask", "hash", "block"] = "mask"):
        """
        Args:
            strategy: PII handling strategy
        """
        super().__init__()
        self.strategy = strategy
        self.redaction_map: Dict[str, str] = {}
        self.pii_detected_count = 0

    @before_model
    def detect_and_redact_pii(self, state):
        """
        Hook before LLM call to detect and redact PII.

        This runs before every model invocation, scanning the latest message
        for PII and applying the configured redaction strategy.
        """
        messages = state.get("messages", [])
        if not messages:
            return {}

        # Get last message (user input)
        last_message = messages[-1]
        content = (
            last_message.content if hasattr(last_message, "content") else str(last_message)
        )

        # Detect PII
        pii_found = self._detect_pii(content)

        if not pii_found:
            return {}

        # Update counter
        self.pii_detected_count += len(pii_found)

        # Block if strategy is block
        if self.strategy == "block":
            pii_types = {pii["type"] for pii in pii_found}
            raise ValueError(
                f"PII detected in input ({', '.join(pii_types)}) - request blocked for security"
            )

        # Redact based on strategy
        redacted_content = self._redact_pii(content, pii_found)

        # Update message
        if hasattr(last_message, "content"):
            last_message.content = redacted_content
        else:
            messages[-1] = redacted_content

        # Set flag in state for audit
        return {
            "messages": messages,
            "pii_detected": True,
        }

    @after_model
    def log_pii_detection(self, state):
        """
        Hook after model response to log PII detection.

        We don't restore PII in output (security best practice),
        but we log the detection for audit purposes.
        """
        if self.redaction_map:
            # Log to audit system
            config = get_settings()
            if config.enable_audit_logging:
                self._audit_log_pii(state, list(self.redaction_map.keys()))

        return {}

    def _detect_pii(self, text: str) -> List[Dict]:
        """
        Detect all PII in text.

        Returns:
            List of dicts with type, value, start, end
        """
        pii_found = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            for match in pattern.finditer(text):
                pii_found.append(
                    {
                        "type": pii_type,
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
        return pii_found

    def _redact_pii(self, text: str, pii_found: List[Dict]) -> str:
        """
        Redact PII based on strategy.

        Args:
            text: Original text
            pii_found: List of detected PII

        Returns:
            Text with PII redacted
        """
        # Process in reverse order to maintain indices
        for pii in reversed(sorted(pii_found, key=lambda x: x["start"])):
            if self.strategy == "redact":
                placeholder = f"[REDACTED_{pii['type'].upper()}]"
                text = text[: pii["start"]] + placeholder + text[pii["end"] :]
                self.redaction_map[placeholder] = pii["value"]

            elif self.strategy == "mask":
                masked = self._mask_value(pii["value"], pii["type"])
                text = text[: pii["start"]] + masked + text[pii["end"] :]

            elif self.strategy == "hash":
                hashed = hashlib.sha256(pii["value"].encode()).hexdigest()[:8]
                placeholder = f"[HASH_{hashed}]"
                text = text[: pii["start"]] + placeholder + text[pii["end"] :]
                self.redaction_map[placeholder] = pii["value"]

        return text

    def _mask_value(self, value: str, pii_type: str) -> str:
        """
        Mask PII value based on type.

        Args:
            value: PII value
            pii_type: Type of PII

        Returns:
            Masked value
        """
        if pii_type == "email":
            parts = value.split("@")
            return f"{parts[0][0]}***@{parts[1]}"
        elif pii_type == "phone":
            return "***-***-" + value[-4:]
        elif pii_type == "nik":
            return value[:4] + "********" + value[-4:]
        elif pii_type == "credit_card":
            return "**** **** **** " + value[-4:]
        elif pii_type == "ip_address":
            parts = value.split(".")
            return f"{parts[0]}.***.***.{parts[3]}"
        else:
            # Generic masking
            if len(value) <= 4:
                return "*" * len(value)
            return value[:2] + "*" * (len(value) - 4) + value[-2:]

    def _audit_log_pii(self, state, pii_types: List[str]) -> None:
        """
        Log PII detection to audit log.

        Args:
            state: Agent state
            pii_types: List of PII types detected
        """
        import json
        from datetime import datetime
        from pathlib import Path

        config = get_settings()
        audit_log_path = Path(config.audit_log_path)
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "pii_detected",
            "pii_types": pii_types,
            "pii_count": self.pii_detected_count,
            "redaction_strategy": self.strategy,
            "conversation_id": state.get("conversation_id"),
            "user_id": state.get("user_profile", {}).get("user_id") if isinstance(state.get("user_profile"), dict) else None,
        }

        with open(audit_log_path, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
