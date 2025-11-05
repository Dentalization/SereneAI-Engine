"""
Runtime Context Management for SereneAI V2
Immutable context passed via config to tools and middleware
"""

from typing import Any
from dataclasses import dataclass, field
from uuid import uuid4
from datetime import datetime

from .config import get_settings


@dataclass(frozen=True)
class Context:
    """
    Immutable runtime context

    Passed via config parameter to graph execution:
    graph.invoke(state, config={"configurable": {"context": context}})

    Accessible in tools via ToolRuntime.context
    """

    # ============================================================================
    # USER IDENTIFIERS
    # ============================================================================

    user_id: str
    session_id: str
    conversation_id: str

    # ============================================================================
    # API KEYS
    # ============================================================================

    api_keys: dict[str, str] = field(default_factory=dict)

    # ============================================================================
    # FEATURE FLAGS
    # ============================================================================

    enable_pubmed: bool = False
    enable_vision: bool = True
    enable_human_approval: bool = False
    enable_claim_validation: bool = True

    # ============================================================================
    # LIMITS
    # ============================================================================

    max_message_length: int = 10000
    max_image_size_mb: int = 5
    max_tool_calls: int = 10
    max_execution_time_s: int = 60

    # ============================================================================
    # OBSERVABILITY
    # ============================================================================

    trace_id: str = field(default_factory=lambda: str(uuid4()))
    langsmith_project: str = "sereneai-v2"
    langsmith_tags: list[str] = field(default_factory=list)

    # ============================================================================
    # METADATA
    # ============================================================================

    created_at: datetime = field(default_factory=datetime.now)
    environment: str = "development"
    client_info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(
        cls,
        user_id: str,
        session_id: str,
        conversation_id: str | None = None,
        **overrides
    ) -> "Context":
        """Create context from settings with optional overrides"""
        settings = get_settings()

        return cls(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id or str(uuid4()),
            api_keys={
                "gemini": settings.gemini_api_key,
                "cohere": settings.cohere_api_key or "",
                "langsmith": settings.langsmith_api_key or "",
            },
            enable_pubmed=settings.rag.enable_pubmed,
            enable_vision=True,
            enable_human_approval=settings.middleware.enable_human_approval,
            enable_claim_validation=settings.rag.enable_claim_validation,
            max_message_length=10000,
            max_image_size_mb=settings.yolo.max_image_size_mb,
            max_tool_calls=10,
            max_execution_time_s=60,
            trace_id=str(uuid4()),
            langsmith_project=settings.observability.langsmith_project,
            environment=settings.environment,
            **overrides
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "trace_id": self.trace_id,
            "environment": self.environment,
            "langsmith_project": self.langsmith_project,
            "langsmith_tags": self.langsmith_tags,
            "created_at": self.created_at.isoformat(),
            # Feature flags
            "enable_pubmed": self.enable_pubmed,
            "enable_vision": self.enable_vision,
            "enable_human_approval": self.enable_human_approval,
            # Limits
            "max_message_length": self.max_message_length,
            "max_image_size_mb": self.max_image_size_mb,
            "max_tool_calls": self.max_tool_calls,
        }


def get_context_from_config(config: dict) -> Context | None:
    """Extract context from LangGraph config"""
    return config.get("configurable", {}).get("context")
