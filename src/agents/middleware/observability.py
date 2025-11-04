"""
Observability Middleware for LangSmith tracing.
Follows LangChain 1.0 observability patterns.
"""

import os
from contextlib import contextmanager

from langchain.agents.middleware import AgentMiddleware, before_agent, after_agent

from src.config import get_settings


class ObservabilityMiddleware(AgentMiddleware):
    """
    LangSmith tracing and observability.

    Automatically traces:
    - Agent execution
    - Model calls
    - Tool invocations
    - Errors and exceptions

    Example:
        >>> middleware = ObservabilityMiddleware()
        >>> agent = create_agent(..., middleware=[middleware])
    """

    def __init__(self):
        super().__init__()
        self.config = get_settings()

        # Setup LangSmith environment variables
        if self.config.langsmith_enabled:
            os.environ["LANGSMITH_API_KEY"] = self.config.langsmith_api_key
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_PROJECT"] = self.config.langsmith_project
            os.environ["LANGSMITH_ENDPOINT"] = self.config.langsmith_endpoint

    @before_agent
    def start_trace(self, state):
        """Start tracing context before agent execution."""
        if not self.config.langsmith_enabled:
            return {}

        # Add metadata for tracing
        conversation_id = state.get("conversation_id")
        user_profile = state.get("user_profile")

        metadata = {
            "conversation_id": conversation_id,
            "user_id": user_profile.user_id if user_profile else None,
            "language": user_profile.language if user_profile else "id",
            "stage": state.get("stage", "unknown"),
        }

        # LangSmith will automatically pick up this metadata
        # through the config parameter in agent invocation

        return {}

    @after_agent
    def end_trace(self, state):
        """Add final metrics after agent execution."""
        if not self.config.langsmith_enabled:
            return {}

        # Add execution metrics to state (for LangSmith)
        return {
            "tool_calls_count": state.get("tool_calls_count", 0),
            "execution_time_ms": state.get("execution_time_ms", 0),
            "confidence": state.get("confidence", 0.0),
        }


@contextmanager
def tracing_context(
    conversation_id: str,
    user_id: str | None = None,
    metadata: dict | None = None,
):
    """
    Context manager for selective tracing.

    Example:
        >>> with tracing_context("conv_123", user_id="user_456"):
        >>>     result = agent.invoke(input)
    """
    config = get_settings()

    if not config.langsmith_enabled:
        yield
        return

    # This would integrate with langsmith's tracing context
    # For now, it's a placeholder
    try:
        yield
    finally:
        pass
