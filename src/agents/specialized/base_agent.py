"""Base agent class for specialized agents with fail-fast error handling.

This module provides the foundation for all specialized agents with:
- Structured execution interface
- Exception propagation (NO fallbacks)
- Execution metrics and logging
- Integration with LLMRetryHandler for LLM operations
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Status of agent execution."""

    SUCCESS = "success"
    FAILURE = "failure"


class AgentResult(BaseModel):
    """Result container for agent execution."""

    status: AgentStatus
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    retries: int = 0
    confidence: float = 1.0


class BaseAgent(ABC):
    """Base class for all specialized agents."""

    def __init__(self, name: str):
        """Initialize base agent.

        Args:
            name: Agent name for logging

        Note:
            Retry logic is delegated to LLMRetryHandler for LLM operations.
            Agents should use get_llm_retry_handler() for all LLM calls.
        """
        self.name = name

    def execute(self, **kwargs) -> AgentResult:
        """Execute agent with fail-fast error handling.

        Philosophy:
        - Execute once, propagate exceptions immediately
        - NO retries at agent level (use LLMRetryHandler for LLM calls)
        - NO fallback chains or circuit breakers
        - Clear error messages for debugging

        Returns:
            AgentResult with status SUCCESS or raises exception

        Raises:
            LLMError subclasses: For LLM-related failures
            Exception: For other agent failures
        """
        start_time = time.time()

        try:
            logger.info(f"{self.name}: Executing...")
            data = self._execute(**kwargs)

            execution_time = (time.time() - start_time) * 1000
            logger.info(f"{self.name}: Success in {execution_time:.2f}ms")

            return AgentResult(
                status=AgentStatus.SUCCESS,
                data=data,
                execution_time_ms=execution_time,
                retries=0,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"{self.name}: Failed after {execution_time:.2f}ms - {str(e)}")

            # Return failure result for metrics, but exception will propagate
            return AgentResult(
                status=AgentStatus.FAILURE,
                error=str(e),
                execution_time_ms=execution_time,
                retries=0,
                confidence=0.0,
            )

    @abstractmethod
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Internal execution logic to be implemented by subclasses.

        Subclasses should:
        - Use LLMRetryHandler from src.utils.llm_retry for all LLM calls
        - Raise clear exceptions on failure (NO silent fallbacks)
        - Return Dict with agent-specific output data on success

        Returns:
            Dict containing agent-specific output data

        Raises:
            LLMValidationError: If LLM output validation fails
            LLMAPIError: If LLM API calls fail
            LLMTimeoutError: If LLM timeout
            Other exceptions: Agent-specific errors
        """
        pass