"""Base agent class for specialized agents with retry and fallback logic.

This module provides the foundation for all specialized agents with:
- Retry logic with exponential backoff
- Circuit breaker pattern
- Fallback chain support
- Structured logging and metrics
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
    FALLBACK = "fallback"
    CIRCUIT_OPEN = "circuit_open"


class AgentResult(BaseModel):
    """Result container for agent execution."""

    status: AgentStatus
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    retries: int = 0
    confidence: float = 1.0


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "half_open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker recovered to closed state")
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            raise e


class BaseAgent(ABC):
    """Base class for all specialized agents."""

    def __init__(
        self,
        name: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_circuit_breaker: bool = True,
    ):
        self.name = name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        self.fallback_agents: list[BaseAgent] = []

    def add_fallback(self, agent: BaseAgent) -> None:
        """Add a fallback agent to be used if this agent fails."""
        self.fallback_agents.append(agent)
        logger.info(f"Added fallback agent {agent.name} to {self.name}")

    def execute(self, **kwargs) -> AgentResult:
        """Execute agent with retry logic and fallback chain."""
        start_time = time.time()
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"{self.name}: Execution attempt {attempt + 1}/{self.max_retries + 1}")

                if self.circuit_breaker:
                    data = self.circuit_breaker.call(self._execute, **kwargs)
                else:
                    data = self._execute(**kwargs)

                execution_time = (time.time() - start_time) * 1000
                logger.info(f"{self.name}: Success in {execution_time:.2f}ms")

                return AgentResult(
                    status=AgentStatus.SUCCESS,
                    data=data,
                    execution_time_ms=execution_time,
                    retries=attempt,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"{self.name}: Attempt {attempt + 1} failed: {last_error}"
                )

                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"{self.name}: Retrying in {delay}s...")
                    time.sleep(delay)

        # All retries exhausted, try fallback chain
        if self.fallback_agents:
            logger.warning(f"{self.name}: All retries failed, trying fallback chain")
            for fallback in self.fallback_agents:
                try:
                    result = fallback.execute(**kwargs)
                    if result.status == AgentStatus.SUCCESS:
                        result.status = AgentStatus.FALLBACK
                        logger.info(f"{self.name}: Fallback {fallback.name} succeeded")
                        return result
                except Exception as fb_error:
                    logger.warning(f"{self.name}: Fallback {fallback.name} failed: {fb_error}")

        # Complete failure
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"{self.name}: Complete failure after {self.max_retries} retries")

        return AgentResult(
            status=AgentStatus.FAILURE,
            error=last_error or "Unknown error",
            execution_time_ms=execution_time,
            retries=self.max_retries,
            confidence=0.0,
        )

    @abstractmethod
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Internal execution logic to be implemented by subclasses.

        Returns:
            Dict containing agent-specific output data.

        Raises:
            Exception: Any error that should trigger retry logic.
        """
        pass