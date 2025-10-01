"""LLM retry handler with structured output and NO fallbacks.

Philosophy: Fail fast with retries, then raise clear errors.
NO degradation to rule-based logic.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from src.utils.exceptions import (
    LLMAPIError,
    LLMTimeoutError,
    LLMValidationError,
)
from src.utils.llm import get_gemini_chat

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMRetryHandler:
    """Pure LLM invocation with structured output, exponential backoff, and NO fallbacks.

    This class embodies fail-fast philosophy:
    - Retry with exponential backoff on failures
    - Validate outputs strictly against Pydantic schemas
    - Raise clear exceptions (NO silent fallbacks)
    - Log all attempts for debugging

    Example:
        handler = LLMRetryHandler(max_retries=3)
        result = handler.invoke_structured(
            messages=[HumanMessage(content="Classify this query")],
            response_schema=TriageDecision,
            strict=True
        )
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,  # Low for consistency
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        timeout: Optional[float] = 30.0,
    ):
        """Initialize retry handler.

        Args:
            model: LLM model to use
            temperature: Sampling temperature (low = consistent)
            max_retries: Maximum retry attempts
            backoff_factor: Exponential backoff multiplier
            timeout: Timeout per request in seconds
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

        self.llm = get_gemini_chat(model=model, temperature=temperature)
        logger.info(
            f"LLMRetryHandler: Initialized {model} with max_retries={max_retries}"
        )

    def invoke_structured(
        self,
        messages: List[BaseMessage],
        response_schema: Type[T],
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Invoke LLM with structured output and automatic retry.

        This method:
        1. Sends prompt to LLM
        2. Parses JSON response
        3. Validates against Pydantic schema
        4. Retries with exponential backoff if validation fails
        5. Raises exception if all retries exhausted (NO fallback)

        Args:
            messages: List of messages to send to LLM
            response_schema: Pydantic model for response validation
            strict: If True, require ALL schema fields (recommended)
            **kwargs: Additional LLM parameters

        Returns:
            Validated Pydantic model instance

        Raises:
            LLMValidationError: If response validation fails after retries
            LLMAPIError: If API calls fail after retries
            LLMTimeoutError: If request times out
        """
        last_error: Optional[Exception] = None
        last_output: Optional[str] = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"LLMRetryHandler: Attempt {attempt + 1}/{self.max_retries}"
                )

                # Step 1: Invoke LLM with JSON mode
                response = self.llm.invoke(
                    messages,
                    config={"response_format": {"type": "json_object"}},
                    timeout=self.timeout,
                    **kwargs,
                )

                raw_content = response.content.strip()
                last_output = raw_content
                logger.debug(f"LLMRetryHandler: Raw response: {raw_content[:200]}...")

                # Step 2: Extract JSON (handle markdown code blocks)
                json_content = self._extract_json(raw_content)

                # Step 3: Parse JSON
                try:
                    data = json.loads(json_content)
                except json.JSONDecodeError as e:
                    raise ValidationError(
                        f"Invalid JSON: {e}",
                        model=response_schema
                    )

                # Step 4: Validate against schema
                validated = response_schema.parse_obj(data)

                # Step 5: Strict mode - check all required fields present
                if strict:
                    missing = self._check_required_fields(validated, response_schema)
                    if missing:
                        raise ValidationError(
                            f"Missing required fields: {missing}",
                            model=response_schema
                        )

                logger.info(
                    f"LLMRetryHandler: Success on attempt {attempt + 1}"
                )
                return validated

            except ValidationError as e:
                last_error = e
                logger.warning(
                    f"LLMRetryHandler: Validation failed (attempt {attempt + 1}): {e}"
                )

                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor ** attempt
                    logger.info(
                        f"LLMRetryHandler: Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    # All retries exhausted
                    raise LLMValidationError(
                        message=f"LLM output validation failed: {str(e)}",
                        attempts=self.max_retries,
                        last_output=last_output,
                        user_action="Please provide clearer or more specific information.",
                    ) from e

            except TimeoutError as e:
                last_error = e
                logger.error(f"LLMRetryHandler: Timeout on attempt {attempt + 1}")

                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor ** attempt
                    time.sleep(wait_time)
                else:
                    raise LLMTimeoutError(
                        timeout_seconds=self.timeout or 30.0,
                        provider=self.model,
                    ) from e

            except Exception as e:
                last_error = e
                logger.error(
                    f"LLMRetryHandler: API error on attempt {attempt + 1}: {e}"
                )

                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor ** attempt
                    time.sleep(wait_time)
                else:
                    raise LLMAPIError(
                        message=str(e),
                        api_provider=self.model,
                    ) from e

        # Should never reach here, but just in case
        raise LLMValidationError(
            message=f"Unexpected error after {self.max_retries} attempts: {last_error}",
            attempts=self.max_retries,
            last_output=last_output,
        )

    def invoke_simple(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Invoke LLM for simple text generation (no structured output).

        Use this for:
        - Free-form text generation
        - When structured output not needed
        - Prompts that don't require validation

        Args:
            prompt: Text prompt
            temperature: Override default temperature
            **kwargs: Additional LLM parameters

        Returns:
            Generated text string

        Raises:
            LLMAPIError: If API calls fail after retries
        """
        messages = [HumanMessage(content=prompt)]
        temp = temperature if temperature is not None else self.temperature

        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(
                    messages, temperature=temp, timeout=self.timeout, **kwargs
                )
                return response.content.strip()

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor ** attempt
                    logger.warning(
                        f"LLMRetryHandler: Simple invoke failed, retry in {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                else:
                    raise LLMAPIError(
                        message=f"Simple invoke failed: {e}",
                        api_provider=self.model,
                    ) from e

        raise LLMAPIError(
            message="Unexpected: exhausted retries",
            api_provider=self.model,
        )

    def _extract_json(self, raw_content: str) -> str:
        """Extract JSON from response that might have markdown wrapping.

        Args:
            raw_content: Raw LLM response

        Returns:
            Clean JSON string
        """
        # Try to find JSON in markdown code block
        code_block_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            raw_content,
            re.DOTALL
        )
        if code_block_match:
            return code_block_match.group(1)

        # Try to find raw JSON
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # Return as-is if no match (will likely fail JSON parsing)
        return raw_content

    def _check_required_fields(
        self,
        instance: BaseModel,
        schema: Type[BaseModel]
    ) -> List[str]:
        """Check if all required fields are present (strict mode).

        Args:
            instance: Validated instance
            schema: Pydantic schema

        Returns:
            List of missing required field names
        """
        missing = []
        for field_name, field_info in schema.__fields__.items():
            if field_info.required:
                value = getattr(instance, field_name, None)
                if value is None:
                    missing.append(field_name)
        return missing


# Global singleton instance
_global_handler: Optional[LLMRetryHandler] = None


def get_llm_retry_handler(
    model: str = "gemini-2.5-flash",
    max_retries: int = 3,
) -> LLMRetryHandler:
    """Get or create global LLM retry handler instance.

    Args:
        model: LLM model to use
        max_retries: Maximum retry attempts

    Returns:
        LLMRetryHandler instance
    """
    global _global_handler
    if _global_handler is None:
        _global_handler = LLMRetryHandler(model=model, max_retries=max_retries)
    return _global_handler
