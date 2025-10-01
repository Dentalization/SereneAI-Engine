"""Fallback LLM chain with graceful degradation.

This module provides:
- Primary LLM with automatic fallback to alternatives
- Model routing based on task complexity
- Graceful degradation when all LLMs fail
- Template-based emergency responses
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from src.utils.llm import get_gemini_chat

logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """Task complexity levels for model routing."""

    SIMPLE = "simple"  # Greetings, simple questions
    MODERATE = "moderate"  # Anamnesis, basic analysis
    COMPLEX = "complex"  # RAG synthesis, medical reasoning


class LLMProvider(str, Enum):
    """Available LLM providers."""

    GEMINI_FLASH = "gemini-2.5-flash"
    GEMINI_PRO = "gemini-2.0-pro"
    # Could add: CLAUDE = "claude-3-5-sonnet" (via Anthropic API)
    # Could add: GPT4 = "gpt-4o" (via OpenAI API)


class FallbackResponse(BaseModel):
    """Response from fallback LLM chain."""

    content: str
    provider: str
    is_fallback: bool = False
    is_template: bool = False


class FallbackLLMChain:
    """LLM chain with automatic fallback and degradation."""

    # Template responses for complete failure scenarios
    EMERGENCY_TEMPLATES = {
        "id": {
            "greeting": "Halo! Saya SereneAI, asisten dental virtual. Ada yang bisa saya bantu dengan kesehatan gigi Anda?",
            "anamnesis": "Bisakah Anda ceritakan lebih detail tentang keluhan gigi Anda? Misalnya lokasi, sejak kapan, dan tingkat sakitnya?",
            "diagnosis": "Maaf, saat ini saya mengalami kesulitan teknis. Untuk keluhan dental, saya sangat menyarankan Anda berkonsultasi langsung dengan dokter gigi terdekat.",
            "error": "Maaf, terjadi kesalahan teknis. Silakan coba lagi atau konsultasikan ke dokter gigi untuk penanganan profesional.",
        },
        "en": {
            "greeting": "Hello! I'm SereneAI, your virtual dental assistant. How can I help with your dental health?",
            "anamnesis": "Could you tell me more details about your dental issue? For example, location, duration, and severity?",
            "diagnosis": "Sorry, I'm experiencing technical difficulties. For dental concerns, I strongly recommend consulting with a dentist directly.",
            "error": "Sorry, a technical error occurred. Please try again or consult a dentist for professional care.",
        }
    }

    def __init__(self):
        """Initialize fallback chain with primary and backup LLMs."""
        self.providers = [
            LLMProvider.GEMINI_FLASH,
            LLMProvider.GEMINI_PRO,
        ]
        self.llm_instances: Dict[str, Any] = {}

    def _get_llm(self, provider: LLMProvider, temperature: float = 0.3):
        """Get or create LLM instance for provider."""
        cache_key = f"{provider.value}_{temperature}"

        if cache_key not in self.llm_instances:
            try:
                if provider in [LLMProvider.GEMINI_FLASH, LLMProvider.GEMINI_PRO]:
                    self.llm_instances[cache_key] = get_gemini_chat(
                        model=provider.value,
                        temperature=temperature,
                    )
                    logger.debug(f"FallbackLLM: Initialized {provider.value}")
                # Could add other providers here
            except Exception as e:
                logger.error(f"FallbackLLM: Failed to initialize {provider.value} - {e}")
                return None

        return self.llm_instances.get(cache_key)

    def invoke(
        self,
        messages: List[BaseMessage],
        complexity: TaskComplexity = TaskComplexity.MODERATE,
        temperature: float = 0.3,
        language: str = "id",
        **kwargs,
    ) -> FallbackResponse:
        """Invoke LLM with automatic fallback.

        Args:
            messages: Messages to send to LLM
            complexity: Task complexity for model routing
            temperature: Sampling temperature
            language: Language for emergency templates (id/en)
            **kwargs: Additional LLM parameters

        Returns:
            FallbackResponse with content and metadata
        """
        # Try each provider in order
        last_error: Optional[str] = None

        for provider in self.providers:
            try:
                logger.debug(f"FallbackLLM: Trying {provider.value}")

                llm = self._get_llm(provider, temperature)
                if llm is None:
                    continue

                response = llm.invoke(messages, **kwargs)

                logger.info(f"FallbackLLM: Success with {provider.value}")
                return FallbackResponse(
                    content=response.content,
                    provider=provider.value,
                    is_fallback=(provider != self.providers[0]),
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"FallbackLLM: {provider.value} failed - {last_error}. "
                    f"Trying next provider..."
                )
                continue

        # All LLMs failed - use template
        logger.error(f"FallbackLLM: All providers failed. Last error: {last_error}")
        return self._get_template_response(complexity, language)

    def _get_template_response(
        self,
        complexity: TaskComplexity,
        language: str,
    ) -> FallbackResponse:
        """Get emergency template response when all LLMs fail."""
        lang = language if language in ["id", "en"] else "id"
        templates = self.EMERGENCY_TEMPLATES[lang]

        # Map complexity to template type
        if complexity == TaskComplexity.SIMPLE:
            template = templates["greeting"]
        elif complexity == TaskComplexity.MODERATE:
            template = templates["anamnesis"]
        else:
            template = templates["diagnosis"]

        logger.info("FallbackLLM: Using emergency template response")
        return FallbackResponse(
            content=template,
            provider="template",
            is_fallback=True,
            is_template=True,
        )

    def route_by_complexity(self, complexity: TaskComplexity) -> LLMProvider:
        """Select optimal model based on task complexity.

        Args:
            complexity: Task complexity level

        Returns:
            Recommended LLM provider
        """
        # Simple tasks -> Fast model
        if complexity == TaskComplexity.SIMPLE:
            return LLMProvider.GEMINI_FLASH

        # Moderate tasks -> Flash (good balance)
        elif complexity == TaskComplexity.MODERATE:
            return LLMProvider.GEMINI_FLASH

        # Complex tasks -> Pro (better reasoning)
        else:
            return LLMProvider.GEMINI_PRO


# Global fallback chain instance
_fallback_chain: Optional[FallbackLLMChain] = None


def get_fallback_chain() -> FallbackLLMChain:
    """Get or create global fallback chain instance."""
    global _fallback_chain
    if _fallback_chain is None:
        _fallback_chain = FallbackLLMChain()
    return _fallback_chain


def invoke_with_fallback(
    prompt: str,
    complexity: TaskComplexity = TaskComplexity.MODERATE,
    temperature: float = 0.3,
    language: str = "id",
    **kwargs,
) -> FallbackResponse:
    """Convenience function to invoke fallback chain.

    Args:
        prompt: Text prompt to send
        complexity: Task complexity
        temperature: Sampling temperature
        language: Response language
        **kwargs: Additional parameters

    Returns:
        FallbackResponse with content and metadata
    """
    chain = get_fallback_chain()
    messages = [HumanMessage(content=prompt)]

    return chain.invoke(
        messages=messages,
        complexity=complexity,
        temperature=temperature,
        language=language,
        **kwargs,
    )