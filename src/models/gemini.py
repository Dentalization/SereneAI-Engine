"""
Google Gemini model initialization.
Follows LangChain 1.0 model initialization patterns.
"""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings


@lru_cache(maxsize=1)
def get_gemini_chat(
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> ChatGoogleGenerativeAI:
    """
    Get Gemini chat model instance (cached).

    Args:
        model: Model identifier (default: from config)
        temperature: Sampling temperature 0-1 (default: 0.1 for deterministic)
        max_tokens: Maximum tokens to generate (default: None for model default)

    Returns:
        Initialized ChatGoogleGenerativeAI instance

    Example:
        >>> model = get_gemini_chat()
        >>> response = model.invoke("Hello, how are you?")
    """
    config = get_settings()

    return ChatGoogleGenerativeAI(
        model=model or config.default_model,
        api_key=config.gemini_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30.0,
        max_retries=3,
        # LangChain 1.0: streaming support
        streaming=True,
    )


@lru_cache(maxsize=1)
def get_gemini_vision(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
) -> ChatGoogleGenerativeAI:
    """
    Get Gemini vision model instance (cached).

    Args:
        model: Model identifier with vision support
        temperature: Sampling temperature 0-1

    Returns:
        Initialized ChatGoogleGenerativeAI instance with vision support

    Example:
        >>> model = get_gemini_vision()
        >>> response = model.invoke([
        ...     {"type": "text", "text": "What's in this image?"},
        ...     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ... ])
    """
    config = get_settings()

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=config.gemini_api_key,
        temperature=temperature,
        timeout=60.0,  # Vision may take longer
        max_retries=3,
    )
