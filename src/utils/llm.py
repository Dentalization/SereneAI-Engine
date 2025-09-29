"""Utilities for constructing LLM clients used across the project.

This module centralizes creation of ChatGoogleGenerativeAI instances to avoid
instantiating them in multiple places with duplicated parameters.
"""
from __future__ import annotations

from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import load_config


def get_gemini_chat(
    model: str = "gemini-2.5-flash",
    *,
    temperature: float = 0.0,
    convert_system_message_to_human: Optional[bool] = None,
) -> ChatGoogleGenerativeAI:
    """Return a configured ChatGoogleGenerativeAI client.

    Args:
        model: Model name to use.
        temperature: Sampling temperature for generation.
        convert_system_message_to_human: Passed through when provided to match
            callers that rely on this behavior.

    Returns:
        A configured ChatGoogleGenerativeAI instance using the GEMINI_API_KEY
        from the environment (via load_config()).
    """
    config = load_config()

    kwargs = {
        "model": model,
        "google_api_key": config["gemini_api_key"],
        "temperature": temperature,
    }
    if convert_system_message_to_human is not None:
        kwargs["convert_system_message_to_human"] = convert_system_message_to_human

    return ChatGoogleGenerativeAI(**kwargs)
