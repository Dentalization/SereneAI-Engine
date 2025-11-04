"""
Dynamic model selection and routing.
Follows LangChain 1.0 configurable models pattern.
"""

from typing import Literal

from langchain_core.language_models import BaseChatModel

from src.config import get_settings
from src.models.gemini import get_gemini_chat


def get_model(
    model_type: Literal["fast", "balanced", "advanced"] = "balanced",
    temperature: float = 0.1,
) -> BaseChatModel:
    """
    Get appropriate model based on task requirements.

    Args:
        model_type:
            - "fast": Fastest, cheapest model (for simple tasks)
            - "balanced": Good balance of speed and capability (default)
            - "advanced": Most capable model (for complex reasoning)
        temperature: Sampling temperature 0-1

    Returns:
        Initialized chat model instance

    Example:
        >>> # For simple classification
        >>> fast_model = get_model("fast")
        >>>
        >>> # For clinical reasoning
        >>> advanced_model = get_model("advanced", temperature=0.2)
    """
    config = get_settings()

    # Model selection based on type
    model_map = {
        "fast": "gemini-2.5-flash",  # Fastest
        "balanced": config.default_model,  # Configurable default
        "advanced": "gemini-2.5-pro",  # Most capable (when available)
    }

    model_name = model_map[model_type]

    # For now, only Gemini is implemented
    # Future: Add Claude, GPT-4, etc.
    return get_gemini_chat(model=model_name, temperature=temperature)


def get_model_for_task(task: str, context_length: int = 0) -> BaseChatModel:
    """
    Get model dynamically based on task and context.

    This implements dynamic model selection as recommended in LangChain 1.0
    context engineering best practices.

    Args:
        task: Task type (triage, diagnosis, synthesis, etc.)
        context_length: Number of tokens in context

    Returns:
        Appropriate model for the task

    Example:
        >>> # For triage (simple classification)
        >>> model = get_model_for_task("triage", context_length=500)
        >>>
        >>> # For diagnosis with long context
        >>> model = get_model_for_task("diagnosis", context_length=8000)
    """
    # Task complexity mapping
    complex_tasks = ["diagnosis", "treatment_planning", "differential_diagnosis"]
    simple_tasks = ["triage", "greeting"]

    if task in simple_tasks and context_length < 1000:
        return get_model("fast")
    elif task in complex_tasks or context_length > 5000:
        return get_model("advanced", temperature=0.2)
    else:
        return get_model("balanced")
