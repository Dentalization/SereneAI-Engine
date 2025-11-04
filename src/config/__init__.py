"""
Modern configuration management using Pydantic Settings.
Follows LangChain 1.0 + LangGraph 1.0 best practices for production deployment.
"""

from src.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
