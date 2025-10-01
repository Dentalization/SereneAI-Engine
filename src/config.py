"""Project configuration and logging utilities.

This module exposes two utilities:
- load_config(): returns a dict of runtime configuration values.
- setup_logging(): sets up a rotating file handler and console logging.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, Tuple, List, Any
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv


def load_config() -> Dict[str, Any]:
    """Load configuration settings from environment and defaults.

    Returns:
        A dictionary containing model/paths thresholds and prompts used across
        the project. Environment variables are loaded via python-dotenv.
    """
    load_dotenv()
    return {
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "cohere_api_key": os.getenv("COHERE_API_KEY"),
        # Vision model path (custom dental model). Fallbacks handled in yolo_tool.
        "model_path": "models/oral_detection_model.pt",
        # Directory for locally indexed documents used by RAG.
        "docs_path": "docs/",
        # SQLite DB path for UI persistence.
        "db_path": "dental_chatbot.db",
        # Detection confidence threshold for YOLO predictions.
        "confidence_threshold": 0.5,
        # Max allowed upload size (in MB) for images.
        "max_file_size_mb": 5,
        # Cache size used by RAG LRU cache.
        "cache_size": 100,
        # RAG controls
        "enable_pubmed": os.getenv("ENABLE_PUBMED", "false").lower() == "true",
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "rag_index_dir": os.getenv("RAG_INDEX_DIR", ".rag/faiss_index"),
        "kg_path": os.getenv("KG_PATH", ".rag/kg.pkl"),
        # Detection classes and colors (BGR) for visualization.
        "classes": [
            "calculus",
            "caries",
            "gingivitis",
            "hypodontia",
            "tooth_discoloration",
            "ulcer",
        ],
        "class_colors": {
            "calculus": (255, 0, 0),
            "caries": (0, 255, 0),
            "gingivitis": (0, 0, 255),
            "hypodontia": (255, 255, 0),
            "tooth_discoloration": (255, 0, 255),
            "ulcer": (0, 255, 255),
        },
        # Prompt used by vision LLM to add spatial context on detections.
        "gemini_spatial_prompt": (
            "Analyze dental image with detections {detections}. Describe spatial relations "
            "(e.g., upper/lower jaw, left/right side, specific tooth position like second molar), "
            "severity, relations to gums/tongue/other teeth. Focus on dental features; "
            "ignore non-dental objects. Provide precise marking insights."
        ),
    }


def setup_logging() -> None:
    """Setup logging with rotation and level from LOG_LEVEL env var.

    The logger writes to both stdout and app.log with rotation.
    UTF-8 encoding is used to prevent unicode issues on Windows.
    """
    if getattr(setup_logging, "_initialized", False):
        return
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()  # DEBUG/INFO/WARNING/...
    level = getattr(logging, log_level, logging.INFO)

    # Configure formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Setup basic config
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Add file handler with UTF-8 encoding
    handler = RotatingFileHandler(
        "app.log",
        maxBytes=1_000_000,
        backupCount=5,
        encoding='utf-8'  # Prevent unicode encoding errors
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Avoid duplicate handlers if Streamlit reruns
    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        root_logger.addHandler(handler)

    # Reduce noisy third-party logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)

    logging.info(f"Logging setup with level: {log_level}")
    setattr(setup_logging, "_initialized", True)
