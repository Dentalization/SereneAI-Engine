"""Centralized embedding model loader with singleton pattern.

This module provides shared embedding model instances to avoid redundant loading
across different components (RAGSystem, SemanticChunker, Reranker, etc.).

Best Practice:
- Load models once, reuse everywhere
- Reduces memory footprint and startup time
- Ensures consistency across components
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model name (can be overridden via config)
DEFAULT_EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"


@lru_cache(maxsize=1)
def get_sentence_transformer(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    """Get or create singleton SentenceTransformer model.

    This is used by components that need direct access to sentence-transformers
    (e.g., SemanticChunker for encoding during chunking).

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Cached SentenceTransformer instance

    Note:
        Uses lru_cache for singleton pattern. Same model_name returns same instance.
    """
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"SentenceTransformer loaded successfully")
    return model


@lru_cache(maxsize=1)
def get_langchain_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """Get or create singleton HuggingFaceEmbeddings for LangChain.

    This is used by components that need LangChain-compatible embeddings
    (e.g., FAISS vectorstore, LangChain retrievers).

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Cached HuggingFaceEmbeddings instance

    Note:
        Uses lru_cache for singleton pattern. Same model_name returns same instance.
        Internally wraps the same underlying model as get_sentence_transformer.
    """
    logger.info(f"Loading HuggingFaceEmbeddings model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    logger.info(f"HuggingFaceEmbeddings loaded successfully")
    return embeddings


def clear_embedding_cache() -> None:
    """Clear cached embedding models.

    Useful for testing or when switching models. Forces reload on next call.
    """
    get_sentence_transformer.cache_clear()
    get_langchain_embeddings.cache_clear()
    logger.info("Embedding model cache cleared")
