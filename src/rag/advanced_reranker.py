"""Advanced reranking with ColBERT-style late interaction and temporal weighting.

ColBERT (Contextualized Late Interaction over BERT) provides better relevance
scoring by comparing query and document token embeddings with maximum similarity.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RerankScore(BaseModel):
    """Reranking score with breakdown."""

    doc_index: int
    semantic_score: float = Field(ge=0.0, le=1.0)
    temporal_score: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class AdvancedReranker:
    """ColBERT-inspired reranker with temporal weighting."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        temporal_weight: float = 0.2,
        semantic_weight: float = 0.8,
        cache_embeddings: bool = True,
    ):
        """Initialize reranker.

        Args:
            model_name: Sentence transformer model for embeddings
            temporal_weight: Weight for recency score (0-1)
            semantic_weight: Weight for semantic similarity (0-1)
            cache_embeddings: Whether to cache document embeddings
        """
        self.model_name = model_name
        self.temporal_weight = temporal_weight
        self.semantic_weight = semantic_weight
        self.cache_embeddings = cache_embeddings

        # Load model
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

        # Cache for document embeddings
        self.embedding_cache: Dict[str, torch.Tensor] = {}

        logger.info(
            f"AdvancedReranker: Initialized with {model_name}, "
            f"weights=(temporal={temporal_weight}, semantic={semantic_weight})"
        )

    def _load_model(self) -> None:
        """Lazy load sentence transformer model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"AdvancedReranker: Loaded model {self.model_name}")
            except Exception as e:
                logger.error(f"AdvancedReranker: Failed to load model - {e}")
                raise

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """Rerank documents using late interaction + temporal weighting.

        Args:
            query: Search query
            documents: Retrieved documents to rerank
            top_k: Number of top documents to return

        Returns:
            Reranked documents (top_k)
        """
        if not documents:
            logger.warning("AdvancedReranker: No documents to rerank")
            return []

        logger.info(f"AdvancedReranker: Reranking {len(documents)} documents for query '{query[:50]}...'")

        try:
            # Encode query
            query_embedding = self._encode_text(query)

            # Score each document
            scores: List[RerankScore] = []

            for i, doc in enumerate(documents):
                # Semantic score (late interaction approximation)
                semantic_score = self._compute_semantic_score(
                    query_embedding,
                    doc.page_content
                )

                # Temporal score (recency for PubMed)
                temporal_score = self._compute_temporal_score(doc.metadata)

                # Combined score
                final_score = (
                    self.semantic_weight * semantic_score +
                    self.temporal_weight * temporal_score
                )

                scores.append(RerankScore(
                    doc_index=i,
                    semantic_score=semantic_score,
                    temporal_score=temporal_score,
                    final_score=final_score,
                    reasoning=f"Semantic: {semantic_score:.3f}, Temporal: {temporal_score:.3f}"
                ))

            # Sort by final score
            scores.sort(key=lambda x: x.final_score, reverse=True)

            # Take top_k
            top_scores = scores[:top_k]

            # Log top results
            for rank, score in enumerate(top_scores[:3], 1):
                logger.debug(
                    f"AdvancedReranker: Rank {rank} - Doc {score.doc_index}, "
                    f"Score={score.final_score:.3f} ({score.reasoning})"
                )

            # Return reranked documents
            reranked_docs = [documents[score.doc_index] for score in top_scores]

            logger.info(f"AdvancedReranker: Returned top {len(reranked_docs)} documents")
            return reranked_docs

        except Exception as e:
            logger.error(f"AdvancedReranker: Reranking failed - {e}")
            # Fallback: return original order
            return documents[:top_k]

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding vector.

        Args:
            text: Text to encode

        Returns:
            Embedding tensor
        """
        # Check cache
        if self.cache_embeddings and text in self.embedding_cache:
            return self.embedding_cache[text]

        # Encode
        embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Cache if enabled
        if self.cache_embeddings:
            self.embedding_cache[text] = embedding

        return embedding

    def _compute_semantic_score(
        self,
        query_embedding: torch.Tensor,
        doc_content: str
    ) -> float:
        """Compute semantic similarity using cosine similarity.

        For true ColBERT, we'd compare token-level embeddings with max similarity.
        This is a simplified version using sentence embeddings.

        Args:
            query_embedding: Query embedding tensor
            doc_content: Document content text

        Returns:
            Similarity score (0-1)
        """
        try:
            # Encode document
            doc_embedding = self._encode_text(doc_content[:512])  # Limit length

            # Cosine similarity
            similarity = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                doc_embedding.unsqueeze(0)
            ).item()

            # Normalize to 0-1 (cosine is -1 to 1)
            normalized_score = (similarity + 1) / 2

            return max(0.0, min(1.0, normalized_score))

        except Exception as e:
            logger.warning(f"AdvancedReranker: Semantic scoring error - {e}")
            return 0.5  # Neutral score

    def _compute_temporal_score(self, metadata: Dict[str, Any]) -> float:
        """Compute recency score for document.

        Args:
            metadata: Document metadata with optional date fields

        Returns:
            Temporal score (0-1), where 1 is most recent
        """
        # Extract date from metadata
        date_fields = [
            'published_date',
            'publication_date',
            'date',
            'year',
        ]

        pub_date = None
        for field in date_fields:
            if field in metadata:
                try:
                    date_str = str(metadata[field])

                    # Parse different date formats
                    if len(date_str) == 4:  # Year only
                        pub_date = datetime(int(date_str), 1, 1)
                    else:
                        # Try ISO format
                        pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    break

                except Exception as e:
                    logger.debug(f"AdvancedReranker: Date parse error for {field}={date_str} - {e}")
                    continue

        # If no date found, neutral score
        if pub_date is None:
            return 0.5

        # Calculate recency score (exponential decay)
        # Recent papers (< 1 year) get high scores
        # Older papers get lower scores
        current_date = datetime.now()
        days_old = (current_date - pub_date).days

        # Exponential decay: score = exp(-days/1825) (5-year half-life)
        import math
        half_life_days = 1825  # 5 years
        temporal_score = math.exp(-days_old / half_life_days)

        logger.debug(f"AdvancedReranker: Temporal score for {pub_date.year} = {temporal_score:.3f}")

        return temporal_score

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.embedding_cache.clear()
        logger.info("AdvancedReranker: Cache cleared")