"""
Semantic Search for Long-term Memory
Embedding-based similarity search in Redis Store

Based on: https://docs.langchain.com/oss/python/langgraph/add-memory.md
"""

import hashlib
import numpy as np
from typing import Optional
from dataclasses import dataclass

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from .store import RedisStore, get_store
from ..core.config import get_settings


@dataclass
class MemorySearchResult:
    """Search result with relevance score"""

    memory_key: str
    memory_data: dict
    similarity_score: float
    created_at: str


class SemanticMemorySearch:
    """
    Semantic search over stored memories using embeddings

    Features:
    - Embedding-based similarity search
    - Caching of embeddings in Redis
    - Filtering by user_id
    - Top-k retrieval
    """

    def __init__(
        self,
        store: Optional[RedisStore] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        """
        Initialize semantic memory search

        Args:
            store: Redis store instance
            embeddings: Embedding model
        """
        self.store = store
        self.embeddings = embeddings
        self.settings = get_settings()

    async def initialize(self) -> None:
        """Initialize store and embeddings"""
        if self.store is None:
            self.store = await get_store()

        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.model.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        print(" Semantic memory search initialized")

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text with caching"""
        if not self.embeddings:
            raise RuntimeError("Embeddings not initialized")

        if not self.store:
            raise RuntimeError("Store not initialized")

        # Create cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"embedding:{text_hash}"

        # Check cache
        cached = await self.store.get_cached_computation(cache_key)
        if cached and "embedding" in cached:
            return cached["embedding"]

        # Compute embedding
        embedding = await self.embeddings.aembed_query(text)

        # Cache for 1 hour
        await self.store.cache_computation(
            cache_key, {"embedding": embedding}, ttl=3600
        )

        return embedding

    async def store_memory_with_embedding(
        self,
        user_id: str,
        memory_key: str,
        memory_text: str,
        memory_data: dict,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store memory with its embedding

        Args:
            user_id: User identifier
            memory_key: Unique memory key
            memory_text: Text to embed (summary of memory)
            memory_data: Full memory data
            ttl: Time to live in seconds
        """
        if not self.store:
            raise RuntimeError("Store not initialized")

        # Get embedding
        embedding = await self._get_embedding(memory_text)

        # Store memory with embedding
        enriched_data = {
            **memory_data,
            "embedding_text": memory_text,
            "embedding": embedding,
        }

        await self.store.store_memory(user_id, memory_key, enriched_data, ttl=ttl)

    async def search_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> list[MemorySearchResult]:
        """
        Search memories by semantic similarity

        Args:
            user_id: User to search memories for
            query: Search query
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of search results sorted by relevance
        """
        if not self.store:
            raise RuntimeError("Store not initialized")

        # Get query embedding
        query_embedding = await self._get_embedding(query)
        query_vec = np.array(query_embedding)

        # Get all user memories
        memories = await self.store.list_user_memories(user_id)

        # Compute similarities
        results = []
        for memory in memories:
            memory_data = memory.get("data", {})

            # Skip if no embedding
            if "embedding" not in memory_data:
                continue

            # Compute cosine similarity
            memory_vec = np.array(memory_data["embedding"])
            similarity = float(np.dot(query_vec, memory_vec))

            # Filter by threshold
            if similarity >= similarity_threshold:
                results.append(
                    MemorySearchResult(
                        memory_key=memory.get("key", ""),
                        memory_data=memory_data,
                        similarity_score=similarity,
                        created_at=memory.get("created_at", ""),
                    )
                )

        # Sort by similarity (descending) and limit to top_k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]

    async def get_relevant_context(
        self,
        user_id: str,
        query: str,
        max_results: int = 3,
    ) -> str:
        """
        Get relevant memory context as formatted string

        Useful for injecting into prompts
        """
        results = await self.search_memories(user_id, query, top_k=max_results)

        if not results:
            return "No relevant past memories found."

        context_parts = ["Relevant information from previous conversations:"]

        for i, result in enumerate(results, 1):
            memory_text = result.memory_data.get("embedding_text", "")
            timestamp = result.created_at
            score = result.similarity_score

            context_parts.append(
                f"\n{i}. [{timestamp}] (relevance: {score:.2f})\n{memory_text}"
            )

        return "\n".join(context_parts)


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_semantic_search_instance: Optional[SemanticMemorySearch] = None


async def get_semantic_search() -> SemanticMemorySearch:
    """Get or create singleton semantic search instance"""
    global _semantic_search_instance

    if _semantic_search_instance is None:
        _semantic_search_instance = SemanticMemorySearch()
        await _semantic_search_instance.initialize()

    return _semantic_search_instance
