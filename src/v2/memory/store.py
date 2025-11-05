"""
Redis Store for Long-term Memory in SereneAI V2
Implements LangGraph Store interface with Redis backend

Based on: https://docs.langchain.com/oss/python/langgraph/add-memory.md
"""

import json
import hashlib
from typing import Any, Optional
from datetime import datetime, timedelta

import redis.asyncio as aioredis
from langchain_core.stores import BaseStore

from ..core.config import get_settings


class RedisStore(BaseStore[str, dict]):
    """
    Redis-backed store for long-term memory

    Features:
    - Namespaced storage (e.g., ("memories", user_id))
    - TTL support
    - Batch operations
    - JSON serialization
    - Connection pooling
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        prefix: str = "sereneai:store:",
        default_ttl: Optional[int] = None,
    ):
        """
        Initialize Redis store

        Args:
            redis_url: Redis connection URL (defaults to settings)
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        settings = get_settings()
        self.redis_url = redis_url or settings.persistence.redis_url
        self.prefix = prefix
        self.default_ttl = default_ttl

        self.client: Optional[aioredis.Redis] = None

    async def initialize(self) -> None:
        """Initialize Redis connection"""
        if self.client is not None:
            return  # Already initialized

        self.client = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )

        # Test connection
        await self.client.ping()
        print(f" Redis store initialized: {self.redis_url}")

    async def close(self) -> None:
        """Close Redis connection"""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
            print(" Redis store closed")

    def _make_key(self, namespace: tuple[str, ...], key: str) -> str:
        """Create Redis key from namespace and key"""
        namespace_str = ":".join(namespace)
        return f"{self.prefix}{namespace_str}:{key}"

    async def mget(self, keys: list[str]) -> list[Optional[dict]]:
        """Get multiple values by keys"""
        if not self.client:
            raise RuntimeError("Store not initialized")

        if not keys:
            return []

        # Get values from Redis
        values = await self.client.mget(keys)

        # Deserialize
        results = []
        for value in values:
            if value is None:
                results.append(None)
            else:
                try:
                    results.append(json.loads(value))
                except json.JSONDecodeError:
                    results.append(None)

        return results

    async def mset(self, key_value_pairs: list[tuple[str, dict]]) -> None:
        """Set multiple key-value pairs"""
        if not self.client:
            raise RuntimeError("Store not initialized")

        if not key_value_pairs:
            return

        # Use pipeline for batch operations
        async with self.client.pipeline() as pipe:
            for key, value in key_value_pairs:
                serialized = json.dumps(value, default=str)
                pipe.set(key, serialized, ex=self.default_ttl)

            await pipe.execute()

    async def mdelete(self, keys: list[str]) -> None:
        """Delete multiple keys"""
        if not self.client:
            raise RuntimeError("Store not initialized")

        if not keys:
            return

        await self.client.delete(*keys)

    async def yield_keys(self, prefix: Optional[str] = None) -> list[str]:
        """List all keys matching prefix"""
        if not self.client:
            raise RuntimeError("Store not initialized")

        pattern = f"{prefix}*" if prefix else f"{self.prefix}*"
        keys = []

        async for key in self.client.scan_iter(match=pattern):
            keys.append(key)

        return keys

    # ============================================================================
    # HIGH-LEVEL MEMORY OPERATIONS
    # ============================================================================

    async def store_memory(
        self,
        user_id: str,
        memory_key: str,
        memory_data: dict,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a memory for a user

        Args:
            user_id: User identifier
            memory_key: Unique key for this memory
            memory_data: Memory content
            ttl: TTL in seconds (overrides default)
        """
        namespace = ("memories", user_id)
        key = self._make_key(namespace, memory_key)

        # Add metadata
        enriched_data = {
            "data": memory_data,
            "created_at": datetime.now().isoformat(),
            "user_id": user_id,
            "key": memory_key,
        }

        if not self.client:
            raise RuntimeError("Store not initialized")

        serialized = json.dumps(enriched_data, default=str)
        await self.client.set(key, serialized, ex=ttl or self.default_ttl)

    async def retrieve_memory(
        self,
        user_id: str,
        memory_key: str,
    ) -> Optional[dict]:
        """Retrieve a specific memory for a user"""
        namespace = ("memories", user_id)
        key = self._make_key(namespace, memory_key)

        if not self.client:
            raise RuntimeError("Store not initialized")

        value = await self.client.get(key)
        if value is None:
            return None

        try:
            memory = json.loads(value)
            return memory.get("data")
        except json.JSONDecodeError:
            return None

    async def list_user_memories(self, user_id: str) -> list[dict]:
        """List all memories for a user"""
        namespace = ("memories", user_id)
        pattern = self._make_key(namespace, "*")

        if not self.client:
            raise RuntimeError("Store not initialized")

        memories = []
        async for key in self.client.scan_iter(match=pattern):
            value = await self.client.get(key)
            if value:
                try:
                    memory = json.loads(value)
                    memories.append(memory)
                except json.JSONDecodeError:
                    continue

        return sorted(memories, key=lambda x: x.get("created_at", ""), reverse=True)

    async def delete_memory(self, user_id: str, memory_key: str) -> None:
        """Delete a specific memory"""
        namespace = ("memories", user_id)
        key = self._make_key(namespace, memory_key)

        if not self.client:
            raise RuntimeError("Store not initialized")

        await self.client.delete(key)

    async def store_user_preference(
        self,
        user_id: str,
        preference_key: str,
        preference_value: Any,
    ) -> None:
        """Store a user preference"""
        namespace = ("preferences", user_id)
        key = self._make_key(namespace, preference_key)

        if not self.client:
            raise RuntimeError("Store not initialized")

        data = {"value": preference_value, "updated_at": datetime.now().isoformat()}
        await self.client.set(key, json.dumps(data, default=str))

    async def get_user_preference(
        self,
        user_id: str,
        preference_key: str,
        default: Any = None,
    ) -> Any:
        """Get a user preference"""
        namespace = ("preferences", user_id)
        key = self._make_key(namespace, preference_key)

        if not self.client:
            raise RuntimeError("Store not initialized")

        value = await self.client.get(key)
        if value is None:
            return default

        try:
            data = json.loads(value)
            return data.get("value", default)
        except json.JSONDecodeError:
            return default

    async def cache_computation(
        self,
        cache_key: str,
        computation_result: dict,
        ttl: int = 3600,
    ) -> None:
        """Cache a computation result (e.g., RAG retrieval, embeddings)"""
        if not self.client:
            raise RuntimeError("Store not initialized")

        key = f"{self.prefix}cache:{cache_key}"
        await self.client.set(key, json.dumps(computation_result, default=str), ex=ttl)

    async def get_cached_computation(self, cache_key: str) -> Optional[dict]:
        """Get cached computation result"""
        if not self.client:
            raise RuntimeError("Store not initialized")

        key = f"{self.prefix}cache:{cache_key}"
        value = await self.client.get(key)

        if value is None:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    async def get_stats(self) -> dict:
        """Get Redis store statistics"""
        if not self.client:
            raise RuntimeError("Store not initialized")

        info = await self.client.info("memory")
        dbsize = await self.client.dbsize()

        return {
            "total_keys": dbsize,
            "used_memory": info.get("used_memory_human", "unknown"),
            "used_memory_peak": info.get("used_memory_peak_human", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
        }

    async def clear_namespace(self, namespace: tuple[str, ...]) -> int:
        """Clear all keys in a namespace"""
        if not self.client:
            raise RuntimeError("Store not initialized")

        namespace_str = ":".join(namespace)
        pattern = f"{self.prefix}{namespace_str}:*"

        deleted = 0
        async for key in self.client.scan_iter(match=pattern):
            await self.client.delete(key)
            deleted += 1

        return deleted


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_store_instance: Optional[RedisStore] = None


async def get_store() -> RedisStore:
    """Get or create singleton store instance"""
    global _store_instance

    if _store_instance is None:
        _store_instance = RedisStore()
        await _store_instance.initialize()

    return _store_instance


async def close_store() -> None:
    """Close singleton store instance"""
    global _store_instance

    if _store_instance is not None:
        await _store_instance.close()
        _store_instance = None
