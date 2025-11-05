"""
PostgreSQL Checkpointer for SereneAI V2
Implements LangGraph 1.0 checkpoint persistence with PostgreSQL

Based on: langgraph-checkpoint-postgres
Documentation: https://docs.langchain.com/oss/python/langgraph/persistence.md
"""

import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from ..core.config import get_settings


class SereneAICheckpointer:
    """
    Wrapper around AsyncPostgresSaver with connection management

    Features:
    - Connection pooling
    - Automatic schema creation
    - Health checks
    - Graceful shutdown
    """

    def __init__(self):
        self.settings = get_settings()
        self.pool: Optional[AsyncConnectionPool] = None
        self.checkpointer: Optional[AsyncPostgresSaver] = None

    async def initialize(self) -> None:
        """
        Initialize PostgreSQL connection pool and checkpointer

        Creates required tables automatically
        """
        if self.pool is not None:
            return  # Already initialized

        # Create connection pool
        self.pool = AsyncConnectionPool(
            conninfo=self.settings.persistence.postgres_url,
            min_size=2,
            max_size=10,
            timeout=30,
            max_idle=600,  # 10 minutes
            max_lifetime=1800,  # 30 minutes
        )

        # Wait for pool to be ready
        await self.pool.wait()

        # Create checkpointer with the pool
        self.checkpointer = AsyncPostgresSaver(self.pool)

        # Setup schema (creates tables if they don't exist)
        await self.checkpointer.setup()

        print(f" PostgreSQL checkpointer initialized: {self.settings.persistence.postgres_host}")

    async def health_check(self) -> bool:
        """Check if PostgreSQL connection is healthy"""
        if self.pool is None:
            return False

        try:
            async with self.pool.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            print(f" PostgreSQL health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close connection pool gracefully"""
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
            self.checkpointer = None
            print(" PostgreSQL checkpointer closed")

    def get_checkpointer(self) -> AsyncPostgresSaver:
        """Get the checkpointer instance (must be initialized first)"""
        if self.checkpointer is None:
            raise RuntimeError(
                "Checkpointer not initialized. Call initialize() first."
            )
        return self.checkpointer

    @asynccontextmanager
    async def lifespan(self):
        """Context manager for application lifespan"""
        await self.initialize()
        try:
            yield self
        finally:
            await self.close()


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_checkpointer_instance: Optional[SereneAICheckpointer] = None


async def get_checkpointer() -> AsyncPostgresSaver:
    """Get or create singleton checkpointer instance"""
    global _checkpointer_instance

    if _checkpointer_instance is None:
        _checkpointer_instance = SereneAICheckpointer()
        await _checkpointer_instance.initialize()

    return _checkpointer_instance.get_checkpointer()


async def close_checkpointer() -> None:
    """Close singleton checkpointer instance"""
    global _checkpointer_instance

    if _checkpointer_instance is not None:
        await _checkpointer_instance.close()
        _checkpointer_instance = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def cleanup_old_checkpoints(days: int = 30) -> int:
    """
    Delete checkpoints older than specified days

    Returns: Number of deleted checkpoints
    """
    checkpointer_wrapper = await get_checkpointer()

    query = """
    DELETE FROM checkpoints
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
    RETURNING checkpoint_id;
    """

    async with checkpointer_wrapper.pool.connection() as conn:  # type: ignore
        result = await conn.execute(query, (days,))
        deleted_count = len(await result.fetchall())

    print(f" Deleted {deleted_count} old checkpoints (older than {days} days)")
    return deleted_count


async def get_checkpoint_stats() -> dict:
    """Get statistics about stored checkpoints"""
    checkpointer_wrapper = _checkpointer_instance
    if checkpointer_wrapper is None or checkpointer_wrapper.pool is None:
        return {"error": "Checkpointer not initialized"}

    query = """
    SELECT
        COUNT(*) as total_checkpoints,
        COUNT(DISTINCT thread_id) as unique_threads,
        MIN(created_at) as oldest_checkpoint,
        MAX(created_at) as newest_checkpoint,
        pg_size_pretty(pg_total_relation_size('checkpoints')) as table_size
    FROM checkpoints;
    """

    async with checkpointer_wrapper.pool.connection() as conn:
        result = await conn.execute(query)
        row = await result.fetchone()

    return {
        "total_checkpoints": row[0] if row else 0,
        "unique_threads": row[1] if row else 0,
        "oldest_checkpoint": row[2].isoformat() if row and row[2] else None,
        "newest_checkpoint": row[3].isoformat() if row and row[3] else None,
        "table_size": row[4] if row else "0 bytes",
    }


# ============================================================================
# CLI COMMANDS (for maintenance)
# ============================================================================

async def main_cleanup():
    """CLI: Cleanup old checkpoints"""
    import sys

    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    await cleanup_old_checkpoints(days)


async def main_stats():
    """CLI: Show checkpoint statistics"""
    stats = await get_checkpoint_stats()
    print("\n=== Checkpoint Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        asyncio.run(main_cleanup())
    elif len(sys.argv) > 1 and sys.argv[1] == "stats":
        asyncio.run(main_stats())
    else:
        print("Usage: python checkpointer.py [cleanup|stats]")
