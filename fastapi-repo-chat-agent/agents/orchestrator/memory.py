"""Redis-backed conversation memory with in-memory fallback."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from shared.utils.logging import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """Manages per-session conversation history with Redis persistence.

    Falls back to an in-memory dict if Redis is unavailable, ensuring
    the orchestrator stays functional in development without Redis.

    Args:
        redis_url: Redis connection URL.
        ttl_seconds: Session expiry time in seconds.
        max_messages: Maximum messages retained per session window.
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 3600,
        max_messages: int = 20,
    ) -> None:
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._max_messages = max_messages
        self._redis: Any = None
        self._fallback: dict[str, list[dict[str, Any]]] = {}

    async def connect(self) -> None:
        """Attempt Redis connection; silently fall back to in-memory on failure."""
        try:
            import redis.asyncio as aioredis
            self._redis = await aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=3,
            )
            await self._redis.ping()
            logger.info("conversation_memory_redis_connected", url=self._redis_url)
        except Exception as exc:
            logger.warning("conversation_memory_redis_unavailable", error=str(exc), fallback="in-memory")
            self._redis = None

    async def close(self) -> None:
        """Close Redis connection if open."""
        if self._redis:
            await self._redis.aclose()

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history.

        Args:
            session_id: Unique session identifier.
            role: Message role — "user" | "assistant" | "system".
            content: Message text content.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        key = f"session:{session_id}:messages"

        if self._redis:
            try:
                await self._redis.rpush(key, json.dumps(message))
                await self._redis.expire(key, self._ttl)
                # Trim to window size
                length = await self._redis.llen(key)
                if length > self._max_messages:
                    await self._redis.ltrim(key, length - self._max_messages, -1)
                return
            except Exception as exc:
                logger.warning("redis_write_failed", error=str(exc))

        # Fallback
        if session_id not in self._fallback:
            self._fallback[session_id] = []
        self._fallback[session_id].append(message)
        if len(self._fallback[session_id]) > self._max_messages:
            self._fallback[session_id] = self._fallback[session_id][-self._max_messages :]

    async def get_messages(
        self, session_id: str, last_n: int = 10
    ) -> list[dict[str, Any]]:
        """Retrieve the most recent N messages for a session.

        Args:
            session_id: Unique session identifier.
            last_n: Number of recent messages to return.

        Returns:
            List of message dicts with role, content, timestamp.
        """
        key = f"session:{session_id}:messages"

        if self._redis:
            try:
                raw = await self._redis.lrange(key, -last_n, -1)
                return [json.loads(r) for r in raw]
            except Exception as exc:
                logger.warning("redis_read_failed", error=str(exc))

        messages = self._fallback.get(session_id, [])
        return messages[-last_n:]

    async def session_exists(self, session_id: str) -> bool:
        """Check whether a session has any recorded history."""
        key = f"session:{session_id}:messages"
        if self._redis:
            try:
                return bool(await self._redis.exists(key))
            except Exception:
                pass
        return session_id in self._fallback

    async def clear_session(self, session_id: str) -> None:
        """Delete all messages for a session."""
        key = f"session:{session_id}:messages"
        if self._redis:
            try:
                await self._redis.delete(key)
            except Exception:
                pass
        self._fallback.pop(session_id, None)

    async def cache_agent_response(
        self, session_id: str, query_hash: str, response: dict[str, Any]
    ) -> None:
        """Cache an agent response to avoid redundant calls for identical queries.

        Args:
            session_id: Session context for the cache key.
            query_hash: Hash of the query string.
            response: Agent response dict to cache.
        """
        key = f"cache:{session_id}:{query_hash}"
        if self._redis:
            try:
                await self._redis.setex(key, 300, json.dumps(response))  # 5 min TTL
            except Exception:
                pass

    async def get_cached_response(
        self, session_id: str, query_hash: str
    ) -> dict[str, Any] | None:
        """Retrieve a cached agent response if available."""
        key = f"cache:{session_id}:{query_hash}"
        if self._redis:
            try:
                raw = await self._redis.get(key)
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        return None