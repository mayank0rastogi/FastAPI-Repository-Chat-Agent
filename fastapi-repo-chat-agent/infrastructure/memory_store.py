"""Redis-backed conversation memory store.

Manages:
  - Conversation history per session
  - Agent response cache (LRU with TTL)
  - Query routing decision log
  - User preferences and context
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import Redis
from tenacity import retry, stop_after_attempt, wait_exponential

from shared.exceptions import MemoryStoreError, SessionNotFoundError
from shared.models.base import (
    ConversationMessage,
    ConversationSession,
    RoutingDecision,
    UserPreferences,
)
from shared.utils.logging import get_logger

logger = get_logger(__name__)

# ── Redis key schema ──────────────────────────────────────────────────────────
# session:{session_id}                 → JSON blob (ConversationSession)
# session:{session_id}:messages        → Redis List of JSON-encoded messages
# session:{session_id}:preferences     → Redis Hash of preference fields
# cache:response:{query_hash}          → JSON blob of cached agent response
# routing:{session_id}                 → Redis List of JSON routing decisions
# context:{session_id}:entities        → Redis Set of entity names mentioned


class MemoryStore:
    """Async Redis-backed store for all conversation memory components.

    Provides:
      - Full session CRUD (create, get, update, delete)
      - Append-only message history with sliding TTL
      - Agent response cache with configurable TTL
      - Routing decision persistence
      - User preference management
      - Entity context tracking per session

    Example:
        >>> store = MemoryStore(settings)
        >>> await store.connect()
        >>> session = await store.get_or_create_session("session-abc")
        >>> await store.append_message("session-abc", ConversationMessage(...))
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._redis: Redis | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to Redis and verify connectivity.

        Raises:
            MemoryStoreError: If Redis is unreachable.
        """
        try:
            self._redis = aioredis.from_url(
                self._settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=self._settings.redis_max_connections,
                socket_connect_timeout=5,
            )
            await self._redis.ping()
            logger.info("redis_connected", url=self._settings.redis_url)
        except Exception as exc:
            raise MemoryStoreError(f"Redis connection failed: {exc}") from exc

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._redis:
            await self._redis.aclose()
            logger.info("redis_closed")

    def _r(self) -> Redis:
        """Return the Redis client, raising if not connected."""
        if not self._redis:
            raise MemoryStoreError("Redis not connected — call connect() first")
        return self._redis

    # ── Session management ────────────────────────────────────────────────────

    async def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Return existing session or create a new one.

        Args:
            session_id: Client-provided session identifier.

        Returns:
            Hydrated ConversationSession with message history.
        """
        key = f"session:{session_id}"
        raw = await self._r().get(key)

        if raw:
            try:
                data = json.loads(raw)
                session = ConversationSession.model_validate(data)
                # Refresh TTL on access (sliding expiration)
                await self._r().expire(key, self._settings.redis_session_ttl)
                return session
            except Exception as exc:
                logger.warning("session_deserialise_error", session_id=session_id, error=str(exc))

        # Create new session
        session = ConversationSession(session_id=session_id)
        await self._save_session(session)
        logger.info("session_created", session_id=session_id)
        return session

    async def save_session(self, session: ConversationSession) -> None:
        """Persist a full session object to Redis.

        Args:
            session: The session to save.
        """
        await self._save_session(session)

    async def delete_session(self, session_id: str) -> None:
        """Delete all Redis keys associated with a session.

        Args:
            session_id: Session to delete.
        """
        keys = await self._r().keys(f"session:{session_id}*")
        keys += await self._r().keys(f"routing:{session_id}")
        keys += await self._r().keys(f"context:{session_id}*")
        if keys:
            await self._r().delete(*keys)
        logger.info("session_deleted", session_id=session_id, keys_removed=len(keys))

    async def _save_session(self, session: ConversationSession) -> None:
        """Internal: serialise and store session with TTL."""
        session.last_active = datetime.utcnow()
        key = f"session:{session.session_id}"
        await self._r().setex(
            key,
            self._settings.redis_session_ttl,
            session.model_dump_json(),
        )

    # ── Message history ───────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.1, max=2))
    async def append_message(
        self, session_id: str, message: ConversationMessage
    ) -> None:
        """Append a message to the session's history list.

        Uses a Redis List with RPUSH for O(1) append. The list is
        capped at max_messages_per_session to prevent unbounded growth.

        Args:
            session_id: Target session.
            message: The message to append.
        """
        key = f"session:{session_id}:messages"
        await self._r().rpush(key, message.model_dump_json())
        # Cap list length — remove oldest if over limit
        await self._r().ltrim(
            key, -self._settings.max_messages_per_session, -1
        )
        await self._r().expire(key, self._settings.redis_session_ttl)

        # Also update session's last_active timestamp
        session_key = f"session:{session_id}"
        if await self._r().exists(session_key):
            raw = await self._r().get(session_key)
            if raw:
                data = json.loads(raw)
                data["last_active"] = datetime.utcnow().isoformat()
                await self._r().setex(
                    session_key, self._settings.redis_session_ttl, json.dumps(data)
                )

    async def get_messages(
        self, session_id: str, last_n: int | None = None
    ) -> list[ConversationMessage]:
        """Retrieve message history for a session.

        Args:
            session_id: Target session.
            last_n: If set, return only the most recent N messages.

        Returns:
            List of ConversationMessage in chronological order.
        """
        key = f"session:{session_id}:messages"
        start = -last_n if last_n else 0
        raw_list = await self._r().lrange(key, start, -1)

        messages = []
        for raw in raw_list:
            try:
                messages.append(ConversationMessage.model_validate(json.loads(raw)))
            except Exception as exc:
                logger.warning("message_parse_error", error=str(exc))
        return messages

    async def get_message_count(self, session_id: str) -> int:
        """Return the total number of messages stored for a session."""
        return await self._r().llen(f"session:{session_id}:messages")

    # ── Agent response cache ──────────────────────────────────────────────────

    @staticmethod
    def _cache_key(query: str, agent: str) -> str:
        """Derive a deterministic Redis key from query + agent name."""
        digest = hashlib.sha256(f"{agent}:{query}".encode()).hexdigest()[:16]
        return f"cache:response:{agent}:{digest}"

    async def cache_response(
        self,
        query: str,
        agent: str,
        response: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Cache an agent's response for a given query.

        Args:
            query: Original query string (used to derive cache key).
            agent: Agent type name (e.g. "graph_query").
            response: JSON-serialisable response dict.
            ttl: TTL in seconds. Defaults to settings.redis_cache_ttl.
        """
        key = self._cache_key(query, agent)
        payload = json.dumps(
            {"response": response, "cached_at": datetime.utcnow().isoformat()}
        )
        await self._r().setex(key, ttl or self._settings.redis_cache_ttl, payload)
        logger.debug("response_cached", agent=agent, key=key[-8:])

    async def get_cached_response(
        self, query: str, agent: str
    ) -> dict[str, Any] | None:
        """Look up a cached agent response.

        Args:
            query: Original query string.
            agent: Agent type name.

        Returns:
            Cached response dict, or None if not found / expired.
        """
        key = self._cache_key(query, agent)
        raw = await self._r().get(key)
        if not raw:
            return None
        try:
            data = json.loads(raw)
            logger.debug("cache_hit", agent=agent, key=key[-8:])
            return data["response"]
        except Exception:
            return None

    async def invalidate_cache(self, agent: str = "") -> int:
        """Delete cached responses, optionally filtered to one agent.

        Args:
            agent: If provided, only delete keys for this agent.
                   If empty, delete all response cache entries.

        Returns:
            Number of keys deleted.
        """
        pattern = f"cache:response:{agent}:*" if agent else "cache:response:*"
        keys = await self._r().keys(pattern)
        if keys:
            await self._r().delete(*keys)
        logger.info("cache_invalidated", agent=agent or "all", count=len(keys))
        return len(keys)

    # ── Query routing decisions ───────────────────────────────────────────────

    async def record_routing_decision(
        self, session_id: str, decision: RoutingDecision
    ) -> None:
        """Persist a routing decision for analytics and debugging.

        Stored as a capped Redis List per session so routing history
        is available for orchestrator tuning and observability.

        Args:
            session_id: Owning session.
            decision: The routing decision to record.
        """
        key = f"routing:{session_id}"
        await self._r().rpush(key, decision.model_dump_json())
        await self._r().ltrim(key, -50, -1)   # keep last 50 decisions
        await self._r().expire(key, self._settings.redis_session_ttl)

    async def get_routing_history(
        self, session_id: str, last_n: int = 10
    ) -> list[RoutingDecision]:
        """Return the most recent routing decisions for a session.

        Args:
            session_id: Target session.
            last_n: Number of decisions to return.

        Returns:
            List of RoutingDecision in chronological order.
        """
        key = f"routing:{session_id}"
        raw_list = await self._r().lrange(key, -last_n, -1)
        decisions = []
        for raw in raw_list:
            try:
                decisions.append(RoutingDecision.model_validate(json.loads(raw)))
            except Exception:
                pass
        return decisions

    # ── User preferences ──────────────────────────────────────────────────────

    async def get_preferences(self, session_id: str) -> UserPreferences:
        """Retrieve user preferences for a session.

        Args:
            session_id: Target session.

        Returns:
            UserPreferences (defaults if not set).
        """
        key = f"session:{session_id}:preferences"
        raw = await self._r().hgetall(key)
        if not raw:
            return UserPreferences()
        try:
            return UserPreferences.model_validate(raw)
        except Exception:
            return UserPreferences()

    async def set_preferences(
        self, session_id: str, preferences: UserPreferences
    ) -> None:
        """Save user preferences for a session.

        Args:
            session_id: Target session.
            preferences: Preference model to save.
        """
        key = f"session:{session_id}:preferences"
        flat: dict[str, str] = {}
        for field, value in preferences.model_dump().items():
            flat[field] = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        await self._r().hset(key, mapping=flat)
        await self._r().expire(key, self._settings.redis_session_ttl)

    async def update_preference(
        self, session_id: str, key: str, value: Any
    ) -> None:
        """Update a single preference field for a session.

        Args:
            session_id: Target session.
            key: Preference field name.
            value: New value (will be JSON-serialised if complex).
        """
        pref_key = f"session:{session_id}:preferences"
        serialised = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        await self._r().hset(pref_key, key, serialised)
        await self._r().expire(pref_key, self._settings.redis_session_ttl)

    # ── Entity context tracking ───────────────────────────────────────────────

    async def add_context_entities(
        self, session_id: str, entities: list[str]
    ) -> None:
        """Record code entities mentioned in this session for context retrieval.

        Args:
            session_id: Target session.
            entities: List of entity names (e.g. ["FastAPI", "APIRouter"]).
        """
        if not entities:
            return
        key = f"context:{session_id}:entities"
        await self._r().sadd(key, *entities)
        await self._r().expire(key, self._settings.redis_session_ttl)

    async def get_context_entities(self, session_id: str) -> list[str]:
        """Return all entity names mentioned in this session.

        Args:
            session_id: Target session.

        Returns:
            Deduplicated list of entity names.
        """
        key = f"context:{session_id}:entities"
        return list(await self._r().smembers(key))

    # ── Utility ───────────────────────────────────────────────────────────────

    async def get_store_stats(self) -> dict[str, Any]:
        """Return Redis memory and key statistics.

        Returns:
            Dict with total keys, memory usage, and hit/miss ratios.
        """
        info = await self._r().info("memory")
        session_keys = len(await self._r().keys("session:*"))
        cache_keys = len(await self._r().keys("cache:response:*"))
        routing_keys = len(await self._r().keys("routing:*"))

        return {
            "total_sessions": session_keys,
            "cached_responses": cache_keys,
            "routing_logs": routing_keys,
            "redis_memory_used": info.get("used_memory_human", "unknown"),
            "redis_peak_memory": info.get("used_memory_peak_human", "unknown"),
        }