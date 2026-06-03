"""Unit tests for the Redis-backed MemoryStore."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infrastructure.memory_store import MemoryStore
from shared.models.base import ConversationMessage, RoutingDecision, UserPreferences


@pytest.fixture
def settings():
    s = MagicMock()
    s.redis_url = "redis://localhost:6379/0"
    s.redis_max_connections = 10
    s.redis_session_ttl = 3600
    s.redis_cache_ttl = 1800
    s.max_messages_per_session = 200
    return s


@pytest.fixture
def store(settings):
    return MemoryStore(settings)


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.ping = AsyncMock(return_value=True)
    r.get = AsyncMock(return_value=None)
    r.set = AsyncMock()
    r.setex = AsyncMock()
    r.rpush = AsyncMock()
    r.ltrim = AsyncMock()
    r.lrange = AsyncMock(return_value=[])
    r.llen = AsyncMock(return_value=0)
    r.hgetall = AsyncMock(return_value={})
    r.hset = AsyncMock()
    r.sadd = AsyncMock()
    r.smembers = AsyncMock(return_value=set())
    r.expire = AsyncMock()
    r.exists = AsyncMock(return_value=False)
    r.delete = AsyncMock()
    r.keys = AsyncMock(return_value=[])
    r.info = AsyncMock(return_value={"used_memory_human": "1MB", "used_memory_peak_human": "2MB"})
    return r


@pytest.mark.asyncio
async def test_get_or_create_session_new(store, mock_redis):
    store._redis = mock_redis
    session = await store.get_or_create_session("test-session")
    assert session.session_id == "test-session"
    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_get_or_create_session_existing(store, mock_redis):
    from shared.models.base import ConversationSession
    existing = ConversationSession(session_id="test-session")
    mock_redis.get = AsyncMock(return_value=existing.model_dump_json())
    store._redis = mock_redis
    session = await store.get_or_create_session("test-session")
    assert session.session_id == "test-session"
    # Should refresh TTL
    mock_redis.expire.assert_called()


@pytest.mark.asyncio
async def test_append_message(store, mock_redis):
    store._redis = mock_redis
    msg = ConversationMessage(role="user", content="How does FastAPI work?")
    await store.append_message("s1", msg)
    mock_redis.rpush.assert_called_once()
    mock_redis.ltrim.assert_called_once_with("session:s1:messages", -200, -1)


@pytest.mark.asyncio
async def test_get_messages_empty(store, mock_redis):
    store._redis = mock_redis
    messages = await store.get_messages("s1")
    assert messages == []


@pytest.mark.asyncio
async def test_cache_response_and_retrieve(store, mock_redis):
    import json
    cached = {"answer": "FastAPI is fast"}
    payload = json.dumps({"response": cached, "cached_at": datetime.utcnow().isoformat()})
    mock_redis.get = AsyncMock(return_value=payload)
    store._redis = mock_redis

    await store.cache_response("what is fastapi", "code_analyst", cached)
    mock_redis.setex.assert_called_once()

    result = await store.get_cached_response("what is fastapi", "code_analyst")
    assert result == cached


@pytest.mark.asyncio
async def test_cache_miss_returns_none(store, mock_redis):
    mock_redis.get = AsyncMock(return_value=None)
    store._redis = mock_redis
    result = await store.get_cached_response("unknown query", "graph_query")
    assert result is None


@pytest.mark.asyncio
async def test_record_routing_decision(store, mock_redis):
    store._redis = mock_redis
    decision = RoutingDecision(
        query_hash="abc123",
        intent="entity_lookup",
        agents_selected=["graph_query"],
    )
    await store.record_routing_decision("s1", decision)
    mock_redis.rpush.assert_called()


@pytest.mark.asyncio
async def test_get_preferences_default(store, mock_redis):
    mock_redis.hgetall = AsyncMock(return_value={})
    store._redis = mock_redis
    prefs = await store.get_preferences("s1")
    assert prefs.response_style == "detailed"
    assert prefs.include_line_numbers is True


@pytest.mark.asyncio
async def test_set_preferences(store, mock_redis):
    store._redis = mock_redis
    prefs = UserPreferences(response_style="brief", include_line_numbers=False)
    await store.set_preferences("s1", prefs)
    mock_redis.hset.assert_called_once()
    mock_redis.expire.assert_called()


@pytest.mark.asyncio
async def test_add_and_get_context_entities(store, mock_redis):
    mock_redis.smembers = AsyncMock(return_value={"FastAPI", "APIRouter", "Depends"})
    store._redis = mock_redis
    await store.add_context_entities("s1", ["FastAPI", "APIRouter", "Depends"])
    mock_redis.sadd.assert_called_once()
    entities = await store.get_context_entities("s1")
    assert "FastAPI" in entities


@pytest.mark.asyncio
async def test_delete_session_cleans_all_keys(store, mock_redis):
    mock_redis.keys = AsyncMock(side_effect=[
        ["session:s1", "session:s1:messages", "session:s1:preferences"],
        ["routing:s1"],
        ["context:s1:entities"],
    ])
    store._redis = mock_redis
    await store.delete_session("s1")
    mock_redis.delete.assert_called()


@pytest.mark.asyncio
async def test_get_store_stats(store, mock_redis):
    mock_redis.keys = AsyncMock(side_effect=[["s1", "s2"], ["c1"], ["r1"]])
    store._redis = mock_redis
    stats = await store.get_store_stats()
    assert stats["total_sessions"] == 2
    assert stats["cached_responses"] == 1
    assert "redis_memory_used" in stats