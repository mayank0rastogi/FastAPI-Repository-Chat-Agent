"""Pytest configuration and shared fixtures for all tests."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ── Event Loop Configuration ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the entire test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ── Mock Settings Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def mock_neo4j_settings() -> MagicMock:
    """Create mock Neo4j settings."""
    settings = MagicMock()
    settings.neo4j_uri = "bolt://localhost:7687"
    settings.neo4j_username = "neo4j"
    settings.neo4j_password = "securepassword123"
    settings.neo4j_database = "neo4j"
    settings.neo4j_max_pool_size = 10
    return settings


@pytest.fixture
def mock_redis_settings() -> MagicMock:
    """Create mock Redis settings."""
    settings = MagicMock()
    settings.redis_url = "redis://localhost:6379/0"
    settings.redis_max_connections = 10
    settings.redis_session_ttl = 3600
    settings.redis_cache_ttl = 1800
    settings.max_messages_per_session = 200
    return settings


@pytest.fixture
def mock_gateway_settings() -> MagicMock:
    """Create mock Gateway settings."""
    settings = MagicMock()
    settings.host = "0.0.0.0"
    settings.port = 8000
    settings.environment = "testing"
    settings.log_level = "INFO"
    settings.orchestrator_url = "http://localhost:8001"
    settings.indexer_url = "http://localhost:8002"
    settings.graph_query_url = "http://localhost:8003"
    settings.code_analyst_url = "http://localhost:8004"
    settings.cors_origins = ["*"]
    settings.ws_allowed_origins = []
    settings.rate_limit_per_minute = 60
    settings.agent_timeout_seconds = 30.0
    settings.default_repo_url = "https://github.com/fastapi/fastapi.git"
    return settings


@pytest.fixture
def mock_orchestrator_settings() -> MagicMock:
    """Create mock Orchestrator settings."""
    settings = MagicMock()
    settings.host = "0.0.0.0"
    settings.port = 8001
    settings.openai_model = "gpt-4o-mini"
    settings.openai_api_key = "sk-test"
    settings.openai_max_tokens = 4096
    settings.synthesis_model = "gpt-4o"
    settings.analysis_model = "gpt-4o-mini"
    settings.agent_timeout_seconds = 30.0
    settings.max_parallel_agents = 3
    settings.context_window_messages = 5
    settings.redis_url = "redis://localhost:6379/0"
    settings.indexer_url = "http://localhost:8002"
    settings.graph_query_url = "http://localhost:8003"
    settings.code_analyst_url = "http://localhost:8004"
    return settings


@pytest.fixture
def mock_indexer_settings() -> MagicMock:
    """Create mock Indexer settings."""
    settings = MagicMock()
    settings.host = "0.0.0.0"
    settings.port = 8002
    settings.repo_url = "https://github.com/fastapi/fastapi.git"
    settings.repo_local_path = "/tmp/test_repo"
    settings.max_file_size_kb = 500
    settings.max_concurrent_files = 10
    settings.include_test_files = False
    settings.include_patterns = ["*.py"]
    settings.exclude_patterns = ["**/test_*.py"]
    return settings


@pytest.fixture
def mock_graph_query_settings() -> MagicMock:
    """Create mock Graph Query settings."""
    settings = MagicMock()
    settings.host = "0.0.0.0"
    settings.port = 8003
    settings.result_limit = 100
    settings.max_query_depth = 5
    return settings


@pytest.fixture
def mock_code_analyst_settings() -> MagicMock:
    """Create mock Code Analyst settings."""
    settings = MagicMock()
    settings.host = "0.0.0.0"
    settings.port = 8004
    settings.analysis_model = "gpt-4o"
    settings.openai_api_key = "sk-test"
    settings.max_source_chars = 3500
    settings.max_snippet_lines = 150
    settings.snippet_context_lines = 10
    return settings


# ── Mock Neo4j Client ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_neo4j_client() -> AsyncMock:
    """Create a mock Neo4j client."""
    client = AsyncMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.run_read = AsyncMock(return_value=[])
    client.run_write = AsyncMock(return_value=[])
    client.run_batch = AsyncMock()
    client.setup_schema = AsyncMock()
    client.get_statistics = AsyncMock(return_value={
        "node_counts": {"Class": 0, "Function": 0},
        "relationship_counts": {"CONTAINS": 0},
        "total_nodes": 0,
        "total_relationships": 0,
    })
    return client


# ── Mock Redis Client ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis() -> AsyncMock:
    """Create a mock Redis client."""
    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    redis.setex = AsyncMock()
    redis.delete = AsyncMock()
    redis.rpush = AsyncMock()
    redis.lrange = AsyncMock(return_value=[])
    redis.llen = AsyncMock(return_value=0)
    redis.ltrim = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.hset = AsyncMock()
    redis.sadd = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())
    redis.expire = AsyncMock()
    redis.close = AsyncMock()
    return redis


# ── Mock OpenAI Client ────────────────────────────────────────────────────────

@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Create a mock OpenAI client."""
    client = AsyncMock()
    
    # Mock chat completion response
    mock_choice = MagicMock()
    mock_choice.message.content = '{"intent": "general", "entities": []}'
    
    mock_usage = MagicMock()
    mock_usage.total_tokens = 100
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    return client


# ── Mock HTTP Client ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_http_client() -> AsyncMock:
    """Create a mock httpx AsyncClient."""
    client = AsyncMock()
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = MagicMock()
    
    client.get = AsyncMock(return_value=mock_response)
    client.post = AsyncMock(return_value=mock_response)
    return client


# ── Sample Data Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_python_source() -> str:
    """Return sample Python source code for testing."""
    return '''
"""Sample module for testing."""
from typing import Any


class BaseClass:
    """A base class."""
    pass


class MyClass(BaseClass):
    """A sample class for testing AST parsing.
    
    Attributes:
        name: The name of the instance.
    """
    
    def __init__(self, name: str) -> None:
        """Initialize MyClass.
        
        Args:
            name: The name to use.
        """
        self.name = name
    
    @property
    def display_name(self) -> str:
        """Return the display name."""
        return f"Display: {self.name}"
    
    async def async_method(self) -> dict[str, Any]:
        """An async method."""
        return {"name": self.name}


def helper_function(value: int) -> int:
    """A helper function.
    
    Args:
        value: Input value.
        
    Returns:
        Doubled value.
    """
    return value * 2


async def async_helper() -> None:
    """An async helper function."""
    pass
'''


@pytest.fixture
def sample_entity_data() -> dict[str, Any]:
    """Return sample entity data for testing."""
    return {
        "name": "MyClass",
        "entity_type": "Class",
        "file_path": "/path/to/file.py",
        "line_start": 10,
        "line_end": 30,
        "docstring": "A sample class.",
        "source_code": "class MyClass: pass",
        "bases": ["BaseClass"],
        "decorators": [],
    }
