"""Unit tests for the custom exception hierarchy."""
import pytest

from shared.exceptions import (
    AgentTimeoutError,
    CodeAnalystError,
    EntityNotFoundError,
    GraphQueryError,
    IndexerError,
    InvalidCypherQueryError,
    LLMProviderError,
    Neo4jConnectionError,
    OrchestratorError,
    RepositoryNotFoundError,
    SessionNotFoundError,
)


def test_orchestrator_error_str() -> None:
    exc = OrchestratorError("test error")
    assert "orchestrator" in str(exc)
    assert "test error" in str(exc)


def test_entity_not_found_message() -> None:
    exc = EntityNotFoundError("MyClass", "Class")
    assert "MyClass" in str(exc)


def test_invalid_cypher_truncates_long_query() -> None:
    long_query = "CREATE " + "x" * 200
    exc = InvalidCypherQueryError(long_query, "Write not allowed")
    assert len(str(exc)) < 300


def test_agent_timeout_contains_seconds() -> None:
    exc = AgentTimeoutError("graph_query", 30.0)
    assert "30.0" in str(exc)


def test_repo_not_found_includes_url() -> None:
    exc = RepositoryNotFoundError("https://github.com/example/repo")
    assert "https://github.com/example/repo" in str(exc)


def test_session_not_found_includes_id() -> None:
    exc = SessionNotFoundError("abc-123")
    assert "abc-123" in str(exc)


def test_llm_provider_error_defaults() -> None:
    exc = LLMProviderError("rate limit exceeded")
    assert exc.agent_type == "openai"
    assert "rate limit" in exc.message