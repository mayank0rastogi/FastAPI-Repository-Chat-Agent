"""Unit tests for all gateway endpoints."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from gateway.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.orchestrator_url = "http://orchestrator:8001"
    s.indexer_url = "http://indexer:8002"
    s.graph_query_url = "http://graph-query:8003"
    s.code_analyst_url = "http://code-analyst:8004"
    s.agent_timeout_seconds = 60.0
    s.rate_limit_per_minute = 1000
    s.cors_origins = ["*"]
    s.ws_allowed_origins = []
    s.log_level = "WARNING"
    s.environment = "testing"
    s.port = 8000
    s.default_repo_url = "https://github.com/fastapi/fastapi.git"
    return s


@pytest.fixture
def client(app, mock_settings):
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.settings = mock_settings
        app.state.http_client = AsyncMock()
        yield c


# ── POST /api/chat ─────────────────────────────────────────────────────────────

class TestChatEndpoint:
    def test_chat_auto_generates_session_id(self, client, mock_settings):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "answer": "FastAPI is fast", "agents_used": ["graph_query"],
            "tokens_used": 100, "latency_ms": 500
        }
        mock_response.raise_for_status = MagicMock()
        client.app.state.http_client.post = AsyncMock(return_value=mock_response)

        resp = client.post("/api/chat", json={"message": "What is FastAPI?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "session_id" in body
        assert len(body["session_id"]) > 0

    def test_chat_preserves_provided_session_id(self, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "answer": "...", "agents_used": [], "tokens_used": 0, "latency_ms": 0
        }
        mock_response.raise_for_status = MagicMock()
        client.app.state.http_client.post = AsyncMock(return_value=mock_response)

        resp = client.post("/api/chat", json={
            "message": "Hello", "session_id": "my-session-123"
        })
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "my-session-123"

    def test_chat_returns_request_id_header(self, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"answer": "", "agents_used": [], "tokens_used": 0, "latency_ms": 0}
        mock_response.raise_for_status = MagicMock()
        client.app.state.http_client.post = AsyncMock(return_value=mock_response)

        resp = client.post("/api/chat", json={"message": "test"})
        assert "X-Request-ID" in resp.headers

    def test_chat_empty_message_returns_422(self, client):
        resp = client.post("/api/chat", json={"message": ""})
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"] == "validation_error"
        assert "field_errors" in body["details"]

    def test_chat_missing_message_returns_422(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_chat_streaming_returns_event_stream(self, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "answer": "Hello world", "agents_used": [], "tokens_used": 10, "latency_ms": 100
        }
        mock_response.raise_for_status = MagicMock()
        client.app.state.http_client.post = AsyncMock(return_value=mock_response)

        resp = client.post("/api/chat", json={"message": "test", "stream": True})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


# ── POST /api/index ────────────────────────────────────────────────────────────

class TestIndexEndpoint:
    def test_index_returns_job_id(self, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "job_id": "job-abc-123", "status": "pending",
            "repo_url": "https://github.com/fastapi/fastapi.git"
        }
        mock_response.raise_for_status = MagicMock()
        client.app.state.http_client.post = AsyncMock(return_value=mock_response)

        resp = client.post("/api/index", json={})
        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "pending"

    def test_index_incremental_flag_forwarded(self, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "job_id": "job-xyz", "status": "pending", "repo_url": ""
        }
        mock_response.raise_for_status = MagicMock()
        client.app.state.http_client.post = AsyncMock(return_value=mock_response)

        resp = client.post("/api/index", json={"incremental": True})
        assert resp.status_code == 202
        assert resp.json()["incremental"] is True

    def test_index_indexer_unavailable_returns_503(self, client):
        import httpx as _httpx
        client.app.state.http_client.post = AsyncMock(
            side_effect=_httpx.ConnectError("refused")
        )
        resp = client.post("/api/index", json={})
        assert resp.status_code == 503
        assert resp.json()["error"] == "agent_unavailable"


# ── GET /api/index/status/{job_id} ────────────────────────────────────────────

class TestIndexStatusEndpoint:
    def test_status_returns_progress(self, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "running", "progress": 45, "errors": [],
            "files_indexed": 120, "entities_created": 2400
        }
        mock_response.raise_for_status = MagicMock()
        client.app.state.http_client.post = AsyncMock(return_value=mock_response)

        resp = client.get("/api/index/status/job-abc-123")
        assert resp.status_code == 200
        assert resp.json()["progress"] == 45

    def test_status_unknown_job_returns_404(self, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "Job not found"}
        mock_response.raise_for_status = MagicMock()
        client.app.state.http_client.post = AsyncMock(return_value=mock_response)

        resp = client.get("/api/index/status/nonexistent-job")
        assert resp.status_code == 404
        assert resp.json()["error"] == "not_found"


# ── GET /api/agents/health ────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_all_agents_healthy(self, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy", "version": "1.0.0"}
        client.app.state.http_client.get = AsyncMock(return_value=mock_response)

        resp = client.get("/api/agents/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["overall"] == "healthy"
        assert set(body["agents"].keys()) == {"orchestrator", "indexer", "graph_query", "code_analyst"}

    def test_health_one_agent_down_returns_unhealthy(self, client):
        import httpx as _httpx

        call_count = 0
        async def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "orchestrator" in url:
                raise _httpx.ConnectError("refused")
            r = AsyncMock()
            r.status_code = 200
            r.json.return_value = {"status": "healthy"}
            return r

        client.app.state.http_client.get = mock_get
        resp = client.get("/api/agents/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["overall"] == "unhealthy"
        assert body["agents"]["orchestrator"]["status"] == "unhealthy"


# ── GET /api/graph/statistics ─────────────────────────────────────────────────

class TestGraphStatisticsEndpoint:
    def test_statistics_returns_counts(self, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "node_counts": {"Class": 120, "Function": 340, "Method": 890},
            "relationship_counts": {"CONTAINS": 500, "CALLS": 200},
        }
        mock_response.raise_for_status = MagicMock()
        client.app.state.http_client.get = AsyncMock(return_value=mock_response)

        resp = client.get("/api/graph/statistics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_nodes"] == 120 + 340 + 890
        assert body["total_relationships"] == 500 + 200


# ── GET /health ────────────────────────────────────────────────────────────────

class TestGatewayHealth:
    def test_gateway_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_security_headers_present(self, client):
        resp = client.get("/health")
        assert "X-Content-Type-Options" in resp.headers
        assert "X-Frame-Options" in resp.headers

    def test_correlation_id_returned(self, client):
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers
        assert resp.headers["X-Request-ID"].startswith("req-")