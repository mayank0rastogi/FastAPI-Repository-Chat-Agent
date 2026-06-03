"""Gateway API request/response models with full OpenAPI documentation."""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── Shared error envelope ─────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """Uniform error response returned for all 4xx/5xx responses."""
    error: str
    message: str
    request_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = {"json_schema_extra": {
        "example": {
            "error": "agent_timeout",
            "message": "The graph_query agent did not respond within 60s",
            "request_id": "req-abc-123",
            "details": {"agent": "graph_query", "timeout_seconds": 60},
        }
    }}


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=8_000,
        description="The user's question about the FastAPI codebase.",
        examples=["How does FastAPI handle dependency injection?"],
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session identifier for multi-turn conversations. "
                    "Auto-generated on first request — reuse across turns.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    stream: bool = Field(
        default=False,
        description="Set true to receive a Server-Sent Events stream instead "
                    "of a single JSON response. Use /ws/chat for WebSocket streaming.",
    )
    response_style: Literal["brief", "detailed", "code-heavy"] = Field(
        default="detailed",
        description="Controls verbosity of the synthesised response.",
    )

    @field_validator("session_id", mode="before")
    @classmethod
    def default_session_id(cls, v: str | None) -> str:
        """Ensure session_id is always a non-empty string."""
        return v or str(uuid.uuid4())


class ChatResponse(BaseModel):
    """Response body for POST /api/chat (non-streaming)."""
    answer: str = Field(description="Synthesised answer from the agent system.")
    session_id: str = Field(description="Session ID — pass back on the next turn.")
    agents_used: list[str] = Field(
        default_factory=list,
        description="Which agents contributed to this response.",
    )
    tokens_used: int = Field(default=0, description="Total LLM tokens consumed.")
    latency_ms: float = Field(default=0.0, description="End-to-end response latency.")
    request_id: str = Field(default="", description="Correlation ID for tracing.")

    model_config = {"json_schema_extra": {
        "example": {
            "answer": "FastAPI's dependency injection works via `Depends()`...",
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "agents_used": ["graph_query", "code_analyst"],
            "tokens_used": 1240,
            "latency_ms": 1823.4,
            "request_id": "req-abc-123",
        }
    }}


# ── Indexing ──────────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    """Request body for POST /api/index."""
    repo_url: str = Field(
        default="",
        description="Git URL of the repository to index. "
                    "Defaults to the configured FastAPI repository.",
        examples=["https://github.com/fastapi/fastapi.git"],
    )
    incremental: bool = Field(
        default=False,
        description="If true, only re-index files changed since last run. "
                    "Use false for a complete fresh index.",
    )
    branch: str = Field(
        default="main",
        description="Git branch to index.",
        examples=["main", "master"],
    )


class IndexResponse(BaseModel):
    """Response body for POST /api/index."""
    job_id: str = Field(description="UUID to poll via GET /api/index/status/{job_id}.")
    status: str = Field(description="Initial job status — always 'pending' on creation.")
    repo_url: str = Field(description="Repository URL being indexed.")
    incremental: bool = Field(description="Whether this is an incremental index.")
    request_id: str = Field(default="")

    model_config = {"json_schema_extra": {
        "example": {
            "job_id": "7f3e4a12-...",
            "status": "pending",
            "repo_url": "https://github.com/fastapi/fastapi.git",
            "incremental": False,
            "request_id": "req-xyz-456",
        }
    }}


class IndexStatusResponse(BaseModel):
    """Response body for GET /api/index/status/{job_id}."""
    job_id: str
    status: str = Field(description="pending | running | completed | failed")
    progress: int = Field(description="Completion percentage 0–100.", ge=0, le=100)
    errors: list[str] = Field(default_factory=list)
    files_indexed: int = Field(default=0)
    entities_created: int = Field(default=0)
    started_at: str = Field(default="")
    completed_at: str = Field(default="")


# ── Health ────────────────────────────────────────────────────────────────────

class AgentHealthStatus(BaseModel):
    """Health status for a single agent."""
    status: Literal["healthy", "degraded", "unhealthy"]
    latency_ms: float = 0.0
    error: str = ""
    version: str = ""


class HealthResponse(BaseModel):
    """Response body for GET /api/agents/health."""
    overall: Literal["healthy", "degraded", "unhealthy"]
    agents: dict[str, AgentHealthStatus]
    gateway_version: str = "1.0.0"
    request_id: str = ""

    model_config = {"json_schema_extra": {
        "example": {
            "overall": "healthy",
            "agents": {
                "orchestrator": {"status": "healthy", "latency_ms": 12.4},
                "indexer": {"status": "healthy", "latency_ms": 9.1},
                "graph_query": {"status": "healthy", "latency_ms": 11.0},
                "code_analyst": {"status": "healthy", "latency_ms": 14.2},
            },
            "gateway_version": "1.0.0",
        }
    }}


# ── Graph statistics ──────────────────────────────────────────────────────────

class GraphStatisticsResponse(BaseModel):
    """Response body for GET /api/graph/statistics."""
    node_counts: dict[str, int] = Field(
        description="Count of nodes per label (File, Class, Function, etc.)"
    )
    relationship_counts: dict[str, int] = Field(
        description="Count of relationships per type (CONTAINS, CALLS, etc.)"
    )
    total_nodes: int = 0
    total_relationships: int = 0
    last_indexed: str = ""
    request_id: str = ""