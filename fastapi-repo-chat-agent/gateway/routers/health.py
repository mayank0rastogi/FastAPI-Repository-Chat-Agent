"""Health and observability router — agents health + graph statistics."""
from __future__ import annotations

import asyncio
import time
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Request

from gateway.dependencies import get_http_client, get_request_id, get_settings
from gateway.models import AgentHealthStatus, ErrorDetail, GraphStatisticsResponse, HealthResponse
from shared.config import GatewaySettings
from shared.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["Operations"])


async def _check_agent(
    name: str,
    base_url: str,
    client: httpx.AsyncClient,
    timeout: float = 5.0,
) -> tuple[str, AgentHealthStatus]:
    """Check health of a single agent with an individual timeout.

    Args:
        name: Agent name (for logging and response key).
        base_url: Base URL of the agent service.
        client: Shared httpx client.
        timeout: Per-agent timeout in seconds.

    Returns:
        Tuple of (agent_name, AgentHealthStatus).
    """
    start = time.monotonic()
    try:
        resp = await asyncio.wait_for(
            client.get(f"{base_url}/health"),
            timeout=timeout,
        )
        latency_ms = (time.monotonic() - start) * 1000
        if resp.status_code == 200:
            body = resp.json()
            return name, AgentHealthStatus(
                status="healthy",
                latency_ms=round(latency_ms, 2),
                version=body.get("version", ""),
            )
        return name, AgentHealthStatus(
            status="degraded",
            latency_ms=round(latency_ms, 2),
            error=f"HTTP {resp.status_code}",
        )
    except asyncio.TimeoutError:
        latency_ms = (time.monotonic() - start) * 1000
        logger.warning("agent_health_timeout", agent=name, timeout=timeout)
        return name, AgentHealthStatus(
            status="unhealthy",
            latency_ms=round(latency_ms, 2),
            error=f"Timeout after {timeout}s",
        )
    except httpx.ConnectError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        logger.warning("agent_health_connect_error", agent=name, error=str(exc))
        return name, AgentHealthStatus(
            status="unhealthy",
            latency_ms=round(latency_ms, 2),
            error=f"Connection refused: {exc}",
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        logger.error("agent_health_error", agent=name, error=str(exc))
        return name, AgentHealthStatus(
            status="unhealthy",
            latency_ms=round(latency_ms, 2),
            error=str(exc)[:100],
        )


@router.get(
    "/agents/health",
    response_model=HealthResponse,
    summary="Health check for all agents",
    description="""
Check the health of all agents in the multi-agent system.

All four agents are checked **concurrently** with individual 5-second timeouts,
so this endpoint always responds within ~5 seconds regardless of agent slowness.

**Overall status**:
- `healthy` — all agents responded with HTTP 200
- `degraded` — at least one agent returned a non-200 response
- `unhealthy` — at least one agent is unreachable
""",
)
async def agents_health(
    request: Request,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    settings: Annotated[GatewaySettings, Depends(get_settings)],
    request_id: Annotated[str, Depends(get_request_id)],
) -> HealthResponse:
    """GET /api/agents/health — concurrent health check of all 4 agents."""
    agent_urls = {
        "orchestrator": settings.orchestrator_url,
        "indexer": settings.indexer_url,
        "graph_query": settings.graph_query_url,
        "code_analyst": settings.code_analyst_url,
    }

    # Run all health checks concurrently with individual 5s timeouts
    results = await asyncio.gather(
        *[
            _check_agent(name, url, client, timeout=5.0)
            for name, url in agent_urls.items()
        ]
    )

    agents: dict[str, AgentHealthStatus] = dict(results)

    statuses = {v.status for v in agents.values()}
    if statuses == {"healthy"}:
        overall = "healthy"
    elif "unhealthy" in statuses:
        overall = "unhealthy"
    else:
        overall = "degraded"

    logger.info(
        "health_check_complete",
        overall=overall,
        request_id=request_id,
        statuses={k: v.status for k, v in agents.items()},
    )

    return HealthResponse(
        overall=overall,
        agents=agents,
        gateway_version="1.0.0",
        request_id=request_id,
    )


@router.get(
    "/graph/statistics",
    response_model=GraphStatisticsResponse,
    responses={
        200: {"description": "Knowledge graph statistics"},
        503: {"model": ErrorDetail, "description": "Graph query agent unavailable"},
    },
    summary="Knowledge graph statistics",
    description="""
Returns node and relationship counts from the Neo4j knowledge graph.

**Node types**: File, Module, Class, Function, Method, Parameter,
Decorator, Import, Docstring

**Relationship types**: CONTAINS, IMPORTS, INHERITS_FROM, CALLS,
DECORATED_BY, HAS_PARAMETER, DOCUMENTED_BY, DEPENDS_ON

Use this to verify that `POST /api/index` completed successfully.
""",
)
async def graph_statistics(
    request: Request,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    settings: Annotated[GatewaySettings, Depends(get_settings)],
    request_id: Annotated[str, Depends(get_request_id)],
) -> GraphStatisticsResponse:
    """GET /api/graph/statistics — node and relationship counts from Neo4j."""
    try:
        resp = await client.get(
            f"{settings.graph_query_url}/statistics",
            headers={"X-Request-ID": request_id},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException:
        from fastapi import HTTPException
        raise HTTPException(status_code=504, detail="Graph query agent timed out.")
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"Graph query agent unavailable: {exc}",
        )

    node_counts: dict[str, int] = data.get("node_counts", {})
    rel_counts: dict[str, int] = data.get("relationship_counts", {})

    return GraphStatisticsResponse(
        node_counts=node_counts,
        relationship_counts=rel_counts,
        total_nodes=sum(node_counts.values()),
        total_relationships=sum(rel_counts.values()),
        last_indexed=data.get("last_indexed", ""),
        request_id=request_id,
    )