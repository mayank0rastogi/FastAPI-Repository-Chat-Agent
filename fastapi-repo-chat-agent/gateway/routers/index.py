"""Index router — POST /api/index and GET /api/index/status/{job_id}."""
from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status

from gateway.dependencies import get_http_client, get_request_id, get_settings
from gateway.models import ErrorDetail, IndexRequest, IndexResponse, IndexStatusResponse
from shared.config import GatewaySettings
from shared.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["Indexing"])


@router.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Indexing job accepted — poll status endpoint for progress"},
        400: {"model": ErrorDetail, "description": "Invalid repository URL"},
        503: {"model": ErrorDetail, "description": "Indexer agent unavailable"},
    },
    summary="Trigger repository indexing",
    description="""
Submit a repository indexing job. Returns immediately with a `job_id`.

**Full index**: Clones the repository and indexes every Python file.
Takes 2–10 minutes depending on repository size.

**Incremental index**: Only re-indexes files changed since the last run.
Suitable for updating the knowledge graph after a `git pull`.

Poll `GET /api/index/status/{job_id}` to track progress.
""",
)
async def trigger_indexing(
    request: Request,
    body: IndexRequest,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    settings: Annotated[GatewaySettings, Depends(get_settings)],
    request_id: Annotated[str, Depends(get_request_id)],
) -> IndexResponse:
    """POST /api/index — submit a full or incremental indexing job."""
    logger.info(
        "index_request",
        repo_url=body.repo_url or settings.default_repo_url,
        incremental=body.incremental,
        request_id=request_id,
    )

    indexer_url = f"{settings.indexer_url}/tools/index_repository"
    payload = {
        "repo_url": body.repo_url or settings.default_repo_url,
        "incremental": body.incremental,
        "branch": body.branch,
    }

    try:
        resp = await client.post(
            indexer_url,
            json=payload,
            headers={"X-Request-ID": request_id},
            timeout=15.0,   # just submitting the job — short timeout
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Indexer agent did not accept the job within 15 seconds.",
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Indexer agent is unreachable. Is it running?",
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:200]
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Indexer returned {exc.response.status_code}: {detail}",
        )

    return IndexResponse(
        job_id=data["job_id"],
        status=data.get("status", "pending"),
        repo_url=data.get("repo_url", body.repo_url),
        incremental=body.incremental,
        request_id=request_id,
    )


@router.get(
    "/index/status/{job_id}",
    response_model=IndexStatusResponse,
    responses={
        200: {"description": "Job status"},
        404: {"model": ErrorDetail, "description": "Job ID not found"},
        503: {"model": ErrorDetail, "description": "Indexer agent unavailable"},
    },
    summary="Get indexing job status",
    description="""
Poll the status of a running or completed indexing job.

**Status values**:
- `pending` — queued, not started yet
- `running` — actively indexing files
- `completed` — all files indexed successfully
- `failed` — indexing failed (check `errors` field)

**Progress** is reported as 0–100 percentage complete.
""",
)
async def get_indexing_status(
    job_id: str,
    request: Request,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    settings: Annotated[GatewaySettings, Depends(get_settings)],
    request_id: Annotated[str, Depends(get_request_id)],
) -> IndexStatusResponse:
    """GET /api/index/status/{job_id} — poll indexing job progress."""
    try:
        resp = await client.post(
            f"{settings.indexer_url}/tools/get_index_status",
            json={"job_id": job_id},
            headers={"X-Request-ID": request_id},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Indexer agent timed out.")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Indexer agent is unreachable.")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Indexer error: {exc.response.text[:200]}",
        )

    if "error" in data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    return IndexStatusResponse(
        job_id=job_id,
        status=data.get("status", "unknown"),
        progress=data.get("progress", 0),
        errors=data.get("errors", []),
        files_indexed=data.get("files_indexed", 0),
        entities_created=data.get("entities_created", 0),
        started_at=data.get("started_at", ""),
        completed_at=data.get("completed_at", ""),
    )