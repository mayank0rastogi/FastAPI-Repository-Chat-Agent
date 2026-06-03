"""Chat router — POST /api/chat and WebSocket /ws/chat."""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse

from gateway.dependencies import get_http_client, get_request_id, get_settings
from gateway.models import ChatRequest, ChatResponse, ErrorDetail
from shared.config import GatewaySettings
from shared.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["Chat"])

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _call_orchestrator(
    payload: dict[str, Any],
    client: httpx.AsyncClient,
    settings: GatewaySettings,
    request_id: str = "",
) -> dict[str, Any]:
    """Forward a query to the Orchestrator Agent.

    Args:
        payload: JSON body to send to /orchestrate.
        client: Shared httpx client.
        settings: Gateway settings.
        request_id: Correlation ID to forward as a header.

    Returns:
        Parsed JSON response from the orchestrator.

    Raises:
        HTTPException 504: On timeout.
        HTTPException 503: On connection error.
        HTTPException 502: On upstream HTTP error.
    """
    url = f"{settings.orchestrator_url}/orchestrate"
    headers = {"X-Request-ID": request_id} if request_id else {}
    try:
        resp = await client.post(
            url, json=payload, headers=headers,
            timeout=settings.agent_timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.TimeoutException:
        logger.error("orchestrator_timeout", request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Orchestrator agent timed out.",
        )
    except httpx.ConnectError:
        logger.error("orchestrator_unreachable", url=url)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator agent is unreachable.",
        )
    except httpx.HTTPStatusError as exc:
        logger.error("orchestrator_http_error", status=exc.response.status_code)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Orchestrator returned {exc.response.status_code}.",
        )


async def _stream_orchestrator(
    payload: dict[str, Any],
    client: httpx.AsyncClient,
    settings: GatewaySettings,
    request_id: str = "",
) -> AsyncGenerator[str, None]:
    """Stream the orchestrator response as Server-Sent Events.

    Fetches the full response then yields it in token-sized SSE chunks
    so the client gets progressive text delivery without a separate
    streaming endpoint on the orchestrator.

    Yields:
        SSE-formatted strings: ``data: <chunk>\\n\\n``
        Final event: ``data: [DONE]\\n\\n``
    """
    start = time.monotonic()
    data = await _call_orchestrator(payload, client, settings, request_id)
    answer: str = data.get("answer", "")

    # Yield metadata first as a special event
    meta = {
        "event": "meta",
        "session_id": payload.get("session_id", ""),
        "agents_used": data.get("agents_used", []),
        "request_id": request_id,
    }
    yield f"data: {json.dumps(meta)}\n\n"

    # Stream answer word-by-word with a small delay for natural feel
    words = answer.split(" ")
    buffer = ""
    for i, word in enumerate(words):
        buffer += word + (" " if i < len(words) - 1 else "")
        # Flush every 5 words or at end of sentence
        if len(buffer) >= 30 or word.endswith((".", "!", "?", "\n")):
            yield f"data: {json.dumps({'text': buffer})}\n\n"
            buffer = ""
            await asyncio.sleep(0.015)  # ~67 chunks/sec

    if buffer:
        yield f"data: {json.dumps({'text': buffer})}\n\n"

    latency_ms = (time.monotonic() - start) * 1000
    done_event = {
        "event": "done",
        "tokens_used": data.get("tokens_used", 0),
        "latency_ms": round(latency_ms, 2),
    }
    yield f"data: {json.dumps(done_event)}\n\n"
    yield "data: [DONE]\n\n"


# ── POST /api/chat ─────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        200: {"description": "Successful response (non-streaming)"},
        206: {"description": "Streaming response (SSE) when stream=true"},
        422: {"model": ErrorDetail, "description": "Validation error"},
        503: {"model": ErrorDetail, "description": "Agent unavailable"},
        504: {"model": ErrorDetail, "description": "Agent timeout"},
    },
    summary="Send a message to the repository chat agent",
    description="""
Send a natural-language question about the FastAPI codebase.

**Session management**: Include `session_id` from a previous response
to continue a multi-turn conversation. Omit it (or send an empty string)
to start a new session — one will be auto-generated and returned.

**Streaming**: Set `stream: true` to receive a Server-Sent Events stream.
Each event has a `data` field containing a JSON object with either
`text` (partial answer), `meta` (session/agent info), or `done` (final stats).
Connect with `EventSource` or `fetch()` with `ReadableStream`.

**Sample queries by complexity**:
- Simple: `"What is the FastAPI class?"`
- Medium: `"What classes inherit from APIRouter?"`
- Complex: `"Explain the complete lifecycle of a FastAPI request"`
""",
)
async def chat(
    request: Request,
    body: ChatRequest,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    settings: Annotated[GatewaySettings, Depends(get_settings)],
    request_id: Annotated[str, Depends(get_request_id)],
) -> ChatResponse | StreamingResponse:
    """POST /api/chat — main chat endpoint with session and streaming support."""
    logger.info(
        "chat_request",
        session_id=body.session_id,
        stream=body.stream,
        message_len=len(body.message),
        request_id=request_id,
    )

    payload = {
        "session_id": body.session_id,
        "query": body.message,
        "response_style": body.response_style,
        "request_id": request_id,
    }

    # ── Streaming path ────────────────────────────────────────────────────────
    if body.stream:
        return StreamingResponse(
            _stream_orchestrator(payload, client, settings, request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",     # disable nginx buffering
                "X-Request-ID": request_id,
                "Connection": "keep-alive",
            },
        )

    # ── Non-streaming path ────────────────────────────────────────────────────
    start = time.monotonic()
    data = await _call_orchestrator(payload, client, settings, request_id)
    latency_ms = (time.monotonic() - start) * 1000

    return ChatResponse(
        answer=data.get("answer", ""),
        session_id=body.session_id,
        agents_used=data.get("agents_used", []),
        tokens_used=data.get("tokens_used", 0),
        latency_ms=round(latency_ms, 2),
        request_id=request_id,
    )


# ── WebSocket /ws/chat ─────────────────────────────────────────────────────────

@router.websocket("/ws/chat")    # note: no /api prefix — websocket is at root
async def websocket_chat(
    websocket: WebSocket,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    settings: Annotated[GatewaySettings, Depends(get_settings)],
) -> None:
    """WebSocket /ws/chat — real-time bidirectional chat with streaming.

    **Protocol**:

    Client sends:
    ```json
    {"message": "How does FastAPI work?", "session_id": "..."}
    ```

    Server sends a sequence of frames:
    ```json
    {"type": "meta",  "session_id": "...", "agents_used": [...]}
    {"type": "chunk", "text": "FastAPI is a modern..."}
    {"type": "chunk", "text": " web framework..."}
    {"type": "done",  "tokens_used": 512, "latency_ms": 1234.5}
    ```

    On error:
    ```json
    {"type": "error", "code": "agent_timeout", "message": "..."}
    ```

    **Keepalive**: Server sends `{"type": "ping"}` every 30 seconds.
    Client should respond with `{"type": "pong"}` to keep the connection alive.
    """
    # Origin check — reject connections from disallowed origins
    origin = websocket.headers.get("origin", "")
    if settings.ws_allowed_origins and origin not in settings.ws_allowed_origins:
        logger.warning("ws_origin_rejected", origin=origin)
        await websocket.close(code=1008, reason="Origin not allowed")
        return

    await websocket.accept()
    ws_id = f"ws-{id(websocket):x}"
    logger.info("ws_connected", ws_id=ws_id, origin=origin)

    # Keepalive task
    async def _keepalive() -> None:
        while True:
            try:
                await asyncio.sleep(30)
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    keepalive_task = asyncio.create_task(_keepalive())

    try:
        while True:
            # Receive with timeout — disconnect idle clients after 5 minutes
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=300)
            except asyncio.TimeoutError:
                logger.info("ws_idle_timeout", ws_id=ws_id)
                await websocket.close(code=1000, reason="Idle timeout")
                break

            # Parse incoming message
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error", "code": "invalid_json",
                    "message": "Message must be valid JSON.",
                })
                continue

            # Handle pong keepalive reply
            if data.get("type") == "pong":
                continue

            message: str = data.get("message", "").strip()
            session_id: str = data.get("session_id", "") or str(__import__("uuid").uuid4())
            request_id = f"ws-{ws_id}-{int(time.monotonic() * 1000)}"

            if not message:
                await websocket.send_json({
                    "type": "error", "code": "empty_message",
                    "message": "message field is required and cannot be empty.",
                })
                continue

            logger.info(
                "ws_message_received",
                ws_id=ws_id, session_id=session_id,
                message_len=len(message),
            )

            start = time.monotonic()
            payload = {
                "session_id": session_id,
                "query": message,
                "request_id": request_id,
            }

            # Call orchestrator
            try:
                orchestrator_data = await _call_orchestrator(
                    payload, client, settings, request_id
                )
            except HTTPException as exc:
                await websocket.send_json({
                    "type": "error",
                    "code": "agent_error",
                    "message": exc.detail,
                    "request_id": request_id,
                })
                continue

            answer: str = orchestrator_data.get("answer", "")
            agents_used: list[str] = orchestrator_data.get("agents_used", [])

            # Send metadata frame
            await websocket.send_json({
                "type": "meta",
                "session_id": session_id,
                "agents_used": agents_used,
                "request_id": request_id,
            })

            # Stream answer word-by-word
            words = answer.split(" ")
            buffer = ""
            for i, word in enumerate(words):
                buffer += word + (" " if i < len(words) - 1 else "")
                if len(buffer) >= 40 or word.endswith((".", "!", "?", "\n")):
                    await websocket.send_json({"type": "chunk", "text": buffer})
                    buffer = ""
                    await asyncio.sleep(0.012)

            if buffer:
                await websocket.send_json({"type": "chunk", "text": buffer})

            # Send done frame
            latency_ms = (time.monotonic() - start) * 1000
            await websocket.send_json({
                "type": "done",
                "tokens_used": orchestrator_data.get("tokens_used", 0),
                "latency_ms": round(latency_ms, 2),
                "session_id": session_id,
            })

    except WebSocketDisconnect as exc:
        logger.info("ws_disconnected", ws_id=ws_id, code=exc.code)
    except Exception as exc:
        logger.error("ws_error", ws_id=ws_id, error=str(exc))
        try:
            await websocket.send_json({
                "type": "error", "code": "server_error",
                "message": "An unexpected error occurred.",
            })
        except Exception:
            pass
    finally:
        keepalive_task.cancel()
        logger.info("ws_cleanup", ws_id=ws_id)