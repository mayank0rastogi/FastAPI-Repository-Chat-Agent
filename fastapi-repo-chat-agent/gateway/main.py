"""FastAPI Gateway — production-grade external interface for the MCP multi-agent system."""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gateway.error_handlers import register_error_handlers
from gateway.middleware import CorrelationIDMiddleware, RateLimitMiddleware, SecurityHeadersMiddleware
from gateway.routers import chat, health, index
from shared.config import GatewaySettings, get_gateway_settings
from shared.utils.logging import configure_logging, get_logger

settings = get_gateway_settings()
configure_logging(settings.log_level, agent_name="gateway")
logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise and tear down the shared HTTP client pool."""
    logger.info("gateway_starting", port=settings.port, environment=settings.environment)

    # Shared HTTP client — connection pool reused across all requests
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=5.0,
            read=settings.agent_timeout_seconds,
            write=10.0,
            pool=5.0,
        ),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        headers={"User-Agent": "fastapi-chat-gateway/1.0.0"},
    )
    app.state.settings = settings

    logger.info("gateway_ready", orchestrator=settings.orchestrator_url)
    yield

    await app.state.http_client.aclose()
    logger.info("gateway_shutdown")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build and configure the FastAPI gateway application.

    Returns:
        Fully configured FastAPI application with all routers,
        middleware, and error handlers registered.
    """
    app = FastAPI(
        title="FastAPI Repository Chat Agent — Gateway",
        description="""
## Multi-Agent Repository Chat System

This gateway provides the external HTTP/WebSocket interface for a multi-agent
system that answers questions about the FastAPI codebase.

### Architecture

### Quick Start

1. **Index the repository**: `POST /api/index`
2. **Poll until complete**: `GET /api/index/status/{job_id}`
3. **Start chatting**: `POST /api/chat`

### Authentication

Currently open. Add `Authorization: Bearer <token>` support via the
`SECRET_KEY` environment variable for production deployments.
        """,
        version="1.0.0",
        contact={"name": "FastAPI Chat Agent", "url": "https://github.com/fastapi/fastapi"},
        license_info={"name": "MIT"},
        openapi_tags=[
            {"name": "Chat", "description": "Send messages and receive codebase answers"},
            {"name": "Indexing", "description": "Manage repository indexing jobs"},
            {"name": "Operations", "description": "Health checks and observability"},
        ],
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware (order matters — outermost executes first) ─────────────────
    # 1. CORS — must be first to handle preflight requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
    )
    # 2. Security headers on all responses
    app.add_middleware(SecurityHeadersMiddleware)
    # 3. Rate limiting by IP
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_per_minute,
    )
    # 4. Correlation ID — innermost, runs last pre-handler / first post-handler
    app.add_middleware(CorrelationIDMiddleware)

    # ── Error handlers ────────────────────────────────────────────────────────
    register_error_handlers(app)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(chat.router)
    app.include_router(index.router)
    app.include_router(health.router)

    # WebSocket is registered directly on the chat router (not /api prefix)
    # Re-register at root level for clean ws:// URL
    app.add_api_websocket_route("/ws/chat", chat.websocket_chat)

    # ── Utility endpoints ─────────────────────────────────────────────────────

    @app.get("/health", tags=["Operations"], include_in_schema=True)
    async def gateway_health(request: Request) -> JSONResponse:
        """Gateway self-health — does NOT check downstream agents."""
        return JSONResponse({
            "status": "healthy",
            "service": "fastapi-repo-chat-gateway",
            "version": "1.0.0",
            "request_id": getattr(request.state, "request_id", ""),
        })

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        """Redirect hint for root path."""
        return JSONResponse({
            "service": "FastAPI Repository Chat Agent",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
        })

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gateway.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
        access_log=False,   # handled by CorrelationIDMiddleware
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )