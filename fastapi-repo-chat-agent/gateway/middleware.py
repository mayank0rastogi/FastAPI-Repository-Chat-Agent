"""Gateway middleware: correlation IDs, request logging, security headers, rate limiting."""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from shared.utils.logging import get_logger

logger = get_logger(__name__)

# ── Correlation ID Middleware ─────────────────────────────────────────────────

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID to every request and response.

    Uses the client-provided X-Request-ID header if present,
    otherwise generates a new UUID. The ID is available via
    `request.state.request_id` throughout the request lifecycle
    and is propagated to all upstream agent calls.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Correlation-ID")
            or f"req-{uuid.uuid4().hex[:12]}"
        )
        request.state.request_id = request_id
        start = time.monotonic()
        response = await call_next(request)
        latency_ms = (time.monotonic() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(round(latency_ms, 2))

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
            request_id=request_id,
            client_ip=request.client.host if request.client else "unknown",
        )
        return response


# ── Security Headers Middleware ───────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every HTTP response.

    Headers applied:
      - X-Content-Type-Options: nosniff
      - X-Frame-Options: DENY
      - X-XSS-Protection: 1; mode=block
      - Referrer-Policy: strict-origin-when-cross-origin
      - Permissions-Policy: restrict browser features
      - Strict-Transport-Security: HSTS for HTTPS deployments
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return response


# ── Rate Limiting Middleware ──────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by client IP address.

    Limits: configurable requests-per-minute per IP.
    Returns 429 with Retry-After header when exceeded.

    Note: This is an in-process limiter suitable for single-instance
    deployments. For multi-instance, use Redis-backed slowapi instead.
    """

    def __init__(self, app: ASGIApp, requests_per_minute: int = 60) -> None:
        super().__init__(app)
        self._rpm = requests_per_minute
        self._window = 60.0  # seconds
        # ip -> list of request timestamps in current window
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/api/agents/health"):
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        # Evict timestamps outside the sliding window
        bucket = self._buckets[ip]
        self._buckets[ip] = [t for t in bucket if now - t < self._window]

        if len(self._buckets[ip]) >= self._rpm:
            oldest = self._buckets[ip][0]
            retry_after = max(1, int(self._window - (now - oldest)))
            logger.warning("rate_limit_exceeded", ip=ip, rpm=self._rpm)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests. Limit: {self._rpm} per minute.",
                    "request_id": getattr(request.state, "request_id", ""),
                    "details": {"retry_after_seconds": retry_after},
                },
                headers={"Retry-After": str(retry_after)},
            )

        self._buckets[ip].append(now)
        return await call_next(request)