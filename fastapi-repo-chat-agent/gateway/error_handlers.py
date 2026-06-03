"""Uniform error handlers for all exception types in the gateway."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from shared.utils.logging import get_logger

logger = get_logger(__name__)

_ERROR_STATUS_MAP: dict[str, int] = {
    "agent_timeout": status.HTTP_504_GATEWAY_TIMEOUT,
    "agent_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
    "entity_not_found": status.HTTP_404_NOT_FOUND,
    "rate_limit_exceeded": status.HTTP_429_TOO_MANY_REQUESTS,
    "invalid_query": status.HTTP_400_BAD_REQUEST,
    "session_not_found": status.HTTP_404_NOT_FOUND,
}


def _error_response(
    request: Request,
    error_code: str,
    message: str,
    http_status: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a uniform error JSONResponse envelope."""
    return JSONResponse(
        status_code=http_status,
        content={
            "error": error_code,
            "message": message,
            "request_id": getattr(request.state, "request_id", ""),
            "details": details or {},
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI app.

    Handlers registered:
      - RequestValidationError → 422 with field-level details
      - StarletteHTTPException → uniform error envelope
      - ValueError → 400
      - Generic Exception → 500 (hides internals in production)
    """

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Convert Pydantic validation errors to a clean 422 response."""
        field_errors = [
            {
                "field": ".".join(str(loc) for loc in err["loc"]),
                "message": err["msg"],
                "type": err["type"],
            }
            for err in exc.errors()
        ]
        logger.warning(
            "request_validation_error",
            path=request.url.path,
            errors=field_errors,
            request_id=getattr(request.state, "request_id", ""),
        )
        return _error_response(
            request,
            error_code="validation_error",
            message="Request body failed validation.",
            http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"field_errors": field_errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Wrap Starlette HTTPExceptions in the uniform envelope."""
        error_code = "http_error"
        if exc.status_code == 404:
            error_code = "not_found"
        elif exc.status_code == 401:
            error_code = "unauthorized"
        elif exc.status_code == 403:
            error_code = "forbidden"
        elif exc.status_code == 504:
            error_code = "agent_timeout"
        elif exc.status_code == 503:
            error_code = "agent_unavailable"

        logger.warning(
            "http_exception",
            status_code=exc.status_code,
            path=request.url.path,
            request_id=getattr(request.state, "request_id", ""),
        )
        return _error_response(
            request,
            error_code=error_code,
            message=str(exc.detail),
            http_status=exc.status_code,
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request, exc: ValueError
    ) -> JSONResponse:
        """Handle ValueError as a 400 Bad Request."""
        logger.warning("value_error", error=str(exc), path=request.url.path)
        return _error_response(
            request,
            error_code="invalid_input",
            message=str(exc),
            http_status=status.HTTP_400_BAD_REQUEST,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all for unhandled exceptions — log internals, hide from client."""
        logger.error(
            "unhandled_exception",
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path,
            request_id=getattr(request.state, "request_id", ""),
        )
        return _error_response(
            request,
            error_code="internal_server_error",
            message="An unexpected error occurred. Check server logs with the request_id.",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )