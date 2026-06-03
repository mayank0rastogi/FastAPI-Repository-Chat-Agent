"""FastAPI dependency injection helpers for the gateway."""
from __future__ import annotations

import httpx
from fastapi import Depends, Request

from shared.config import GatewaySettings, get_gateway_settings


def get_http_client(request: Request) -> httpx.AsyncClient:
    """Return the shared async HTTP client from app state."""
    return request.app.state.http_client


def get_settings(request: Request) -> GatewaySettings:
    """Return the gateway settings from app state."""
    return request.app.state.settings


def get_request_id(request: Request) -> str:
    """Return the correlation request ID attached by middleware."""
    return getattr(request.state, "request_id", "")