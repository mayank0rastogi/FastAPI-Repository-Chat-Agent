"""Gateway routers for chat, indexing, and health endpoints."""
from gateway.routers.chat import router as chat_router
from gateway.routers.index import router as index_router
from gateway.routers.health import router as health_router

__all__ = ["chat_router", "index_router", "health_router"]
