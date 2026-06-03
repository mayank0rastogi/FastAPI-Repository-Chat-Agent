"""Indexer Agent MCP Server — repository parsing and graph population."""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

from agents.indexer.tools import register_indexer_tools
from infrastructure.neo4j_client import Neo4jClient
from shared.config import get_indexer_settings
from shared.utils.logging import configure_logging, get_logger

settings = get_indexer_settings()
configure_logging(settings.log_level, agent_name="indexer")
logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Build the FastAPI + MCP application for the Indexer Agent."""
    mcp = FastMCP("indexer-agent", version="1.0.0")
    neo4j = Neo4jClient(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("indexer_starting", port=settings.port)
        await neo4j.connect()
        await neo4j.setup_schema()
        register_indexer_tools(mcp, settings, neo4j)
        logger.info("indexer_ready")
        yield
        await neo4j.close()
        logger.info("indexer_shutdown")

    app = FastAPI(title="Indexer Agent", lifespan=lifespan)

    @app.post("/query")
    async def query(body: dict[str, Any]) -> dict[str, Any]:
        """Handle query routing from the orchestrator."""
        intent = body.get("intent", "")
        query_str = body.get("query", "")
        entities = body.get("entities", [])

        # Route to appropriate tool based on intent
        from agents.indexer.tools import _jobs
        if "status" in query_str.lower():
            return {"index_jobs": list(_jobs.values())[-5:], "intent": intent}
        return {
            "message": "Indexer ready. Use /tools/index_repository to start indexing.",
            "intent": intent,
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy", "agent": "indexer"}

    @app.get("/statistics")
    async def statistics() -> dict[str, Any]:
        return await neo4j.get_statistics()

    app.mount("/mcp", mcp.sse_app())
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)