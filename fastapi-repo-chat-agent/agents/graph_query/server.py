"""Graph Query Agent MCP Server — knowledge graph traversal and queries."""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import FastMCP

from agents.graph_query.tools import register_graph_query_tools
from infrastructure.neo4j_client import Neo4jClient
from shared.config import get_graph_query_settings
from shared.exceptions import EntityNotFoundError, InvalidCypherQueryError
from shared.utils.logging import configure_logging, get_logger

settings = get_graph_query_settings()
configure_logging(settings.log_level, agent_name="graph_query")
logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Build the FastAPI + MCP application for the Graph Query Agent."""
    mcp = FastMCP("graph-query-agent", version="1.0.0")
    neo4j = Neo4jClient(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("graph_query_starting", port=settings.port)
        await neo4j.connect()
        register_graph_query_tools(mcp, settings, neo4j)
        logger.info("graph_query_ready")
        yield
        await neo4j.close()
        logger.info("graph_query_shutdown")

    app = FastAPI(title="Graph Query Agent", lifespan=lifespan)

    # ── Main query endpoint called by orchestrator ────────────────────────────

    @app.post("/query")
    async def query(body: dict[str, Any]) -> dict[str, Any]:
        """Route orchestrator queries to the correct graph tool.

        Dispatches based on query intent to the most appropriate tool.
        Also accepts prior_context from sequential orchestration.

        Args:
            body: Dict with session_id, query, intent, entities, prior_context.

        Returns:
            Tool result dict enriched with agent metadata.
        """
        intent: str = body.get("intent", "general")
        query_str: str = body.get("query", "")
        entities: list[str] = body.get("entities", [])
        prior_context: dict[str, Any] = body.get("prior_context", {})

        logger.info(
            "graph_query_dispatch",
            intent=intent,
            entities=entities,
            has_prior_context=bool(prior_context),
        )

        try:
            # Import tools module to call tool functions directly
            from agents.graph_query import tools as t

            # ── Intent-based dispatch ─────────────────────────────────────────
            if intent == "entity_lookup" and entities:
                result = await _call_find_entity(entities[0], neo4j, settings)

            elif intent == "dependency_analysis" and entities:
                result = await _call_get_dependencies(entities[0], neo4j, settings)

            elif intent == "relationship_query" and entities:
                # Extract relationship type from query text
                rel_type = _extract_relationship_type(query_str)
                result = await _call_find_related(entities[0], rel_type, neo4j, settings)

            elif intent == "pattern_detection":
                result = await _call_find_usage_patterns(neo4j)

            elif "import" in query_str.lower() and entities:
                result = await _call_trace_imports(entities[0], neo4j, settings)

            elif "depend" in query_str.lower() and entities:
                # "what depends on X" vs "what does X depend on"
                if "on" in query_str.lower() and entities:
                    result = await _call_get_dependents(entities[0], neo4j, settings)
                else:
                    result = await _call_get_dependencies(entities[0], neo4j, settings)

            elif entities:
                # Default: find the entity and its direct relationships
                result = await _call_find_entity(entities[0], neo4j, settings)

            else:
                # No entities — return top usage patterns
                result = await _call_find_usage_patterns(neo4j)

            return {
                "agent": "graph_query",
                "intent": intent,
                "entities_queried": entities,
                "data": result,
            }

        except EntityNotFoundError as exc:
            logger.warning("entity_not_found", error=str(exc))
            return {
                "agent": "graph_query",
                "intent": intent,
                "data": {"message": str(exc), "entities_queried": entities},
                "not_found": True,
            }
        except InvalidCypherQueryError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("graph_query_error", error=str(exc))
            return {
                "agent": "graph_query",
                "intent": intent,
                "error": str(exc),
                "data": {},
            }

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "agent": "graph_query"}

    @app.get("/statistics")
    async def statistics() -> dict[str, Any]:
        """Return knowledge graph node and relationship counts.

        Called by the gateway's GET /api/graph/statistics endpoint.
        """
        try:
            stats = await neo4j.get_statistics()
            return {"status": "ok", "graph": stats}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}")

    app.mount("/mcp", mcp.sse_app())
    return app


# ── Direct call helpers (avoid MCP transport overhead for HTTP routing) ────────

async def _call_find_entity(
    name: str, neo4j: Neo4jClient, settings: Any
) -> dict[str, Any]:
    results = await neo4j.run_read(
        """
        MATCH (e) WHERE toLower(e.name) CONTAINS toLower($name)
        OPTIONAL MATCH (e)<-[:CONTAINS]-(parent)
        OPTIONAL MATCH (e)<-[used:CALLS|INHERITS_FROM|IMPORTS]-()
        RETURN labels(e)[0] AS entity_type, e.name AS name,
               e.file_path AS file_path, e.line_start AS line_start,
               e.docstring AS docstring, e.decorators AS decorators,
               e.bases AS bases, e.return_type AS return_type,
               parent.name AS parent_name, count(used) AS usage_count
        ORDER BY usage_count DESC
        LIMIT $limit
        """,
        {"name": name, "limit": settings.result_limit},
    )
    return {"entities": results, "count": len(results), "search_term": name}


async def _call_get_dependencies(
    entity_name: str, neo4j: Neo4jClient, settings: Any
) -> dict[str, Any]:
    results = await neo4j.run_read(
        """
        MATCH (e {name: $name})-[r:IMPORTS|INHERITS_FROM|CALLS|DEPENDS_ON]->(dep)
        RETURN type(r) AS relationship, labels(dep)[0] AS dep_type,
               dep.name AS dep_name, dep.file_path AS dep_file
        ORDER BY relationship, dep_name
        LIMIT $limit
        """,
        {"name": entity_name, "limit": settings.result_limit},
    )
    return {"entity": entity_name, "dependencies": results, "count": len(results)}


async def _call_get_dependents(
    entity_name: str, neo4j: Neo4jClient, settings: Any
) -> dict[str, Any]:
    results = await neo4j.run_read(
        """
        MATCH (dep)-[r:IMPORTS|INHERITS_FROM|CALLS|DEPENDS_ON]->(e {name: $name})
        RETURN type(r) AS relationship, labels(dep)[0] AS dep_type,
               dep.name AS dep_name, dep.file_path AS dep_file
        ORDER BY relationship, dep_name
        LIMIT $limit
        """,
        {"name": entity_name, "limit": settings.result_limit},
    )
    return {"entity": entity_name, "dependents": results, "count": len(results)}


async def _call_find_related(
    entity_name: str, rel_type: str, neo4j: Neo4jClient, settings: Any
) -> dict[str, Any]:
    safe_rel = rel_type.upper()
    results = await neo4j.run_read(
        f"""
        MATCH (e {{name: $name}})-[:{safe_rel}]->(related)
        RETURN labels(related)[0] AS type, related.name AS name,
               related.file_path AS file_path
        UNION
        MATCH (related)-[:{safe_rel}]->(e {{name: $name}})
        RETURN labels(related)[0] AS type, related.name AS name,
               related.file_path AS file_path
        LIMIT $limit
        """,
        {"name": entity_name, "limit": settings.result_limit},
    )
    return {"entity": entity_name, "relationship": safe_rel, "related": results}


async def _call_trace_imports(
    module_name: str, neo4j: Neo4jClient, settings: Any
) -> dict[str, Any]:
    results = await neo4j.run_read(
        """
        MATCH (source)-[:IMPORTS*1..3]->(imported)
        WHERE source.name CONTAINS $module OR source.path CONTAINS $module
        RETURN imported.name AS imported, imported.module AS module,
               imported.file_path AS file_path
        LIMIT $limit
        """,
        {"module": module_name, "limit": settings.result_limit},
    )
    return {"module": module_name, "imports": results, "count": len(results)}


async def _call_find_usage_patterns(neo4j: Neo4jClient) -> dict[str, Any]:
    results = await neo4j.run_read(
        """
        MATCH (fn)<-[:CALLS]-(caller)
        WITH fn, count(caller) AS call_count
        WHERE call_count > 0
        RETURN labels(fn)[0] AS type, fn.name AS name,
               fn.file_path AS file, call_count
        ORDER BY call_count DESC LIMIT 15
        """
    )
    return {"pattern": "most_called", "results": results}


def _extract_relationship_type(query: str) -> str:
    """Heuristically extract a relationship type from a natural language query."""
    query_lower = query.lower()
    if "inherit" in query_lower or "subclass" in query_lower or "extend" in query_lower:
        return "INHERITS_FROM"
    if "call" in query_lower or "invoke" in query_lower or "use" in query_lower:
        return "CALLS"
    if "import" in query_lower:
        return "IMPORTS"
    if "decorator" in query_lower or "decorate" in query_lower:
        return "DECORATED_BY"
    if "parameter" in query_lower or "argument" in query_lower:
        return "HAS_PARAMETER"
    if "depend" in query_lower:
        return "DEPENDS_ON"
    return "CONTAINS"


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)