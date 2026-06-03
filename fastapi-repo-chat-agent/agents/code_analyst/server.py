"""Code Analyst Agent MCP Server — deep code understanding and pattern analysis."""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

from agents.code_analyst.tools import register_code_analyst_tools
from infrastructure.neo4j_client import Neo4jClient
from shared.config import get_code_analyst_settings
from shared.exceptions import CodeAnalystError, EntityNotFoundError
from shared.utils.logging import configure_logging, get_logger

settings = get_code_analyst_settings()
configure_logging(settings.log_level, agent_name="code_analyst")
logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Build the FastAPI + MCP application for the Code Analyst Agent."""
    mcp = FastMCP("code-analyst-agent", version="1.0.0")
    neo4j = Neo4jClient(settings)
    openai_client: AsyncOpenAI | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal openai_client
        logger.info("code_analyst_starting", port=settings.port)

        await neo4j.connect()
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        register_code_analyst_tools(mcp, settings, openai_client, neo4j)
        logger.info("code_analyst_ready", model=settings.analysis_model)

        yield

        await neo4j.close()
        logger.info("code_analyst_shutdown")

    app = FastAPI(title="Code Analyst Agent", lifespan=lifespan)

    # ── Main query endpoint called by orchestrator ────────────────────────────

    @app.post("/query")
    async def query(body: dict[str, Any]) -> dict[str, Any]:
        """Route orchestrator queries to the correct analyst tool.

        Dispatches based on intent and entity types to the most
        appropriate analysis tool.

        Args:
            body: Dict with session_id, query, intent, entities, prior_context.

        Returns:
            Analysis result dict enriched with agent metadata.
        """
        intent: str = body.get("intent", "general")
        query_str: str = body.get("query", "")
        entities: list[str] = body.get("entities", [])
        prior_context: dict[str, Any] = body.get("prior_context", {})

        logger.info(
            "code_analyst_dispatch",
            intent=intent,
            entities=entities,
            has_prior_context=bool(prior_context),
        )

        try:
            result: dict[str, Any] = {}

            if intent == "comparison" and len(entities) >= 2:
                result = await _direct_compare(entities[0], entities[1], neo4j, settings, openai_client)

            elif intent == "pattern_detection":
                result = await _direct_find_patterns(entities, neo4j, settings, openai_client)

            elif intent == "code_explanation" and entities:
                result = await _direct_explain(entities[0], neo4j, settings, openai_client)

            elif intent in ("lifecycle_analysis", "dependency_analysis") and entities:
                # For lifecycle — explain the primary entity deeply
                result = await _direct_explain(entities[0], neo4j, settings, openai_client)

            elif entities:
                # Default: analyze the primary entity
                entity_type = await _detect_entity_type(entities[0], neo4j)
                if entity_type == "Class":
                    result = await _direct_analyze_class(entities[0], neo4j, settings, openai_client)
                else:
                    result = await _direct_analyze_function(entities[0], neo4j, settings, openai_client)

            else:
                # No entities — broad pattern scan
                result = await _direct_find_patterns([], neo4j, settings, openai_client)

            return {
                "agent": "code_analyst",
                "intent": intent,
                "entities_analysed": entities,
                "data": result,
            }

        except EntityNotFoundError as exc:
            logger.warning("entity_not_found", error=str(exc))
            return {
                "agent": "code_analyst",
                "intent": intent,
                "data": {"message": str(exc)},
                "not_found": True,
            }
        except CodeAnalystError as exc:
            logger.error("code_analyst_error", error=str(exc))
            return {
                "agent": "code_analyst",
                "intent": intent,
                "error": str(exc),
                "data": {},
            }
        except Exception as exc:
            logger.error("unexpected_error", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "agent": "code_analyst"}

    app.mount("/mcp", mcp.sse_app())
    return app


# ── Direct call helpers ───────────────────────────────────────────────────────

async def _detect_entity_type(name: str, neo4j: Neo4jClient) -> str:
    """Quick Neo4j check to determine if entity is a Class, Function, or Method."""
    results = await neo4j.run_read(
        "MATCH (e {name: $name}) RETURN labels(e)[0] AS label LIMIT 1",
        {"name": name},
    )
    return results[0]["label"] if results else "Function"


async def _direct_analyze_function(
    name: str, neo4j: Neo4jClient, settings: Any, client: Any
) -> dict[str, Any]:
    from agents.code_analyst.tools import register_code_analyst_tools  # noqa: F401
    from mcp.server.fastmcp import FastMCP as _MCP
    import json
    from agents.code_analyst.prompts import ANALYZE_FUNCTION_PROMPT, SYSTEM_PROMPT

    results = await neo4j.run_read(
        "MATCH (e {name: $name}) RETURN e LIMIT 1", {"name": name}
    )
    if not results:
        raise EntityNotFoundError(name)
    entity = dict(results[0]["e"])
    source = entity.get("source_code") or ""
    if not source:
        return {"warning": f"No source stored for '{name}'"}

    prompt = ANALYZE_FUNCTION_PROMPT.format(
        file_path=entity.get("file_path", ""), line_start=entity.get("line_start", 0),
        line_end=entity.get("line_end", 0), source_code=source[:3500],
        params=json.dumps(entity.get("params", []), default=str),
        decorators=json.dumps(entity.get("decorators", []), default=str),
    )
    resp = await client.chat.completions.create(
        model=settings.analysis_model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=0.1, max_tokens=2500,
    )
    return json.loads(resp.choices[0].message.content or "{}")


async def _direct_analyze_class(
    name: str, neo4j: Neo4jClient, settings: Any, client: Any
) -> dict[str, Any]:
    import json
    from agents.code_analyst.prompts import ANALYZE_CLASS_PROMPT, SYSTEM_PROMPT

    results = await neo4j.run_read(
        "MATCH (c:Class {name: $name}) RETURN c LIMIT 1", {"name": name}
    )
    if not results:
        raise EntityNotFoundError(name, "Class")
    entity = dict(results[0]["c"])
    source = entity.get("source_code") or ""

    prompt = ANALYZE_CLASS_PROMPT.format(
        file_path=entity.get("file_path", ""), line_start=entity.get("line_start", 0),
        line_end=entity.get("line_end", 0), source_code=source[:3500],
        bases=json.dumps(entity.get("bases", []), default=str),
        decorators=json.dumps(entity.get("decorators", []), default=str),
        method_count=entity.get("method_count", 0),
    )
    resp = await client.chat.completions.create(
        model=settings.analysis_model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=0.1, max_tokens=3000,
    )
    return json.loads(resp.choices[0].message.content or "{}")


async def _direct_explain(
    name: str, neo4j: Neo4jClient, settings: Any, client: Any
) -> dict[str, Any]:
    import json
    from agents.code_analyst.prompts import EXPLAIN_IMPLEMENTATION_PROMPT, SYSTEM_PROMPT

    results = await neo4j.run_read(
        "MATCH (e {name: $name}) RETURN e, labels(e)[0] AS lbl LIMIT 1", {"name": name}
    )
    if not results:
        raise EntityNotFoundError(name)
    entity = dict(results[0]["e"])
    source = entity.get("source_code") or ""
    if not source:
        return {"warning": f"No source stored for '{name}'"}

    prompt = EXPLAIN_IMPLEMENTATION_PROMPT.format(
        file_path=entity.get("file_path", ""), entity_name=name,
        entity_type=results[0]["lbl"], source_code=source[:3500],
    )
    resp = await client.chat.completions.create(
        model=settings.analysis_model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=0.1, max_tokens=2500,
    )
    return json.loads(resp.choices[0].message.content or "{}")


async def _direct_compare(
    name_a: str, name_b: str, neo4j: Neo4jClient, settings: Any, client: Any
) -> dict[str, Any]:
    import json
    from agents.code_analyst.prompts import COMPARE_PROMPT, SYSTEM_PROMPT

    results = await neo4j.run_read(
        "MATCH (e) WHERE e.name IN [$a, $b] RETURN e, labels(e)[0] AS lbl",
        {"a": name_a, "b": name_b},
    )
    if len(results) < 2:
        return {"error": f"Could not find both entities: {name_a}, {name_b}"}

    a = dict(results[0]["e"]); b = dict(results[1]["e"])
    prompt = COMPARE_PROMPT.format(
        name_a=name_a, type_a=results[0]["lbl"],
        file_a=(a.get("file_path") or "").split("/")[-1],
        source_a=(a.get("source_code") or "")[:1800],
        name_b=name_b, type_b=results[1]["lbl"],
        file_b=(b.get("file_path") or "").split("/")[-1],
        source_b=(b.get("source_code") or "")[:1800],
    )
    resp = await client.chat.completions.create(
        model=settings.analysis_model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=0.1, max_tokens=2500,
    )
    return json.loads(resp.choices[0].message.content or "{}")


async def _direct_find_patterns(
    entity_names: list[str], neo4j: Neo4jClient, settings: Any, client: Any
) -> dict[str, Any]:
    import json
    from agents.code_analyst.prompts import FIND_PATTERNS_PROMPT, SYSTEM_PROMPT

    if entity_names:
        entities = await neo4j.run_read(
            "MATCH (e) WHERE e.name IN $names RETURN labels(e)[0] AS type, e.name AS name, e.bases AS bases, e.decorators AS decorators, e.source_code AS source LIMIT 20",
            {"names": entity_names},
        )
    else:
        entities = await neo4j.run_read(
            "MATCH (e)-[r]-() WHERE e:Class OR e:Function WITH e, count(r) AS rc ORDER BY rc DESC LIMIT 20 RETURN labels(e)[0] AS type, e.name AS name, e.bases AS bases, e.decorators AS decorators, e.source_code AS source"
        )

    summaries = [{"name": e["name"], "type": e["type"], "bases": e.get("bases") or [], "decorators": e.get("decorators") or [], "source_preview": (e.get("source") or "")[:300]} for e in entities]
    prompt = FIND_PATTERNS_PROMPT.format(entity_summaries=json.dumps(summaries, indent=2, default=str)[:5000])
    resp = await client.chat.completions.create(
        model=settings.analysis_model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=0.1, max_tokens=3000,
    )
    return json.loads(resp.choices[0].message.content or "{}")


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)