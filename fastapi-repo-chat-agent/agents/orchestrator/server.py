"""Orchestrator Agent MCP Server — production wiring with lifespan management."""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

from agents.orchestrator.memory import ConversationMemory
from agents.orchestrator.tools import register_orchestrator_tools
from shared.config import get_orchestrator_settings
from shared.utils.logging import configure_logging, get_logger

settings = get_orchestrator_settings()
configure_logging(settings.log_level, agent_name="orchestrator")
logger = get_logger(__name__)

# Services — initialized in lifespan
_memory: ConversationMemory | None = None
_openai_client: AsyncOpenAI | None = None


def create_app() -> FastAPI:
    """Build the FastAPI + MCP application for the orchestrator agent.

    Returns:
        Configured FastAPI application with MCP and HTTP routes mounted.
    """
    mcp = FastMCP("orchestrator-agent", version="1.0.0")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Initialize OpenAI client, Redis memory, and register MCP tools."""
        global _memory, _openai_client
        logger.info("orchestrator_starting", host=settings.host, port=settings.port)

        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        _memory = ConversationMemory(
            redis_url=settings.redis_url,
            ttl_seconds=settings.redis_ttl_seconds,
            max_messages=settings.context_window_messages,
        )
        await _memory.connect()

        agent_urls: dict[str, str] = {
            "graph_query": settings.graph_query_url if hasattr(settings, "graph_query_url")
                           else "http://graph-query:8003",
            "code_analyst": settings.code_analyst_url if hasattr(settings, "code_analyst_url")
                            else "http://code-analyst:8004",
            "indexer": settings.indexer_url if hasattr(settings, "indexer_url")
                       else "http://indexer:8002",
        }

        register_orchestrator_tools(mcp, settings, _openai_client, _memory, agent_urls)
        logger.info("orchestrator_ready", agent_urls=agent_urls)

        yield

        logger.info("orchestrator_shutting_down")
        await _memory.close()

    app = FastAPI(title="Orchestrator Agent", lifespan=lifespan)

    # ── Main orchestration HTTP endpoint (called by gateway) ─────────────────
    @app.post("/orchestrate")
    async def orchestrate(body: dict[str, Any]) -> dict[str, Any]:
        """Full orchestration pipeline: analyze → route → synthesize.

        Args:
            body: Dict with session_id and query keys.

        Returns:
            Synthesized response dict.
        """
        session_id: str = body.get("session_id", "")
        query: str = body.get("query", "")

        if not query:
            raise HTTPException(status_code=422, detail="query is required")

        # Step 1: Get conversation context
        context = await _memory.get_messages(session_id, last_n=settings.context_window_messages)

        # Step 2: Analyze query intent
        from agents.orchestrator.tools import register_orchestrator_tools  # noqa: F401
        # Access registered tools via MCP directly
        analysis = await _analyze_query_direct(session_id, query)

        # Step 3: Route to agents
        routing_result = await _route_direct(
            session_id=session_id,
            query=query,
            intent=analysis.get("intent", "general"),
            entities=analysis.get("entities", []),
        )

        # Step 4: Synthesize
        from agents.orchestrator.tools import _call_single_agent  # noqa: F401
        final = await _synthesize_direct(
            session_id=session_id,
            original_query=query,
            agent_responses=routing_result.get("agent_responses", {}),
            context_messages=context,
        )

        return {
            **final,
            "query_analysis": analysis,
            "routing": routing_result.get("routing_plan"),
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "agent": "orchestrator"}

    # Mount MCP SSE server at /mcp
    app.mount("/mcp", mcp.sse_app())
    return app


# ── Direct function call wrappers (bypass MCP transport for internal use) ─────

async def _analyze_query_direct(session_id: str, query: str) -> dict[str, Any]:
    """Call analyze_query logic directly without MCP transport overhead."""
    import json
    from shared.models.base import QueryIntent

    if not _openai_client:
        return {"intent": "general", "entities": [], "complexity": "medium", "requires_agents": ["graph_query", "code_analyst"]}

    history = await _memory.get_messages(session_id, last_n=3)  # type: ignore[union-attr]
    history_str = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)

    try:
        resp = await _openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": """Classify the FastAPI codebase query. Return JSON only:
{"intent": "code_explanation|dependency_analysis|pattern_detection|entity_lookup|relationship_query|lifecycle_analysis|comparison|general",
 "entities": ["code entity names"],
 "complexity": "simple|medium|complex",
 "requires_agents": ["graph_query", "code_analyst"]}"""},
                {"role": "user", "content": f"History:\n{history_str}\n\nQuery: {query}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:
        logger.error("analyze_query_direct_failed", error=str(exc))
        return {"intent": "general", "entities": [], "complexity": "medium",
                "requires_agents": ["graph_query", "code_analyst"]}


async def _route_direct(
    session_id: str, query: str, intent: str, entities: list[str]
) -> dict[str, Any]:
    """Route to agents and return combined responses."""
    from agents.orchestrator.router import build_routing_plan, ExecutionMode
    from agents.orchestrator.tools import _call_agents_parallel, _call_agents_sequential

    agent_urls: dict[str, str] = {
        "graph_query": "http://graph-query:8003",
        "code_analyst": "http://code-analyst:8004",
    }

    plan = build_routing_plan(intent, entities)
    payload = {"session_id": session_id, "query": query, "intent": intent, "entities": entities}

    if plan.mode == ExecutionMode.PARALLEL:
        responses = await _call_agents_parallel(plan.agents, payload, plan.fallback_agents, settings, agent_urls)
    else:
        responses = await _call_agents_sequential(plan.agents, payload, plan.fallback_agents, settings, agent_urls)

    all_failed = all(r.get("error") for r in responses.values())
    if all_failed and plan.fallback_agents:
        fallback_resp = await _call_agents_parallel(
            plan.fallback_agents, {**payload, "is_fallback": True}, [], settings, agent_urls
        )
        responses.update(fallback_resp)

    return {
        "agent_responses": responses,
        "routing_plan": {"agents": plan.agents, "mode": plan.mode.value, "reasoning": plan.reasoning},
    }


async def _synthesize_direct(
    session_id: str,
    original_query: str,
    agent_responses: dict[str, Any],
    context_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Synthesize final answer from agent responses."""
    import json, time
    from shared.exceptions import LLMProviderError

    if not _openai_client:
        return {"answer": "LLM not configured.", "session_id": session_id}

    successful = {k: v for k, v in agent_responses.items() if not v.get("error")}
    failed = [k for k, v in agent_responses.items() if v.get("error")]

    if not successful:
        return {
            "answer": "I couldn't reach the knowledge agents. Please try again.",
            "session_id": session_id, "agents_used": [], "failed_agents": failed,
        }

    agent_sections = [
        f"### {name.upper()} AGENT\n{json.dumps(data, indent=2, default=str)[:2000]}"
        for name, data in successful.items()
    ]
    history_str = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in context_messages[-4:])

    try:
        start = time.monotonic()
        resp = await _openai_client.chat.completions.create(
            model=settings.synthesis_model,
            messages=[
                {"role": "system", "content": "You are an expert FastAPI codebase assistant. Synthesize the agent findings into a clear, developer-friendly Markdown answer with code citations."},
                {"role": "user", "content": f"History:\n{history_str or '(none)'}\n\nQuestion: {original_query}\n\nAgent data:\n{chr(10).join(agent_sections)}"},
            ],
            temperature=0.2,
            max_tokens=settings.openai_max_tokens,
        )
        answer = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        await _memory.add_message(session_id, "user", original_query)  # type: ignore[union-attr]
        await _memory.add_message(session_id, "assistant", answer)  # type: ignore[union-attr]
        return {
            "answer": answer, "session_id": session_id,
            "agents_used": list(successful.keys()), "failed_agents": failed,
            "tokens_used": tokens, "latency_ms": round((time.monotonic() - start) * 1000, 2),
        }
    except Exception as exc:
        raise LLMProviderError(f"Synthesis failed: {exc}") from exc


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)