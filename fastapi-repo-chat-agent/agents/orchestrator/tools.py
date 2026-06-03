"""MCP tool implementations for the Orchestrator Agent — complete implementation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agents.orchestrator.memory import ConversationMemory
from agents.orchestrator.router import ExecutionMode, RoutingPlan, build_routing_plan
from shared.config import OrchestratorSettings
from shared.exceptions import AgentTimeoutError, LLMProviderError, SessionNotFoundError
from shared.models.base import QueryIntent
from shared.utils.logging import get_logger

logger = get_logger(__name__)


def register_orchestrator_tools(
    mcp: FastMCP,
    settings: OrchestratorSettings,
    openai_client: AsyncOpenAI,
    memory: ConversationMemory,
    agent_urls: dict[str, str],
) -> None:
    """Register all five Orchestrator MCP tools onto the FastMCP server.

    Args:
        mcp: FastMCP server instance.
        settings: Orchestrator configuration.
        openai_client: Configured AsyncOpenAI client.
        memory: Redis-backed conversation memory store.
        agent_urls: Dict mapping agent names → base HTTP URLs.
    """

    # ── 1. analyze_query ─────────────────────────────────────────────────────

    @mcp.tool()
    async def analyze_query(session_id: str, query: str) -> dict[str, Any]:
        """Classify query intent and extract key code entities.

        Determines what kind of question is being asked and what FastAPI
        code entities (classes, functions, modules) are referenced.

        Args:
            session_id: Active conversation session ID.
            query: Raw user query string.

        Returns:
            Dict with intent, entities, complexity, and required_agents.
        """
        # Pull last 3 messages for context-aware classification
        history = await memory.get_messages(session_id, last_n=3)
        history_str = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)

        system_prompt = """You are a query analyzer for a FastAPI codebase assistant.
Classify the user's query and extract code entity names.

Return ONLY valid JSON (no markdown):
{
  "intent": "<one of: code_explanation | dependency_analysis | pattern_detection | entity_lookup | relationship_query | lifecycle_analysis | comparison | general>",
  "entities": ["list", "of", "code", "entity", "names"],
  "complexity": "<simple | medium | complex>",
  "requires_agents": ["graph_query and/or code_analyst"],
  "reasoning": "one sentence explaining your classification"
}"""

        user_content = f"""Prior conversation:
{history_str or "(none)"}

Current query: {query}"""

        try:
            resp = await openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=256,
            )
            result = json.loads(resp.choices[0].message.content or "{}")
            logger.info(
                "query_analyzed",
                session_id=session_id,
                intent=result.get("intent"),
                complexity=result.get("complexity"),
                entities=result.get("entities", []),
            )
            return result
        except Exception as exc:
            logger.error("analyze_query_failed", error=str(exc))
            # Graceful fallback — don't crash the orchestration pipeline
            return {
                "intent": QueryIntent.GENERAL.value,
                "entities": [],
                "complexity": "medium",
                "requires_agents": ["graph_query", "code_analyst"],
                "reasoning": f"Fallback due to classification error: {exc}",
            }

    # ── 2. route_to_agents ───────────────────────────────────────────────────

    @mcp.tool()
    async def route_to_agents(
        session_id: str,
        query: str,
        intent: str,
        entities: list[str],
        required_agents: list[str],
    ) -> dict[str, Any]:
        """Invoke the appropriate agents in parallel or sequential order.

        Builds a routing plan from the classified intent, then executes
        agents with retry logic and graceful fallback on failure.

        Args:
            session_id: Active conversation session ID.
            query: Original user query string.
            intent: Classified intent from analyze_query.
            entities: Extracted entity names from analyze_query.
            required_agents: Agent list from analyze_query (used as hint).

        Returns:
            Dict with agent_responses, execution_mode, and routing_plan.
        """
        plan = build_routing_plan(intent, entities)
        logger.info(
            "routing_plan_built",
            session_id=session_id,
            agents=plan.agents,
            mode=plan.mode,
            reasoning=plan.reasoning,
        )

        payload = {
            "session_id": session_id,
            "query": query,
            "intent": intent,
            "entities": entities,
        }

        if plan.mode == ExecutionMode.PARALLEL:
            agent_responses = await _call_agents_parallel(
                plan.agents, payload, plan.fallback_agents, settings, agent_urls
            )
        else:
            agent_responses = await _call_agents_sequential(
                plan.agents, payload, plan.fallback_agents, settings, agent_urls
            )

        # Check if all primary agents failed — try fallbacks
        all_failed = all(r.get("error") for r in agent_responses.values())
        if all_failed and plan.fallback_agents:
            logger.warning("all_primary_agents_failed", trying_fallbacks=plan.fallback_agents)
            fallback_payload = {**payload, "is_fallback": True}
            fallback_responses = await _call_agents_parallel(
                plan.fallback_agents, fallback_payload, [], settings, agent_urls
            )
            agent_responses.update(fallback_responses)

        return {
            "agent_responses": agent_responses,
            "execution_mode": plan.mode.value,
            "routing_plan": {
                "agents": plan.agents,
                "fallback_agents": plan.fallback_agents,
                "reasoning": plan.reasoning,
            },
            "agents_called": [k for k, v in agent_responses.items() if not v.get("error")],
        }

    # ── 3. get_conversation_context ──────────────────────────────────────────

    @mcp.tool()
    async def get_conversation_context(
        session_id: str, last_n: int = 5
    ) -> dict[str, Any]:
        """Retrieve relevant conversation history for context injection.

        Returns the most recent N message turns for the session,
        plus metadata about session age and total turn count.

        Args:
            session_id: Session identifier (creates new session if absent).
            last_n: Number of recent messages to retrieve (default 5).

        Returns:
            Dict with messages, is_new_session, total_messages, session_id.
        """
        is_new = not await memory.session_exists(session_id)

        if is_new:
            logger.info("new_session_created", session_id=session_id)
            return {
                "session_id": session_id,
                "messages": [],
                "is_new_session": True,
                "total_messages": 0,
            }

        messages = await memory.get_messages(session_id, last_n=last_n)
        all_messages = await memory.get_messages(session_id, last_n=1000)

        return {
            "session_id": session_id,
            "messages": messages,
            "is_new_session": False,
            "total_messages": len(all_messages),
        }

    # ── 4. synthesize_response ───────────────────────────────────────────────

    @mcp.tool()
    async def synthesize_response(
        session_id: str,
        original_query: str,
        agent_responses: dict[str, Any],
        context_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Combine multiple agent outputs into a single coherent developer response.

        Filters out failed agents, formats successful data into a prompt,
        and calls GPT-4o to synthesize a final answer with source attribution.

        Args:
            session_id: Active session ID.
            original_query: The user's original question.
            agent_responses: Dict of agent_name → response data from route_to_agents.
            context_messages: Prior conversation messages for multi-turn coherence.

        Returns:
            Dict with answer, session_id, agents_used, tokens_used, latency_ms.
        """
        # Separate successful and failed agent responses
        successful: dict[str, Any] = {}
        failed: list[str] = []

        for agent_name, resp in agent_responses.items():
            if resp.get("error"):
                failed.append(agent_name)
                logger.warning("agent_response_excluded", agent=agent_name, error=resp["error"])
            else:
                successful[agent_name] = resp

        if not successful:
            # All agents failed — return a graceful error message
            return {
                "answer": (
                    "I was unable to retrieve information from the knowledge agents at this time. "
                    f"Failed agents: {', '.join(failed)}. Please try rephrasing or try again shortly."
                ),
                "session_id": session_id,
                "agents_used": [],
                "failed_agents": failed,
                "tokens_used": 0,
                "latency_ms": 0.0,
                "fallback_used": True,
            }

        # Build agent findings section
        agent_sections = []
        for agent_name, resp in successful.items():
            content = json.dumps(resp, indent=2, default=str)[:2000]
            agent_sections.append(f"### {agent_name.upper().replace('_', ' ')} AGENT\n{content}")

        # Build conversation history section
        history_str = ""
        if context_messages:
            history_str = "\n".join(
                f"{m['role'].upper()}: {m['content'][:300]}" for m in context_messages[-4:]
            )

        system_prompt = """You are an expert FastAPI codebase assistant. 
You have structured data from specialized code analysis agents.
Synthesize this into a clear, accurate, developer-friendly Markdown response.

Rules:
- Lead with the direct answer to the question
- Use code blocks for any code snippets  
- Cite which file/class/function supports each claim (e.g. "in `fastapi/applications.py`")
- If agents disagreed, note the discrepancy
- If data was incomplete, state what's known and what's uncertain
- Be thorough but concise — no filler"""

        user_content = f"""Previous conversation:
{history_str or "(new conversation)"}

User question: {original_query}

Agent findings:
{chr(10).join(agent_sections)}

Provide a comprehensive, well-cited answer."""

        try:
            start = time.monotonic()
            resp = await openai_client.chat.completions.create(
                model=settings.synthesis_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=settings.openai_max_tokens,
            )
            latency_ms = round((time.monotonic() - start) * 1000, 2)
            answer = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else 0

            # Persist to memory
            await memory.add_message(session_id, "user", original_query)
            await memory.add_message(session_id, "assistant", answer)

            logger.info(
                "response_synthesized",
                session_id=session_id,
                tokens=tokens,
                latency_ms=latency_ms,
                agents_used=list(successful.keys()),
            )

            return {
                "answer": answer,
                "session_id": session_id,
                "agents_used": list(successful.keys()),
                "failed_agents": failed,
                "tokens_used": tokens,
                "latency_ms": latency_ms,
                "fallback_used": bool(failed),
            }

        except Exception as exc:
            raise LLMProviderError(f"Response synthesis failed: {exc}") from exc


# ── Agent invocation helpers ──────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(httpx.TransportError),
    reraise=False,
)
async def _call_single_agent(
    agent_name: str,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Call a single agent HTTP endpoint with retry on transport errors.

    Args:
        agent_name: Human-readable agent identifier for logging.
        url: Full endpoint URL to POST to.
        payload: JSON request body.
        timeout: Per-request timeout in seconds.

    Returns:
        Parsed JSON response dict, or error dict on failure.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        logger.error("agent_timeout", agent=agent_name, timeout=timeout)
        return {"error": f"Agent '{agent_name}' timed out after {timeout}s", "agent": agent_name}
    except httpx.HTTPStatusError as exc:
        logger.error("agent_http_error", agent=agent_name, status=exc.response.status_code)
        return {"error": f"Agent '{agent_name}' returned HTTP {exc.response.status_code}", "agent": agent_name}
    except Exception as exc:
        logger.error("agent_call_error", agent=agent_name, error=str(exc))
        return {"error": str(exc), "agent": agent_name}


async def _call_agents_parallel(
    agents: list[str],
    payload: dict[str, Any],
    fallback_agents: list[str],
    settings: OrchestratorSettings,
    agent_urls: dict[str, str],
) -> dict[str, Any]:
    """Call all agents simultaneously with asyncio.gather.

    Args:
        agents: List of agent names to call.
        payload: Shared request payload for all agents.
        fallback_agents: Fallback agent list (for logging only here).
        settings: Orchestrator settings for URL and timeout config.
        agent_urls: Agent name → base URL mapping.

    Returns:
        Dict mapping agent_name → response.
    """
    async def _invoke(name: str) -> tuple[str, dict[str, Any]]:
        url = f"{agent_urls.get(name, '')}/query"
        if not agent_urls.get(name):
            return name, {"error": f"No URL configured for agent: {name}"}
        result = await _call_single_agent(name, url, payload, settings.agent_timeout_seconds)
        return name, result

    results = await asyncio.gather(*[_invoke(a) for a in agents], return_exceptions=True)

    responses: dict[str, Any] = {}
    for item in results:
        if isinstance(item, tuple):
            responses[item[0]] = item[1]
        elif isinstance(item, Exception):
            logger.error("gather_exception", error=str(item))

    return responses


async def _call_agents_sequential(
    agents: list[str],
    payload: dict[str, Any],
    fallback_agents: list[str],
    settings: OrchestratorSettings,
    agent_urls: dict[str, str],
) -> dict[str, Any]:
    """Call agents in order, feeding each agent the prior agent's output.

    This is used for multi-step queries where agent B needs agent A's
    results as additional context (e.g., graph lookup → code analysis).

    Args:
        agents: Ordered list of agent names.
        payload: Initial request payload.
        fallback_agents: Agents to try if a step fails.
        settings: Orchestrator settings.
        agent_urls: Agent name → base URL mapping.

    Returns:
        Dict mapping agent_name → response (all steps included).
    """
    responses: dict[str, Any] = {}
    accumulated_context: dict[str, Any] = {}

    for agent_name in agents:
        url = f"{agent_urls.get(agent_name, '')}/query"
        if not agent_urls.get(agent_name):
            responses[agent_name] = {"error": f"No URL configured for: {agent_name}"}
            continue

        # Enrich payload with prior results
        enriched_payload = {**payload, "prior_context": accumulated_context}
        result = await _call_single_agent(
            agent_name, url, enriched_payload, settings.agent_timeout_seconds
        )
        responses[agent_name] = result

        if not result.get("error"):
            accumulated_context[agent_name] = result
        else:
            # Try fallback for this step
            for fallback in fallback_agents:
                if fallback != agent_name and fallback not in responses:
                    logger.warning("sequential_fallback", failed=agent_name, trying=fallback)
                    fb_url = f"{agent_urls.get(fallback, '')}/query"
                    fb_result = await _call_single_agent(
                        fallback, fb_url, enriched_payload, settings.agent_timeout_seconds
                    )
                    if not fb_result.get("error"):
                        responses[f"{agent_name}_fallback"] = fb_result
                        accumulated_context[fallback] = fb_result
                        break

    return responses