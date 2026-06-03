"""Query routing logic — determines which agents to invoke and in what order."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from shared.models.base import QueryIntent


class ExecutionMode(str, Enum):
    """How agents should be invoked for a given query."""

    PARALLEL = "parallel"       # All agents called simultaneously
    SEQUENTIAL = "sequential"   # Agents called in order, each gets prior results
    SINGLE = "single"           # Only one agent needed


@dataclass
class RoutingPlan:
    """Execution plan produced by the router for a single query.

    Attributes:
        agents: Ordered list of agent names to invoke.
        mode: Parallel or sequential execution.
        fallback_agents: Agents to try if primary agents fail.
        reasoning: Human-readable explanation of routing decision.
    """

    agents: list[str]
    mode: ExecutionMode
    fallback_agents: list[str] = field(default_factory=list)
    reasoning: str = ""


# Maps query intent → routing plan
_INTENT_ROUTING: dict[QueryIntent, RoutingPlan] = {
    QueryIntent.ENTITY_LOOKUP: RoutingPlan(
        agents=["graph_query"],
        mode=ExecutionMode.SINGLE,
        fallback_agents=["code_analyst"],
        reasoning="Simple entity lookup — graph query is sufficient",
    ),
    QueryIntent.DEPENDENCY_ANALYSIS: RoutingPlan(
        agents=["graph_query", "code_analyst"],
        mode=ExecutionMode.PARALLEL,
        fallback_agents=["graph_query"],
        reasoning="Dependencies need both graph traversal and code understanding",
    ),
    QueryIntent.CODE_EXPLANATION: RoutingPlan(
        agents=["graph_query", "code_analyst"],
        mode=ExecutionMode.SEQUENTIAL,
        fallback_agents=["code_analyst"],
        reasoning="Fetch entity context from graph first, then deep analysis",
    ),
    QueryIntent.PATTERN_DETECTION: RoutingPlan(
        agents=["graph_query", "code_analyst"],
        mode=ExecutionMode.SEQUENTIAL,
        fallback_agents=["code_analyst"],
        reasoning="Graph finds candidate classes; analyst detects patterns",
    ),
    QueryIntent.RELATIONSHIP_QUERY: RoutingPlan(
        agents=["graph_query"],
        mode=ExecutionMode.SINGLE,
        fallback_agents=["code_analyst"],
        reasoning="Pure graph traversal question",
    ),
    QueryIntent.LIFECYCLE_ANALYSIS: RoutingPlan(
        agents=["graph_query", "code_analyst"],
        mode=ExecutionMode.SEQUENTIAL,
        fallback_agents=["code_analyst"],
        reasoning="Lifecycle requires full dependency chain + LLM synthesis",
    ),
    QueryIntent.COMPARISON: RoutingPlan(
        agents=["graph_query", "code_analyst"],
        mode=ExecutionMode.PARALLEL,
        fallback_agents=["code_analyst"],
        reasoning="Both entities fetched in parallel, then compared",
    ),
    QueryIntent.GENERAL: RoutingPlan(
        agents=["graph_query", "code_analyst"],
        mode=ExecutionMode.PARALLEL,
        fallback_agents=["code_analyst"],
        reasoning="General query — cast wide net across agents",
    ),
}


def build_routing_plan(intent: str, entities: list[str]) -> RoutingPlan:
    """Build an agent execution plan from classified intent.

    Args:
        intent: String value matching a QueryIntent enum.
        entities: List of code entity names extracted from the query.

    Returns:
        RoutingPlan with agents, mode, and fallback strategy.
    """
    try:
        intent_enum = QueryIntent(intent)
    except ValueError:
        intent_enum = QueryIntent.GENERAL

    plan = _INTENT_ROUTING.get(intent_enum, _INTENT_ROUTING[QueryIntent.GENERAL])

    # If no entities found, sequential doesn't help — switch to parallel
    if not entities and plan.mode == ExecutionMode.SEQUENTIAL:
        return RoutingPlan(
            agents=plan.agents,
            mode=ExecutionMode.PARALLEL,
            fallback_agents=plan.fallback_agents,
            reasoning=plan.reasoning + " (no entities → switched to parallel)",
        )

    return plan