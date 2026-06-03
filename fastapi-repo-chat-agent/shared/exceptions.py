"""Custom exception hierarchy for the multi-agent system.

All exceptions follow a consistent pattern:
  - Each has a descriptive __str__ for logging
  - Each stores structured attributes for programmatic access
  - Agent-specific errors inherit from a common base
"""
from __future__ import annotations


# ── Base Exceptions ───────────────────────────────────────────────────────────

class BaseAgentError(Exception):
    """Base class for all agent-related errors."""
    agent_type: str = "unknown"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f"[{self.agent_type}] {message}")


# ── Agent-Specific Errors ─────────────────────────────────────────────────────

class OrchestratorError(BaseAgentError):
    """Raised when the Orchestrator agent encounters an error."""
    agent_type = "orchestrator"


class IndexerError(BaseAgentError):
    """Raised when the Indexer agent encounters an error."""
    agent_type = "indexer"


class GraphQueryError(BaseAgentError):
    """Raised when the Graph Query agent encounters an error."""
    agent_type = "graph_query"


class CodeAnalystError(BaseAgentError):
    """Raised when the Code Analyst agent encounters an error."""
    agent_type = "code_analyst"


# ── LLM/OpenAI Errors ─────────────────────────────────────────────────────────

class LLMProviderError(Exception):
    """Raised when an LLM API call fails."""
    def __init__(self, message: str, agent_type: str = "openai") -> None:
        self.message = message
        self.agent_type = agent_type
        super().__init__(f"[{agent_type}] LLM error: {message}")


# ── Entity/Query Errors ───────────────────────────────────────────────────────

class EntityNotFoundError(Exception):
    """Raised when a requested code entity is not found in the graph."""
    def __init__(self, entity_name: str, entity_type: str = "entity") -> None:
        self.entity_name = entity_name
        self.entity_type = entity_type
        super().__init__(f"{entity_type} '{entity_name}' not found in knowledge graph")


class InvalidCypherQueryError(Exception):
    """Raised when a Cypher query fails safety validation."""
    def __init__(self, query: str, reason: str) -> None:
        # Truncate long queries to avoid log spam
        self.query_preview = query[:100] + "..." if len(query) > 100 else query
        self.reason = reason
        super().__init__(f"[cypher] Invalid query — {reason}: '{self.query_preview}'")


# ── Session/Memory Errors ─────────────────────────────────────────────────────

class SessionNotFoundError(Exception):
    """Raised when a session ID is not found in the memory store."""
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' not found")


class MemoryStoreError(Exception):
    """Raised when the Redis memory store operation fails."""
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f"[memory_store] {message}")


# ── Infrastructure Errors ─────────────────────────────────────────────────────

class Neo4jQueryError(Exception):
    """Raised when a Cypher query fails at runtime."""
    def __init__(self, query_preview: str, reason: str) -> None:
        self.query_preview = query_preview
        self.reason = reason
        super().__init__(f"[neo4j] Query failed — '{query_preview}': {reason}")


class Neo4jConnectionError(Exception):
    """Raised when Neo4j driver cannot connect or verify connectivity."""
    def __init__(self, uri: str, reason: str) -> None:
        self.uri = uri
        self.reason = reason
        super().__init__(f"[neo4j] Cannot connect to {uri}: {reason}")


# ── Timeout/Availability Errors ───────────────────────────────────────────────

class AgentTimeoutError(Exception):
    """Raised when an agent does not respond within the timeout."""
    def __init__(self, agent_name: str, timeout_seconds: float) -> None:
        self.agent_name = agent_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Agent '{agent_name}' did not respond within {timeout_seconds} seconds"
        )


class RepositoryNotFoundError(Exception):
    """Raised when a repository URL cannot be cloned or accessed."""
    def __init__(self, repo_url: str) -> None:
        self.repo_url = repo_url
        super().__init__(f"Repository not found or inaccessible: {repo_url}")