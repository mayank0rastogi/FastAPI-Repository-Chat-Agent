"""Centralised Pydantic Settings for the entire multi-agent system.

Architecture:
  - Shared leaf settings (Neo4jSettings, OpenAISettings, RedisSettings)
    are composed into each agent's own Settings class.
  - Every Settings class has a distinct env_prefix so variables never clash.
  - Secrets use pydantic.SecretStr — they are NEVER logged or serialised
    as plain text.
  - lru_cache factories ensure a single Settings instance per process.
  - Environment enum drives dev/test/prod divergence (CORS, debug, reload).

Environment variable naming convention:
  <PREFIX>_<FIELD>  e.g.  NEO4J_URI, OPENAI_API_KEY, INDEXER_PORT

All settings can be overridden by:
  1. Environment variables (highest priority)
  2. .env file in project root (or path set by DOTENV_PATH)
  3. Model defaults (lowest priority — never contain secrets)
"""
from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from typing import Annotated, Any

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Environment enum ──────────────────────────────────────────────────────────

class Environment(str, Enum):
    """Deployment environment — drives defaults for debug, CORS, reload."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


# ── Shared leaf settings (composed into agent settings) ───────────────────────

class Neo4jSettings(BaseSettings):
    """Neo4j connection settings — shared by all agents that need graph access.

    Environment variables (prefix: NEO4J_):
        NEO4J_URI              bolt://localhost:7687
        NEO4J_USERNAME         neo4j
        NEO4J_PASSWORD         <secret>
        NEO4J_DATABASE         neo4j
        NEO4J_MAX_POOL_SIZE    50
        NEO4J_CONN_TIMEOUT     30
        NEO4J_MAX_TX_RETRY     3
    """
    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt URI",
        examples=["bolt://localhost:7687", "neo4j+s://abc.databases.neo4j.io"],
    )
    username: str = Field(default="neo4j", description="Neo4j username")
    password: SecretStr = Field(
        default=SecretStr(""),
        description="Neo4j password — stored as SecretStr, never logged",
    )
    database: str = Field(default="neo4j", description="Target Neo4j database name")
    max_pool_size: int = Field(default=50, ge=1, le=500, description="Connection pool size")
    conn_timeout: int = Field(default=30, ge=1, le=300, description="Connection timeout in seconds")
    max_tx_retry: int = Field(default=3, ge=1, le=10, description="Max transaction retry attempts")

    @field_validator("uri")
    @classmethod
    def validate_neo4j_uri(cls, v: str) -> str:
        """Ensure URI uses a recognised Neo4j scheme."""
        valid_schemes = ("bolt://", "bolt+s://", "neo4j://", "neo4j+s://")
        if not any(v.startswith(s) for s in valid_schemes):
            raise ValueError(
                f"Neo4j URI must start with one of {valid_schemes}, got: {v!r}"
            )
        return v

    @property
    def safe_uri(self) -> str:
        """URI with credentials stripped — safe for logging."""
        return re.sub(r"//[^@]+@", "//***:***@", self.uri)


class OpenAISettings(BaseSettings):
    """OpenAI API settings — shared by Orchestrator and Code Analyst agents.

    Environment variables (prefix: OPENAI_):
        OPENAI_API_KEY         sk-...
        OPENAI_MODEL           gpt-4o-mini
        OPENAI_MAX_TOKENS      4096
        OPENAI_TEMPERATURE     0.1
        OPENAI_TIMEOUT         60
        OPENAI_MAX_RETRIES     3
    """
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key — stored as SecretStr",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Default chat completion model",
        examples=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    )
    max_tokens: int = Field(default=4096, ge=64, le=128_000)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    timeout: float = Field(default=60.0, ge=5.0, le=600.0, description="API call timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max API call retries on transient errors")

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v: Any) -> Any:
        """Warn (but don't fail) if key is empty — allows test environments."""
        if not v or str(v) in ("", "sk-placeholder"):
            import warnings
            warnings.warn(
                "OPENAI_API_KEY is not set. LLM-dependent tools will fail at runtime.",
                stacklevel=2,
            )
        return v


class RedisSettings(BaseSettings):
    """Redis connection and TTL settings — shared by memory store consumers.

    Environment variables (prefix: REDIS_):
        REDIS_URL                  redis://localhost:6379/0
        REDIS_MAX_CONNECTIONS      20
        REDIS_SESSION_TTL          86400
        REDIS_CACHE_TTL            3600
        REDIS_MAX_MESSAGES         200
        REDIS_SOCKET_TIMEOUT       5
    """
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
        examples=["redis://localhost:6379/0", "rediss://user:pass@host:6379/0"],
    )
    max_connections: int = Field(default=20, ge=1, le=200)
    session_ttl: int = Field(
        default=86_400, ge=60, description="Session TTL in seconds (sliding)"
    )
    cache_ttl: int = Field(
        default=3_600, ge=60, description="Response cache TTL in seconds"
    )
    max_messages_per_session: int = Field(
        default=200, ge=10, le=10_000,
        description="Max messages stored per session before oldest are dropped",
    )
    socket_timeout: int = Field(default=5, ge=1, le=30)

    @field_validator("url")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        if not (v.startswith("redis://") or v.startswith("rediss://")):
            raise ValueError(f"Redis URL must start with redis:// or rediss://, got: {v!r}")
        return v


class RetrySettings(BaseSettings):
    """Configurable retry policy for agent HTTP calls and DB operations.

    Environment variables (prefix: RETRY_):
        RETRY_MAX_ATTEMPTS     3
        RETRY_WAIT_MIN         1.0
        RETRY_WAIT_MAX         10.0
        RETRY_MULTIPLIER       1.5
        RETRY_ON_TIMEOUT       true
    """
    model_config = SettingsConfigDict(
        env_prefix="RETRY_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    max_attempts: int = Field(default=3, ge=1, le=20, description="Max total attempts (1 = no retry)")
    wait_min: float = Field(default=1.0, ge=0.1, description="Min wait between retries in seconds")
    wait_max: float = Field(default=10.0, ge=1.0, description="Max wait between retries in seconds")
    multiplier: float = Field(default=1.5, ge=1.0, description="Exponential backoff multiplier")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout errors")
    retry_on_server_error: bool = Field(default=True, description="Retry on 5xx responses")

    @model_validator(mode="after")
    def validate_wait_range(self) -> "RetrySettings":
        if self.wait_min > self.wait_max:
            raise ValueError(
                f"RETRY_WAIT_MIN ({self.wait_min}) must be <= RETRY_WAIT_MAX ({self.wait_max})"
            )
        return self


# ── Per-agent Settings classes ────────────────────────────────────────────────

class GatewaySettings(BaseSettings):
    """FastAPI Gateway configuration.

    Environment variables (prefix: GATEWAY_):
        GATEWAY_HOST                    0.0.0.0
        GATEWAY_PORT                    8000
        GATEWAY_ENVIRONMENT             development
        GATEWAY_LOG_LEVEL               INFO
        GATEWAY_SECRET_KEY              <secret>
        GATEWAY_CORS_ORIGINS            ["*"]
        GATEWAY_WS_ALLOWED_ORIGINS      []
        GATEWAY_RATE_LIMIT_PER_MINUTE   60
        GATEWAY_AGENT_TIMEOUT_SECONDS   90.0
        GATEWAY_STREAM_CHUNK_CHARS      40
        GATEWAY_DEFAULT_REPO_URL        https://github.com/fastapi/fastapi.git

    Downstream agent URLs:
        GATEWAY_ORCHESTRATOR_URL        http://orchestrator:8001
        GATEWAY_INDEXER_URL             http://indexer:8002
        GATEWAY_GRAPH_QUERY_URL         http://graph-query:8003
        GATEWAY_CODE_ANALYST_URL        http://code-analyst:8004
    """
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # ── Security ──────────────────────────────────────────────────────────────
    secret_key: SecretStr = Field(
        default=SecretStr("change-me-in-production"),
        description="Used for session signing — MUST be changed in production",
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description='Allowed CORS origins. Use ["*"] only in development.',
    )
    ws_allowed_origins: list[str] = Field(
        default_factory=list,
        description="WebSocket allowed origins. Empty = allow all (dev only).",
    )
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10_000)

    # ── Agent routing ─────────────────────────────────────────────────────────
    orchestrator_url: str = Field(default="http://orchestrator:8001")
    indexer_url: str = Field(default="http://indexer:8002")
    graph_query_url: str = Field(default="http://graph-query:8003")
    code_analyst_url: str = Field(default="http://code-analyst:8004")
    agent_timeout_seconds: float = Field(default=90.0, ge=5.0, le=600.0)
    stream_chunk_chars: int = Field(default=40, ge=1, le=500)
    default_repo_url: str = Field(default="https://github.com/fastapi/fastapi.git")

    @model_validator(mode="after")
    def production_safety_checks(self) -> "GatewaySettings":
        """Enforce production-safe defaults when environment=production."""
        if self.environment == Environment.PRODUCTION:
            if "*" in self.cors_origins:
                raise ValueError(
                    "GATEWAY_CORS_ORIGINS cannot contain '*' in production. "
                    "Set explicit allowed origins."
                )
            if self.secret_key.get_secret_value() == "change-me-in-production":
                raise ValueError(
                    "GATEWAY_SECRET_KEY must be changed from the default in production."
                )
        return self

    @property
    def is_debug(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @property
    def reload(self) -> bool:
        return self.environment == Environment.DEVELOPMENT


class OrchestratorSettings(BaseSettings):
    """Orchestrator Agent MCP server configuration.

    Environment variables (prefix: ORCHESTRATOR_):
        ORCHESTRATOR_HOST                   0.0.0.0
        ORCHESTRATOR_PORT                   8001
        ORCHESTRATOR_ENVIRONMENT            development
        ORCHESTRATOR_LOG_LEVEL              INFO
        ORCHESTRATOR_MAX_PARALLEL_AGENTS    3
        ORCHESTRATOR_AGENT_TIMEOUT_SECONDS  60.0
        ORCHESTRATOR_SYNTHESIS_MODEL        gpt-4o
        ORCHESTRATOR_ANALYSIS_MODEL         gpt-4o-mini

    Composed settings (use their own prefixes):
        NEO4J_*     (unused directly — passed to agents)
        OPENAI_*
        REDIS_*
        RETRY_*
    """
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001, ge=1024, le=65535)
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # ── LLM models ────────────────────────────────────────────────────────────
    # These override OPENAI_MODEL for specific orchestrator tasks
    synthesis_model: str = Field(
        default="gpt-4o",
        description="Model for final response synthesis (higher quality needed)",
    )
    analysis_model: str = Field(
        default="gpt-4o-mini",
        description="Model for query intent classification (speed matters)",
    )
    openai_max_tokens: int = Field(default=4096, ge=64, le=128_000)

    # ── Agent coordination ────────────────────────────────────────────────────
    max_parallel_agents: int = Field(
        default=3, ge=1, le=4,
        description="Max agents invoked concurrently per query",
    )
    agent_timeout_seconds: float = Field(
        default=60.0, ge=5.0, le=300.0,
        description="Timeout for each downstream agent call",
    )

    # ── Downstream agent URLs ─────────────────────────────────────────────────
    indexer_url: str = Field(default="http://indexer:8002")
    graph_query_url: str = Field(default="http://graph-query:8003")
    code_analyst_url: str = Field(default="http://code-analyst:8004")

    # ── Context window ────────────────────────────────────────────────────────
    context_window_messages: int = Field(
        default=5, ge=1, le=50,
        description="Number of prior messages included in synthesis context",
    )
    max_entity_context: int = Field(
        default=10, ge=1, le=100,
        description="Max entity names included in context from session history",
    )

    # ── Secrets (loaded from shared OPENAI_* prefix) ──────────────────────────
    # These are read at runtime from the composed OpenAISettings
    @property
    def openai_api_key(self) -> str:
        return get_openai_settings().api_key.get_secret_value()


class IndexerSettings(BaseSettings):
    """Indexer Agent MCP server configuration.

    Environment variables (prefix: INDEXER_):
        INDEXER_HOST                    0.0.0.0
        INDEXER_PORT                    8002
        INDEXER_ENVIRONMENT             development
        INDEXER_LOG_LEVEL               INFO
        INDEXER_REPO_URL                https://github.com/fastapi/fastapi.git
        INDEXER_REPO_LOCAL_PATH         /tmp/fastapi_repo
        INDEXER_MAX_FILE_SIZE_KB        500
        INDEXER_MAX_CONCURRENT_FILES    10
        INDEXER_INCLUDE_TEST_FILES      false
        INDEXER_INCLUDE_PATTERNS        ["*.py"]
        INDEXER_EXCLUDE_PATTERNS        ["**/test_*.py", "**/__pycache__/**"]
        INDEXER_SOURCE_SNIPPET_MAX_KB   8
    """
    model_config = SettingsConfigDict(
        env_prefix="INDEXER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8002, ge=1024, le=65535)
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # ── Repository ────────────────────────────────────────────────────────────
    repo_url: str = Field(
        default="https://github.com/fastapi/fastapi.git",
        description="Default repository to index",
    )
    repo_local_path: str = Field(
        default="/tmp/fastapi_repo",
        description="Local filesystem path where the repo is cloned",
    )
    git_depth: int = Field(
        default=1, ge=1,
        description="Git clone depth (1 = shallow clone for speed)",
    )
    git_timeout: int = Field(
        default=300, ge=30, le=3600,
        description="Git clone/pull timeout in seconds",
    )

    # ── File filtering ────────────────────────────────────────────────────────
    max_file_size_kb: int = Field(
        default=500, ge=1, le=10_000,
        description="Files larger than this are skipped during indexing",
    )
    max_concurrent_files: int = Field(
        default=10, ge=1, le=50,
        description="Semaphore limit for concurrent file parsing",
    )
    include_test_files: bool = Field(
        default=False,
        description="Whether to index test_*.py files",
    )
    include_patterns: list[str] = Field(
        default=["*.py"],
        description="Glob patterns for files to include",
    )
    exclude_patterns: list[str] = Field(
        default=["**/test_*.py", "**/__pycache__/**", "**/migrations/**"],
        description="Glob patterns for files to exclude",
    )
    source_snippet_max_kb: int = Field(
        default=8, ge=1, le=100,
        description="Max size of source_code stored per entity in Neo4j",
    )

    # ── Neo4j properties (loaded from shared NEO4J_* prefix) ──────────────────
    @property
    def neo4j_uri(self) -> str:
        return get_neo4j_settings().uri

    @property
    def neo4j_username(self) -> str:
        return get_neo4j_settings().username

    @property
    def neo4j_password(self) -> str:
        return get_neo4j_settings().password.get_secret_value()

    @property
    def neo4j_database(self) -> str:
        return get_neo4j_settings().database

    @property
    def neo4j_max_pool_size(self) -> int:
        return get_neo4j_settings().max_pool_size


class GraphQuerySettings(BaseSettings):
    """Graph Query Agent MCP server configuration.

    Environment variables (prefix: GRAPH_QUERY_):
        GRAPH_QUERY_HOST                    0.0.0.0
        GRAPH_QUERY_PORT                    8003
        GRAPH_QUERY_ENVIRONMENT             development
        GRAPH_QUERY_LOG_LEVEL               INFO
        GRAPH_QUERY_RESULT_LIMIT            50
        GRAPH_QUERY_MAX_QUERY_DEPTH         5
        GRAPH_QUERY_ALLOWED_QUERY_PREFIXES  ["MATCH"]
        GRAPH_QUERY_CACHE_QUERY_RESULTS     true
        GRAPH_QUERY_CACHE_TTL_SECONDS       300
    """
    model_config = SettingsConfigDict(
        env_prefix="GRAPH_QUERY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8003, ge=1024, le=65535)
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # ── Query controls ────────────────────────────────────────────────────────
    result_limit: int = Field(
        default=50, ge=1, le=1000,
        description="Max records returned by any graph query",
    )
    max_query_depth: int = Field(
        default=5, ge=1, le=20,
        description="Max relationship traversal depth for path queries",
    )
    allowed_query_prefixes: list[str] = Field(
        default=["MATCH"],
        description="Only queries starting with these keywords are allowed via execute_query",
    )

    # ── Result caching ────────────────────────────────────────────────────────
    cache_query_results: bool = Field(
        default=True,
        description="Cache Cypher query results in Redis to reduce Neo4j load",
    )
    cache_ttl_seconds: int = Field(
        default=300, ge=10,
        description="TTL for cached query results",
    )

    # ── Timeout ───────────────────────────────────────────────────────────────
    query_timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=300.0,
        description="Per-query execution timeout",
    )

    # ── Neo4j properties (loaded from shared NEO4J_* prefix) ──────────────────
    @property
    def neo4j_uri(self) -> str:
        return get_neo4j_settings().uri

    @property
    def neo4j_username(self) -> str:
        return get_neo4j_settings().username

    @property
    def neo4j_password(self) -> str:
        return get_neo4j_settings().password.get_secret_value()

    @property
    def neo4j_database(self) -> str:
        return get_neo4j_settings().database

    @property
    def neo4j_max_pool_size(self) -> int:
        return get_neo4j_settings().max_pool_size


class CodeAnalystSettings(BaseSettings):
    """Code Analyst Agent MCP server configuration.

    Environment variables (prefix: CODE_ANALYST_):
        CODE_ANALYST_HOST                   0.0.0.0
        CODE_ANALYST_PORT                   8004
        CODE_ANALYST_ENVIRONMENT            development
        CODE_ANALYST_LOG_LEVEL              INFO
        CODE_ANALYST_ANALYSIS_MODEL         gpt-4o
        CODE_ANALYST_MAX_SOURCE_CHARS       3500
        CODE_ANALYST_MAX_SNIPPET_LINES      150
        CODE_ANALYST_SNIPPET_CONTEXT_LINES  10
        CODE_ANALYST_ENABLE_CACHE           true
        CODE_ANALYST_CACHE_TTL              1800
    """
    model_config = SettingsConfigDict(
        env_prefix="CODE_ANALYST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8004, ge=1024, le=65535)
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # ── LLM model ─────────────────────────────────────────────────────────────
    analysis_model: str = Field(
        default="gpt-4o",
        description="Model used for all code analysis tasks (quality critical)",
    )
    max_source_chars: int = Field(
        default=3500, ge=500, le=50_000,
        description="Max source code characters sent to LLM per request",
    )

    # ── Snippet settings ──────────────────────────────────────────────────────
    max_snippet_lines: int = Field(
        default=150, ge=10, le=1000,
        description="Max lines returned by get_code_snippet",
    )
    snippet_context_lines: int = Field(
        default=10, ge=0, le=50,
        description="Default context lines before/after in get_code_snippet",
    )

    # ── Cache ─────────────────────────────────────────────────────────────────
    enable_cache: bool = Field(
        default=True,
        description="Cache LLM analysis results (expensive to recompute)",
    )
    cache_ttl: int = Field(
        default=1800, ge=60,
        description="TTL for cached analysis results in seconds",
    )

    @property
    def openai_api_key(self) -> str:
        return get_openai_settings().api_key.get_secret_value()

    # ── Neo4j properties (loaded from shared NEO4J_* prefix) ──────────────────
    @property
    def neo4j_uri(self) -> str:
        return get_neo4j_settings().uri

    @property
    def neo4j_username(self) -> str:
        return get_neo4j_settings().username

    @property
    def neo4j_password(self) -> str:
        return get_neo4j_settings().password.get_secret_value()

    @property
    def neo4j_database(self) -> str:
        return get_neo4j_settings().database

    @property
    def neo4j_max_pool_size(self) -> int:
        return get_neo4j_settings().max_pool_size


# ── Environment-specific config overrides ────────────────────────────────────

class EnvironmentDefaults:
    """Canonical default overrides per environment.

    These are applied as documentation / reference — the actual
    override mechanism is environment variables in .env.development,
    .env.testing, .env.production loaded via DOTENV_PATH.
    """

    DEVELOPMENT: dict[str, str] = {
        "GATEWAY_LOG_LEVEL": "DEBUG",
        "GATEWAY_CORS_ORIGINS": '["*"]',
        "GATEWAY_RATE_LIMIT_PER_MINUTE": "1000",
        "OPENAI_MODEL": "gpt-4o-mini",
        "CODE_ANALYST_ANALYSIS_MODEL": "gpt-4o-mini",
        "ORCHESTRATOR_SYNTHESIS_MODEL": "gpt-4o-mini",
        "REDIS_SESSION_TTL": "3600",          # 1 hour in dev
        "NEO4J_MAX_POOL_SIZE": "10",
    }

    TESTING: dict[str, str] = {
        "GATEWAY_LOG_LEVEL": "WARNING",
        "GATEWAY_CORS_ORIGINS": '["http://testserver"]',
        "GATEWAY_RATE_LIMIT_PER_MINUTE": "10000",
        "OPENAI_API_KEY": "sk-test-placeholder",
        "NEO4J_URI": "bolt://localhost:7688",  # separate test DB
        "REDIS_URL": "redis://localhost:6380/1",  # separate test Redis
        "INDEXER_REPO_LOCAL_PATH": "/tmp/test_repo",
        "INDEXER_MAX_FILE_SIZE_KB": "50",
    }

    PRODUCTION: dict[str, str] = {
        "GATEWAY_LOG_LEVEL": "INFO",
        "GATEWAY_RATE_LIMIT_PER_MINUTE": "60",
        "OPENAI_MODEL": "gpt-4o",
        "CODE_ANALYST_ANALYSIS_MODEL": "gpt-4o",
        "ORCHESTRATOR_SYNTHESIS_MODEL": "gpt-4o",
        "REDIS_SESSION_TTL": "86400",
        "NEO4J_MAX_POOL_SIZE": "50",
    }


# ── Singleton factories (lru_cache ensures one instance per process) ──────────

@lru_cache(maxsize=1)
def get_neo4j_settings() -> Neo4jSettings:
    """Return the singleton Neo4j settings instance."""
    return Neo4jSettings()


@lru_cache(maxsize=1)
def get_openai_settings() -> OpenAISettings:
    """Return the singleton OpenAI settings instance."""
    return OpenAISettings()


@lru_cache(maxsize=1)
def get_redis_settings() -> RedisSettings:
    """Return the singleton Redis settings instance."""
    return RedisSettings()


@lru_cache(maxsize=1)
def get_retry_settings() -> RetrySettings:
    """Return the singleton retry policy settings instance."""
    return RetrySettings()


@lru_cache(maxsize=1)
def get_gateway_settings() -> GatewaySettings:
    """Return the singleton Gateway settings instance."""
    return GatewaySettings()


@lru_cache(maxsize=1)
def get_orchestrator_settings() -> OrchestratorSettings:
    """Return the singleton Orchestrator settings instance."""
    return OrchestratorSettings()


@lru_cache(maxsize=1)
def get_indexer_settings() -> IndexerSettings:
    """Return the singleton Indexer settings instance."""
    return IndexerSettings()


@lru_cache(maxsize=1)
def get_graph_query_settings() -> GraphQuerySettings:
    """Return the singleton Graph Query settings instance."""
    return GraphQuerySettings()


@lru_cache(maxsize=1)
def get_code_analyst_settings() -> CodeAnalystSettings:
    """Return the singleton Code Analyst settings instance."""
    return CodeAnalystSettings()


def clear_settings_cache() -> None:
    """Clear all lru_cache caches — use in tests to reset settings between cases.

    Example:
        >>> from shared.config import clear_settings_cache
        >>> clear_settings_cache()
        >>> os.environ["GATEWAY_PORT"] = "9000"
        >>> settings = get_gateway_settings()  # reloaded
    """
    get_neo4j_settings.cache_clear()
    get_openai_settings.cache_clear()
    get_redis_settings.cache_clear()
    get_retry_settings.cache_clear()
    get_gateway_settings.cache_clear()
    get_orchestrator_settings.cache_clear()
    get_indexer_settings.cache_clear()
    get_graph_query_settings.cache_clear()
    get_code_analyst_settings.cache_clear()