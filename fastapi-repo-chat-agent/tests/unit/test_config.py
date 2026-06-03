"""Unit tests for Pydantic Settings — validation, env overrides, secrets."""
from __future__ import annotations

import os
import pytest

from shared.config import (
    CodeAnalystSettings,
    Environment,
    GatewaySettings,
    GraphQuerySettings,
    IndexerSettings,
    Neo4jSettings,
    OrchestratorSettings,
    OpenAISettings,
    RedisSettings,
    RetrySettings,
    clear_settings_cache,
    get_gateway_settings,
    get_neo4j_settings,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear lru_cache between every test."""
    clear_settings_cache()
    yield
    clear_settings_cache()


# ── Neo4jSettings ─────────────────────────────────────────────────────────────

class TestNeo4jSettings:
    def test_default_uri(self):
        s = Neo4jSettings()
        assert s.uri == "bolt://localhost:7687"

    def test_invalid_uri_scheme_raises(self):
        with pytest.raises(Exception, match="Neo4j URI must start"):
            Neo4jSettings(uri="http://localhost:7687")

    def test_password_is_secret_str(self):
        s = Neo4jSettings(password="supersecret")
        assert "supersecret" not in repr(s)
        assert "supersecret" not in str(s)
        assert s.password.get_secret_value() == "supersecret"

    def test_safe_uri_masks_credentials(self):
        s = Neo4jSettings(uri="bolt://user:pass@host:7687")
        assert "pass" not in s.safe_uri
        assert "***" in s.safe_uri

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "bolt://remotehost:7687")
        monkeypatch.setenv("NEO4J_DATABASE", "mydb")
        s = Neo4jSettings()
        assert s.uri == "bolt://remotehost:7687"
        assert s.database == "mydb"

    def test_pool_size_bounds(self):
        with pytest.raises(Exception):
            Neo4jSettings(max_pool_size=0)
        with pytest.raises(Exception):
            Neo4jSettings(max_pool_size=501)


# ── OpenAISettings ────────────────────────────────────────────────────────────

class TestOpenAISettings:
    def test_api_key_is_secret_str(self):
        s = OpenAISettings(api_key="sk-real-key")
        assert "sk-real-key" not in repr(s)
        assert s.api_key.get_secret_value() == "sk-real-key"

    def test_temperature_bounds(self):
        with pytest.raises(Exception):
            OpenAISettings(temperature=2.1)
        with pytest.raises(Exception):
            OpenAISettings(temperature=-0.1)

    def test_env_override_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4-turbo")
        s = OpenAISettings()
        assert s.model == "gpt-4-turbo"


# ── RetrySettings ─────────────────────────────────────────────────────────────

class TestRetrySettings:
    def test_defaults(self):
        s = RetrySettings()
        assert s.max_attempts == 3
        assert s.wait_min < s.wait_max

    def test_wait_min_greater_than_max_raises(self):
        with pytest.raises(Exception, match="RETRY_WAIT_MIN"):
            RetrySettings(wait_min=15.0, wait_max=5.0)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("RETRY_MAX_ATTEMPTS", "5")
        monkeypatch.setenv("RETRY_WAIT_MIN", "0.5")
        s = RetrySettings()
        assert s.max_attempts == 5
        assert s.wait_min == 0.5


# ── GatewaySettings ───────────────────────────────────────────────────────────

class TestGatewaySettings:
    def test_production_rejects_wildcard_cors(self):
        with pytest.raises(Exception, match="GATEWAY_CORS_ORIGINS"):
            GatewaySettings(
                environment=Environment.PRODUCTION,
                cors_origins=["*"],
                secret_key="a-real-secret-key-for-production",
            )

    def test_production_rejects_default_secret_key(self):
        with pytest.raises(Exception, match="GATEWAY_SECRET_KEY"):
            GatewaySettings(
                environment=Environment.PRODUCTION,
                cors_origins=["https://example.com"],
                secret_key="change-me-in-production",
            )

    def test_production_valid_config(self):
        s = GatewaySettings(
            environment=Environment.PRODUCTION,
            cors_origins=["https://example.com"],
            secret_key="a-long-random-production-key-123",
        )
        assert s.environment == Environment.PRODUCTION

    def test_port_bounds(self):
        with pytest.raises(Exception):
            GatewaySettings(port=80)    # below 1024
        with pytest.raises(Exception):
            GatewaySettings(port=99999)

    def test_secret_key_is_secret_str(self):
        s = GatewaySettings(secret_key="mysecret")
        assert "mysecret" not in repr(s)

    def test_is_debug_true_in_development(self):
        s = GatewaySettings(environment=Environment.DEVELOPMENT)
        assert s.is_debug is True

    def test_is_debug_false_in_production(self):
        s = GatewaySettings(
            environment=Environment.PRODUCTION,
            cors_origins=["https://example.com"],
            secret_key="prod-secret-key-long-enough",
        )
        assert s.is_debug is False

    def test_env_override_via_monkeypatch(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PORT", "9000")
        monkeypatch.setenv("GATEWAY_LOG_LEVEL", "DEBUG")
        s = GatewaySettings()
        assert s.port == 9000
        assert s.log_level == "DEBUG"


# ── Per-agent settings ────────────────────────────────────────────────────────

class TestOrchestratorSettings:
    def test_defaults(self):
        s = OrchestratorSettings()
        assert s.port == 8001
        assert s.max_parallel_agents == 3
        assert s.agent_timeout_seconds == 60.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_MAX_PARALLEL_AGENTS", "2")
        monkeypatch.setenv("ORCHESTRATOR_AGENT_TIMEOUT_SECONDS", "30.0")
        s = OrchestratorSettings()
        assert s.max_parallel_agents == 2
        assert s.agent_timeout_seconds == 30.0


class TestIndexerSettings:
    def test_defaults(self):
        s = IndexerSettings()
        assert s.port == 8002
        assert s.max_file_size_kb == 500
        assert s.include_test_files is False

    def test_exclude_patterns_default(self):
        s = IndexerSettings()
        assert any("pycache" in p for p in s.exclude_patterns)


class TestGraphQuerySettings:
    def test_defaults(self):
        s = GraphQuerySettings()
        assert s.port == 8003
        assert s.result_limit == 50
        assert "MATCH" in s.allowed_query_prefixes

    def test_result_limit_bounds(self):
        with pytest.raises(Exception):
            GraphQuerySettings(result_limit=0)


class TestCodeAnalystSettings:
    def test_defaults(self):
        s = CodeAnalystSettings()
        assert s.port == 8004
        assert s.analysis_model == "gpt-4o"
        assert s.snippet_context_lines == 10


# ── lru_cache singleton behaviour ─────────────────────────────────────────────

class TestSingletonFactories:
    def test_same_instance_returned(self):
        s1 = get_neo4j_settings()
        s2 = get_neo4j_settings()
        assert s1 is s2

    def test_cache_clear_allows_new_instance(self, monkeypatch):
        _ = get_gateway_settings()
        clear_settings_cache()
        monkeypatch.setenv("GATEWAY_PORT", "9999")
        s = get_gateway_settings()
        assert s.port == 9999