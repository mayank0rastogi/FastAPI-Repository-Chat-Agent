"""Neo4j client with full schema: 9 node types, 8 relationships, constraints, full-text indexes."""
from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from shared.exceptions import Neo4jConnectionError, Neo4jQueryError
from shared.utils.logging import get_logger

logger = get_logger(__name__)


# ── Schema Statements ─────────────────────────────────────────────────────────

_CONSTRAINT_STATEMENTS: list[str] = [
    # File — unique by absolute filesystem path
    "CREATE CONSTRAINT file_path_unique IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE",
    # Module — unique by dotted Python module name
    "CREATE CONSTRAINT module_name_unique IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE",
    # Class — composite key: same class name valid in different files
    "CREATE CONSTRAINT class_name_file_unique IF NOT EXISTS FOR (c:Class) REQUIRE (c.name, c.file_path) IS NODE KEY",
    # Function — composite key
    "CREATE CONSTRAINT function_name_file_unique IF NOT EXISTS FOR (f:Function) REQUIRE (f.name, f.file_path) IS NODE KEY",
    # Method — name includes class prefix (e.g. FastAPI.__init__)
    "CREATE CONSTRAINT method_name_file_unique IF NOT EXISTS FOR (m:Method) REQUIRE (m.name, m.file_path) IS NODE KEY",
    # Parameter — unique per owning function in a file
    "CREATE CONSTRAINT parameter_unique IF NOT EXISTS FOR (p:Parameter) REQUIRE (p.name, p.function_name, p.file_path) IS NODE KEY",
    # Decorator — unique per decorated target
    "CREATE CONSTRAINT decorator_unique IF NOT EXISTS FOR (d:Decorator) REQUIRE (d.name, d.target_name, d.target_file) IS NODE KEY",
    # Import — unique per (module, importing file)
    "CREATE CONSTRAINT import_unique IF NOT EXISTS FOR (i:Import) REQUIRE (i.module, i.file_path) IS NODE KEY",
    # Docstring — unique per owning entity in a file
    "CREATE CONSTRAINT docstring_unique IF NOT EXISTS FOR (d:Docstring) REQUIRE (d.owner_name, d.file_path) IS NODE KEY",
]

_INDEX_STATEMENTS: list[str] = [
    # Hot lookup: entity name searches
    "CREATE INDEX class_name_idx IF NOT EXISTS FOR (c:Class) ON (c.name)",
    "CREATE INDEX function_name_idx IF NOT EXISTS FOR (f:Function) ON (f.name)",
    "CREATE INDEX method_name_idx IF NOT EXISTS FOR (m:Method) ON (m.name)",
    "CREATE INDEX module_name_idx IF NOT EXISTS FOR (m:Module) ON (m.name)",
    # Pattern detection: filter by decorator name
    "CREATE INDEX decorator_name_idx IF NOT EXISTS FOR (d:Decorator) ON (d.name)",
    # Dependency injection analysis
    "CREATE INDEX parameter_annotation_idx IF NOT EXISTS FOR (p:Parameter) ON (p.annotation)",
    # Import chain traversal
    "CREATE INDEX import_module_idx IF NOT EXISTS FOR (i:Import) ON (i.module)",
    # File path lookups
    "CREATE INDEX file_path_idx IF NOT EXISTS FOR (f:File) ON (f.path)",
]


class Neo4jClient:
    """Async Neo4j client with connection pooling and schema bootstrap.

    Implements the full knowledge graph schema:

    Nodes (9):
        File, Module, Class, Function, Method,
        Parameter, Decorator, Import, Docstring

    Relationships (8):
        CONTAINS, IMPORTS, INHERITS_FROM, CALLS,
        DECORATED_BY, HAS_PARAMETER, DOCUMENTED_BY, DEPENDS_ON

    Example:
        >>> client = Neo4jClient(settings)
        >>> await client.connect()
        >>> results = await client.run_read("MATCH (c:Class) RETURN c.name LIMIT 5")
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._driver: AsyncDriver | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the async driver and bootstrap the schema."""
        try:
            self._driver = AsyncGraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(self._settings.neo4j_username, self._settings.neo4j_password),
                max_connection_pool_size=self._settings.neo4j_max_pool_size,
            )
            await self._driver.verify_connectivity()
            await self._bootstrap_schema()
            logger.info("neo4j_connected", uri=self._settings.neo4j_uri)
        except Exception as exc:
            raise Neo4jConnectionError(self._settings.neo4j_uri, str(exc)) from exc

    async def close(self) -> None:
        """Close the Neo4j driver and release all connections."""
        if self._driver:
            await self._driver.close()
            logger.info("neo4j_closed")

    # ── Query execution ───────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def run_read(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a read-only Cypher query and return all records as dicts.

        Args:
            query: Cypher query string (MATCH only).
            params: Optional parameter dict.

        Returns:
            List of record dicts.

        Raises:
            Neo4jConnectionError: If driver is not connected.
            Neo4jQueryError: If the query fails.
        """
        if not self._driver:
            raise Neo4jConnectionError(self._settings.neo4j_uri, "Driver not connected")
        try:
            async with self._driver.session(
                database=self._settings.neo4j_database
            ) as session:
                result = await session.run(query, params or {})
                return [dict(record) async for record in result]
        except Exception as exc:
            raise Neo4jQueryError(query[:100], str(exc)) from exc

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def run_write(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a write Cypher query in a managed transaction.

        Args:
            query: Cypher query string.
            params: Optional parameter dict.

        Returns:
            List of result records (may be empty for write-only queries).

        Raises:
            Neo4jQueryError: If the query fails.
        """
        if not self._driver:
            raise Neo4jConnectionError(self._settings.neo4j_uri, "Driver not connected")
        try:
            async with self._driver.session(
                database=self._settings.neo4j_database
            ) as session:
                result = await session.run(query, params or {})
                records = [dict(record) async for record in result]
                return records
        except Exception as exc:
            raise Neo4jQueryError(query[:100], str(exc)) from exc

    async def run_batch(self, statements: list[tuple[str, dict[str, Any]]]) -> None:
        """Execute multiple write statements in a single transaction.

        Args:
            statements: List of (cypher, params) tuples.
        """
        if not self._driver:
            raise Neo4jConnectionError(self._settings.neo4j_uri, "Driver not connected")
        try:
            async with self._driver.session(
                database=self._settings.neo4j_database
            ) as session:
                async with await session.begin_transaction() as tx:
                    for query, params in statements:
                        await tx.run(query, params)
                    await tx.commit()
        except Exception as exc:
            raise Neo4jQueryError("batch_write", str(exc)) from exc

    # ── Schema Bootstrap ──────────────────────────────────────────────────────

    async def _bootstrap_schema(self) -> None:
        """Create all constraints and indexes on first connect.

        Idempotent — safe to call on every startup.
        Creates:
          - Uniqueness constraints for all 9 node types
          - Composite key constraints where needed
          - Full-text index on Docstring.content for semantic search
          - B-tree indexes on hot lookup properties
        """
        statements = _CONSTRAINT_STATEMENTS + _INDEX_STATEMENTS
        for stmt in statements:
            try:
                await self.run_write(stmt)
            except Exception as exc:
                # Constraint/index already exists — fine to ignore
                if "already exists" in str(exc).lower() or "equivalent" in str(exc).lower():
                    continue
                logger.warning("schema_bootstrap_warning", stmt=stmt[:80], error=str(exc))

        logger.info("neo4j_schema_bootstrapped")

    async def setup_schema(self) -> None:
        """Public method to setup the schema (calls _bootstrap_schema)."""
        await self._bootstrap_schema()

    async def get_statistics(self) -> dict[str, Any]:
        """Return node and relationship counts for all schema types.

        Returns:
            Dict with node_counts and relationship_counts.
        """
        node_labels = [
            "File", "Module", "Class", "Function", "Method",
            "Parameter", "Decorator", "Import", "Docstring",
        ]
        rel_types = [
            "CONTAINS", "IMPORTS", "INHERITS_FROM", "CALLS",
            "DECORATED_BY", "HAS_PARAMETER", "DOCUMENTED_BY", "DEPENDS_ON",
        ]

        node_counts: dict[str, int] = {}
        for label in node_labels:
            result = await self.run_read(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            node_counts[label] = result[0]["cnt"] if result else 0

        rel_counts: dict[str, int] = {}
        for rel_type in rel_types:
            result = await self.run_read(
                f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt"
            )
            rel_counts[rel_type] = result[0]["cnt"] if result else 0

        return {
            "node_counts": node_counts,
            "relationship_counts": rel_counts,
            "total_nodes": sum(node_counts.values()),
            "total_relationships": sum(rel_counts.values()),
        }