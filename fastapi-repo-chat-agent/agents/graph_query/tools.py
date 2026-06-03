"""MCP tool implementations for the Graph Query Agent — all 6 required tools."""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from agents.graph_query.cypher_safety import MAX_RESULT_LIMIT, validate_read_only
from infrastructure.neo4j_client import Neo4jClient
from shared.config import GraphQuerySettings
from shared.exceptions import EntityNotFoundError, InvalidCypherQueryError
from shared.utils.logging import get_logger

logger = get_logger(__name__)


def register_graph_query_tools(
    mcp: FastMCP, settings: GraphQuerySettings, neo4j: Neo4jClient
) -> None:
    """Register all 6 Graph Query Agent MCP tools.

    Args:
        mcp: FastMCP server instance.
        settings: Graph query configuration.
        neo4j: Initialized Neo4j client.
    """

    # ── Tool 1: find_entity ───────────────────────────────────────────────────

    @mcp.tool()
    async def find_entity(
        name: str,
        entity_type: str = "",
        exact_match: bool = False,
    ) -> dict[str, Any]:
        """Locate a class, function, method, or module by name.

        Supports partial/fuzzy matching by default. Searches across all
        node labels unless entity_type is specified.

        Args:
            name: Entity name to search (supports partial match).
            entity_type: Optional label filter: Class, Function, Method,
                         Module, File, Decorator, Import, Parameter.
            exact_match: If True, only return exact name matches.

        Returns:
            Dict with matching entities, their file locations, docstrings,
            decorator lists, and usage counts.

        Raises:
            EntityNotFoundError: If no entities match the search.
        """
        label_clause = f":{entity_type}" if entity_type else ""
        match_clause = (
            "e.name = $name" if exact_match
            else "toLower(e.name) CONTAINS toLower($name)"
        )

        results = await neo4j.run_read(
            f"""
            MATCH (e{label_clause})
            WHERE {match_clause}
            OPTIONAL MATCH (e)<-[:CONTAINS]-(parent)
            OPTIONAL MATCH (e)<-[used_in:CALLS|INHERITS_FROM|IMPORTS]-()
            RETURN
                labels(e)[0]          AS entity_type,
                e.name                AS name,
                e.file_path           AS file_path,
                e.line_start          AS line_start,
                e.line_end            AS line_end,
                e.docstring           AS docstring,
                e.decorators          AS decorators,
                e.return_type         AS return_type,
                e.is_async            AS is_async,
                e.bases               AS bases,
                parent.name           AS parent_name,
                labels(parent)[0]     AS parent_type,
                count(used_in)        AS usage_count
            ORDER BY usage_count DESC, e.name
            LIMIT $limit
            """,
            {"name": name, "limit": settings.result_limit},
        )

        if not results:
            raise EntityNotFoundError(name, entity_type or "entity")

        return {
            "entities": results,
            "count": len(results),
            "search_term": name,
            "exact_match": exact_match,
        }

    # ── Tool 2: get_dependencies ──────────────────────────────────────────────

    @mcp.tool()
    async def get_dependencies(
        entity_name: str,
        depth: int = 1,
        relationship_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find everything a given entity depends on.

        Traverses outgoing IMPORTS, INHERITS_FROM, CALLS, and DEPENDS_ON
        relationships from the entity. Supports multi-hop traversal.

        Args:
            entity_name: Name of the class, function, or module.
            depth: Traversal depth (1 = direct only, up to max_query_depth).
            relationship_types: Optional filter list e.g. ["IMPORTS", "CALLS"].

        Returns:
            Dict with dependency list, relationship types, and traversal depth.
        """
        depth = min(max(depth, 1), settings.max_query_depth)
        rel_filter = "|".join(relationship_types) if relationship_types else \
                     "IMPORTS|INHERITS_FROM|CALLS|DEPENDS_ON|DECORATED_BY"

        results = await neo4j.run_read(
            f"""
            MATCH (e {{name: $name}})
            CALL {{
                WITH e
                MATCH path = (e)-[:{rel_filter}*1..{depth}]->(dep)
                RETURN
                    dep.name              AS dep_name,
                    labels(dep)[0]        AS dep_type,
                    dep.file_path         AS dep_file,
                    dep.docstring         AS dep_docstring,
                    length(path)          AS hop_count,
                    [r in relationships(path) | type(r)] AS relationship_chain
                ORDER BY hop_count, dep_name
                LIMIT $limit
            }}
            RETURN dep_name, dep_type, dep_file, dep_docstring,
                   hop_count, relationship_chain
            """,
            {"name": entity_name, "limit": settings.result_limit},
        )

        # Group by hop count for clarity
        by_depth: dict[int, list[dict[str, Any]]] = {}
        for r in results:
            hop = r.get("hop_count", 1)
            if hop not in by_depth:
                by_depth[hop] = []
            by_depth[hop].append({
                "name": r.get("dep_name"),
                "type": r.get("dep_type"),
                "file": r.get("dep_file"),
                "docstring": r.get("dep_docstring"),
                "via": r.get("relationship_chain", []),
            })

        return {
            "entity": entity_name,
            "total_dependencies": len(results),
            "max_depth_searched": depth,
            "dependencies_by_depth": by_depth,
            "flat_list": results,
        }

    # ── Tool 3: get_dependents ────────────────────────────────────────────────

    @mcp.tool()
    async def get_dependents(
        entity_name: str,
        depth: int = 1,
        relationship_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find everything that depends on a given entity (reverse lookup).

        Traverses INCOMING relationships — who imports, inherits from,
        calls, or decorates the target entity.

        Args:
            entity_name: Name of the target entity.
            depth: Reverse traversal depth (1 = direct dependents only).
            relationship_types: Optional filter e.g. ["INHERITS_FROM"].

        Returns:
            Dict with dependent entities grouped by relationship type.
        """
        depth = min(max(depth, 1), settings.max_query_depth)
        rel_filter = "|".join(relationship_types) if relationship_types else \
                     "IMPORTS|INHERITS_FROM|CALLS|DEPENDS_ON|DECORATED_BY"

        results = await neo4j.run_read(
            f"""
            MATCH (e {{name: $name}})
            CALL {{
                WITH e
                MATCH path = (dep)-[:{rel_filter}*1..{depth}]->(e)
                RETURN
                    dep.name              AS dep_name,
                    labels(dep)[0]        AS dep_type,
                    dep.file_path         AS dep_file,
                    length(path)          AS hop_count,
                    type(relationships(path)[0]) AS direct_relationship
                ORDER BY hop_count, dep_name
                LIMIT $limit
            }}
            RETURN dep_name, dep_type, dep_file, hop_count, direct_relationship
            """,
            {"name": entity_name, "limit": settings.result_limit},
        )

        # Group by relationship type
        by_rel: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            rel_type = r.get("direct_relationship", "UNKNOWN")
            if rel_type not in by_rel:
                by_rel[rel_type] = []
            by_rel[rel_type].append({
                "name": r.get("dep_name"),
                "type": r.get("dep_type"),
                "file": r.get("dep_file"),
                "hop_count": r.get("hop_count"),
            })

        return {
            "entity": entity_name,
            "total_dependents": len(results),
            "max_depth_searched": depth,
            "dependents_by_relationship": by_rel,
            "flat_list": results,
        }

    # ── Tool 4: trace_imports ─────────────────────────────────────────────────

    @mcp.tool()
    async def trace_imports(
        module_name: str,
        depth: int = 3,
        direction: str = "outgoing",
    ) -> dict[str, Any]:
        """Follow the complete import chain for a module.

        Traces either what a module imports (outgoing) or what imports
        it (incoming), up to the specified depth. Detects circular imports.

        Args:
            module_name: Module or file name to trace (e.g. "applications").
            depth: Maximum chain depth to follow (capped at max_query_depth).
            direction: "outgoing" (what it imports) or "incoming" (who imports it).

        Returns:
            Dict with import chains as paths, cycle detection flag,
            and a flat unique list of all modules in the chain.
        """
        depth = min(max(depth, 1), settings.max_query_depth)

        if direction == "incoming":
            cypher = f"""
            MATCH (importer)-[:IMPORTS*1..{depth}]->(target)
            WHERE target.name CONTAINS $module OR target.module CONTAINS $module
            WITH importer, target,
                 shortestPath((importer)-[:IMPORTS*]->(target)) AS sp
            RETURN
                importer.name          AS importer,
                importer.file_path     AS importer_file,
                target.name            AS target,
                length(sp)             AS depth,
                [n in nodes(sp) | coalesce(n.name, n.path)] AS chain
            ORDER BY depth, importer
            LIMIT $limit
            """
        else:
            cypher = f"""
            MATCH (source)
            WHERE source.name CONTAINS $module OR source.path CONTAINS $module
            CALL {{
                WITH source
                MATCH path = (source)-[:IMPORTS*1..{depth}]->(imported)
                RETURN
                    imported.name      AS imported_name,
                    imported.module    AS imported_module,
                    length(path)       AS depth,
                    [n in nodes(path) | coalesce(n.name, n.path, n.module)] AS chain
                ORDER BY depth
                LIMIT $limit
            }}
            RETURN imported_name, imported_module, depth, chain
            """

        results = await neo4j.run_read(
            cypher,
            {"module": module_name, "limit": settings.result_limit},
        )

        # Detect potential circular imports
        all_names = set()
        chains: list[dict[str, Any]] = []
        has_cycles = False

        for r in results:
            chain = r.get("chain", [])
            if len(chain) != len(set(chain)):
                has_cycles = True
            all_names.update(n for n in chain if n)
            chains.append({
                "chain": chain,
                "depth": r.get("depth", 0),
                "imported": r.get("imported_name") or r.get("target"),
            })

        return {
            "module": module_name,
            "direction": direction,
            "depth_searched": depth,
            "import_chains": chains,
            "unique_modules": sorted(all_names - {None}),
            "total_imports_found": len(results),
            "circular_imports_detected": has_cycles,
        }

    # ── Tool 5: find_related ──────────────────────────────────────────────────

    @mcp.tool()
    async def find_related(
        entity_name: str,
        relationship_type: str,
        depth: int = 1,
        direction: str = "both",
    ) -> dict[str, Any]:
        """Get entities related to a target by a specific relationship type.

        Supports all 8 schema relationship types:
        CONTAINS, IMPORTS, INHERITS_FROM, CALLS, DECORATED_BY,
        HAS_PARAMETER, DOCUMENTED_BY, DEPENDS_ON

        Args:
            entity_name: Source entity name to start from.
            relationship_type: Neo4j relationship type string.
            depth: Traversal depth (default 1).
            direction: "outgoing", "incoming", or "both".

        Returns:
            Dict with related entities grouped by direction, with file paths.
        """
        depth = min(max(depth, 1), settings.max_query_depth)

        # Sanitize relationship type — only allow schema-defined types
        allowed_rels = {
            "CONTAINS", "IMPORTS", "INHERITS_FROM", "CALLS",
            "DECORATED_BY", "HAS_PARAMETER", "DOCUMENTED_BY", "DEPENDS_ON",
        }
        safe_rel = relationship_type.upper().replace(" ", "_").replace("-", "_")
        if safe_rel not in allowed_rels:
            return {
                "error": f"Unknown relationship type: '{relationship_type}'",
                "allowed_types": sorted(allowed_rels),
            }

        outgoing: list[dict[str, Any]] = []
        incoming: list[dict[str, Any]] = []

        if direction in ("outgoing", "both"):
            out_results = await neo4j.run_read(
                f"""
                MATCH (e {{name: $name}})-[:{safe_rel}*1..{depth}]->(related)
                RETURN DISTINCT
                    labels(related)[0]  AS entity_type,
                    related.name        AS name,
                    related.file_path   AS file_path,
                    related.line_start  AS line_start,
                    related.docstring   AS docstring
                ORDER BY name
                LIMIT $limit
                """,
                {"name": entity_name, "limit": settings.result_limit},
            )
            outgoing = out_results

        if direction in ("incoming", "both"):
            in_results = await neo4j.run_read(
                f"""
                MATCH (related)-[:{safe_rel}*1..{depth}]->(e {{name: $name}})
                RETURN DISTINCT
                    labels(related)[0]  AS entity_type,
                    related.name        AS name,
                    related.file_path   AS file_path,
                    related.line_start  AS line_start,
                    related.docstring   AS docstring
                ORDER BY name
                LIMIT $limit
                """,
                {"name": entity_name, "limit": settings.result_limit},
            )
            incoming = in_results

        return {
            "entity": entity_name,
            "relationship_type": safe_rel,
            "direction": direction,
            "depth": depth,
            "outgoing": outgoing,
            "incoming": incoming,
            "total_outgoing": len(outgoing),
            "total_incoming": len(incoming),
            "total_related": len(outgoing) + len(incoming),
        }

    # ── Tool 6: execute_query ─────────────────────────────────────────────────

    @mcp.tool()
    async def execute_query(
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a custom read-only Cypher query with full safety validation.

        The query is validated through three safety checks before execution:
        1. Must start with a read-only clause (MATCH, OPTIONAL, WITH, etc.)
        2. No write keywords (CREATE, MERGE, SET, DELETE, etc.)
        3. No dangerous APOC procedures or multi-statement injection

        A LIMIT clause is automatically appended if not present.

        Args:
            cypher: A read-only Cypher query string.
            params: Optional query parameters dict for parameterized queries.

        Returns:
            Dict with results list, count, and the validated query used.

        Raises:
            InvalidCypherQueryError: If safety validation fails.

        Example:
            cypher = "MATCH (c:Class) WHERE c.bases <> [] RETURN c.name, c.bases"
        """
        validated_query = validate_read_only(cypher)

        results = await neo4j.run_read(validated_query, params or {})

        logger.info(
            "custom_query_executed",
            result_count=len(results),
            query_preview=validated_query[:80],
        )

        return {
            "results": results,
            "count": len(results),
            "query_used": validated_query,
            "params_used": params or {},
            "truncated": len(results) >= MAX_RESULT_LIMIT,
        }

    # ── Bonus Tool: find_usage_patterns ──────────────────────────────────────
    # Covers the "identify usage patterns across the codebase" responsibility

    @mcp.tool()
    async def find_usage_patterns(
        pattern_type: str = "most_called",
    ) -> dict[str, Any]:
        """Identify usage patterns across the entire codebase.

        Covers the Graph Query Agent's responsibility to identify usage
        patterns — which entities are most called, most inherited from,
        most imported, and which files are most connected.

        Args:
            pattern_type: One of:
                - "most_called"      — functions/methods called most often
                - "most_inherited"   — classes most used as base classes
                - "most_imported"    — modules imported most frequently
                - "most_connected"   — entities with most total relationships
                - "decorator_usage"  — which decorators are used most
                - "async_coverage"   — ratio of async vs sync functions

        Returns:
            Dict with ranked entity list and pattern statistics.
        """
        queries: dict[str, str] = {
            "most_called": """
                MATCH (fn)<-[:CALLS]-(caller)
                WITH fn, count(caller) AS call_count
                WHERE call_count > 0
                RETURN labels(fn)[0] AS type, fn.name AS name,
                       fn.file_path AS file_path, call_count
                ORDER BY call_count DESC LIMIT 20
            """,
            "most_inherited": """
                MATCH (child:Class)-[:INHERITS_FROM]->(parent)
                WITH parent, count(child) AS child_count
                WHERE child_count > 0
                RETURN labels(parent)[0] AS type, parent.name AS name,
                       parent.file_path AS file_path, child_count AS usage_count
                ORDER BY child_count DESC LIMIT 20
            """,
            "most_imported": """
                MATCH (f:File)-[:IMPORTS]->(i:Import)
                WITH i, count(f) AS import_count
                WHERE import_count > 0
                RETURN 'Import' AS type, i.name AS name,
                       i.module AS module, import_count AS usage_count
                ORDER BY import_count DESC LIMIT 20
            """,
            "most_connected": """
                MATCH (e)-[r]-()
                WITH e, count(r) AS rel_count
                WHERE rel_count > 2
                RETURN labels(e)[0] AS type, e.name AS name,
                       e.file_path AS file_path, rel_count AS connection_count
                ORDER BY rel_count DESC LIMIT 20
            """,
            "decorator_usage": """
                MATCH (e)-[:DECORATED_BY]->(d:Decorator)
                WITH d, count(e) AS usage_count
                RETURN 'Decorator' AS type, d.name AS name,
                       usage_count
                ORDER BY usage_count DESC LIMIT 20
            """,
            "async_coverage": """
                MATCH (fn)
                WHERE fn:Function OR fn:Method
                WITH
                    count(fn) AS total,
                    sum(CASE WHEN fn.is_async = true THEN 1 ELSE 0 END) AS async_count
                RETURN total, async_count,
                       round(toFloat(async_count) / total * 100, 1) AS async_percentage
            """,
        }

        if pattern_type not in queries:
            return {
                "error": f"Unknown pattern_type: '{pattern_type}'",
                "available_patterns": list(queries.keys()),
            }

        results = await neo4j.run_read(queries[pattern_type])

        return {
            "pattern_type": pattern_type,
            "results": results,
            "count": len(results),
        }