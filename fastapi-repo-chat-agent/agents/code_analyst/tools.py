"""MCP tool implementations for the Code Analyst Agent — all 6 required tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

from agents.code_analyst.prompts import (
    ANALYZE_CLASS_PROMPT,
    ANALYZE_FUNCTION_PROMPT,
    COMPARE_PROMPT,
    EXPLAIN_IMPLEMENTATION_PROMPT,
    FIND_PATTERNS_PROMPT,
    SYSTEM_PROMPT,
)
from infrastructure.neo4j_client import Neo4jClient
from shared.config import CodeAnalystSettings
from shared.exceptions import CodeAnalystError, EntityNotFoundError
from shared.utils.logging import get_logger

logger = get_logger(__name__)


def register_code_analyst_tools(
    mcp: FastMCP,
    settings: CodeAnalystSettings,
    openai_client: AsyncOpenAI,
    neo4j: Neo4jClient,
) -> None:
    """Register all 6 Code Analyst Agent MCP tools.

    Args:
        mcp: FastMCP server instance.
        settings: Code analyst configuration.
        openai_client: Configured AsyncOpenAI client.
        neo4j: Neo4j client for fetching stored source code.
    """

    async def _llm(prompt: str, max_tokens: int = 2048) -> dict[str, Any]:
        """Call the LLM with JSON response mode and return parsed dict.

        Args:
            prompt: Full user-facing prompt (system prompt is always prepended).
            max_tokens: Maximum tokens for the completion.

        Returns:
            Parsed JSON dict from the model's response.

        Raises:
            CodeAnalystError: If LLM call or JSON parsing fails.
        """
        try:
            resp = await openai_client.chat.completions.create(
                model=settings.analysis_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content or "{}"
            result = json.loads(raw)
            result["_tokens_used"] = resp.usage.total_tokens if resp.usage else 0
            return result
        except json.JSONDecodeError as exc:
            raise CodeAnalystError(f"LLM returned invalid JSON: {exc}") from exc
        except Exception as exc:
            raise CodeAnalystError(f"LLM call failed: {exc}") from exc

    async def _fetch_entity(
        name: str, label: str = ""
    ) -> dict[str, Any]:
        """Fetch entity record from Neo4j by name.

        Args:
            name: Entity name to look up.
            label: Optional Neo4j label filter (Class, Function, Method).

        Returns:
            Dict of entity properties.

        Raises:
            EntityNotFoundError: If no matching entity is found.
        """
        label_clause = f":{label}" if label else ""
        results = await neo4j.run_read(
            f"""
            MATCH (e{label_clause})
            WHERE e.name = $name OR e.name ENDS WITH ('.' + $name)
            RETURN e {{
                .name, .file_path, .line_start, .line_end,
                .docstring, .source_code, .bases, .decorators,
                .return_type, .is_async, .method_count, .class_name,
                .is_property, .params
            }} AS entity,
            labels(e)[0] AS entity_type
            ORDER BY size(e.name)   // prefer shorter (exact) match
            LIMIT 1
            """,
            {"name": name},
        )
        if not results:
            raise EntityNotFoundError(name, label or "entity")
        record = dict(results[0]["entity"])
        record["entity_type"] = results[0]["entity_type"]
        return record

    async def _get_surrounding_lines(
        file_path: str,
        line_start: int,
        line_end: int,
        context_lines: int,
    ) -> dict[str, Any]:
        """Read a file from disk and extract source with surrounding context.

        Args:
            file_path: Absolute path to the Python source file.
            line_start: First line of the entity (1-indexed).
            line_end: Last line of the entity (1-indexed).
            context_lines: Lines of context before and after.

        Returns:
            Dict with snippet, context_before, context_after, and full_range.
        """
        path = Path(file_path)
        if not path.exists():
            return {
                "snippet": "(file not available on disk)",
                "context_before": [],
                "context_after": [],
            }

        all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(all_lines)

        before_start = max(0, line_start - 1 - context_lines)
        after_end = min(total, line_end + context_lines)

        return {
            "context_before": all_lines[before_start : line_start - 1],
            "snippet": "\n".join(all_lines[line_start - 1 : line_end]),
            "context_after": all_lines[line_end : after_end],
            "full_range": f"lines {before_start + 1}–{after_end}",
            "total_file_lines": total,
        }

    # ── Tool 1: analyze_function ──────────────────────────────────────────────

    @mcp.tool()
    async def analyze_function(function_name: str) -> dict[str, Any]:
        """Perform deep analysis of a function's logic and behaviour.

        Covers: complexity, parameters, return value, side effects,
        error handling, design patterns, anti-patterns, best practices,
        FastAPI concepts demonstrated, and execution flow.

        Args:
            function_name: Name of the function or method (e.g. "Depends",
                           "APIRouter.include_router", "get_openapi").

        Returns:
            Structured analysis dict — see ANALYZE_FUNCTION_PROMPT schema.

        Raises:
            EntityNotFoundError: If function not found in knowledge graph.
            CodeAnalystError: If LLM analysis fails.
        """
        entity = await _fetch_entity(function_name)
        source = entity.get("source_code") or ""

        if not source:
            return {
                "warning": f"No source stored for '{function_name}'. "
                           "Run index_repository first.",
                "entity": entity,
            }

        prompt = ANALYZE_FUNCTION_PROMPT.format(
            file_path=entity.get("file_path", "unknown"),
            line_start=entity.get("line_start", 0),
            line_end=entity.get("line_end", 0),
            source_code=source[:3500],
            params=json.dumps(entity.get("params", []), default=str),
            decorators=json.dumps(entity.get("decorators", []), default=str),
        )

        result = await _llm(prompt, max_tokens=2500)
        result["entity_name"] = function_name
        result["file_path"] = entity.get("file_path")
        result["line_start"] = entity.get("line_start")
        result["line_end"] = entity.get("line_end")

        logger.info(
            "function_analyzed",
            name=function_name,
            complexity=result.get("complexity"),
            patterns=result.get("patterns_used", []),
        )
        return result

    # ── Tool 2: analyze_class ─────────────────────────────────────────────────

    @mcp.tool()
    async def analyze_class(class_name: str) -> dict[str, Any]:
        """Perform comprehensive analysis of a class.

        Covers: inheritance, public/private API, state management,
        design patterns, anti-patterns, SOLID compliance assessment,
        cohesion, coupling, and improvement suggestions.

        Args:
            class_name: Name of the class (e.g. "FastAPI", "APIRouter",
                        "HTTPException", "Depends").

        Returns:
            Structured analysis dict — see ANALYZE_CLASS_PROMPT schema.

        Raises:
            EntityNotFoundError: If class not found in knowledge graph.
        """
        entity = await _fetch_entity(class_name, label="Class")
        source = entity.get("source_code") or ""

        # Also fetch methods for richer context
        methods = await neo4j.run_read(
            """
            MATCH (c:Class {name: $name})-[:CONTAINS]->(m:Method)
            RETURN m.name AS name, m.is_async AS is_async,
                   m.decorators AS decorators, m.return_type AS return_type,
                   m.docstring AS docstring
            ORDER BY m.line_start
            LIMIT 40
            """,
            {"name": class_name},
        )

        prompt = ANALYZE_CLASS_PROMPT.format(
            file_path=entity.get("file_path", "unknown"),
            line_start=entity.get("line_start", 0),
            line_end=entity.get("line_end", 0),
            source_code=source[:3500],
            bases=json.dumps(entity.get("bases", []), default=str),
            decorators=json.dumps(entity.get("decorators", []), default=str),
            method_count=entity.get("method_count", len(methods)),
        )

        result = await _llm(prompt, max_tokens=3000)
        result["entity_name"] = class_name
        result["file_path"] = entity.get("file_path")
        result["methods_found"] = [m["name"] for m in methods]
        result["bases"] = entity.get("bases", [])

        logger.info(
            "class_analyzed",
            name=class_name,
            cohesion=result.get("cohesion"),
            patterns=len(result.get("design_patterns", [])),
        )
        return result

    # ── Tool 3: find_patterns ─────────────────────────────────────────────────

    @mcp.tool()
    async def find_patterns(
        file_path: str = "",
        entity_names: list[str] | None = None,
        include_anti_patterns: bool = True,
    ) -> dict[str, Any]:
        """Detect design patterns and anti-patterns across the codebase.

        Can scope analysis to a single file, specific entities, or the
        entire repository. Always checks for both patterns and anti-patterns.

        Args:
            file_path: Optional file path to limit scope.
            entity_names: Optional list of specific entity names to analyse.
            include_anti_patterns: Include anti-pattern detection (default True).

        Returns:
            Structured pattern report — see FIND_PATTERNS_PROMPT schema.
        """
        # Build query to fetch entities for analysis
        if entity_names:
            entities = await neo4j.run_read(
                """
                MATCH (e) WHERE e.name IN $names
                RETURN labels(e)[0] AS type, e.name AS name,
                       e.bases AS bases, e.decorators AS decorators,
                       e.source_code AS source, e.file_path AS file_path,
                       e.docstring AS docstring
                LIMIT 30
                """,
                {"names": entity_names},
            )
        elif file_path:
            entities = await neo4j.run_read(
                """
                MATCH (e) WHERE e.file_path = $path
                AND (e:Class OR e:Function OR e:Method)
                RETURN labels(e)[0] AS type, e.name AS name,
                       e.bases AS bases, e.decorators AS decorators,
                       e.source_code AS source, e.file_path AS file_path,
                       e.docstring AS docstring
                LIMIT 30
                """,
                {"path": file_path},
            )
        else:
            # Broad repository scan — use most-connected entities
            entities = await neo4j.run_read(
                """
                MATCH (e)-[r]-()
                WHERE e:Class OR e:Function
                WITH e, count(r) AS rel_count
                WHERE rel_count > 2
                RETURN labels(e)[0] AS type, e.name AS name,
                       e.bases AS bases, e.decorators AS decorators,
                       e.source_code AS source, e.file_path AS file_path,
                       e.docstring AS docstring
                ORDER BY rel_count DESC
                LIMIT 25
                """
            )

        if not entities:
            return {
                "message": "No entities found for pattern analysis. "
                           "Run index_repository first.",
                "design_patterns": [],
                "anti_patterns": [],
            }

        # Build concise entity summaries for LLM (avoid token overflow)
        summaries = []
        for e in entities:
            summary: dict[str, Any] = {
                "name": e["name"],
                "type": e["type"],
                "file": (e.get("file_path") or "").split("/")[-1],
                "bases": e.get("bases") or [],
                "decorators": e.get("decorators") or [],
                "docstring_preview": (e.get("docstring") or "")[:100],
            }
            # Include first 400 chars of source for pattern evidence
            if e.get("source"):
                summary["source_preview"] = e["source"][:400]
            summaries.append(summary)

        prompt = FIND_PATTERNS_PROMPT.format(
            entity_summaries=json.dumps(summaries, indent=2, default=str)[:5000],
        )

        result = await _llm(prompt, max_tokens=3000)
        result["entities_analysed"] = len(entities)
        result["scope"] = file_path or (", ".join(entity_names or [])) or "repository-wide"

        logger.info(
            "patterns_found",
            patterns=len(result.get("design_patterns", [])),
            anti_patterns=len(result.get("anti_patterns", [])),
            scope=result["scope"],
        )
        return result

    # ── Tool 4: get_code_snippet ──────────────────────────────────────────────

    @mcp.tool()
    async def get_code_snippet(
        entity_name: str,
        context_lines: int | None = None,
        include_docstring: bool = True,
    ) -> dict[str, Any]:
        """Extract a code entity with surrounding context lines.

        Reads the actual file from disk to provide accurate context,
        falling back to the stored source_code if the file is unavailable.

        Args:
            entity_name: Name of the function, class, or method.
            context_lines: Lines of context before/after (defaults to
                           settings.snippet_context_lines).
            include_docstring: Include the parsed docstring separately.

        Returns:
            Dict with snippet, context_before, context_after, file metadata,
            and optional docstring.

        Raises:
            EntityNotFoundError: If entity not found in knowledge graph.
        """
        ctx = context_lines if context_lines is not None else settings.snippet_context_lines
        ctx = min(ctx, 50)  # Safety cap

        entity = await _fetch_entity(entity_name)
        file_path = entity.get("file_path") or ""
        line_start = entity.get("line_start") or 1
        line_end = entity.get("line_end") or line_start

        # Try to get real file context
        context_data = await _get_surrounding_lines(
            file_path, line_start, line_end, ctx
        )

        # Fall back to stored source if file unavailable
        if context_data["snippet"] == "(file not available on disk)":
            stored = entity.get("source_code") or "(no source stored)"
            context_data["snippet"] = stored[:settings.max_snippet_lines * 80]

        result: dict[str, Any] = {
            "entity_name": entity_name,
            "entity_type": entity.get("entity_type"),
            "file_path": file_path,
            "line_start": line_start,
            "line_end": line_end,
            "context_before": context_data.get("context_before", []),
            "snippet": context_data.get("snippet", ""),
            "context_after": context_data.get("context_after", []),
            "full_range": context_data.get("full_range", ""),
            "total_file_lines": context_data.get("total_file_lines", 0),
            "snippet_lines": line_end - line_start + 1,
        }

        if include_docstring:
            result["docstring"] = entity.get("docstring") or ""

        return result

    # ── Tool 5: explain_implementation ───────────────────────────────────────

    @mcp.tool()
    async def explain_implementation(entity_name: str) -> dict[str, Any]:
        """Generate a developer-friendly step-by-step explanation of code.

        Tailored for developers new to FastAPI — explains prerequisites,
        execution flow, FastAPI-specific magic, gotchas, and related components.

        Args:
            entity_name: The function, class, or method to explain.

        Returns:
            Structured explanation — see EXPLAIN_IMPLEMENTATION_PROMPT schema.

        Raises:
            EntityNotFoundError: If entity not found in knowledge graph.
        """
        entity = await _fetch_entity(entity_name)
        source = entity.get("source_code") or ""

        if not source:
            return {
                "warning": f"No source stored for '{entity_name}'. "
                           "Run index_repository first.",
                "entity": entity,
            }

        prompt = EXPLAIN_IMPLEMENTATION_PROMPT.format(
            file_path=entity.get("file_path", "unknown"),
            entity_name=entity_name,
            entity_type=entity.get("entity_type", "entity"),
            source_code=source[:3500],
        )

        result = await _llm(prompt, max_tokens=2500)
        result["entity_name"] = entity_name
        result["entity_type"] = entity.get("entity_type")
        result["file_path"] = entity.get("file_path")

        logger.info("implementation_explained", name=entity_name)
        return result

    # ── Tool 6: compare_implementations ──────────────────────────────────────

    @mcp.tool()
    async def compare_implementations(
        entity_a: str, entity_b: str
    ) -> dict[str, Any]:
        """Compare two code entities highlighting similarities and differences.

        Provides a structured diff covering: purpose, complexity, performance,
        error handling, design patterns, and when to use each one.

        Args:
            entity_a: Name of the first entity.
            entity_b: Name of the second entity.

        Returns:
            Structured comparison dict — see COMPARE_PROMPT schema.

        Raises:
            EntityNotFoundError: If either entity is not found.
            CodeAnalystError: If LLM analysis fails.
        """
        # Fetch both entities (fail fast if either is missing)
        entity_a_data = await _fetch_entity(entity_a)
        entity_b_data = await _fetch_entity(entity_b)

        source_a = entity_a_data.get("source_code") or ""
        source_b = entity_b_data.get("source_code") or ""

        if not source_a and not source_b:
            return {
                "error": "No source code stored for either entity. "
                         "Run index_repository first.",
                "entity_a": entity_a,
                "entity_b": entity_b,
            }

        prompt = COMPARE_PROMPT.format(
            name_a=entity_a,
            type_a=entity_a_data.get("entity_type", "entity"),
            file_a=(entity_a_data.get("file_path") or "").split("/")[-1],
            source_a=source_a[:1800],
            name_b=entity_b,
            type_b=entity_b_data.get("entity_type", "entity"),
            file_b=(entity_b_data.get("file_path") or "").split("/")[-1],
            source_b=source_b[:1800],
        )

        result = await _llm(prompt, max_tokens=2500)
        result["entity_a"] = {
            "name": entity_a,
            "type": entity_a_data.get("entity_type"),
            "file": entity_a_data.get("file_path"),
        }
        result["entity_b"] = {
            "name": entity_b,
            "type": entity_b_data.get("entity_type"),
            "file": entity_b_data.get("file_path"),
        }

        logger.info(
            "implementations_compared",
            entity_a=entity_a,
            entity_b=entity_b,
            are_interchangeable=result.get("are_interchangeable"),
        )
        return result