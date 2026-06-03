"""MCP tool implementations for the Indexer Agent — all 5 required tools."""
from __future__ import annotations

import asyncio
import hashlib
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from agents.indexer.ast_parser import CodeRelationship, ParseResult, PythonASTParser
from infrastructure.neo4j_client import Neo4jClient
from shared.config import IndexerSettings
from shared.models.base import CodeEntityType, IndexingStatus
from shared.utils.logging import get_logger

logger = get_logger(__name__)

# Job store — use Redis in production
_jobs: dict[str, dict[str, Any]] = {}


def register_indexer_tools(
    mcp: FastMCP, settings: IndexerSettings, neo4j: Neo4jClient
) -> None:
    """Register all 5 required Indexer Agent MCP tools.

    Args:
        mcp: FastMCP server instance.
        settings: Indexer configuration.
        neo4j: Initialized Neo4j client.
    """
    parser = PythonASTParser()

    # ── Tool 1: index_repository ──────────────────────────────────────────────

    @mcp.tool()
    async def index_repository(
        repo_url: str = "", incremental: bool = False
    ) -> dict[str, Any]:
        """Trigger full or incremental repository indexing.

        Clones the FastAPI repository if not present, then indexes every
        Python file into the Neo4j knowledge graph. For incremental mode,
        only files changed since the last index run are re-processed.

        Args:
            repo_url: Git URL to clone. Defaults to configured FastAPI repo.
            incremental: If True, only re-index files modified since last run.

        Returns:
            Dict with job_id, status, and repo_url.
        """
        job_id = str(uuid.uuid4())
        url = repo_url or settings.repo_url
        _jobs[job_id] = {
            "status": IndexingStatus.PENDING,
            "progress": 0,
            "total_files": 0,
            "indexed_files": 0,
            "failed_files": 0,
            "entities_created": 0,
            "relationships_created": 0,
            "errors": [],
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "incremental": incremental,
            "repo_url": url,
        }
        asyncio.create_task(
            _run_indexing_job(job_id, url, incremental, settings, neo4j, parser)
        )
        logger.info("indexing_job_created", job_id=job_id, url=url, incremental=incremental)
        return {"job_id": job_id, "status": IndexingStatus.PENDING, "repo_url": url}

    # ── Tool 2: index_file ────────────────────────────────────────────────────

    @mcp.tool()
    async def index_file(file_path: str) -> dict[str, Any]:
        """Index a single Python file into the knowledge graph.

        Useful for re-indexing one changed file without a full repository scan.
        Also rebuilds all relationships for entities in this file.

        Args:
            file_path: Absolute path to the .py file.

        Returns:
            Dict with entity count, relationship count, and any errors.
        """
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        if path.suffix != ".py":
            return {"success": False, "error": f"Not a Python file: {file_path}"}
        if path.stat().st_size > settings.max_file_size_kb * 1024:
            return {"success": False, "error": f"File too large (>{settings.max_file_size_kb}KB)"}

        result = parser.parse_file(path)

        if result.errors and not result.entities:
            return {"success": False, "errors": result.errors, "file": file_path}

        entity_count, rel_count = await _persist_parse_result(result, neo4j)

        logger.info(
            "file_indexed",
            file=file_path,
            entities=entity_count,
            relationships=rel_count,
        )
        return {
            "success": True,
            "file": file_path,
            "file_hash": result.file_hash,
            "entities_created": entity_count,
            "relationships_created": rel_count,
            "parse_errors": result.errors,
        }

    # ── Tool 3: parse_python_ast ──────────────────────────────────────────────

    @mcp.tool()
    async def parse_python_ast(
        source_code: str, file_name: str = "snippet.py"
    ) -> dict[str, Any]:
        """Extract full AST information from raw Python source code.

        Does NOT write to Neo4j — returns structured entity and relationship
        data for inspection or preview purposes.

        Args:
            source_code: Python source string to parse.
            file_name: Virtual filename for entity path references.

        Returns:
            Dict with entities grouped by type and all relationships found.
        """
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="ast_parse_", delete=False
        ) as f:
            f.write(source_code)
            tmp_path = Path(f.name)

        try:
            result = parser.parse_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        # Group entities by type
        by_type: dict[str, list[dict[str, Any]]] = {}
        for entity in result.entities:
            key = entity.entity_type.value
            if key not in by_type:
                by_type[key] = []
            by_type[key].append({
                "name": entity.name,
                "line_start": entity.line_start,
                "line_end": entity.line_end,
                "has_docstring": bool(entity.docstring),
                "metadata": entity.metadata,
            })

        return {
            "file_name": file_name,
            "file_hash": result.file_hash,
            "entities_by_type": by_type,
            "total_entities": len(result.entities),
            "relationships": [
                {
                    "source": r.source_name,
                    "target": r.target_name,
                    "type": r.rel_type,
                    "metadata": r.metadata,
                }
                for r in result.relationships
            ],
            "total_relationships": len(result.relationships),
            "parse_errors": result.errors,
        }

    # ── Tool 4: extract_entities ──────────────────────────────────────────────

    @mcp.tool()
    async def extract_entities(file_path: str) -> dict[str, Any]:
        """Identify all code entities and their relationships in a file.

        Returns a detailed breakdown of every entity type found and every
        relationship connecting them — including CALLS, DECORATED_BY,
        HAS_PARAMETER, INHERITS_FROM, and DOCUMENTED_BY.

        Args:
            file_path: Path to a Python source file.

        Returns:
            Dict with entities by type, relationship summary, and statistics.
        """
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        result = parser.parse_file(path)

        # Relationship breakdown by type
        rel_summary: dict[str, list[dict[str, Any]]] = {}
        for rel in result.relationships:
            if rel.rel_type not in rel_summary:
                rel_summary[rel.rel_type] = []
            rel_summary[rel.rel_type].append({
                "source": rel.source_name,
                "target": rel.target_name,
                "metadata": rel.metadata,
            })

        # Entity breakdown by type
        entity_summary: dict[str, list[dict[str, Any]]] = {}
        for entity in result.entities:
            key = entity.entity_type.value
            if key not in entity_summary:
                entity_summary[key] = []
            entry: dict[str, Any] = {
                "name": entity.name,
                "line_start": entity.line_start,
                "line_end": entity.line_end,
                "has_docstring": bool(entity.docstring),
            }
            if entity.metadata:
                entry["metadata"] = entity.metadata
            entity_summary[key].append(entry)

        return {
            "file": file_path,
            "file_hash": result.file_hash,
            "entities_by_type": entity_summary,
            "relationships_by_type": rel_summary,
            "statistics": {
                "total_entities": len(result.entities),
                "total_relationships": len(result.relationships),
                "classes": len(entity_summary.get("Class", [])),
                "functions": len(entity_summary.get("Function", [])),
                "methods": len(entity_summary.get("Method", [])),
                "imports": len(entity_summary.get("Import", [])),
                "decorators": len(entity_summary.get("Decorator", [])),
                "parameters": len(entity_summary.get("Parameter", [])),
                "calls_relationships": len(rel_summary.get("CALLS", [])),
                "inheritance_relationships": len(rel_summary.get("INHERITS_FROM", [])),
            },
            "parse_errors": result.errors,
        }

    # ── Tool 5: get_index_status ──────────────────────────────────────────────

    @mcp.tool()
    async def get_index_status(job_id: str = "") -> dict[str, Any]:
        """Report indexing progress, statistics, and Neo4j graph counts.

        If job_id is empty, returns overall graph statistics from Neo4j.
        If job_id is provided, returns real-time progress for that job.

        Args:
            job_id: UUID from index_repository. Pass empty string for graph stats.

        Returns:
            Dict with status, progress, entity counts, and Neo4j statistics.
        """
        # Graph-level statistics (always included)
        try:
            graph_stats = await neo4j.get_statistics()
        except Exception as exc:
            graph_stats = {"error": str(exc)}

        if not job_id:
            return {
                "graph_statistics": graph_stats,
                "active_jobs": [
                    {"job_id": jid, "status": j["status"], "progress": j["progress"]}
                    for jid, j in _jobs.items()
                    if j["status"] == IndexingStatus.RUNNING
                ],
                "total_jobs": len(_jobs),
            }

        if job_id not in _jobs:
            return {"error": f"Job '{job_id}' not found", "graph_statistics": graph_stats}

        job = _jobs[job_id]
        return {
            **job,
            "graph_statistics": graph_stats,
        }


# ── Background job runner ─────────────────────────────────────────────────────

async def _run_indexing_job(
    job_id: str,
    repo_url: str,
    incremental: bool,
    settings: IndexerSettings,
    neo4j: Neo4jClient,
    parser: PythonASTParser,
) -> None:
    """Background task: clone repo, index all Python files, build relationships.

    For incremental mode, compares each file's current SHA-256 hash against
    the stored hash in Neo4j — only re-indexes files whose hash has changed.

    Args:
        job_id: Job identifier for progress tracking.
        repo_url: Git repository URL.
        incremental: Skip unchanged files if True.
        settings: Indexer configuration.
        neo4j: Neo4j client.
        parser: AST parser instance.
    """
    job = _jobs[job_id]
    job["status"] = IndexingStatus.RUNNING

    try:
        repo_path = Path(settings.repo_local_path)

        # ── Clone or update repo ─────────────────────────────────────────────
        if not (repo_path / ".git").exists():
            logger.info("cloning_repo", url=repo_url, path=str(repo_path))
            repo_path.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                ["git", "clone", "--depth=1", repo_url, str(repo_path)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"git clone failed: {proc.stderr[:300]}")
        else:
            if not incremental:
                logger.info("pulling_repo", path=str(repo_path))
                subprocess.run(
                    ["git", "-C", str(repo_path), "pull", "--ff-only"],
                    capture_output=True, text=True,
                )

        # ── Collect files to index ───────────────────────────────────────────
        all_py_files = [
            f for f in repo_path.rglob("*.py")
            if f.stat().st_size <= settings.max_file_size_kb * 1024
            and "__pycache__" not in str(f)
        ]

        if incremental:
            all_py_files = await _filter_changed_files(all_py_files, neo4j)
            logger.info("incremental_files_to_index", count=len(all_py_files))

        total = len(all_py_files)
        job["total_files"] = total
        logger.info("indexing_started", job_id=job_id, total_files=total, incremental=incremental)

        total_entities = 0
        total_rels = 0

        # ── Index files in batches ───────────────────────────────────────────
        batch_size = settings.batch_size
        for batch_start in range(0, total, batch_size):
            batch = all_py_files[batch_start : batch_start + batch_size]
            for py_file in batch:
                try:
                    result = parser.parse_file(py_file)
                    entity_count, rel_count = await _persist_parse_result(result, neo4j)
                    total_entities += entity_count
                    total_rels += rel_count
                    job["indexed_files"] += 1
                except Exception as exc:
                    job["failed_files"] += 1
                    job["errors"].append(f"{py_file.name}: {str(exc)[:100]}")
                    logger.warning("file_index_error", file=str(py_file), error=str(exc))

            job["progress"] = int((batch_start + len(batch)) / total * 100)
            job["entities_created"] = total_entities
            job["relationships_created"] = total_rels
            # Yield control between batches to avoid blocking event loop
            await asyncio.sleep(0)

        job["status"] = IndexingStatus.COMPLETED
        job["progress"] = 100
        job["completed_at"] = datetime.utcnow().isoformat()
        logger.info(
            "indexing_complete",
            job_id=job_id,
            files=total,
            entities=total_entities,
            relationships=total_rels,
        )

    except Exception as exc:
        job["status"] = IndexingStatus.FAILED
        job["errors"].append(str(exc))
        job["completed_at"] = datetime.utcnow().isoformat()
        logger.error("indexing_job_failed", job_id=job_id, error=str(exc))


async def _filter_changed_files(
    files: list[Path], neo4j: Neo4jClient
) -> list[Path]:
    """Return only files whose SHA-256 hash differs from what's in Neo4j.

    Args:
        files: All candidate Python files.
        neo4j: Neo4j client to check stored hashes.

    Returns:
        Subset of files that are new or modified.
    """
    stored_hashes = await neo4j.run_read(
        "MATCH (f:File) WHERE f.file_hash IS NOT NULL RETURN f.path AS path, f.file_hash AS hash"
    )
    hash_map = {r["path"]: r["hash"] for r in stored_hashes}

    changed: list[Path] = []
    for file_path in files:
        try:
            content = file_path.read_bytes()
            current_hash = hashlib.sha256(content).hexdigest()
            if hash_map.get(str(file_path)) != current_hash:
                changed.append(file_path)
        except OSError:
            changed.append(file_path)  # Include if unreadable to be safe

    return changed


async def _persist_parse_result(
    result: ParseResult, neo4j: Neo4jClient
) -> tuple[int, int]:
    """Write all entities and relationships from a ParseResult into Neo4j.

    Uses MERGE to ensure idempotent writes — safe to call multiple times
    on the same file without creating duplicate nodes.

    Args:
        result: Parsed entities and relationships.
        neo4j: Neo4j client.

    Returns:
        Tuple of (entity_count, relationship_count) written.
    """
    entity_count = 0
    rel_count = 0

    for entity in result.entities:
        try:
            if entity.entity_type == CodeEntityType.FILE:
                await neo4j.run_write(
                    """
                    MERGE (f:File {path: $path})
                    SET f.name = $name, f.size_bytes = $size_bytes,
                        f.file_hash = $file_hash, f.line_count = $line_count,
                        f.last_indexed = datetime()
                    """,
                    {
                        "path": entity.file_path,
                        "name": entity.name,
                        "size_bytes": entity.metadata.get("size_bytes", 0),
                        "file_hash": entity.metadata.get("file_hash", ""),
                        "line_count": entity.metadata.get("line_count", 0),
                    },
                )
            elif entity.entity_type == CodeEntityType.CLASS:
                await neo4j.run_write(
                    """
                    MERGE (c:Class {name: $name, file_path: $file_path})
                    SET c.line_start = $line_start, c.line_end = $line_end,
                        c.docstring = $docstring, c.bases = $bases,
                        c.decorators = $decorators, c.method_count = $method_count,
                        c.source_code = $source_code
                    """,
                    {
                        "name": entity.name,
                        "file_path": entity.file_path,
                        "line_start": entity.line_start,
                        "line_end": entity.line_end,
                        "docstring": entity.docstring or "",
                        "bases": entity.metadata.get("bases", []),
                        "decorators": entity.metadata.get("decorators", []),
                        "method_count": entity.metadata.get("method_count", 0),
                        "source_code": entity.source_code or "",
                    },
                )
            elif entity.entity_type in (CodeEntityType.FUNCTION, CodeEntityType.METHOD):
                label = "Method" if entity.entity_type == CodeEntityType.METHOD else "Function"
                await neo4j.run_write(
                    f"""
                    MERGE (fn:{label} {{name: $name, file_path: $file_path}})
                    SET fn.line_start = $line_start, fn.line_end = $line_end,
                        fn.docstring = $docstring, fn.is_async = $is_async,
                        fn.decorators = $decorators, fn.return_type = $return_type,
                        fn.simple_name = $simple_name, fn.class_name = $class_name,
                        fn.is_property = $is_property, fn.source_code = $source_code
                    """,
                    {
                        "name": entity.name,
                        "file_path": entity.file_path,
                        "line_start": entity.line_start,
                        "line_end": entity.line_end,
                        "docstring": entity.docstring or "",
                        "is_async": entity.metadata.get("is_async", False),
                        "decorators": entity.metadata.get("decorators", []),
                        "return_type": entity.metadata.get("return_type", ""),
                        "simple_name": entity.metadata.get("simple_name", entity.name),
                        "class_name": entity.metadata.get("class_name", ""),
                        "is_property": entity.metadata.get("is_property", False),
                        "source_code": entity.source_code or "",
                    },
                )
            elif entity.entity_type == CodeEntityType.IMPORT:
                await neo4j.run_write(
                    """
                    MERGE (i:Import {name: $name, file_path: $file_path})
                    SET i.module = $module, i.symbol = $symbol,
                        i.alias = $alias, i.import_type = $import_type,
                        i.line_start = $line_start
                    """,
                    {
                        "name": entity.name,
                        "file_path": entity.file_path,
                        "module": entity.metadata.get("module", ""),
                        "symbol": entity.metadata.get("symbol", ""),
                        "alias": entity.metadata.get("alias", ""),
                        "import_type": entity.metadata.get("import_type", "direct"),
                        "line_start": entity.line_start,
                    },
                )
            elif entity.entity_type == CodeEntityType.DECORATOR:
                await neo4j.run_write(
                    """
                    MERGE (d:Decorator {name: $name})
                    SET d.file_path = $file_path, d.line_start = $line_start
                    """,
                    {"name": entity.name, "file_path": entity.file_path, "line_start": entity.line_start},
                )
            elif entity.entity_type == CodeEntityType.PARAMETER:
                await neo4j.run_write(
                    """
                    MERGE (p:Parameter {name: $name, file_path: $file_path})
                    SET p.annotation = $annotation, p.has_default = $has_default,
                        p.line_start = $line_start
                    """,
                    {
                        "name": entity.name,
                        "file_path": entity.file_path,
                        "annotation": entity.metadata.get("annotation", ""),
                        "has_default": entity.metadata.get("has_default", False),
                        "line_start": entity.line_start,
                    },
                )
            elif entity.entity_type == CodeEntityType.DOCSTRING:
                await neo4j.run_write(
                    """
                    MERGE (ds:Docstring {name: $name, file_path: $file_path})
                    SET ds.content = $content, ds.line_start = $line_start
                    """,
                    {
                        "name": entity.name,
                        "file_path": entity.file_path,
                        "content": entity.source_code or "",
                        "line_start": entity.line_start,
                    },
                )
            entity_count += 1
        except Exception as exc:
            logger.warning("entity_persist_error", name=entity.name, error=str(exc))

    # ── Persist relationships ─────────────────────────────────────────────────
    for rel in result.relationships:
        try:
            await _persist_relationship(rel, neo4j)
            rel_count += 1
        except Exception as exc:
            logger.debug("rel_persist_skip", rel=rel.rel_type, error=str(exc))

    return entity_count, rel_count


async def _persist_relationship(rel: CodeRelationship, neo4j: Neo4jClient) -> None:
    """Write a single relationship into Neo4j using a dynamic label-agnostic MERGE.

    Matches source and target nodes by name regardless of their label,
    then creates the relationship if it doesn't already exist.

    Args:
        rel: CodeRelationship with source, target, and type.
        neo4j: Neo4j client.
    """
    # All 8 schema relationship types handled here
    query = f"""
    MATCH (source {{name: $source_name}})
    MATCH (target {{name: $target_name}})
    MERGE (source)-[r:{rel.rel_type}]->(target)
    SET r += $props
    """
    await neo4j.run_write(
        query,
        {
            "source_name": rel.source_name,
            "target_name": rel.target_name,
            "props": rel.metadata,
        },
    )