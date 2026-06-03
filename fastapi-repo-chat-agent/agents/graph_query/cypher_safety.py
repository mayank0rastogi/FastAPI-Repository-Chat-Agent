"""Cypher query safety validator — prevents write operations via prompt injection."""
from __future__ import annotations

import re

from shared.exceptions import InvalidCypherQueryError
from shared.utils.logging import get_logger

logger = get_logger(__name__)

# All Cypher clauses that mutate the graph
_WRITE_CLAUSES = frozenset([
    "CREATE", "MERGE", "SET", "DELETE", "REMOVE",
    "DROP", "DETACH", "LOAD", "FOREACH", "CALL",
])

# Allowed read-only clause starters
_READ_STARTERS = frozenset([
    "MATCH", "OPTIONAL", "WITH", "RETURN",
    "UNWIND", "UNION", "CALL {",
])

# Dangerous substrings even in read queries
_DANGEROUS_PATTERNS = [
    r"apoc\.schema\.",
    r"apoc\.periodic\.",
    r"apoc\.cypher\.run",
    r"apoc\.load\.",
    r"db\.index\.",
    r"dbms\.",
    r";\s*\w",          # Multiple statements via semicolon
    r"\/\/.*\n.*CREATE",  # Comment obfuscation
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _DANGEROUS_PATTERNS]

# Hard cap on result size to prevent resource exhaustion
MAX_RESULT_LIMIT = 500


def validate_read_only(query: str) -> str:
    """Validate that a Cypher query is safe to execute (read-only).

    Performs three checks:
    1. Query must start with an allowed read clause
    2. No write keywords anywhere in the query
    3. No dangerous APOC/DBMS procedure calls

    Args:
        query: Raw Cypher query string from user input.

    Returns:
        Cleaned query string with enforced LIMIT if missing.

    Raises:
        InvalidCypherQueryError: If any safety check fails.
    """
    stripped = query.strip()

    if not stripped:
        raise InvalidCypherQueryError(query, "Empty query")

    # Check 1: Must start with a read clause
    upper = stripped.upper()
    starts_valid = any(upper.startswith(clause) for clause in _READ_STARTERS)
    if not starts_valid:
        raise InvalidCypherQueryError(
            query,
            f"Query must start with one of: {sorted(_READ_STARTERS)}",
        )

    # Check 2: No write keywords (word-boundary match to avoid false positives)
    for keyword in _WRITE_CLAUSES:
        # Use word boundary — avoids matching "CREATE" inside "RECREATE" etc.
        pattern = rf"\b{keyword}\b"
        if re.search(pattern, upper):
            raise InvalidCypherQueryError(
                query,
                f"Write operation '{keyword}' is not permitted in custom queries",
            )

    # Check 3: No dangerous patterns
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(stripped):
            raise InvalidCypherQueryError(
                query,
                f"Dangerous pattern detected: {pattern.pattern[:40]}",
            )

    # Enforce LIMIT if not present — prevent accidental full-graph scans
    if "LIMIT" not in upper:
        stripped = stripped.rstrip(";") + f"\nLIMIT {MAX_RESULT_LIMIT}"
        logger.debug("cypher_limit_injected", query=stripped[:80])

    logger.debug("cypher_validated", query=stripped[:80])
    return stripped