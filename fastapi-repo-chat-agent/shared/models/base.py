"""Shared Pydantic models for all agents — 9 Neo4j node types and all data contracts."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class CodeEntityType(str, Enum):
    """All 9 node types in the knowledge graph schema."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    PARAMETER = "parameter"
    DECORATOR = "decorator"
    IMPORT = "import"
    DOCSTRING = "docstring"
    FILE = "file"


class RelationshipType(str, Enum):
    """All 8 relationship types in the knowledge graph schema."""
    CONTAINS = "CONTAINS"
    IMPORTS = "IMPORTS"
    INHERITS_FROM = "INHERITS_FROM"
    CALLS = "CALLS"
    DECORATED_BY = "DECORATED_BY"
    HAS_PARAMETER = "HAS_PARAMETER"
    DOCUMENTED_BY = "DOCUMENTED_BY"
    DEPENDS_ON = "DEPENDS_ON"


class QueryIntent(str, Enum):
    CODE_EXPLANATION = "code_explanation"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    PATTERN_DETECTION = "pattern_detection"
    ENTITY_LOOKUP = "entity_lookup"
    RELATIONSHIP_QUERY = "relationship_query"
    LIFECYCLE_ANALYSIS = "lifecycle_analysis"
    COMPARISON = "comparison"
    GENERAL = "general"


class AgentType(str, Enum):
    ORCHESTRATOR = "orchestrator"
    INDEXER = "indexer"
    GRAPH_QUERY = "graph_query"
    CODE_ANALYST = "code_analyst"


class IndexingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Knowledge Graph Node Models ───────────────────────────────────────────────

class FileNode(BaseModel):
    """Represents a Python source file in the knowledge graph."""
    path: str
    name: str
    module_path: str = ""        # dotted module path, e.g. fastapi.routing
    size_bytes: int = 0
    line_count: int = 0
    last_indexed: datetime = Field(default_factory=datetime.utcnow)


class ModuleNode(BaseModel):
    """Represents a Python module (package or single file module).

    A Module is a higher-level concept than a File — one module may
    consist of multiple files (package) or map 1:1 to a file.
    """
    name: str                    # dotted module name, e.g. fastapi.routing
    package: str = ""            # top-level package, e.g. fastapi
    is_package: bool = False     # True if __init__.py present
    description: str = ""        # extracted from module docstring


class ClassNode(BaseModel):
    """Represents a Python class definition."""
    name: str
    file_path: str
    line_start: int = 0
    line_end: int = 0
    bases: list[str] = Field(default_factory=list)
    docstring: str = ""
    source_code: str = ""
    decorator_names: list[str] = Field(default_factory=list)
    is_abstract: bool = False
    is_dataclass: bool = False
    is_pydantic_model: bool = False
    method_count: int = 0


class FunctionNode(BaseModel):
    """Represents a module-level Python function."""
    name: str
    file_path: str
    line_start: int = 0
    line_end: int = 0
    docstring: str = ""
    source_code: str = ""
    return_type: str = ""
    is_async: bool = False
    decorator_names: list[str] = Field(default_factory=list)


class MethodNode(BaseModel):
    """Represents a method belonging to a class."""
    name: str                    # format: ClassName.method_name
    class_name: str
    file_path: str
    line_start: int = 0
    line_end: int = 0
    docstring: str = ""
    source_code: str = ""
    return_type: str = ""
    is_async: bool = False
    is_property: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_abstract: bool = False
    decorator_names: list[str] = Field(default_factory=list)


class ParameterNode(BaseModel):
    """Represents a single parameter of a function or method.

    Stored as a separate node so queries like
    'find all functions that accept a Depends parameter'
    can be answered via graph traversal.
    """
    name: str
    function_name: str           # owning function/method name
    file_path: str
    annotation: str = ""         # type annotation as string
    default_value: str = ""      # string repr of default, empty = no default
    has_default: bool = False
    is_positional_only: bool = False
    is_keyword_only: bool = False
    is_var_positional: bool = False  # *args
    is_var_keyword: bool = False     # **kwargs
    position: int = 0            # 0-indexed position in signature


class DecoratorNode(BaseModel):
    """Represents a decorator applied to a class, function, or method.

    Stored separately so pattern queries like
    'find all functions decorated with @app.get'
    traverse DECORATED_BY relationships rather than scanning strings.
    """
    name: str                    # e.g. "app.get", "property", "lru_cache"
    full_expression: str = ""    # full decorator text including args
    target_name: str = ""        # the decorated entity's name
    target_file: str = ""
    target_line: int = 0
    arguments: list[str] = Field(default_factory=list)  # string repr of args


class ImportNode(BaseModel):
    """Represents an import statement."""
    module: str                  # imported module or dotted path
    file_path: str
    line_start: int = 0
    alias: str = ""              # as-alias if present
    import_type: str = "direct"  # "direct" | "from"
    level: int = 0               # relative import dots (0 = absolute)
    names: list[str] = Field(default_factory=list)  # from X import A, B, C


class DocstringNode(BaseModel):
    """Represents a docstring attached to a module, class, function, or method.

    Stored as a node so full-text search queries can be run directly
    on the docstring content via Neo4j full-text indexes.
    """
    owner_name: str              # name of the owning entity
    owner_type: str              # Class | Function | Method | Module
    file_path: str
    content: str                 # full docstring text
    style: str = "unknown"       # "google" | "numpy" | "rest" | "plain"
    has_args_section: bool = False
    has_returns_section: bool = False
    has_raises_section: bool = False
    has_examples_section: bool = False


# ── Generic wrapper used by the AST parser ────────────────────────────────────

class CodeEntity(BaseModel):
    """Generic container used during parsing before typed node creation."""
    name: str
    entity_type: CodeEntityType
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    docstring: str | None = None
    source_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Conversation memory models ────────────────────────────────────────────────

class ConversationMessage(BaseModel):
    """A single turn in a conversation."""
    role: str                    # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_used: list[str] = Field(default_factory=list)
    tokens_used: int = 0


class UserPreferences(BaseModel):
    """Persisted user preferences for a session."""
    response_style: str = "detailed"   # "brief" | "detailed" | "code-heavy"
    preferred_agents: list[str] = Field(default_factory=list)
    code_format: str = "python"
    include_line_numbers: bool = True
    explain_concepts: bool = True


class RoutingDecision(BaseModel):
    """Records how the orchestrator routed a specific query."""
    query_hash: str
    intent: str
    agents_selected: list[str]
    agents_succeeded: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationSession(BaseModel):
    """Full session state stored in Redis."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    messages: list[ConversationMessage] = Field(default_factory=list)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    routing_history: list[RoutingDecision] = Field(default_factory=list)
    context_entities: list[str] = Field(default_factory=list)  # entities mentioned in session
    metadata: dict[str, Any] = Field(default_factory=dict)