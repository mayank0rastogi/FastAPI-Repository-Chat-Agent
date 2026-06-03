"""Python AST parser — extracts all code entities AND relationships from source files."""
from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.models.base import CodeEntity, CodeEntityType
from shared.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CodeRelationship:
    """Represents a directed relationship between two code entities.

    Attributes:
        source_name: Name of the source entity.
        target_name: Name of the target entity.
        rel_type: Neo4j relationship type string (e.g. CALLS, INHERITS_FROM).
        metadata: Extra properties stored on the relationship.
    """

    source_name: str
    target_name: str
    rel_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """Full parse output for a single Python file.

    Attributes:
        entities: All code entities found in the file.
        relationships: All directed relationships between entities.
        file_hash: SHA-256 of the source for incremental change detection.
        errors: Any non-fatal parse warnings.
    """

    entities: list[CodeEntity] = field(default_factory=list)
    relationships: list[CodeRelationship] = field(default_factory=list)
    file_hash: str = ""
    errors: list[str] = field(default_factory=list)


class CallVisitor(ast.NodeVisitor):
    """AST visitor that collects all function/method call names in a scope."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Record every function call encountered."""
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                self.calls.append(f"{node.func.value.id}.{node.func.attr}")
            else:
                self.calls.append(node.func.attr)
        self.generic_visit(node)


class PythonASTParser:
    """Extract structured code entities AND relationships from Python source.

    Handles:
    - Classes, functions, methods, parameters, decorators, imports, docstrings
    - Relationships: CONTAINS, IMPORTS, INHERITS_FROM, CALLS,
                     DECORATED_BY, HAS_PARAMETER, DOCUMENTED_BY, DEPENDS_ON

    Example:
        >>> parser = PythonASTParser()
        >>> result = parser.parse_file(Path("fastapi/applications.py"))
        >>> print(len(result.entities), len(result.relationships))
    """

    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse a single Python file into entities and relationships.

        Args:
            file_path: Absolute path to the .py file.

        Returns:
            ParseResult with entities, relationships, hash, and errors.
        """
        result = ParseResult()
        try:
            source = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            result.errors.append(f"Read error: {exc}")
            return result

        result.file_hash = hashlib.sha256(source.encode()).hexdigest()

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            result.errors.append(f"SyntaxError at line {exc.lineno}: {exc.msg}")
            return result

        lines = source.splitlines()
        fp = str(file_path)

        # ── File node ────────────────────────────────────────────────────────
        file_entity = CodeEntity(
            name=file_path.name,
            entity_type=CodeEntityType.FILE,
            file_path=fp,
            line_start=1,
            line_end=len(lines),
            metadata={
                "size_bytes": len(source),
                "file_hash": result.file_hash,
                "line_count": len(lines),
            },
        )
        result.entities.append(file_entity)

        # ── Module docstring ─────────────────────────────────────────────────
        module_doc = ast.get_docstring(tree)
        if module_doc:
            doc_entity = CodeEntity(
                name=f"{file_path.stem}.__module_doc__",
                entity_type=CodeEntityType.DOCSTRING,
                file_path=fp,
                line_start=1,
                source_code=module_doc[:1000],
            )
            result.entities.append(doc_entity)
            result.relationships.append(
                CodeRelationship(
                    source_name=file_path.name,
                    target_name=doc_entity.name,
                    rel_type="DOCUMENTED_BY",
                )
            )

        # ── Walk top-level nodes ─────────────────────────────────────────────
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._process_class(node, file_path, lines, result)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._process_function(node, file_path, lines, result, parent_class=None)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._process_import(node, file_path, result)

        return result

    # ── Class processing ──────────────────────────────────────────────────────

    def _process_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
        lines: list[str],
        result: ParseResult,
    ) -> None:
        """Extract class entity + all its methods, decorators, and relationships."""
        fp = str(file_path)
        source_snippet = self._extract_source(lines, node.lineno, node.end_lineno or node.lineno)
        docstring = ast.get_docstring(node)

        bases = []
        for base in node.bases:
            base_name = self._node_to_name(base)
            if base_name:
                bases.append(base_name)

        decorators = [self._decorator_name(d) for d in node.decorator_list]

        class_entity = CodeEntity(
            name=node.name,
            entity_type=CodeEntityType.CLASS,
            file_path=fp,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=docstring,
            source_code=source_snippet[:3000],
            metadata={
                "bases": bases,
                "decorators": decorators,
                "method_count": sum(
                    1 for n in ast.walk(node)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not node
                ),
            },
        )
        result.entities.append(class_entity)

        # FILE -[CONTAINS]-> CLASS
        result.relationships.append(CodeRelationship(
            source_name=file_path.name,
            target_name=node.name,
            rel_type="CONTAINS",
            metadata={"line": node.lineno},
        ))

        # CLASS -[INHERITS_FROM]-> base classes
        for base_name in bases:
            result.relationships.append(CodeRelationship(
                source_name=node.name,
                target_name=base_name,
                rel_type="INHERITS_FROM",
            ))

        # CLASS -[DECORATED_BY]-> decorators
        for dec_name in decorators:
            dec_entity = CodeEntity(
                name=dec_name,
                entity_type=CodeEntityType.DECORATOR,
                file_path=fp,
                line_start=node.lineno,
            )
            result.entities.append(dec_entity)
            result.relationships.append(CodeRelationship(
                source_name=node.name,
                target_name=dec_name,
                rel_type="DECORATED_BY",
            ))

        # CLASS docstring -[DOCUMENTED_BY]->
        if docstring:
            doc_entity = CodeEntity(
                name=f"{node.name}.__doc__",
                entity_type=CodeEntityType.DOCSTRING,
                file_path=fp,
                line_start=node.lineno,
                source_code=docstring[:500],
            )
            result.entities.append(doc_entity)
            result.relationships.append(CodeRelationship(
                source_name=node.name,
                target_name=doc_entity.name,
                rel_type="DOCUMENTED_BY",
            ))

        # Process methods inside class
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._process_function(child, file_path, lines, result, parent_class=node.name)

    # ── Function/Method processing ────────────────────────────────────────────

    def _process_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        lines: list[str],
        result: ParseResult,
        parent_class: str | None,
    ) -> None:
        """Extract function/method entity + parameters, decorators, calls."""
        fp = str(file_path)
        is_method = parent_class is not None
        entity_name = f"{parent_class}.{node.name}" if is_method else node.name
        entity_type = CodeEntityType.METHOD if is_method else CodeEntityType.FUNCTION

        source_snippet = self._extract_source(lines, node.lineno, node.end_lineno or node.lineno)
        docstring = ast.get_docstring(node)
        decorators = [self._decorator_name(d) for d in node.decorator_list]
        params = self._extract_params(node)
        return_annotation = self._annotation_to_str(node.returns)

        fn_entity = CodeEntity(
            name=entity_name,
            entity_type=entity_type,
            file_path=fp,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=docstring,
            source_code=source_snippet[:2000],
            metadata={
                "simple_name": node.name,
                "class_name": parent_class or "",
                "params": params,
                "decorators": decorators,
                "return_type": return_annotation,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "is_property": "property" in decorators,
                "is_classmethod": "classmethod" in decorators,
                "is_staticmethod": "staticmethod" in decorators,
                "is_dunder": node.name.startswith("__") and node.name.endswith("__"),
            },
        )
        result.entities.append(fn_entity)

        # Parent container -[CONTAINS]-> function
        container = parent_class if parent_class else file_path.name
        result.relationships.append(CodeRelationship(
            source_name=container,
            target_name=entity_name,
            rel_type="CONTAINS",
            metadata={"line": node.lineno},
        ))

        # FUNCTION -[DECORATED_BY]-> each decorator
        for dec_name in decorators:
            dec_entity = CodeEntity(
                name=dec_name,
                entity_type=CodeEntityType.DECORATOR,
                file_path=fp,
                line_start=node.lineno,
            )
            result.entities.append(dec_entity)
            result.relationships.append(CodeRelationship(
                source_name=entity_name,
                target_name=dec_name,
                rel_type="DECORATED_BY",
            ))

        # FUNCTION -[HAS_PARAMETER]-> each parameter (skip self/cls)
        for param in params:
            pname = param["name"].lstrip("*")
            if pname in ("self", "cls"):
                continue
            param_entity = CodeEntity(
                name=f"{entity_name}.{pname}",
                entity_type=CodeEntityType.PARAMETER,
                file_path=fp,
                line_start=node.lineno,
                metadata={
                    "annotation": param.get("annotation", ""),
                    "has_default": param.get("has_default", False),
                    "keyword_only": param.get("keyword_only", False),
                },
            )
            result.entities.append(param_entity)
            result.relationships.append(CodeRelationship(
                source_name=entity_name,
                target_name=param_entity.name,
                rel_type="HAS_PARAMETER",
                metadata={"position": params.index(param)},
            ))

        # FUNCTION -[DOCUMENTED_BY]-> docstring
        if docstring:
            doc_entity = CodeEntity(
                name=f"{entity_name}.__doc__",
                entity_type=CodeEntityType.DOCSTRING,
                file_path=fp,
                line_start=node.lineno,
                source_code=docstring[:500],
            )
            result.entities.append(doc_entity)
            result.relationships.append(CodeRelationship(
                source_name=entity_name,
                target_name=doc_entity.name,
                rel_type="DOCUMENTED_BY",
            ))

        # FUNCTION -[CALLS]-> called functions
        call_visitor = CallVisitor()
        call_visitor.visit(node)
        for called_name in set(call_visitor.calls):
            if called_name not in (entity_name, node.name):
                result.relationships.append(CodeRelationship(
                    source_name=entity_name,
                    target_name=called_name,
                    rel_type="CALLS",
                    metadata={"caller_file": fp},
                ))

    # ── Import processing ─────────────────────────────────────────────────────

    def _process_import(
        self,
        node: ast.Import | ast.ImportFrom,
        file_path: Path,
        result: ParseResult,
    ) -> None:
        """Extract import entities and FILE -[IMPORTS]-> relationships."""
        fp = str(file_path)

        if isinstance(node, ast.Import):
            for alias in node.names:
                imp_entity = CodeEntity(
                    name=alias.name,
                    entity_type=CodeEntityType.IMPORT,
                    file_path=fp,
                    line_start=node.lineno,
                    metadata={"alias": alias.asname or "", "import_type": "direct"},
                )
                result.entities.append(imp_entity)
                result.relationships.append(CodeRelationship(
                    source_name=file_path.name,
                    target_name=alias.name,
                    rel_type="IMPORTS",
                    metadata={"line": node.lineno},
                ))

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                full_name = f"{module}.{alias.name}" if module else alias.name
                imp_entity = CodeEntity(
                    name=full_name,
                    entity_type=CodeEntityType.IMPORT,
                    file_path=fp,
                    line_start=node.lineno,
                    metadata={
                        "module": module,
                        "symbol": alias.name,
                        "alias": alias.asname or "",
                        "import_type": "from",
                        "relative_level": node.level,
                    },
                )
                result.entities.append(imp_entity)
                result.relationships.append(CodeRelationship(
                    source_name=file_path.name,
                    target_name=full_name,
                    rel_type="IMPORTS",
                    metadata={"line": node.lineno, "module": module},
                ))
                # Cross-file DEPENDS_ON between modules
                if module:
                    result.relationships.append(CodeRelationship(
                        source_name=file_path.stem,
                        target_name=module,
                        rel_type="DEPENDS_ON",
                    ))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_params(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[dict[str, Any]]:
        """Return structured parameter list with name, annotation, defaults."""
        params: list[dict[str, Any]] = []
        args = node.args
        default_offset = len(args.args) - len(args.defaults)

        for i, arg in enumerate(args.args):
            params.append({
                "name": arg.arg,
                "annotation": PythonASTParser._annotation_to_str(arg.annotation),
                "has_default": i >= default_offset,
                "positional_only": False,
            })
        for arg in args.posonlyargs:
            params.append({
                "name": arg.arg,
                "annotation": PythonASTParser._annotation_to_str(arg.annotation),
                "positional_only": True,
            })
        if args.vararg:
            params.append({"name": f"*{args.vararg.arg}", "annotation": PythonASTParser._annotation_to_str(args.vararg.annotation)})
        for arg in args.kwonlyargs:
            params.append({"name": arg.arg, "annotation": PythonASTParser._annotation_to_str(arg.annotation), "keyword_only": True})
        if args.kwarg:
            params.append({"name": f"**{args.kwarg.arg}", "annotation": PythonASTParser._annotation_to_str(args.kwarg.annotation)})

        return params

    @staticmethod
    def _annotation_to_str(annotation: ast.expr | None) -> str:
        """Convert an AST annotation to its source string representation."""
        if annotation is None:
            return ""
        try:
            return ast.unparse(annotation)
        except Exception:
            return ""

    @staticmethod
    def _decorator_name(node: ast.expr) -> str:
        """Extract string name from a decorator AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
            return node.attr
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            if isinstance(node.func, ast.Attribute):
                return node.func.attr
        try:
            return ast.unparse(node)
        except Exception:
            return "unknown_decorator"

    @staticmethod
    def _node_to_name(node: ast.expr) -> str:
        """Convert a base class AST node to its string name."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
            return node.attr
        try:
            return ast.unparse(node)
        except Exception:
            return ""

    @staticmethod
    def _extract_source(lines: list[str], start: int, end: int) -> str:
        """Extract and dedent source lines for a node."""
        snippet = lines[max(0, start - 1) : end]
        return textwrap.dedent("\n".join(snippet))