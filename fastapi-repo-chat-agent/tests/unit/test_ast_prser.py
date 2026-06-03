"""Unit tests for the Python AST parser."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agents.indexer.ast_parser import PythonASTParser
from shared.models.base import CodeEntityType


@pytest.fixture
def parser() -> PythonASTParser:
    return PythonASTParser()


@pytest.fixture
def sample_python_file(tmp_path: Path) -> Path:
    source = '''"""Module docstring."""
import os
from typing import Optional

class MyClass(BaseClass):
    """A sample class."""

    def __init__(self, name: str) -> None:
        """Initialize."""
        self.name = name

    @property
    def display_name(self) -> str:
        """Return display name."""
        return self.name.title()

def standalone_function(x: int, y: int = 0) -> int:
    """Add two numbers."""
    return x + y

async def async_handler(request: dict) -> dict:
    """Handle async request."""
    return {}
'''
    file_path = tmp_path / "sample.py"
    file_path.write_text(source)
    return file_path


def test_parse_file_returns_entities(parser: PythonASTParser, sample_python_file: Path) -> None:
    entities = parser.parse_file(sample_python_file)
    assert len(entities) > 0


def test_extracts_class(parser: PythonASTParser, sample_python_file: Path) -> None:
    entities = parser.parse_file(sample_python_file)
    classes = [e for e in entities if e.entity_type == CodeEntityType.CLASS]
    assert len(classes) == 1
    assert classes[0].name == "MyClass"
    assert "BaseClass" in classes[0].metadata["bases"]


def test_extracts_methods(parser: PythonASTParser, sample_python_file: Path) -> None:
    entities = parser.parse_file(sample_python_file)
    methods = [e for e in entities if e.entity_type == CodeEntityType.METHOD]
    method_names = [e.name for e in methods]
    assert "MyClass.__init__" in method_names
    assert "MyClass.display_name" in method_names


def test_extracts_functions(parser: PythonASTParser, sample_python_file: Path) -> None:
    entities = parser.parse_file(sample_python_file)
    functions = [e for e in entities if e.entity_type == CodeEntityType.FUNCTION]
    func_names = [e.name for e in functions]
    assert "standalone_function" in func_names
    assert "async_handler" in func_names


def test_async_function_marked(parser: PythonASTParser, sample_python_file: Path) -> None:
    entities = parser.parse_file(sample_python_file)
    async_fn = next(e for e in entities if e.name == "async_handler")
    assert async_fn.metadata["is_async"] is True


def test_property_decorator_detected(parser: PythonASTParser, sample_python_file: Path) -> None:
    entities = parser.parse_file(sample_python_file)
    prop = next(e for e in entities if e.name == "MyClass.display_name")
    assert prop.metadata["is_property"] is True


def test_imports_extracted(parser: PythonASTParser, sample_python_file: Path) -> None:
    entities = parser.parse_file(sample_python_file)
    imports = [e for e in entities if e.entity_type == CodeEntityType.IMPORT]
    import_names = [e.name for e in imports]
    assert "os" in import_names
    assert any("Optional" in n for n in import_names)


def test_docstrings_captured(parser: PythonASTParser, sample_python_file: Path) -> None:
    entities = parser.parse_file(sample_python_file)
    cls = next(e for e in entities if e.entity_type == CodeEntityType.CLASS)
    assert cls.docstring == "A sample class."


def test_invalid_syntax_returns_empty(parser: PythonASTParser, tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("def broken(:\n  pass")
    entities = parser.parse_file(bad_file)
    assert entities == []


def test_file_entity_created(parser: PythonASTParser, sample_python_file: Path) -> None:
    entities = parser.parse_file(sample_python_file)
    file_entities = [e for e in entities if e.entity_type == CodeEntityType.FILE]
    assert len(file_entities) == 1
    assert file_entities[0].name == "sample.py"