"""LLM prompt templates for the Code Analyst Agent."""
from __future__ import annotations

SYSTEM_PROMPT = """You are an expert Python code analyst specialising in FastAPI internals.

Your expertise covers:
- FastAPI request/response lifecycle, dependency injection, routing, middleware
- Python design patterns: Strategy, Observer, Decorator, Factory, Singleton,
  Repository, Adapter, Proxy, Chain of Responsibility, Template Method
- SOLID principles, Clean Code, DRY, YAGNI
- Async/await patterns, event loops, concurrency pitfalls
- Type system: generics, Protocols, TypeVars, overloads
- Pydantic v2 models, validators, serialisation
- Anti-patterns: God Object, Shotgun Surgery, Feature Envy, Primitive Obsession,
  Long Method, Divergent Change, Speculative Generality

Rules:
- Always return valid JSON matching the requested schema exactly
- Base every claim on the actual source code provided — never hallucinate
- When source is truncated, note it in your response
- Cite file paths and line numbers when given"""

# ── Per-tool prompt templates ─────────────────────────────────────────────────

ANALYZE_FUNCTION_PROMPT = """Analyse this Python function from the FastAPI codebase:

File: {file_path} (lines {line_start}–{line_end})
```python
{source_code}
```
Parameters metadata: {params}
Decorators: {decorators}

Return JSON matching EXACTLY:
{{
  "summary": "one-sentence description",
  "purpose": "what problem this solves in the FastAPI context",
  "complexity": "low | medium | high",
  "complexity_reasons": ["reason 1", "reason 2"],
  "is_async": true/false,
  "parameters": [
    {{"name": "...", "type": "...", "purpose": "...", "is_fastapi_special": true/false}}
  ],
  "return_value": {{
    "type": "...",
    "description": "what it returns and when",
    "can_be_none": true/false
  }},
  "side_effects": ["list of observable side effects"],
  "error_handling": {{
    "raises": ["ExceptionType: when"],
    "catches": ["what it catches and how"]
  }},
  "patterns_used": ["pattern name: how it's used here"],
  "anti_patterns": ["anti-pattern: description if found"],
  "best_practices_followed": ["practice: example from code"],
  "best_practices_violated": ["practice: what should be done instead"],
  "dependencies": ["external functions/classes called"],
  "fastapi_concepts": ["FastAPI concepts demonstrated"],
  "execution_flow": ["step 1: ...", "step 2: ...", "step 3: ..."],
  "testability": "easy | moderate | hard",
  "testability_notes": "why and how to test this"
}}"""

ANALYZE_CLASS_PROMPT = """Analyse this Python class from the FastAPI codebase:

File: {file_path} (lines {line_start}–{line_end})
```python
{source_code}
```
Base classes: {bases}
Decorators: {decorators}
Method count: {method_count}

Return JSON matching EXACTLY:
{{
  "summary": "one-sentence description",
  "purpose": "what problem this class solves",
  "inheritance": {{
    "bases": ["base class names"],
    "pattern": "mixin | abstract_base | concrete | dataclass | enum | protocol",
    "mro_notes": "notable aspects of method resolution order"
  }},
  "public_api": [
    {{
      "name": "method/property name",
      "type": "method | property | classmethod | staticmethod",
      "purpose": "...",
      "is_async": true/false
    }}
  ],
  "private_api": ["list of private method names and their roles"],
  "state": {{
    "instance_attributes": ["name: type — purpose"],
    "class_attributes": ["name: type — purpose"],
    "is_stateful": true/false
  }},
  "design_patterns": [
    {{"pattern": "...", "evidence": "how you can see it in the code"}}
  ],
  "anti_patterns": [
    {{"pattern": "...", "evidence": "...", "suggestion": "how to fix"}}
  ],
  "cohesion": "high | medium | low",
  "cohesion_notes": "why",
  "coupling": "tight | moderate | loose",
  "coupling_notes": "what it couples to and why",
  "solid_assessment": {{
    "single_responsibility": "pass | partial | fail — explanation",
    "open_closed": "pass | partial | fail — explanation",
    "liskov_substitution": "pass | partial | fail — explanation",
    "interface_segregation": "pass | partial | fail — explanation",
    "dependency_inversion": "pass | partial | fail — explanation"
  }},
  "fastapi_role": "how this class fits into FastAPI's architecture",
  "testability": "easy | moderate | hard",
  "improvement_suggestions": ["concrete suggestion 1", "suggestion 2"]
}}"""

FIND_PATTERNS_PROMPT = """Identify design patterns and anti-patterns in these FastAPI code entities:

{entity_summaries}

Return JSON matching EXACTLY:
{{
  "design_patterns": [
    {{
      "pattern_name": "...",
      "category": "creational | structural | behavioural | architectural",
      "entities": ["entity names implementing this pattern"],
      "description": "how this pattern is implemented here",
      "purpose": "why this pattern is used in FastAPI",
      "evidence": "specific code evidence (method names, inheritance, etc.)"
    }}
  ],
  "anti_patterns": [
    {{
      "pattern_name": "...",
      "entities": ["affected entity names"],
      "description": "what the anti-pattern is",
      "impact": "performance | maintainability | testability | readability",
      "suggestion": "how to refactor"
    }}
  ],
  "architectural_style": "description of the overall architecture",
  "solid_compliance": {{
    "overall": "high | medium | low",
    "notes": "key observations"
  }},
  "code_quality_score": {{
    "score": 0-10,
    "rationale": "justification"
  }},
  "summary": "key takeaway about FastAPI's design philosophy"
}}"""

EXPLAIN_IMPLEMENTATION_PROMPT = """Explain how this FastAPI code works for a developer
who is familiar with Python but new to FastAPI:

File: {file_path}
Entity: {entity_name} ({entity_type})
```python
{source_code}
```

Return JSON matching EXACTLY:
{{
  "overview": "high-level description in plain English",
  "prerequisite_concepts": [
    {{"concept": "...", "what_it_is": "brief explanation"}}
  ],
  "step_by_step": [
    {{"step": 1, "what_happens": "...", "why_it_matters": "..."}}
  ],
  "fastapi_magic": [
    {{"feature": "...", "how_it_works_here": "..."}}
  ],
  "data_flow": "how data enters, transforms, and exits",
  "when_is_this_called": "lifecycle context — when FastAPI calls this",
  "gotchas": [
    {{"issue": "...", "explanation": "...", "how_to_avoid": "..."}}
  ],
  "related_components": [
    {{"name": "...", "relationship": "how it interacts with this code"}}
  ],
  "analogies": ["plain-English analogy if helpful"],
  "further_reading": ["FastAPI docs topics to explore"]
}}"""

COMPARE_PROMPT = """Compare these two FastAPI code entities:

Entity A — {name_a} ({type_a}) in {file_a}:
```python
{source_a}
```

Entity B — {name_b} ({type_b}) in {file_b}:
```python
{source_b}
```

Return JSON matching EXACTLY:
{{
  "overview": "one-sentence summary of what each does",
  "similarities": [
    {{"aspect": "...", "description": "..."}}
  ],
  "differences": [
    {{
      "aspect": "...",
      "entity_a": "how A handles this",
      "entity_b": "how B handles this",
      "significance": "why this difference matters"
    }}
  ],
  "complexity_comparison": {{
    "entity_a": "low | medium | high",
    "entity_b": "low | medium | high",
    "more_complex": "A | B | equal",
    "reason": "..."
  }},
  "performance": {{
    "entity_a": "performance characteristics",
    "entity_b": "performance characteristics",
    "recommendation": "which is faster/lighter and why"
  }},
  "when_to_use_a": ["scenario 1", "scenario 2"],
  "when_to_use_b": ["scenario 1", "scenario 2"],
  "recommendation": "definitive guidance on choosing between them",
  "are_interchangeable": true/false,
  "migration_notes": "if replacing one with the other, what to watch for"
}}"""