// ── FastAPI Repo Chat Agent — Knowledge Graph Schema ─────────────────────────
// Run this once to bootstrap a fresh Neo4j instance.
// neo4j 5.x syntax — IF NOT EXISTS makes all statements idempotent.

// ═══════════════════════════════════════════════════════════════════════════════
// NODE CONSTRAINTS (9 node types)
// ═══════════════════════════════════════════════════════════════════════════════

// 1. File — unique by absolute filesystem path
CREATE CONSTRAINT file_path_unique IF NOT EXISTS
FOR (f:File) REQUIRE f.path IS UNIQUE;

// 2. Module — unique by dotted Python module name (e.g. fastapi.routing)
CREATE CONSTRAINT module_name_unique IF NOT EXISTS
FOR (m:Module) REQUIRE m.name IS UNIQUE;

// 3. Class — composite key: same class name valid in different files
CREATE CONSTRAINT class_name_file_unique IF NOT EXISTS
FOR (c:Class) REQUIRE (c.name, c.file_path) IS NODE KEY;

// 4. Function — composite key: same function name valid in different files
CREATE CONSTRAINT function_name_file_unique IF NOT EXISTS
FOR (f:Function) REQUIRE (f.name, f.file_path) IS NODE KEY;

// 5. Method — name includes class prefix (e.g. FastAPI.__init__)
CREATE CONSTRAINT method_name_file_unique IF NOT EXISTS
FOR (m:Method) REQUIRE (m.name, m.file_path) IS NODE KEY;

// 6. Parameter — unique per owning function in a file
CREATE CONSTRAINT parameter_unique IF NOT EXISTS
FOR (p:Parameter) REQUIRE (p.name, p.function_name, p.file_path) IS NODE KEY;

// 7. Decorator — unique per decorated target
CREATE CONSTRAINT decorator_unique IF NOT EXISTS
FOR (d:Decorator) REQUIRE (d.name, d.target_name, d.target_file) IS NODE KEY;

// 8. Import — unique per (module, importing file)
CREATE CONSTRAINT import_unique IF NOT EXISTS
FOR (i:Import) REQUIRE (i.module, i.file_path) IS NODE KEY;

// 9. Docstring — unique per owning entity in a file
CREATE CONSTRAINT docstring_unique IF NOT EXISTS
FOR (d:Docstring) REQUIRE (d.owner_name, d.file_path) IS NODE KEY;


// ═══════════════════════════════════════════════════════════════════════════════
// INDEXES
// ═══════════════════════════════════════════════════════════════════════════════

// Hot lookup: entity name searches
CREATE INDEX class_name_idx    IF NOT EXISTS FOR (c:Class)    ON (c.name);
CREATE INDEX function_name_idx IF NOT EXISTS FOR (f:Function) ON (f.name);
CREATE INDEX method_name_idx   IF NOT EXISTS FOR (m:Method)   ON (m.name);
CREATE INDEX module_name_idx   IF NOT EXISTS FOR (m:Module)   ON (m.name);

// Pattern detection: filter by decorator name
CREATE INDEX decorator_name_idx IF NOT EXISTS FOR (d:Decorator) ON (d.name);

// Dependency injection analysis: find all Depends() parameters
CREATE INDEX parameter_annotation_idx IF NOT EXISTS FOR (p:Parameter) ON (p.annotation);

// Import chain traversal
CREATE INDEX import_module_idx IF NOT EXISTS FOR (i:Import) ON (i.module);

// File path lookups (per-file indexing)
CREATE INDEX file_path_idx IF NOT EXISTS FOR (f:File) ON (f.path);


// ═══════════════════════════════════════════════════════════════════════════════
// FULL-TEXT INDEXES
// ═══════════════════════════════════════════════════════════════════════════════

// Semantic search across all docstrings
// Usage: CALL db.index.fulltext.queryNodes("docstring_search", "dependency injection") YIELD node
CREATE FULLTEXT INDEX docstring_search IF NOT EXISTS
FOR (d:Docstring) ON EACH [d.content];

// Cross-entity name + docstring search
CREATE FULLTEXT INDEX entity_name_search IF NOT EXISTS
FOR (n:Class|Function|Method|Module) ON EACH [n.name, n.docstring];


// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLE RELATIONSHIP PATTERNS (for documentation — not runnable DDL)
// ═══════════════════════════════════════════════════════════════════════════════
//
// (File)-[:CONTAINS]->(Class)
// (File)-[:CONTAINS]->(Function)
// (File)-[:CONTAINS]->(Import)
// (File)-[:CONTAINS]->(Module)
//
// (Class)-[:CONTAINS]->(Method)
// (Module)-[:CONTAINS]->(Class)
// (Module)-[:CONTAINS]->(Function)
//
// (File)-[:IMPORTS]->(Import)
// (Module)-[:DEPENDS_ON]->(Module)
//
// (Class)-[:INHERITS_FROM]->(Class)
//
// (Function)-[:CALLS]->(Function)
// (Method)-[:CALLS]->(Function)
// (Method)-[:CALLS]->(Method)
//
// (Class)-[:DECORATED_BY]->(Decorator)
// (Function)-[:DECORATED_BY]->(Decorator)
// (Method)-[:DECORATED_BY]->(Decorator)
//
// (Function)-[:HAS_PARAMETER]->(Parameter)
// (Method)-[:HAS_PARAMETER]->(Parameter)
//
// (Class)-[:DOCUMENTED_BY]->(Docstring)
// (Function)-[:DOCUMENTED_BY]->(Docstring)
// (Method)-[:DOCUMENTED_BY]->(Docstring)
// (Module)-[:DOCUMENTED_BY]->(Docstring)