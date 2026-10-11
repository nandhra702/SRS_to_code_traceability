"""
PostgreSQL schema for the traceability package.

Tables (in creation order)
--------------------------
tr_schema_migrations      versioned migration log; idempotency guard
tr_srs_documents          one row per SRS source file; gives requirements a scope
tr_requirements           one row per requirement; PK is (document_pk, requirement_id)
                          so the same label (e.g. FR-01.1) can appear in two docs
tr_code_components        one row per CodeComponent; carries repo/snapshot identity
tr_docstrings             keeps original and generated docstrings as distinct rows
tr_relationships          flat, deduplicated typed edges between components
tr_mappings               traceability links (populated by Mapping Agent)

Design decisions
----------------
- embed_dim is never hard-coded.  All vector(N) columns are built from the
  integer returned by Embedder.dim at runtime.
- embed_model_version + embed_text_hash on each embedding row allow stale
  embedding detection when the model changes.
- requirements.requirement_id is NOT the primary key because the same label
  (FR-01.1) can legitimately appear in multiple documents.  The surrogate PK
  is a BIGSERIAL; the natural composite key is (document_pk, requirement_id).
- tr_docstrings separates original source docstrings from LLM-generated ones
  by a docstring_kind column ('original' | 'generated'), matching the data
  that DocAgent's CodeComponent actually carries.
- All table names are configurable via TraceabilityConfig so the package can
  coexist with other tables in the same database.
- IF NOT EXISTS on every CREATE statement — safe to call multiple times.
- No DROP TABLE anywhere.
"""

from __future__ import annotations

from typing import List

from .config import TraceabilityConfig

# ---------------------------------------------------------------------------
# Current schema version — bump when adding columns or tables
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------


def create_extension_ddl() -> str:
    return "CREATE EXTENSION IF NOT EXISTS vector;"


# ---------------------------------------------------------------------------
# Migrations table — idempotency guard
# ---------------------------------------------------------------------------


def create_migrations_table_ddl(cfg: TraceabilityConfig) -> str:
    t = cfg.table_migrations
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    version         TEXT        PRIMARY KEY,
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description     TEXT
);
""".strip()


# ---------------------------------------------------------------------------
# SRS documents table — document-scope parent for requirements
# ---------------------------------------------------------------------------


def create_srs_documents_table_ddl(cfg: TraceabilityConfig) -> str:
    t = cfg.table_srs_documents
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    document_pk     BIGSERIAL   PRIMARY KEY,
    -- Logical document identifier (file stem, UUID, or user-supplied label)
    document_id     TEXT        NOT NULL,
    file_name       TEXT        NOT NULL,
    source_type     TEXT        NOT NULL,   -- 'docx', 'pdf', 'text'
    ingest_path     TEXT,                   -- filesystem path at ingest time
    total_requirements INTEGER  NOT NULL DEFAULT 0,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- A document is uniquely identified by its file name within a logical scope.
    -- Use document_id as the de-duplication key (e.g. file stem hash).
    UNIQUE (document_id)
);
""".strip()


# ---------------------------------------------------------------------------
# Requirements table
# ---------------------------------------------------------------------------


def create_requirements_table_ddl(cfg: TraceabilityConfig, embed_dim: int) -> str:
    t = cfg.table_requirements
    docs = cfg.table_srs_documents
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    -- Surrogate PK — requirement_id alone is not unique across documents
    req_pk              BIGSERIAL   PRIMARY KEY,

    -- Document scope (FK to tr_srs_documents)
    document_pk         BIGINT      NOT NULL REFERENCES {docs}(document_pk) ON DELETE CASCADE,

    -- Requirement identity within its document
    requirement_id      TEXT        NOT NULL,
    id_source           TEXT        NOT NULL CHECK (id_source IN ('source', 'generated')),

    -- Natural uniqueness: same label cannot appear twice in one document
    UNIQUE (document_pk, requirement_id),

    -- Core text (raw_text is the verbatim original; never modified)
    raw_text            TEXT        NOT NULL,
    normalized_text     TEXT        NOT NULL DEFAULT '',
    retrieval_text      TEXT        NOT NULL DEFAULT '',
    title               TEXT        NOT NULL DEFAULT '',

    -- Classification
    type                TEXT        NOT NULL CHECK (type IN ('functional', 'non-functional', 'other')),
    subtype             TEXT        NOT NULL DEFAULT 'general',

    -- Semantic arrays (JSONB, not ARRAY, to allow future nesting)
    actor               JSONB       NOT NULL DEFAULT '[]'::jsonb,
    actions             JSONB       NOT NULL DEFAULT '[]'::jsonb,
    entities            JSONB       NOT NULL DEFAULT '[]'::jsonb,
    inputs              JSONB       NOT NULL DEFAULT '[]'::jsonb,
    outputs             JSONB       NOT NULL DEFAULT '[]'::jsonb,
    preconditions       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    postconditions      JSONB       NOT NULL DEFAULT '[]'::jsonb,
    conditions          JSONB       NOT NULL DEFAULT '[]'::jsonb,
    constraints         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    error_conditions    JSONB       NOT NULL DEFAULT '[]'::jsonb,
    domain_terms        JSONB       NOT NULL DEFAULT '[]'::jsonb,
    synonyms            JSONB       NOT NULL DEFAULT '[]'::jsonb,
    sub_requirements    JSONB       NOT NULL DEFAULT '[]'::jsonb,

    -- Source location within the document (SourceLocation flattened)
    src_section         TEXT,
    src_subsection      TEXT,
    src_page            INTEGER,
    src_paragraph       INTEGER,
    src_table           INTEGER,
    src_row             INTEGER,
    src_column          INTEGER,

    -- Embedding + provenance
    embedding           vector({embed_dim}),
    embed_model_version TEXT,
    embed_text_hash     TEXT,   -- SHA-256 of the text that was embedded

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""".strip()


# ---------------------------------------------------------------------------
# Code components table
# ---------------------------------------------------------------------------


def create_code_components_table_ddl(cfg: TraceabilityConfig, embed_dim: int) -> str:
    t = cfg.table_components
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    -- component_id is the DocAgent canonical ID: module.ClassName.method_name
    component_id        TEXT        PRIMARY KEY,
    component_type      TEXT        NOT NULL CHECK (component_type IN ('class', 'function', 'method')),

    -- Repository / snapshot identity
    repo_path           TEXT,       -- absolute repo path at parse time
    repo_snapshot       TEXT,       -- git commit SHA or tag, NULL if unknown

    -- File location
    file_path           TEXT        NOT NULL,
    relative_path       TEXT        NOT NULL,
    start_line          INTEGER     NOT NULL DEFAULT 0,
    end_line            INTEGER     NOT NULL DEFAULT 0,

    -- V1 metadata
    signature           TEXT,
    return_type         TEXT,
    containing_class    TEXT,
    parameters          JSONB       NOT NULL DEFAULT '[]'::jsonb,
    depends_on          JSONB       NOT NULL DEFAULT '[]'::jsonb,

    -- Source code (stored for context; may be large)
    source_code         TEXT,

    -- Embedding + provenance
    embedding           vector({embed_dim}),
    embed_model_version TEXT,
    embed_text_hash     TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""".strip()


# ---------------------------------------------------------------------------
# Docstrings table — keeps original and generated docstrings as distinct rows
# ---------------------------------------------------------------------------


def create_docstrings_table_ddl(cfg: TraceabilityConfig) -> str:
    t = cfg.table_docstrings
    comp = cfg.table_components
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    id                  BIGSERIAL   PRIMARY KEY,
    component_id        TEXT        NOT NULL REFERENCES {comp}(component_id) ON DELETE CASCADE,

    -- 'original'  = docstring present in source at parse time (has_docstring=True)
    -- 'generated' = docstring produced by DocAgent's Writer agent
    docstring_kind      TEXT        NOT NULL CHECK (docstring_kind IN ('original', 'generated')),

    content             TEXT        NOT NULL DEFAULT '',

    -- Only one row per (component, kind) is meaningful
    UNIQUE (component_id, docstring_kind),

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""".strip()


# ---------------------------------------------------------------------------
# Relationships table
# ---------------------------------------------------------------------------


def create_relationships_table_ddl(cfg: TraceabilityConfig) -> str:
    t = cfg.table_relationships
    comp = cfg.table_components
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    id                  BIGSERIAL   PRIMARY KEY,
    source_id           TEXT        NOT NULL REFERENCES {comp}(component_id) ON DELETE CASCADE,
    target_id           TEXT        NOT NULL REFERENCES {comp}(component_id) ON DELETE CASCADE,
    -- rel_type is open-ended TEXT to survive future DocAgent relationship types;
    -- the application layer validates against RELATIONSHIP_TYPES.
    rel_type            TEXT        NOT NULL,

    UNIQUE (source_id, target_id, rel_type)
);
""".strip()


# ---------------------------------------------------------------------------
# Mappings table
# ---------------------------------------------------------------------------


def create_mappings_table_ddl(cfg: TraceabilityConfig) -> str:
    t = cfg.table_mappings
    req = cfg.table_requirements
    comp = cfg.table_components
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    id                  BIGSERIAL   PRIMARY KEY,
    req_pk              BIGINT      NOT NULL REFERENCES {req}(req_pk) ON DELETE CASCADE,
    component_id        TEXT        NOT NULL REFERENCES {comp}(component_id) ON DELETE CASCADE,

    confidence_score    REAL        CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    -- 'vector_search' | 'llm_verified' | 'manual'
    mapping_type        TEXT        NOT NULL DEFAULT 'vector_search',
    rationale           TEXT,

    UNIQUE (req_pk, component_id),

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""".strip()


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------


def btree_index_ddl(cfg: TraceabilityConfig) -> List[str]:
    """B-tree indexes — safe to run on empty tables."""
    req = cfg.table_requirements
    comp = cfg.table_components
    docs = cfg.table_srs_documents
    rel = cfg.table_relationships
    ds = cfg.table_docstrings
    mp = cfg.table_mappings
    return [
        f"CREATE INDEX IF NOT EXISTS idx_{docs}_document_id ON {docs}(document_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{req}_document_pk ON {req}(document_pk);",
        f"CREATE INDEX IF NOT EXISTS idx_{req}_requirement_id ON {req}(requirement_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{req}_type ON {req}(type);",
        f"CREATE INDEX IF NOT EXISTS idx_{req}_subtype ON {req}(subtype);",
        f"CREATE INDEX IF NOT EXISTS idx_{req}_embed_version ON {req}(embed_model_version);",
        f"CREATE INDEX IF NOT EXISTS idx_{comp}_type ON {comp}(component_type);",
        f"CREATE INDEX IF NOT EXISTS idx_{comp}_file ON {comp}(relative_path);",
        f"CREATE INDEX IF NOT EXISTS idx_{comp}_embed_version ON {comp}(embed_model_version);",
        f"CREATE INDEX IF NOT EXISTS idx_{ds}_component ON {ds}(component_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{rel}_source ON {rel}(source_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{rel}_target ON {rel}(target_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{rel}_type ON {rel}(rel_type);",
        f"CREATE INDEX IF NOT EXISTS idx_{mp}_req ON {mp}(req_pk);",
        f"CREATE INDEX IF NOT EXISTS idx_{mp}_comp ON {mp}(component_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{mp}_score ON {mp}(confidence_score DESC NULLS LAST);",
    ]


def ivfflat_index_ddl(cfg: TraceabilityConfig, lists: int = 100) -> List[str]:
    """
    IVFFlat ANN indexes on vector columns.

    Must be called AFTER data is loaded — IVFFlat requires rows to exist.
    """
    req = cfg.table_requirements
    comp = cfg.table_components
    return [
        (
            f"CREATE INDEX IF NOT EXISTS idx_{req}_embedding "
            f"ON {req} USING ivfflat (embedding vector_cosine_ops) "
            f"WITH (lists = {lists});"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS idx_{comp}_embedding "
            f"ON {comp} USING ivfflat (embedding vector_cosine_ops) "
            f"WITH (lists = {lists});"
        ),
    ]


# ---------------------------------------------------------------------------
# Ordered DDL list
# ---------------------------------------------------------------------------


def all_create_ddl(cfg: TraceabilityConfig, embed_dim: int) -> List[str]:
    """
    Return all DDL statements in strict dependency order:
      extension → migrations → documents → requirements → components →
      docstrings → relationships → mappings → btree indexes.

    IVFFlat indexes are excluded — call ivfflat_index_ddl() after loading data.

    Args:
        cfg:       TraceabilityConfig instance.
        embed_dim: Resolved embedding dimension (must be > 0).
    """
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}")
    stmts = [
        create_extension_ddl(),
        create_migrations_table_ddl(cfg),
        create_srs_documents_table_ddl(cfg),
        create_requirements_table_ddl(cfg, embed_dim),
        create_code_components_table_ddl(cfg, embed_dim),
        create_docstrings_table_ddl(cfg),
        create_relationships_table_ddl(cfg),
        create_mappings_table_ddl(cfg),
    ]
    stmts.extend(btree_index_ddl(cfg))
    return stmts
