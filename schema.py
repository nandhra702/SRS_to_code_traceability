"""
PostgreSQL schema for the traceability package (schema version 2.0.0).

Tables (in creation order)
--------------------------
tr_schema_migrations   schema version + the embedding dimension the vector
                       columns were created with (guards against model changes)
tr_srs_documents       one row per SRS source; gives requirements a scope
tr_requirements        natural key (document_pk, requirement_id) — the same
                       label (FR-01.1) can appear in two documents
tr_repositories        one row per analysed code repository / snapshot
tr_code_components     natural key (repo_pk, component_id) — the same
                       component ID (main.main) can appear in two repositories
tr_docstrings          docstrings as their own entity, with their own embedding;
                       'existing' (in source when DocAgent parsed) and
                       'generated' (written by DocAgent afterwards) kept apart
tr_relationships       DocAgent's typed edges, constrained to its five types
tr_trace_links         requirement ↔ docstring ↔ component links (Phase 3)
tr_verifications       append-only verification history (Phase 4)

Design decisions
----------------
- The vector dimension is never hard-coded; vector(N) columns are built from the
  embedder's resolved dimension and N is recorded in tr_schema_migrations.
  Store.init_schema() refuses a different N instead of silently keeping the
  old columns (CREATE TABLE IF NOT EXISTS would otherwise do exactly that).
- Requirements, components and docstrings each carry their own embedding,
  model version and text hash, so stale vectors can be detected per entity.
- Each searchable entity has a generated tsvector column + GIN index for the
  Phase 2 lexical baseline.
- Rows that disappear from a re-ingested source are flagged is_current = FALSE
  rather than deleted, so trace links and verification history survive.
- IF NOT EXISTS on every CREATE; no DROP anywhere.
"""

from __future__ import annotations

from typing import List

from .config import TraceabilityConfig

# Bump when tables/columns change.  Store.init_schema() refuses to run against
# a database initialised with a different version.
SCHEMA_VERSION = "2.0.0"

RELATIONSHIP_TYPES = ("calls", "imports", "inherits", "instantiates", "contains")
COMPONENT_TYPES = ("class", "function", "method")
DOCSTRING_KINDS = ("existing", "generated")
LINK_TYPES = (
    "requirement_to_docstring",
    "requirement_to_component",
    "docstring_to_component",
)
# Verdict statuses (plan Phase 4).  not_found = the pipeline found no suitable
# candidate, not that the requirement is definitely unimplemented.
VERIFICATION_STATUSES = (
    "supported",
    "partially_supported",
    "not_supported_by_candidate",
    "insufficient_evidence",
    "not_found",
)
LINK_STATUSES = ("candidate", "pending_verification") + VERIFICATION_STATUSES


def _sql_list(values) -> str:
    return ", ".join(f"'{v}'" for v in values)


# ---------------------------------------------------------------------------
# Extension + migrations
# ---------------------------------------------------------------------------


def create_extension_ddl() -> str:
    return "CREATE EXTENSION IF NOT EXISTS vector;"


def create_migrations_table_ddl(cfg: TraceabilityConfig) -> str:
    t = cfg.table_migrations
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    version         TEXT        PRIMARY KEY,
    embed_dim       INTEGER     NOT NULL CHECK (embed_dim > 0),
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description     TEXT
);
""".strip()


# ---------------------------------------------------------------------------
# SRS side
# ---------------------------------------------------------------------------


def create_srs_documents_table_ddl(cfg: TraceabilityConfig) -> str:
    t = cfg.table_srs_documents
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    document_pk         BIGSERIAL   PRIMARY KEY,
    -- Logical document identifier (defaults to the output file stem)
    document_id         TEXT        NOT NULL UNIQUE,
    file_name           TEXT        NOT NULL,
    source_type         TEXT        NOT NULL,   -- 'docx' | 'pdf' | 'text' | 'unknown'
    ingest_path         TEXT,
    total_requirements  INTEGER     NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""".strip()


def create_requirements_table_ddl(cfg: TraceabilityConfig, embed_dim: int) -> str:
    t = cfg.table_requirements
    docs = cfg.table_srs_documents
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    req_pk              BIGSERIAL   PRIMARY KEY,
    document_pk         BIGINT      NOT NULL REFERENCES {docs}(document_pk) ON DELETE CASCADE,
    requirement_id      TEXT        NOT NULL,
    id_source           TEXT        NOT NULL CHECK (id_source IN ('source', 'generated')),
    UNIQUE (document_pk, requirement_id),

    -- raw_text is the verbatim original and is never modified
    raw_text            TEXT        NOT NULL,
    normalized_text     TEXT        NOT NULL DEFAULT '',
    retrieval_text      TEXT        NOT NULL DEFAULT '',
    title               TEXT        NOT NULL DEFAULT '',
    type                TEXT        NOT NULL CHECK (type IN ('functional', 'non-functional', 'other')),
    subtype             TEXT        NOT NULL DEFAULT 'general',
    review_required     BOOLEAN     NOT NULL DEFAULT FALSE,

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

    -- Full SourceLocation as emitted by the SRS Agent (nothing dropped)
    source_location     JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
    -- Detection / candidate-judge fields (detection_confidence, candidate_*, …)
    extraction_meta     JSONB       NOT NULL DEFAULT '{{}}'::jsonb,

    is_current          BOOLEAN     NOT NULL DEFAULT TRUE,

    embedding           vector({embed_dim}),
    embed_model_version TEXT,
    embed_text_hash     TEXT,

    search_tsv          tsvector GENERATED ALWAYS AS (
        to_tsvector('english',
            title || ' ' || raw_text || ' ' || normalized_text || ' ' || retrieval_text)
    ) STORED,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""".strip()


# ---------------------------------------------------------------------------
# Code side
# ---------------------------------------------------------------------------


def create_repositories_table_ddl(cfg: TraceabilityConfig) -> str:
    t = cfg.table_repositories
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    repo_pk             BIGSERIAL   PRIMARY KEY,
    -- Logical repository identifier (defaults to the graph file stem)
    repo_id             TEXT        NOT NULL UNIQUE,
    repo_path           TEXT,
    repo_snapshot       TEXT,       -- git commit SHA or tag; NULL if unknown
    graph_path          TEXT,
    graph_format        TEXT        NOT NULL CHECK (graph_format IN ('v1', 'legacy')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""".strip()


def create_code_components_table_ddl(cfg: TraceabilityConfig, embed_dim: int) -> str:
    t = cfg.table_components
    repos = cfg.table_repositories
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    comp_pk             BIGSERIAL   PRIMARY KEY,
    repo_pk             BIGINT      NOT NULL REFERENCES {repos}(repo_pk) ON DELETE CASCADE,
    -- DocAgent canonical ID: module.ClassName.method_name
    component_id        TEXT        NOT NULL,
    UNIQUE (repo_pk, component_id),
    component_type      TEXT        NOT NULL CHECK (component_type IN ({_sql_list(COMPONENT_TYPES)})),

    file_path           TEXT        NOT NULL,
    relative_path       TEXT        NOT NULL,
    start_line          INTEGER     NOT NULL DEFAULT 0,
    end_line            INTEGER     NOT NULL DEFAULT 0,

    signature           TEXT,
    parameters          JSONB       NOT NULL DEFAULT '[]'::jsonb,
    return_type         TEXT,
    containing_class    TEXT,
    -- DocAgent's untyped ordering dependencies, kept verbatim (noisy; typed
    -- edges live in the relationships table)
    depends_on          JSONB       NOT NULL DEFAULT '[]'::jsonb,

    -- Source as parsed by DocAgent; line numbers refer to this snapshot
    source_code         TEXT,
    source_hash         TEXT,

    is_current          BOOLEAN     NOT NULL DEFAULT TRUE,

    -- Code embedding (signature + docstring-stripped source)
    embedding           vector({embed_dim}),
    embed_model_version TEXT,
    embed_text_hash     TEXT,

    -- Identifiers split on _ and . so 'process_data' matches 'process data'
    search_tsv          tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', regexp_replace(
            component_id || ' ' || coalesce(signature, '') || ' ' || coalesce(source_code, ''),
            '[_.]', ' ', 'g'))
    ) STORED,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""".strip()


def create_docstrings_table_ddl(cfg: TraceabilityConfig, embed_dim: int) -> str:
    t = cfg.table_docstrings
    comp = cfg.table_components
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    docstring_pk        BIGSERIAL   PRIMARY KEY,
    comp_pk             BIGINT      NOT NULL REFERENCES {comp}(comp_pk) ON DELETE CASCADE,

    -- 'existing'  = present in source when DocAgent parsed the repo (this may
    --               itself be the output of an earlier DocAgent run)
    -- 'generated' = differs from / absent in the parsed source and found in
    --               the post-generation source tree
    docstring_kind      TEXT        NOT NULL CHECK (docstring_kind IN ({_sql_list(DOCSTRING_KINDS)})),
    UNIQUE (comp_pk, docstring_kind),

    content             TEXT        NOT NULL,
    content_hash        TEXT        NOT NULL,
    provenance          JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
    is_current          BOOLEAN     NOT NULL DEFAULT TRUE,

    embedding           vector({embed_dim}),
    embed_model_version TEXT,
    embed_text_hash     TEXT,

    search_tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""".strip()


def create_relationships_table_ddl(cfg: TraceabilityConfig) -> str:
    t = cfg.table_relationships
    comp = cfg.table_components
    repos = cfg.table_repositories
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    rel_pk              BIGSERIAL   PRIMARY KEY,
    repo_pk             BIGINT      NOT NULL REFERENCES {repos}(repo_pk) ON DELETE CASCADE,
    source_comp_pk      BIGINT      NOT NULL REFERENCES {comp}(comp_pk) ON DELETE CASCADE,
    target_comp_pk      BIGINT      NOT NULL REFERENCES {comp}(comp_pk) ON DELETE CASCADE,
    rel_type            TEXT        NOT NULL CHECK (rel_type IN ({_sql_list(RELATIONSHIP_TYPES)})),
    UNIQUE (source_comp_pk, target_comp_pk, rel_type)
);
""".strip()


# ---------------------------------------------------------------------------
# Mapping + verification (populated by Phases 3 and 4)
# ---------------------------------------------------------------------------


def create_trace_links_table_ddl(cfg: TraceabilityConfig) -> str:
    t = cfg.table_trace_links
    req = cfg.table_requirements
    ds = cfg.table_docstrings
    comp = cfg.table_components
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    link_pk             BIGSERIAL   PRIMARY KEY,
    link_type           TEXT        NOT NULL CHECK (link_type IN ({_sql_list(LINK_TYPES)})),
    req_pk              BIGINT      REFERENCES {req}(req_pk) ON DELETE CASCADE,
    docstring_pk        BIGINT      REFERENCES {ds}(docstring_pk) ON DELETE CASCADE,
    comp_pk             BIGINT      REFERENCES {comp}(comp_pk) ON DELETE CASCADE,

    -- Every link type must reference exactly the two entities it joins
    CHECK (
        (link_type = 'requirement_to_docstring'
            AND req_pk IS NOT NULL AND docstring_pk IS NOT NULL AND comp_pk IS NULL)
     OR (link_type = 'requirement_to_component'
            AND req_pk IS NOT NULL AND comp_pk IS NOT NULL AND docstring_pk IS NULL)
     OR (link_type = 'docstring_to_component'
            AND docstring_pk IS NOT NULL AND comp_pk IS NOT NULL AND req_pk IS NULL)
    ),

    -- Links stay pending until the verifier assigns a verdict
    status              TEXT        NOT NULL DEFAULT 'pending_verification'
                                    CHECK (status IN ({_sql_list(LINK_STATUSES)})),
    retrieval_method    TEXT,                       -- 'lexical' | 'vector' | 'hybrid' | 'ownership' | 'graph'
    scores              JSONB       NOT NULL DEFAULT '{{}}'::jsonb,   -- per-method score breakdown
    evidence            JSONB       NOT NULL DEFAULT '{{}}'::jsonb,   -- snippets, edge paths, locations
    pipeline_version    TEXT        NOT NULL,

    UNIQUE NULLS NOT DISTINCT (link_type, req_pk, docstring_pk, comp_pk, pipeline_version),

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""".strip()


def create_verifications_table_ddl(cfg: TraceabilityConfig) -> str:
    t = cfg.table_verifications
    links = cfg.table_trace_links
    req = cfg.table_requirements
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    verification_pk     BIGSERIAL   PRIMARY KEY,
    -- A verdict is about a proposed link, or (for not_found) a requirement alone
    link_pk             BIGINT      REFERENCES {links}(link_pk) ON DELETE CASCADE,
    req_pk              BIGINT      REFERENCES {req}(req_pk) ON DELETE CASCADE,
    CHECK (link_pk IS NOT NULL OR req_pk IS NOT NULL),

    status              TEXT        NOT NULL CHECK (status IN ({_sql_list(VERIFICATION_STATUSES)})),
    rationale           TEXT,
    claims              JSONB       NOT NULL DEFAULT '[]'::jsonb,   -- claim-level findings
    evidence_refs       JSONB       NOT NULL DEFAULT '[]'::jsonb,   -- validated component/line refs
    model               TEXT,
    prompt_version      TEXT,
    pipeline_version    TEXT        NOT NULL,
    -- Append-only: one row per verification run, never updated
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""".strip()


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------


def btree_index_ddl(cfg: TraceabilityConfig) -> List[str]:
    """B-tree and GIN indexes — safe to run on empty tables."""
    req = cfg.table_requirements
    comp = cfg.table_components
    ds = cfg.table_docstrings
    rel = cfg.table_relationships
    tl = cfg.table_trace_links
    vf = cfg.table_verifications
    return [
        f"CREATE INDEX IF NOT EXISTS idx_{req}_document_pk ON {req}(document_pk);",
        f"CREATE INDEX IF NOT EXISTS idx_{req}_requirement_id ON {req}(requirement_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{req}_type ON {req}(type, subtype);",
        f"CREATE INDEX IF NOT EXISTS idx_{req}_tsv ON {req} USING GIN (search_tsv);",
        f"CREATE INDEX IF NOT EXISTS idx_{comp}_repo ON {comp}(repo_pk);",
        f"CREATE INDEX IF NOT EXISTS idx_{comp}_file ON {comp}(repo_pk, relative_path);",
        f"CREATE INDEX IF NOT EXISTS idx_{comp}_tsv ON {comp} USING GIN (search_tsv);",
        f"CREATE INDEX IF NOT EXISTS idx_{ds}_comp ON {ds}(comp_pk);",
        f"CREATE INDEX IF NOT EXISTS idx_{ds}_tsv ON {ds} USING GIN (search_tsv);",
        f"CREATE INDEX IF NOT EXISTS idx_{rel}_source ON {rel}(source_comp_pk);",
        f"CREATE INDEX IF NOT EXISTS idx_{rel}_target ON {rel}(target_comp_pk);",
        f"CREATE INDEX IF NOT EXISTS idx_{tl}_req ON {tl}(req_pk);",
        f"CREATE INDEX IF NOT EXISTS idx_{tl}_comp ON {tl}(comp_pk);",
        f"CREATE INDEX IF NOT EXISTS idx_{tl}_status ON {tl}(status);",
        f"CREATE INDEX IF NOT EXISTS idx_{vf}_link ON {vf}(link_pk);",
    ]


def vector_index_ddl(cfg: TraceabilityConfig) -> List[str]:
    """
    Optional HNSW ANN indexes (cosine).  Exact search is used until a dataset
    is large enough to justify these; HNSW, unlike IVFFlat, can be built on
    an empty table.
    """
    return [
        f"CREATE INDEX IF NOT EXISTS idx_{t}_embedding "
        f"ON {t} USING hnsw (embedding vector_cosine_ops);"
        for t in (cfg.table_requirements, cfg.table_components, cfg.table_docstrings)
    ]


# ---------------------------------------------------------------------------
# Ordered DDL list
# ---------------------------------------------------------------------------


def all_create_ddl(cfg: TraceabilityConfig, embed_dim: int) -> List[str]:
    """
    All entity DDL in dependency order (the extension and migrations table are
    created separately by Store.init_schema so it can check the recorded
    version first).  Vector indexes are excluded — see vector_index_ddl().
    """
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}")
    stmts = [
        create_srs_documents_table_ddl(cfg),
        create_requirements_table_ddl(cfg, embed_dim),
        create_repositories_table_ddl(cfg),
        create_code_components_table_ddl(cfg, embed_dim),
        create_docstrings_table_ddl(cfg, embed_dim),
        create_relationships_table_ddl(cfg),
        create_trace_links_table_ddl(cfg),
        create_verifications_table_ddl(cfg),
    ]
    stmts.extend(btree_index_ddl(cfg))
    return stmts
