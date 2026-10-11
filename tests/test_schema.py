"""
Tests for traceability.schema — DDL generation.

No DB required for structural tests.
Live DDL execution tests skip if PostgreSQL is unavailable.
"""

from __future__ import annotations

import pytest

from traceability.config import TraceabilityConfig
from traceability.schema import (
    SCHEMA_VERSION,
    all_create_ddl,
    btree_index_ddl,
    create_code_components_table_ddl,
    create_docstrings_table_ddl,
    create_extension_ddl,
    create_mappings_table_ddl,
    create_migrations_table_ddl,
    create_relationships_table_ddl,
    create_requirements_table_ddl,
    create_srs_documents_table_ddl,
    ivfflat_index_ddl,
)
from traceability.tests.conftest import FAKE_DIM, requires_pg

DIM = 384


@pytest.fixture()
def cfg():
    return TraceabilityConfig(embed_dim=DIM)


# ============================================================================
# SCHEMA_VERSION
# ============================================================================

def test_schema_version_string():
    assert isinstance(SCHEMA_VERSION, str)
    assert SCHEMA_VERSION  # non-empty


# ============================================================================
# Extension
# ============================================================================

def test_extension_contains_vector(cfg):
    assert "vector" in create_extension_ddl().lower()

def test_extension_idempotent(cfg):
    assert "if not exists" in create_extension_ddl().lower()


# ============================================================================
# Migrations table
# ============================================================================

def test_migrations_table_name(cfg):
    sql = create_migrations_table_ddl(cfg)
    assert cfg.table_migrations in sql

def test_migrations_primary_key(cfg):
    sql = create_migrations_table_ddl(cfg).lower()
    assert "version" in sql and "primary key" in sql


# ============================================================================
# SRS Documents table
# ============================================================================

def test_srs_documents_table_name(cfg):
    assert cfg.table_srs_documents in create_srs_documents_table_ddl(cfg)

def test_srs_documents_unique_document_id(cfg):
    sql = create_srs_documents_table_ddl(cfg).lower()
    assert "unique" in sql and "document_id" in sql

def test_srs_documents_if_not_exists(cfg):
    assert "if not exists" in create_srs_documents_table_ddl(cfg).lower()


# ============================================================================
# Requirements table
# ============================================================================

def test_requirements_surrogate_pk(cfg):
    sql = create_requirements_table_ddl(cfg, DIM).lower()
    assert "req_pk" in sql and "bigserial" in sql

def test_requirements_natural_unique(cfg):
    sql = create_requirements_table_ddl(cfg, DIM).lower()
    # Natural key: (document_pk, requirement_id)
    assert "document_pk" in sql and "requirement_id" in sql
    assert "unique" in sql

def test_requirements_no_primary_key_on_requirement_id(cfg):
    sql = create_requirements_table_ddl(cfg, DIM)
    # requirement_id must NOT be PRIMARY KEY alone
    lines = [l.strip().lower() for l in sql.splitlines()]
    for line in lines:
        if "requirement_id" in line and "primary key" in line:
            pytest.fail("requirement_id must not be PRIMARY KEY")

def test_requirements_fk_to_documents(cfg):
    sql = create_requirements_table_ddl(cfg, DIM).lower()
    assert cfg.table_srs_documents.lower() in sql

def test_requirements_vector_dim(cfg):
    sql = create_requirements_table_ddl(cfg, DIM)
    assert f"vector({DIM})" in sql

def test_requirements_embed_metadata_columns(cfg):
    sql = create_requirements_table_ddl(cfg, DIM).lower()
    assert "embed_model_version" in sql
    assert "embed_text_hash" in sql

def test_requirements_type_check(cfg):
    sql = create_requirements_table_ddl(cfg, DIM).lower()
    for t in ("functional", "non-functional", "other"):
        assert t in sql

def test_requirements_jsonb_columns(cfg):
    sql = create_requirements_table_ddl(cfg, DIM).lower()
    for col in ("actor", "actions", "entities", "domain_terms", "synonyms"):
        assert col in sql

def test_requirements_source_columns(cfg):
    sql = create_requirements_table_ddl(cfg, DIM).lower()
    for col in ("src_section", "src_paragraph", "src_page", "src_table"):
        assert col in sql

def test_requirements_different_dim(cfg):
    sql = create_requirements_table_ddl(cfg, 768)
    assert "vector(768)" in sql and "vector(384)" not in sql

def test_requirements_if_not_exists(cfg):
    assert "if not exists" in create_requirements_table_ddl(cfg, DIM).lower()


# ============================================================================
# Code components table
# ============================================================================

def test_components_component_id_pk(cfg):
    sql = create_code_components_table_ddl(cfg, DIM).lower()
    assert "component_id" in sql and "primary key" in sql

def test_components_repo_columns(cfg):
    sql = create_code_components_table_ddl(cfg, DIM).lower()
    assert "repo_path" in sql and "repo_snapshot" in sql

def test_components_vector_dim(cfg):
    assert f"vector({DIM})" in create_code_components_table_ddl(cfg, DIM)

def test_components_embed_metadata(cfg):
    sql = create_code_components_table_ddl(cfg, DIM).lower()
    assert "embed_model_version" in sql and "embed_text_hash" in sql

def test_components_v1_columns(cfg):
    sql = create_code_components_table_ddl(cfg, DIM).lower()
    for col in ("signature", "return_type", "containing_class", "parameters"):
        assert col in sql

def test_components_type_check(cfg):
    sql = create_code_components_table_ddl(cfg, DIM).lower()
    for ct in ("class", "function", "method"):
        assert ct in sql


# ============================================================================
# Docstrings table
# ============================================================================

def test_docstrings_table_name(cfg):
    assert cfg.table_docstrings in create_docstrings_table_ddl(cfg)

def test_docstrings_fk_to_components(cfg):
    sql = create_docstrings_table_ddl(cfg).lower()
    assert cfg.table_components.lower() in sql

def test_docstrings_kind_check(cfg):
    sql = create_docstrings_table_ddl(cfg).lower()
    assert "original" in sql and "generated" in sql

def test_docstrings_unique_per_component_kind(cfg):
    sql = create_docstrings_table_ddl(cfg).lower()
    assert "unique" in sql and "docstring_kind" in sql


# ============================================================================
# Relationships table
# ============================================================================

def test_relationships_table_name(cfg):
    assert cfg.table_relationships in create_relationships_table_ddl(cfg)

def test_relationships_fk_to_components(cfg):
    sql = create_relationships_table_ddl(cfg).lower()
    assert cfg.table_components.lower() in sql

def test_relationships_unique_triple(cfg):
    sql = create_relationships_table_ddl(cfg).lower()
    assert "unique" in sql

def test_relationships_rel_type_open_text(cfg):
    # rel_type must be TEXT without a CHECK constraint
    # so future types don't break ingestion
    sql = create_relationships_table_ddl(cfg).lower()
    assert "rel_type" in sql
    # Must NOT have 'check' on the same line as rel_type limiting values
    for line in sql.splitlines():
        if "rel_type" in line and "check" in line:
            pytest.fail("rel_type must not have a CHECK constraint in schema")


# ============================================================================
# Mappings table
# ============================================================================

def test_mappings_fk_req(cfg):
    sql = create_mappings_table_ddl(cfg).lower()
    assert cfg.table_requirements.lower() in sql

def test_mappings_fk_comp(cfg):
    sql = create_mappings_table_ddl(cfg).lower()
    assert cfg.table_components.lower() in sql

def test_mappings_confidence_check(cfg):
    sql = create_mappings_table_ddl(cfg).lower()
    assert "confidence_score" in sql and "between 0" in sql

def test_mappings_uses_req_pk_not_requirement_id(cfg):
    sql = create_mappings_table_ddl(cfg).lower()
    assert "req_pk" in sql


# ============================================================================
# Indexes
# ============================================================================

def test_btree_returns_list(cfg):
    stmts = btree_index_ddl(cfg)
    assert isinstance(stmts, list) and len(stmts) > 0

def test_btree_all_if_not_exists(cfg):
    for s in btree_index_ddl(cfg):
        assert "if not exists" in s.lower()

def test_btree_covers_all_tables(cfg):
    combined = " ".join(btree_index_ddl(cfg)).lower()
    for t in (cfg.table_requirements, cfg.table_components,
              cfg.table_relationships, cfg.table_mappings):
        assert t.lower() in combined

def test_ivfflat_two_stmts(cfg):
    assert len(ivfflat_index_ddl(cfg)) == 2

def test_ivfflat_cosine(cfg):
    for s in ivfflat_index_ddl(cfg):
        assert "vector_cosine_ops" in s

def test_ivfflat_custom_lists(cfg):
    for s in ivfflat_index_ddl(cfg, lists=50):
        assert "50" in s


# ============================================================================
# all_create_ddl — ordering
# ============================================================================

def test_extension_first(cfg):
    stmts = all_create_ddl(cfg, DIM)
    assert "CREATE EXTENSION" in stmts[0]

def test_migrations_before_documents(cfg):
    stmts = all_create_ddl(cfg, DIM)
    mig = next(i for i, s in enumerate(stmts) if cfg.table_migrations in s and "CREATE TABLE" in s)
    doc = next(i for i, s in enumerate(stmts) if cfg.table_srs_documents in s and "CREATE TABLE" in s)
    assert mig < doc

def test_documents_before_requirements(cfg):
    stmts = all_create_ddl(cfg, DIM)
    doc = next(i for i, s in enumerate(stmts) if cfg.table_srs_documents in s and "CREATE TABLE" in s)
    req = next(i for i, s in enumerate(stmts) if cfg.table_requirements in s and "CREATE TABLE" in s)
    assert doc < req

def test_components_before_docstrings(cfg):
    stmts = all_create_ddl(cfg, DIM)
    comp = next(i for i, s in enumerate(stmts) if cfg.table_components in s and "CREATE TABLE" in s)
    ds   = next(i for i, s in enumerate(stmts) if cfg.table_docstrings in s and "CREATE TABLE" in s)
    assert comp < ds

def test_components_before_relationships(cfg):
    stmts = all_create_ddl(cfg, DIM)
    comp = next(i for i, s in enumerate(stmts) if cfg.table_components in s and "CREATE TABLE" in s)
    rel  = next(i for i, s in enumerate(stmts) if cfg.table_relationships in s and "CREATE TABLE" in s)
    assert comp < rel

def test_no_ivfflat_in_all_create(cfg):
    assert "ivfflat" not in " ".join(all_create_ddl(cfg, DIM)).lower()

def test_zero_dim_raises(cfg):
    with pytest.raises(ValueError, match="embed_dim"):
        all_create_ddl(cfg, 0)

def test_custom_table_names():
    cfg2 = TraceabilityConfig(
        table_requirements="custom_req",
        table_components="custom_comp",
        table_docstrings="custom_ds",
        table_relationships="custom_rel",
        table_mappings="custom_map",
        table_srs_documents="custom_docs",
        embed_dim=DIM,
    )
    combined = " ".join(all_create_ddl(cfg2, DIM))
    for name in ("custom_req", "custom_comp", "custom_ds",
                 "custom_rel", "custom_map", "custom_docs"):
        assert name in combined


# ============================================================================
# Live DDL execution
# ============================================================================

@requires_pg
class TestLiveDDL:

    @pytest.fixture(autouse=True)
    def _setup(self, live_store):
        # live_store fixture already calls init_schema
        self.store = live_store

    def test_init_schema_twice_no_error(self):
        self.store.init_schema(FAKE_DIM)   # second call is idempotent

    def test_tables_exist(self, cfg):
        import psycopg2
        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        for t in (cfg.table_migrations, cfg.table_srs_documents,
                  cfg.table_requirements, cfg.table_components,
                  cfg.table_docstrings, cfg.table_relationships,
                  cfg.table_mappings):
            cur.execute("SELECT to_regclass(%s);", (t,))
            result = cur.fetchone()[0]
            assert result is not None, f"Table {t!r} was not created"
        cur.close()
        conn.close()
