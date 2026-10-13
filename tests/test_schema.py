"""
Tests for traceability.schema.

Structural tests check the generated DDL without a database.  The live tests
execute the DDL on the disposable test database and check that constraints
are actually *enforced*, which string matching cannot show.
"""

from __future__ import annotations

import psycopg2
import pytest

from traceability.schema import (
    LINK_TYPES,
    RELATIONSHIP_TYPES,
    SCHEMA_VERSION,
    all_create_ddl,
    create_code_components_table_ddl,
    create_requirements_table_ddl,
    vector_index_ddl,
)
from traceability.tests.conftest import FAKE_DIM, make_cfg, requires_pg


# ============================================================================
# Structural (no database)
# ============================================================================


def test_schema_version_is_v2():
    assert SCHEMA_VERSION.startswith("2.")


def test_relationship_types_match_docagent_v1():
    # DocAgent V1 RELATIONSHIP_TYPES
    assert set(RELATIONSHIP_TYPES) == {"calls", "imports", "inherits", "instantiates", "contains"}


@pytest.mark.parametrize("dim", [4, 384, 1536])
def test_vector_dimension_comes_from_argument(dim):
    cfg = make_cfg()
    assert f"vector({dim})" in create_requirements_table_ddl(cfg, dim)
    assert f"vector({dim})" in create_code_components_table_ddl(cfg, dim)


def test_zero_dim_raises():
    with pytest.raises(ValueError):
        all_create_ddl(make_cfg(), 0)


def test_no_drop_statements():
    ddl = " ".join(all_create_ddl(make_cfg(), FAKE_DIM)).upper()
    assert "DROP " not in ddl and "TRUNCATE" not in ddl


def test_every_create_is_idempotent():
    for stmt in all_create_ddl(make_cfg(), FAKE_DIM) + vector_index_ddl(make_cfg()):
        assert "IF NOT EXISTS" in stmt, stmt[:80]


def test_custom_table_names_used():
    cfg = make_cfg(table_components="custom_comps")
    ddl = " ".join(all_create_ddl(cfg, FAKE_DIM))
    assert "custom_comps" in ddl
    assert "tr_code_components" not in ddl


def test_vector_indexes_are_hnsw_cosine():
    stmts = vector_index_ddl(make_cfg())
    assert len(stmts) == 3
    assert all("hnsw" in s and "vector_cosine_ops" in s for s in stmts)


# ============================================================================
# Live: constraints are enforced
# ============================================================================


def _insert(cur, sql, params=None):
    cur.execute(sql, params)
    return cur.fetchone()[0] if cur.description else None


@requires_pg
class TestLiveConstraints:
    """Each test runs against freshly initialised test tables."""

    @pytest.fixture(autouse=True)
    def _db(self, live_store, cfg):
        self.cfg = cfg
        self.conn = psycopg2.connect(cfg.build_dsn())
        cur = self.conn.cursor()
        self.repo = _insert(cur,
            f"INSERT INTO {cfg.table_repositories} (repo_id, graph_format) "
            f"VALUES ('r', 'v1') RETURNING repo_pk")
        self.doc = _insert(cur,
            f"INSERT INTO {cfg.table_srs_documents} (document_id, file_name, source_type) "
            f"VALUES ('d', 'd.docx', 'docx') RETURNING document_pk")
        self.conn.commit()
        yield
        self.conn.rollback()
        self.conn.close()

    def _comp(self, cid, ctype="function", repo=None):
        cur = self.conn.cursor()
        return _insert(cur,
            f"INSERT INTO {self.cfg.table_components} "
            f"(repo_pk, component_id, component_type, file_path, relative_path) "
            f"VALUES (%s, %s, %s, 'f.py', 'f.py') RETURNING comp_pk",
            (repo or self.repo, cid, ctype))

    def _req(self, rid, rtype="functional", doc=None):
        cur = self.conn.cursor()
        return _insert(cur,
            f"INSERT INTO {self.cfg.table_requirements} "
            f"(document_pk, requirement_id, id_source, raw_text, type) "
            f"VALUES (%s, %s, 'source', 'text', %s) RETURNING req_pk",
            (doc or self.doc, rid, rtype))

    def test_same_component_id_allowed_in_two_repos(self):
        cur = self.conn.cursor()
        other = _insert(cur, f"INSERT INTO {self.cfg.table_repositories} (repo_id, graph_format) "
                             f"VALUES ('r2', 'v1') RETURNING repo_pk")
        self._comp("main.main")
        self._comp("main.main", repo=other)  # no error

    def test_duplicate_component_in_one_repo_rejected(self):
        self._comp("main.main")
        with pytest.raises(psycopg2.errors.UniqueViolation):
            self._comp("main.main")

    def test_same_requirement_id_allowed_in_two_documents(self):
        cur = self.conn.cursor()
        other = _insert(cur, f"INSERT INTO {self.cfg.table_srs_documents} "
                             f"(document_id, file_name, source_type) "
                             f"VALUES ('d2', 'd2.pdf', 'pdf') RETURNING document_pk")
        self._req("FR-01.1")
        self._req("FR-01.1", doc=other)

    def test_invalid_requirement_type_rejected(self):
        with pytest.raises(psycopg2.errors.CheckViolation):
            self._req("X", rtype="mystery")

    def test_invalid_component_type_rejected(self):
        with pytest.raises(psycopg2.errors.CheckViolation):
            self._comp("m.x", ctype="module")

    def test_unknown_relationship_type_rejected(self):
        a, b = self._comp("m.a"), self._comp("m.b")
        with pytest.raises(psycopg2.errors.CheckViolation):
            self.conn.cursor().execute(
                f"INSERT INTO {self.cfg.table_relationships} "
                f"(repo_pk, source_comp_pk, target_comp_pk, rel_type) VALUES (%s,%s,%s,'uses')",
                (self.repo, a, b))

    def test_relationship_to_missing_component_rejected(self):
        a = self._comp("m.a")
        with pytest.raises(psycopg2.errors.ForeignKeyViolation):
            self.conn.cursor().execute(
                f"INSERT INTO {self.cfg.table_relationships} "
                f"(repo_pk, source_comp_pk, target_comp_pk, rel_type) VALUES (%s,%s,99999,'calls')",
                (self.repo, a))

    def test_wrong_vector_dimension_rejected_by_db(self):
        comp = self._comp("m.a")
        with pytest.raises(psycopg2.errors.DataException):
            self.conn.cursor().execute(
                f"UPDATE {self.cfg.table_components} SET embedding = %s::vector WHERE comp_pk = %s",
                ([0.1] * (FAKE_DIM + 1), comp))

    def test_trace_link_requires_its_entities(self):
        req = self._req("FR-1")
        # requirement_to_component without a component
        with pytest.raises(psycopg2.errors.CheckViolation):
            self.conn.cursor().execute(
                f"INSERT INTO {self.cfg.table_trace_links} (link_type, req_pk, pipeline_version) "
                f"VALUES ('requirement_to_component', %s, 'v1')", (req,))

    def test_trace_link_defaults_to_pending_and_is_unique(self):
        req, comp = self._req("FR-1"), self._comp("m.a")
        cur = self.conn.cursor()
        sql = (f"INSERT INTO {self.cfg.table_trace_links} "
               f"(link_type, req_pk, comp_pk, pipeline_version) "
               f"VALUES ('requirement_to_component', %s, %s, 'v1') RETURNING status")
        assert _insert(cur, sql, (req, comp)) == "pending_verification"
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute(sql, (req, comp))  # NULL docstring_pk still counts as equal

    def test_verification_needs_link_or_requirement(self):
        with pytest.raises(psycopg2.errors.CheckViolation):
            self.conn.cursor().execute(
                f"INSERT INTO {self.cfg.table_verifications} (status, pipeline_version) "
                f"VALUES ('not_found', 'v1')")

    def test_full_text_columns_populated(self):
        comp = self._comp("helper.process_data")
        cur = self.conn.cursor()
        cur.execute(f"SELECT search_tsv @@ plainto_tsquery('simple', 'process data') "
                    f"FROM {self.cfg.table_components} WHERE comp_pk = %s", (comp,))
        assert cur.fetchone()[0] is True

    def test_link_types_cover_plan(self):
        assert set(LINK_TYPES) == {
            "requirement_to_docstring", "requirement_to_component", "docstring_to_component"}
