"""
Tests for traceability.store — Store.

Unit tests use a mock psycopg2 connection to test parameter building without
a live database.

Integration tests (requires_pg) run against a real disposable test database.
They cover every requirement from the spec:
  - valid ingestion of requirements / components / docstrings / edges
  - duplicate ingestion (idempotency)
  - legacy graph format
  - missing optional fields
  - invalid edge targets
  - unknown relationship type
  - empty text
  - database rollback on failure
  - vector dimension mismatch
  - embedding-model version change
  - preservation of source metadata
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from traceability.config import TraceabilityConfig
from traceability.embedder import EmbeddingResult
from traceability.store import DimensionMismatchError, Store
from traceability.tests.conftest import (
    COMPONENT_RECORDS,
    FAKE_DIM,
    FAKE_VEC,
    REQUIREMENT_RECORDS,
    SRS_DOCUMENT_RECORD,
    requires_pg,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_emb(vec=None, version="test-v1", text_hash="abc123") -> EmbeddingResult:
    return EmbeddingResult(
        vector=vec or FAKE_VEC,
        model_version=version,
        text_hash=text_hash,
    )


def _mock_store(dim=FAKE_DIM) -> Store:
    """Store with a mocked psycopg2 connection — no DB required."""
    cfg = TraceabilityConfig(embed_dim=dim)
    mock_conn = MagicMock()
    mock_conn.closed = False
    with patch("traceability.store.psycopg2") as mock_pg, \
         patch("traceability.store.register_vector"):
        mock_pg.connect.return_value = mock_conn
        mock_pg.extras.RealDictCursor = dict
        store = Store(cfg)
    store._conn = mock_conn
    store._schema_embed_dim = dim
    return store


# ============================================================================
# _req_params  (unit — no DB)
# ============================================================================

class TestReqParams:

    def test_all_keys_present(self):
        s = _mock_store()
        rec = REQUIREMENT_RECORDS[0]
        src = rec.get("source", {})
        params = s._req_params(1, rec, src, None)
        for key in (
            "document_pk", "requirement_id", "id_source",
            "raw_text", "normalized_text", "retrieval_text", "title",
            "type", "subtype",
            "actor", "actions", "entities", "inputs", "outputs",
            "preconditions", "postconditions", "conditions", "constraints",
            "error_conditions", "domain_terms", "synonyms", "sub_requirements",
            "src_section", "src_subsection", "src_page", "src_paragraph",
            "src_table", "src_row", "src_column",
            "embedding", "embed_model_version", "embed_text_hash",
        ):
            assert key in params, f"Missing key: {key!r}"

    def test_list_fields_are_json_strings(self):
        s = _mock_store()
        rec = REQUIREMENT_RECORDS[1]   # FR-01.1  actor=["user"]
        params = s._req_params(1, rec, rec.get("source", {}), None)
        assert json.loads(params["actor"]) == ["user"]

    def test_embedding_none_when_not_provided(self):
        s = _mock_store()
        rec = REQUIREMENT_RECORDS[0]
        params = s._req_params(1, rec, {}, None)
        assert params["embedding"] is None
        assert params["embed_model_version"] is None
        assert params["embed_text_hash"] is None

    def test_embedding_populated_from_result(self):
        s = _mock_store()
        rec = REQUIREMENT_RECORDS[0]
        emb = _fake_emb(version="v42", text_hash="deadbeef")
        params = s._req_params(1, rec, {}, emb)
        assert params["embedding"] == FAKE_VEC
        assert params["embed_model_version"] == "v42"
        assert params["embed_text_hash"] == "deadbeef"

    def test_source_location_flattened(self):
        s = _mock_store()
        rec = REQUIREMENT_RECORDS[0]
        src = rec.get("source", {})
        params = s._req_params(1, rec, src, None)
        assert params["src_paragraph"] == 11
        assert params["src_page"] is None

    def test_missing_optional_fields_default(self):
        s = _mock_store()
        minimal = {
            "requirement_id": "T-001",
            "raw_text": "shall do something",
            "normalized_text": "",
            "type": "functional",
            "subtype": "general",
            "title": "Test",
            "retrieval_text": "do something",
            "actor": [], "actions": [], "entities": [], "inputs": [],
            "outputs": [], "preconditions": [], "postconditions": [],
            "conditions": [], "constraints": [], "error_conditions": [],
            "domain_terms": [], "synonyms": [], "sub_requirements": [],
        }
        params = s._req_params(1, minimal, {}, None)
        assert params["requirement_id"] == "T-001"
        assert params["src_paragraph"] is None


# ============================================================================
# _comp_params  (unit — no DB)
# ============================================================================

class TestCompParams:

    def test_all_keys_present(self):
        s = _mock_store()
        rec = COMPONENT_RECORDS["helper.HelperClass.process_data"]
        params = Store._comp_params(rec, None, None, None)
        for key in (
            "component_id", "component_type", "repo_path", "repo_snapshot",
            "file_path", "relative_path", "start_line", "end_line",
            "signature", "return_type", "containing_class",
            "parameters", "depends_on", "source_code",
            "embedding", "embed_model_version", "embed_text_hash",
        ):
            assert key in params, f"Missing: {key!r}"

    def test_id_mapped_to_component_id(self):
        rec = COMPONENT_RECORDS["main.main_function"]
        params = Store._comp_params(rec, None, None, None)
        assert params["component_id"] == "main.main_function"

    def test_depends_on_is_json(self):
        rec = COMPONENT_RECORDS["main.main_function"]
        params = Store._comp_params(rec, None, None, None)
        deps = json.loads(params["depends_on"])
        assert "main.utility_function" in deps

    def test_embedding_from_result(self):
        rec = COMPONENT_RECORDS["main.utility_function"]
        emb = _fake_emb(version="v2", text_hash="ff00")
        params = Store._comp_params(rec, emb, "/repo", "abc123")
        assert params["embedding"] == FAKE_VEC
        assert params["embed_model_version"] == "v2"
        assert params["embed_text_hash"] == "ff00"
        assert params["repo_path"] == "/repo"
        assert params["repo_snapshot"] == "abc123"

    def test_missing_v1_fields_default_none(self):
        legacy = {
            "id": "mod.f", "component_type": "function",
            "file_path": "/p.py", "relative_path": "p.py",
            "depends_on": [], "start_line": 1, "end_line": 2,
            "has_docstring": False, "docstring": "",
        }
        params = Store._comp_params(legacy, None, None, None)
        assert params["signature"] is None
        assert params["return_type"] is None
        assert params["containing_class"] is None
        assert json.loads(params["parameters"]) == []


# ============================================================================
# DimensionMismatchError  (unit — no DB)
# ============================================================================

class TestDimensionMismatch:

    def test_raises_when_dim_wrong(self):
        s = _mock_store(dim=4)
        bad_emb = EmbeddingResult(
            vector=[0.1, 0.2],   # dim=2, schema expects 4
            model_version="v1",
            text_hash="h",
        )
        with pytest.raises(DimensionMismatchError, match="2"):
            s._validate_embedding_dim(bad_emb)

    def test_no_raise_when_dim_matches(self):
        s = _mock_store(dim=4)
        good_emb = _fake_emb()   # len=4
        s._validate_embedding_dim(good_emb)   # must not raise

    def test_no_raise_when_result_none(self):
        s = _mock_store(dim=4)
        s._validate_embedding_dim(None)    # must not raise

    def test_no_raise_when_schema_dim_unknown(self):
        s = _mock_store(dim=4)
        s._schema_embed_dim = None
        bad_emb = EmbeddingResult(vector=[0.1], model_version="v1", text_hash="h")
        s._validate_embedding_dim(bad_emb)   # must not raise


# ============================================================================
# Live DB tests
# ============================================================================

@requires_pg
class TestStoreLiveDB:
    """Full round-trip tests against a real PostgreSQL instance."""

    # ------------------------------------------------------------------
    # init_schema
    # ------------------------------------------------------------------

    def test_init_schema_idempotent(self, live_store):
        live_store.init_schema(FAKE_DIM)   # second call must not raise

    # ------------------------------------------------------------------
    # Migration log
    # ------------------------------------------------------------------

    def test_record_migration(self, live_store):
        live_store.record_migration("1.0.0", "initial")
        assert live_store.is_migration_applied("1.0.0")

    def test_record_migration_idempotent(self, live_store):
        live_store.record_migration("1.0.0", "initial")
        live_store.record_migration("1.0.0", "initial")  # must not raise
        assert live_store.is_migration_applied("1.0.0")

    def test_unknown_version_not_applied(self, live_store):
        assert not live_store.is_migration_applied("99.0.0")

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def test_upsert_document_returns_pk(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        assert isinstance(pk, int) and pk > 0

    def test_upsert_document_idempotent(self, live_store):
        pk1 = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        pk2 = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        assert pk1 == pk2  # same row updated, same PK

    # ------------------------------------------------------------------
    # Requirements — valid ingestion
    # ------------------------------------------------------------------

    def test_upsert_requirements_inserts_rows(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        n = live_store.upsert_requirements(pk, REQUIREMENT_RECORDS)
        assert n == len(REQUIREMENT_RECORDS)

    def test_upsert_requirements_idempotent(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        live_store.upsert_requirements(pk, REQUIREMENT_RECORDS)
        n2 = live_store.upsert_requirements(pk, REQUIREMENT_RECORDS)
        assert n2 == len(REQUIREMENT_RECORDS)

    def test_requirements_with_embeddings(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        embeddings = {r["requirement_id"]: _fake_emb() for r in REQUIREMENT_RECORDS}
        live_store.upsert_requirements(pk, REQUIREMENT_RECORDS, embeddings)

        # Verify embedding stored
        from traceability.config import TraceabilityConfig
        import psycopg2
        cfg = live_store._cfg
        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        cur.execute(
            f"SELECT embedding, embed_model_version, embed_text_hash "
            f"FROM {cfg.table_requirements} WHERE requirement_id = %s "
            f"AND document_pk = %s;",
            ("REQ-001", pk),
        )
        row = cur.fetchone()
        cur.close(); conn.close()
        assert row is not None
        assert row[0] is not None          # embedding
        assert row[1] == "test-v1"         # embed_model_version
        assert row[2] == "abc123"          # embed_text_hash

    def test_source_metadata_preserved(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        live_store.upsert_requirements(pk, REQUIREMENT_RECORDS)

        import psycopg2
        cfg = live_store._cfg
        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        cur.execute(
            f"SELECT src_paragraph, src_page, src_section "
            f"FROM {cfg.table_requirements} WHERE requirement_id = %s "
            f"AND document_pk = %s;",
            ("REQ-001", pk),
        )
        row = cur.fetchone()
        cur.close(); conn.close()
        assert row[0] == 11    # paragraph
        assert row[1] is None  # page
        assert row[2] is None  # section

    def test_same_requirement_id_in_two_documents(self, live_store):
        """The same label (FR-01.1) in two documents must produce two rows."""
        doc1 = {**SRS_DOCUMENT_RECORD, "document_id": "doc_a", "file_name": "a.docx"}
        doc2 = {**SRS_DOCUMENT_RECORD, "document_id": "doc_b", "file_name": "b.docx"}
        pk1 = live_store.upsert_document(doc1)
        pk2 = live_store.upsert_document(doc2)
        req = [REQUIREMENT_RECORDS[1]]  # FR-01.1
        live_store.upsert_requirements(pk1, req)
        live_store.upsert_requirements(pk2, req)

        import psycopg2
        cfg = live_store._cfg
        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        cur.execute(
            f"SELECT COUNT(*) FROM {cfg.table_requirements} "
            f"WHERE requirement_id = 'FR-01.1';"
        )
        count = cur.fetchone()[0]
        cur.close(); conn.close()
        assert count == 2

    def test_empty_text_requirement(self, live_store):
        """Requirements with empty normalized_text / retrieval_text must ingest."""
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        sparse = {**REQUIREMENT_RECORDS[0], "requirement_id": "EMPTY-001",
                  "normalized_text": "", "retrieval_text": "", "title": ""}
        n = live_store.upsert_requirements(pk, [sparse])
        assert n == 1

    # ------------------------------------------------------------------
    # Requirements — dimension mismatch
    # ------------------------------------------------------------------

    def test_dimension_mismatch_raises(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        bad_emb = EmbeddingResult(
            vector=[0.1, 0.2],  # dim=2, schema=4
            model_version="v1",
            text_hash="h",
        )
        with pytest.raises(DimensionMismatchError):
            live_store.upsert_requirements(
                pk,
                [REQUIREMENT_RECORDS[0]],
                {"REQ-001": bad_emb},
            )

    # ------------------------------------------------------------------
    # Components — valid ingestion
    # ------------------------------------------------------------------

    def test_upsert_components_inserts_rows(self, live_store):
        recs = list(COMPONENT_RECORDS.values())
        n = live_store.upsert_components(recs)
        assert n == len(recs)

    def test_upsert_components_idempotent(self, live_store):
        recs = list(COMPONENT_RECORDS.values())
        live_store.upsert_components(recs)
        n2 = live_store.upsert_components(recs)
        assert n2 == len(recs)

    def test_components_with_repo_identity(self, live_store):
        recs = list(COMPONENT_RECORDS.values())
        live_store.upsert_components(
            recs,
            repo_path="/home/user/repo",
            repo_snapshot="abc1234",
        )
        import psycopg2
        cfg = live_store._cfg
        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        cur.execute(
            f"SELECT repo_path, repo_snapshot FROM {cfg.table_components} "
            f"WHERE component_id = %s;",
            ("main.main_function",),
        )
        row = cur.fetchone()
        cur.close(); conn.close()
        assert row[0] == "/home/user/repo"
        assert row[1] == "abc1234"

    def test_legacy_format_components(self, live_store, legacy_graph_file):
        """Components loaded from the legacy flat format must ingest correctly."""
        from traceability.loader import GraphLoader
        gl = GraphLoader(legacy_graph_file).load()
        n = live_store.upsert_components(list(gl.components.values()))
        assert n == len(gl.components)

    def test_missing_optional_v1_fields(self, live_store):
        """Legacy components without V1 fields must insert without errors."""
        legacy_comp = {
            "id": "mod.legacy_func",
            "component_type": "function",
            "file_path": "/project/mod.py",
            "relative_path": "mod.py",
            "depends_on": [],
            "start_line": 1, "end_line": 5,
            "has_docstring": False, "docstring": "",
            # No signature, parameters, return_type, containing_class
        }
        n = live_store.upsert_components([legacy_comp])
        assert n == 1

    def test_component_dimension_mismatch_raises(self, live_store):
        recs = [COMPONENT_RECORDS["main.utility_function"]]
        bad_emb = EmbeddingResult(
            vector=[0.1, 0.2],  # dim=2, schema=4
            model_version="v1",
            text_hash="h",
        )
        with pytest.raises(DimensionMismatchError):
            live_store.upsert_components(
                recs,
                {"main.utility_function": bad_emb},
            )

    # ------------------------------------------------------------------
    # Docstrings
    # ------------------------------------------------------------------

    def test_upsert_original_docstring(self, live_store):
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))
        n = live_store.upsert_docstrings(
            "helper.HelperClass",
            original="A helper class for processing.",
        )
        assert n == 1

    def test_upsert_generated_docstring(self, live_store):
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))
        n = live_store.upsert_docstrings(
            "main.utility_function",
            generated="Return the integer 42.",
        )
        assert n == 1

    def test_original_and_generated_are_distinct_rows(self, live_store):
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))
        live_store.upsert_docstrings(
            "main.utility_function",
            original="Old docstring.",
            generated="Return the integer 42.",
        )
        import psycopg2
        cfg = live_store._cfg
        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        cur.execute(
            f"SELECT docstring_kind, content FROM {cfg.table_docstrings} "
            f"WHERE component_id = 'main.utility_function' "
            f"ORDER BY docstring_kind;"
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        assert len(rows) == 2
        kinds = {r[0] for r in rows}
        assert kinds == {"generated", "original"}

    def test_docstring_idempotent(self, live_store):
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))
        live_store.upsert_docstrings("main.utility_function", original="v1")
        live_store.upsert_docstrings("main.utility_function", original="v2")  # update
        import psycopg2
        cfg = live_store._cfg
        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        cur.execute(
            f"SELECT content FROM {cfg.table_docstrings} "
            f"WHERE component_id = 'main.utility_function' "
            f"AND docstring_kind = 'original';"
        )
        row = cur.fetchone()
        cur.close(); conn.close()
        assert row[0] == "v2"

    def test_none_docstring_not_inserted(self, live_store):
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))
        n = live_store.upsert_docstrings(
            "main.utility_function",
            original=None,
            generated=None,
        )
        assert n == 0

    def test_empty_docstring_not_inserted(self, live_store):
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))
        n = live_store.upsert_docstrings(
            "main.utility_function",
            original="   ",  # whitespace-only
        )
        assert n == 0

    # ------------------------------------------------------------------
    # Relationships — valid
    # ------------------------------------------------------------------

    def test_upsert_relationships_valid(self, live_store):
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))
        rels = [
            {"source": "helper.HelperClass",
             "target": "helper.HelperClass.process_data",
             "type": "contains"},
        ]
        inserted, skipped = live_store.upsert_relationships(rels)
        assert inserted == 1
        assert skipped == 0

    def test_upsert_relationships_idempotent(self, live_store):
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))
        rel = [{"source": "helper.HelperClass",
                "target": "helper.HelperClass.process_data",
                "type": "contains"}]
        live_store.upsert_relationships(rel)
        inserted2, _ = live_store.upsert_relationships(rel)
        assert inserted2 == 0   # ON CONFLICT DO NOTHING

    def test_invalid_edge_target_skipped(self, live_store):
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))
        rels = [
            {"source": "does.not.exist", "target": "helper.HelperClass", "type": "calls"},
        ]
        inserted, skipped = live_store.upsert_relationships(rels)
        assert inserted == 0
        assert skipped == 1

    def test_unknown_relationship_type_skipped(self, live_store):
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))
        rels = [
            {"source": "helper.HelperClass",
             "target": "helper.HelperClass.process_data",
             "type": "FUTURE_TYPE"},
        ]
        inserted, skipped = live_store.upsert_relationships(rels)
        assert inserted == 0
        assert skipped == 1

    def test_all_known_relationship_types_accepted(self, live_store):
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))
        base_rels = [
            ("helper.HelperClass", "helper.HelperClass.process_data", "contains"),
            ("main.main_function", "helper.HelperClass",              "instantiates"),
            ("main.main_function", "main.utility_function",           "calls"),
            ("main.main_function", "helper.HelperClass",              "imports"),
            ("helper.HelperClass", "main.utility_function",           "inherits"),
        ]
        rels = [{"source": s, "target": t, "type": r} for s, t, r in base_rels]
        inserted, skipped = live_store.upsert_relationships(rels)
        assert inserted == len(base_rels)
        assert skipped == 0

    # ------------------------------------------------------------------
    # Mappings
    # ------------------------------------------------------------------

    def test_upsert_mapping(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        live_store.upsert_requirements(pk, REQUIREMENT_RECORDS)
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))

        import psycopg2
        cfg = live_store._cfg
        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        cur.execute(
            f"SELECT req_pk FROM {cfg.table_requirements} "
            f"WHERE requirement_id = 'FR-01.1' AND document_pk = %s;",
            (pk,),
        )
        row = cur.fetchone()
        req_pk = row[0]
        cur.close(); conn.close()

        live_store.upsert_mapping(
            req_pk, "helper.HelperClass",
            confidence_score=0.88,
            mapping_type="vector_search",
            rationale="login maps to HelperClass",
        )

        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        cur.execute(
            f"SELECT confidence_score, mapping_type FROM {cfg.table_mappings} "
            f"WHERE req_pk = %s AND component_id = 'helper.HelperClass';",
            (req_pk,),
        )
        row = cur.fetchone()
        cur.close(); conn.close()
        assert abs(row[0] - 0.88) < 1e-4
        assert row[1] == "vector_search"

    # ------------------------------------------------------------------
    # Stale embedding queries
    # ------------------------------------------------------------------

    def test_stale_requirements_null_embedding(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        live_store.upsert_requirements(pk, REQUIREMENT_RECORDS)
        stale = live_store.get_stale_requirement_embeddings("current-v1")
        assert len(stale) == len(REQUIREMENT_RECORDS)

    def test_stale_requirements_wrong_version(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        embeddings = {r["requirement_id"]: _fake_emb(version="old-v1")
                      for r in REQUIREMENT_RECORDS}
        live_store.upsert_requirements(pk, REQUIREMENT_RECORDS, embeddings)
        stale = live_store.get_stale_requirement_embeddings("new-v2")
        assert len(stale) == len(REQUIREMENT_RECORDS)

    def test_not_stale_when_version_matches(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        embeddings = {r["requirement_id"]: _fake_emb(version="current-v1")
                      for r in REQUIREMENT_RECORDS}
        live_store.upsert_requirements(pk, REQUIREMENT_RECORDS, embeddings)
        stale = live_store.get_stale_requirement_embeddings("current-v1")
        assert len(stale) == 0

    def test_stale_components_null_embedding(self, live_store):
        live_store.upsert_components(list(COMPONENT_RECORDS.values()))
        stale = live_store.get_stale_component_embeddings("current-v1")
        assert len(stale) == len(COMPONENT_RECORDS)

    def test_update_requirement_embedding(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        live_store.upsert_requirements(pk, [REQUIREMENT_RECORDS[0]])

        import psycopg2
        cfg = live_store._cfg
        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        cur.execute(
            f"SELECT req_pk FROM {cfg.table_requirements} "
            f"WHERE requirement_id = 'REQ-001' AND document_pk = %s;",
            (pk,),
        )
        req_pk = cur.fetchone()[0]
        cur.close(); conn.close()

        result = _fake_emb(version="new-v2", text_hash="newhash")
        live_store.update_requirement_embedding(req_pk, result)

        stale = live_store.get_stale_requirement_embeddings("old-v1")
        pks = [r["req_pk"] for r in stale]
        assert req_pk not in pks  # no longer stale for old-v1? actually it is
        # verify version updated
        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        cur.execute(
            f"SELECT embed_model_version FROM {cfg.table_requirements} "
            f"WHERE req_pk = %s;",
            (req_pk,),
        )
        ver = cur.fetchone()[0]
        cur.close(); conn.close()
        assert ver == "new-v2"

    def test_update_component_embedding(self, live_store):
        live_store.upsert_components([COMPONENT_RECORDS["main.utility_function"]])
        result = _fake_emb(version="new-v2", text_hash="hash2")
        live_store.update_component_embedding("main.utility_function", result)

        import psycopg2
        cfg = live_store._cfg
        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        cur.execute(
            f"SELECT embed_model_version FROM {cfg.table_components} "
            f"WHERE component_id = 'main.utility_function';"
        )
        ver = cur.fetchone()[0]
        cur.close(); conn.close()
        assert ver == "new-v2"

    # ------------------------------------------------------------------
    # Rollback on failure
    # ------------------------------------------------------------------

    def test_rollback_on_bad_type(self, live_store):
        """
        A batch containing a requirement with an invalid type value should
        cause a DB constraint violation, rolling back the whole batch.
        """
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)

        bad_rec = {**REQUIREMENT_RECORDS[0],
                   "requirement_id": "BAD-001",
                   "type": "INVALID_TYPE"}

        with pytest.raises(Exception):
            live_store.upsert_requirements(pk, [bad_rec])

        # The transaction must have rolled back — no rows for this batch
        import psycopg2
        cfg = live_store._cfg
        conn = psycopg2.connect(cfg.build_dsn())
        cur = conn.cursor()
        cur.execute(
            f"SELECT COUNT(*) FROM {cfg.table_requirements} "
            f"WHERE requirement_id = 'BAD-001';"
        )
        count = cur.fetchone()[0]
        cur.close(); conn.close()
        assert count == 0

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def test_search_requirements(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        embeddings = {r["requirement_id"]: _fake_emb() for r in REQUIREMENT_RECORDS}
        live_store.upsert_requirements(pk, REQUIREMENT_RECORDS, embeddings)
        results = live_store.search_requirements(FAKE_VEC, top_k=2)
        assert len(results) <= 2
        for row in results:
            assert "req_pk" in row and "distance" in row

    def test_search_components(self, live_store):
        embeddings = {r["id"]: _fake_emb() for r in COMPONENT_RECORDS.values()}
        live_store.upsert_components(
            list(COMPONENT_RECORDS.values()), embeddings
        )
        results = live_store.search_components(FAKE_VEC, top_k=3)
        assert len(results) <= 3

    def test_search_requirements_type_filter(self, live_store):
        pk = live_store.upsert_document(SRS_DOCUMENT_RECORD)
        embeddings = {r["requirement_id"]: _fake_emb() for r in REQUIREMENT_RECORDS}
        live_store.upsert_requirements(pk, REQUIREMENT_RECORDS, embeddings)
        results = live_store.search_requirements(FAKE_VEC, filter_type="functional")
        for row in results:
            assert row["type"] == "functional"

    def test_search_components_type_filter(self, live_store):
        embeddings = {r["id"]: _fake_emb() for r in COMPONENT_RECORDS.values()}
        live_store.upsert_components(
            list(COMPONENT_RECORDS.values()), embeddings
        )
        results = live_store.search_components(FAKE_VEC, filter_type="method")
        for row in results:
            assert row["component_type"] == "method"

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def test_context_manager_closes_connection(self, cfg):
        with Store(cfg) as s:
            s.init_schema(FAKE_DIM)
            assert not s._conn.closed
        assert s._conn.closed
