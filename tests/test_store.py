"""
Tests for traceability.store — run against the disposable PostgreSQL + pgvector
test database (skipped with a reason when it is unavailable).

Covers the Phase 1 acceptance criteria: valid ingestion, idempotence, scoping,
relationship validation, metadata preservation, transaction rollback, vector
dimension / schema version guards, re-ingest semantics and embedding upkeep.
"""

from __future__ import annotations

import json

import psycopg2
import pytest

from traceability.embedder import EmbeddingResult
from traceability.loader import GraphLoader, SRSLoader
from traceability.schema import SCHEMA_VERSION
from traceability.store import DimensionMismatchError, SchemaVersionError, Store
from traceability.tests.conftest import (
    FAKE_DIM,
    make_cfg,
    requirement,
    requires_pg,
    reset_tables,
    v1_component,
)

pytestmark = requires_pg


def emb(vec=None, version="test-v1", text_hash="h") -> EmbeddingResult:
    return EmbeddingResult(vector=vec or [0.1, 0.2, 0.3, 0.4],
                           model_version=version, text_hash=text_hash)


def rows(store: Store, sql: str, params=None):
    with store.transaction() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def count(store: Store, table: str, where: str = "TRUE") -> int:
    return rows(store, f"SELECT count(*) AS n FROM {table} WHERE {where}")[0]["n"]


@pytest.fixture()
def doc_pk(live_store):
    return live_store.upsert_document({"document_id": "srs-1", "file_name": "srs.docx",
                                       "source_type": "docx", "total_requirements": 2})


@pytest.fixture()
def repo_pk(live_store):
    return live_store.upsert_repository({"repo_id": "repo-1", "graph_format": "v1"})


# ============================================================================
# Schema guards
# ============================================================================


class TestSchemaGuards:

    def test_init_schema_idempotent(self, live_store):
        live_store.init_schema(FAKE_DIM)
        live_store.init_schema(FAKE_DIM)
        assert count(live_store, live_store._cfg.table_migrations) == 1

    def test_recorded_dimension(self, live_store):
        assert live_store.get_schema_embed_dim() == FAKE_DIM

    def test_different_dimension_refused(self, live_store):
        with pytest.raises(DimensionMismatchError, match="dimension"):
            live_store.init_schema(FAKE_DIM + 1)

    def test_other_schema_version_refused(self, live_store):
        t = live_store._cfg.table_migrations
        with live_store.transaction() as cur:
            cur.execute(f"UPDATE {t} SET version = '1.0.0'")
        with pytest.raises(SchemaVersionError):
            live_store.init_schema(FAKE_DIM)

    def test_pre_v2_migrations_table_refused(self, cfg):
        reset_tables(cfg)
        conn = psycopg2.connect(cfg.build_dsn())
        conn.autocommit = True
        conn.cursor().execute(
            f"CREATE TABLE {cfg.table_migrations} (version TEXT PRIMARY KEY, "
            f"applied_at TIMESTAMPTZ, description TEXT)")
        conn.close()
        with Store(cfg) as store, pytest.raises(SchemaVersionError, match="predates"):
            store.init_schema(FAKE_DIM)

    def test_new_store_learns_dimension_from_db(self, live_store, cfg):
        # A new connection that never called init_schema still validates vectors
        with Store(cfg) as fresh:
            assert fresh.get_schema_embed_dim() == FAKE_DIM
            repo = fresh.upsert_repository({"repo_id": "r", "graph_format": "v1"})
            with pytest.raises(DimensionMismatchError):
                fresh.upsert_components(repo, [v1_component("m.f")], {"m.f": emb([0.1] * 9)})

    def test_schema_version_recorded(self, live_store):
        t = live_store._cfg.table_migrations
        assert rows(live_store, f"SELECT version FROM {t}")[0]["version"] == SCHEMA_VERSION


# ============================================================================
# Requirements
# ============================================================================


class TestRequirements:

    def test_insert_returns_pks(self, live_store, doc_pk):
        pks = live_store.upsert_requirements(doc_pk, [requirement("FR-1"), requirement("FR-2")])
        assert set(pks) == {"FR-1", "FR-2"}
        assert len(set(pks.values())) == 2

    def test_idempotent(self, live_store, doc_pk):
        a = live_store.upsert_requirements(doc_pk, [requirement("FR-1")])
        b = live_store.upsert_requirements(doc_pk, [requirement("FR-1")])
        assert a == b
        assert count(live_store, live_store._cfg.table_requirements) == 1

    def test_source_and_extraction_metadata_preserved(self, live_store, doc_pk):
        rec = requirement("FR-1", review_required=True, future_field="kept")
        live_store.upsert_requirements(doc_pk, [rec])
        row = rows(live_store, f"SELECT * FROM {live_store._cfg.table_requirements}")[0]
        assert row["raw_text"] == rec["raw_text"]
        assert row["review_required"] is True
        assert row["source_location"] == rec["source"]
        assert row["extraction_meta"]["candidate_reason"] == rec["candidate_reason"]
        assert row["extraction_meta"]["future_field"] == "kept"
        assert row["synonyms"] == ["authenticate"]

    def test_missing_optional_fields(self, live_store, doc_pk):
        minimal = {"requirement_id": "R", "raw_text": "The system shall work.", "type": "other"}
        live_store.upsert_requirements(doc_pk, [minimal])
        row = rows(live_store, f"SELECT * FROM {live_store._cfg.table_requirements}")[0]
        assert row["normalized_text"] == "" and row["actions"] == [] and row["embedding"] is None

    def test_invalid_type_rolls_back_whole_batch(self, live_store, doc_pk):
        with pytest.raises(psycopg2.errors.CheckViolation):
            live_store.upsert_requirements(
                doc_pk, [requirement("FR-1"), requirement("FR-2", type="mystery")])
        assert count(live_store, live_store._cfg.table_requirements) == 0

    def test_removed_requirement_flagged_not_deleted(self, live_store, doc_pk):
        live_store.upsert_requirements(doc_pk, [requirement("FR-1"), requirement("FR-2")])
        live_store.upsert_requirements(doc_pk, [requirement("FR-1")])
        t = live_store._cfg.table_requirements
        assert rows(live_store, f"SELECT requirement_id FROM {t} WHERE NOT is_current") == [
            {"requirement_id": "FR-2"}]
        live_store.upsert_requirements(doc_pk, [requirement("FR-1"), requirement("FR-2")])
        assert count(live_store, t, "NOT is_current") == 0

    def test_same_id_in_two_documents_kept_apart(self, live_store, doc_pk):
        other = live_store.upsert_document({"document_id": "srs-2", "file_name": "b.pdf",
                                            "source_type": "pdf"})
        a = live_store.upsert_requirements(doc_pk, [requirement("FR-1")])
        b = live_store.upsert_requirements(other, [requirement("FR-1")])
        assert a["FR-1"] != b["FR-1"]
        # Re-ingesting one document does not flag the other's rows
        assert count(live_store, live_store._cfg.table_requirements, "is_current") == 2

    def test_wrong_vector_length_rejected(self, live_store, doc_pk):
        with pytest.raises(DimensionMismatchError):
            live_store.upsert_requirements(doc_pk, [requirement("FR-1")], {"FR-1": emb([0.1] * 3)})

    def test_upsert_without_embedding_keeps_existing_vector(self, live_store, doc_pk):
        live_store.upsert_requirements(doc_pk, [requirement("FR-1")], {"FR-1": emb()})
        live_store.upsert_requirements(doc_pk, [requirement("FR-1", title="changed")])
        row = rows(live_store, f"SELECT title, embedding IS NOT NULL AS has, embed_text_hash "
                               f"FROM {live_store._cfg.table_requirements}")[0]
        assert row == {"title": "changed", "has": True, "embed_text_hash": "h"}

    def test_real_srs_output_ingests(self, live_store, doc_pk, real_srs):
        srs = SRSLoader(real_srs).load()
        assert len(live_store.upsert_requirements(doc_pk, srs.requirements)) == 62


# ============================================================================
# Components, docstrings, relationships
# ============================================================================


class TestCodeEntities:

    def test_real_v1_graph_ingests(self, live_store, repo_pk, real_v1_graph):
        g = GraphLoader(real_v1_graph).load()
        comp_pks = live_store.upsert_components(repo_pk, list(g.components.values()))
        n_doc = live_store.upsert_docstrings(comp_pks, g.docstring_records())
        inserted, skipped = live_store.replace_relationships(repo_pk, comp_pks, g.relationships)
        assert (len(comp_pks), n_doc, inserted) == (20, 18, 22)
        assert skipped == {"unknown_endpoint": 0, "unknown_type": 0}
        cfg = live_store._cfg
        row = rows(live_store, f"SELECT * FROM {cfg.table_components} "
                               f"WHERE component_id = 'helper.HelperClass.process_data'")[0]
        assert row["containing_class"] == "helper.HelperClass"
        assert row["source_code"] and row["source_hash"]

    def test_real_legacy_graph_ingests(self, live_store, repo_pk, real_legacy_graph):
        g = GraphLoader(real_legacy_graph).load()
        comp_pks = live_store.upsert_components(repo_pk, list(g.components.values()))
        assert len(comp_pks) == 20
        assert live_store.replace_relationships(repo_pk, comp_pks, g.relationships)[0] == 0

    def test_components_idempotent(self, live_store, repo_pk):
        a = live_store.upsert_components(repo_pk, [v1_component("m.f")])
        b = live_store.upsert_components(repo_pk, [v1_component("m.f")])
        assert a == b and count(live_store, live_store._cfg.table_components) == 1

    def test_same_component_id_in_two_repos(self, live_store, repo_pk):
        other = live_store.upsert_repository({"repo_id": "repo-2", "graph_format": "v1"})
        a = live_store.upsert_components(repo_pk, [v1_component("main.main")])
        b = live_store.upsert_components(other, [v1_component("main.main")])
        assert a["main.main"] != b["main.main"]

    def test_removed_component_flagged(self, live_store, repo_pk):
        live_store.upsert_components(repo_pk, [v1_component("m.a"), v1_component("m.b")])
        live_store.upsert_components(repo_pk, [v1_component("m.a")])
        t = live_store._cfg.table_components
        assert rows(live_store, f"SELECT component_id FROM {t} WHERE NOT is_current") == [
            {"component_id": "m.b"}]

    def test_docstring_kinds_are_distinct_rows(self, live_store, repo_pk):
        pks = live_store.upsert_components(repo_pk, [v1_component("m.f")])
        live_store.upsert_docstrings(pks, [
            {"component_id": "m.f", "kind": "existing", "content": "Old."},
            {"component_id": "m.f", "kind": "generated", "content": "New."},
        ])
        assert count(live_store, live_store._cfg.table_docstrings) == 2

    def test_docstring_for_unknown_component_rejected(self, live_store, repo_pk):
        with pytest.raises(KeyError):
            live_store.upsert_docstrings({}, [
                {"component_id": "m.ghost", "kind": "existing", "content": "x"}])

    def test_unobserved_docstring_kind_left_alone(self, live_store, repo_pk):
        pks = live_store.upsert_components(repo_pk, [v1_component("m.f")])
        live_store.upsert_docstrings(pks, [
            {"component_id": "m.f", "kind": "existing", "content": "Old."},
            {"component_id": "m.f", "kind": "generated", "content": "New."},
        ], observed_kinds=("existing", "generated"))
        # Re-ingest without a post-run source tree: generated was not looked for
        live_store.upsert_docstrings(pks, [
            {"component_id": "m.f", "kind": "existing", "content": "Old."}])
        t = live_store._cfg.table_docstrings
        assert count(live_store, t, "is_current") == 2
        # ...but when generated *was* looked for and is gone, it is flagged
        live_store.upsert_docstrings(pks, [
            {"component_id": "m.f", "kind": "existing", "content": "Old."}],
            observed_kinds=("existing", "generated"))
        assert rows(live_store, f"SELECT docstring_kind FROM {t} WHERE NOT is_current") == [
            {"docstring_kind": "generated"}]

    def test_invalid_relationships_skipped_and_counted(self, live_store, repo_pk):
        pks = live_store.upsert_components(repo_pk, [v1_component("m.a"), v1_component("m.b")])
        inserted, skipped = live_store.replace_relationships(repo_pk, pks, [
            {"source": "m.a", "target": "m.b", "type": "calls"},
            {"source": "m.a", "target": "m.ghost", "type": "calls"},
            {"source": "m.a", "target": "m.b", "type": "uses"},
        ])
        assert inserted == 1
        assert skipped == {"unknown_endpoint": 1, "unknown_type": 1}

    def test_relationship_endpoint_in_other_repo_rejected(self, live_store, repo_pk):
        other = live_store.upsert_repository({"repo_id": "repo-2", "graph_format": "v1"})
        mine = live_store.upsert_components(repo_pk, [v1_component("m.a")])
        live_store.upsert_components(other, [v1_component("m.b")])
        _, skipped = live_store.replace_relationships(
            repo_pk, mine, [{"source": "m.a", "target": "m.b", "type": "calls"}])
        assert skipped["unknown_endpoint"] == 1

    def test_relationships_replaced_on_reingest(self, live_store, repo_pk):
        pks = live_store.upsert_components(repo_pk, [v1_component("m.a"), v1_component("m.b")])
        live_store.replace_relationships(repo_pk, pks, [
            {"source": "m.a", "target": "m.b", "type": "calls"},
            {"source": "m.b", "target": "m.a", "type": "calls"}])
        live_store.replace_relationships(repo_pk, pks, [
            {"source": "m.a", "target": "m.b", "type": "calls"}])
        assert count(live_store, live_store._cfg.table_relationships) == 1


# ============================================================================
# Transactions
# ============================================================================


class TestTransactions:

    def test_outer_transaction_rolls_back_all_steps(self, live_store):
        with pytest.raises(DimensionMismatchError):
            with live_store.transaction():
                repo = live_store.upsert_repository({"repo_id": "r", "graph_format": "v1"})
                live_store.upsert_components(repo, [v1_component("m.f")],
                                             {"m.f": emb([0.1] * 9)})
        cfg = live_store._cfg
        assert count(live_store, cfg.table_repositories) == 0
        assert count(live_store, cfg.table_components) == 0

    def test_store_usable_after_rollback(self, live_store):
        with pytest.raises(RuntimeError):
            with live_store.transaction():
                live_store.upsert_repository({"repo_id": "r", "graph_format": "v1"})
                raise RuntimeError("boom")
        live_store.upsert_repository({"repo_id": "r", "graph_format": "v1"})
        assert count(live_store, live_store._cfg.table_repositories) == 1


# ============================================================================
# Embedding upkeep + search
# ============================================================================


class TestEmbeddingsAndSearch:

    @pytest.fixture()
    def populated(self, live_store, repo_pk, doc_pk):
        pks = live_store.upsert_components(
            repo_pk, [v1_component("m.near"), v1_component("m.far")],
            {"m.near": emb([1, 0, 0, 0]), "m.far": emb([0, 1, 0, 0])})
        live_store.upsert_docstrings(
            pks, [{"component_id": "m.near", "kind": "existing", "content": "Near."}],
            {("m.near", "existing"): emb([1, 0, 0, 0])})
        live_store.upsert_requirements(
            doc_pk, [requirement("FR-1"), requirement("FR-2")],
            {"FR-1": emb([1, 0, 0, 0]), "FR-2": emb([0, 0, 1, 0], version="old")})
        return pks

    def test_component_search_orders_by_distance(self, live_store, populated):
        hits = live_store.search_components([1, 0, 0, 0], 5)
        assert [h["component_id"] for h in hits] == ["m.near", "m.far"]
        assert hits[0]["distance"] == pytest.approx(0.0)

    def test_docstring_search_returns_owner(self, live_store, populated):
        hit = live_store.search_docstrings([1, 0, 0, 0], 5)[0]
        assert hit["component_id"] == "m.near" and hit["docstring_kind"] == "existing"
        assert "embedding" not in hit

    def test_search_scoped_to_repo(self, live_store, populated):
        assert live_store.search_components([1, 0, 0, 0], 5, repo_pk=99999) == []

    def test_search_filters_model_version(self, live_store, populated):
        hits = live_store.search_requirements([1, 0, 0, 0], 5, model_version="test-v1")
        assert [h["requirement_id"] for h in hits] == ["FR-1"]

    def test_search_excludes_non_current(self, live_store, populated, repo_pk):
        live_store.upsert_components(repo_pk, [v1_component("m.near")])  # m.far gone
        assert [h["component_id"] for h in live_store.search_components([0, 1, 0, 0], 5)] == ["m.near"]

    def test_query_dimension_checked(self, live_store, populated):
        with pytest.raises(DimensionMismatchError):
            live_store.search_components([1, 0, 0], 5)

    def test_get_embedding_rows_and_update(self, live_store, populated):
        reqs = {r["requirement_id"]: r for r in live_store.get_embedding_rows("requirement")}
        assert reqs["FR-2"]["embed_model_version"] == "old" and reqs["FR-2"]["has_embedding"]
        live_store.update_embedding("requirement", reqs["FR-2"]["req_pk"], emb(version="new"))
        live_store.update_embedding("requirement", reqs["FR-1"]["req_pk"], None)
        after = {r["requirement_id"]: r for r in live_store.get_embedding_rows("requirement")}
        assert after["FR-2"]["embed_model_version"] == "new"
        assert after["FR-1"]["has_embedding"] is False

    def test_docstring_rows_include_owner_fields(self, live_store, populated):
        row = live_store.get_embedding_rows("docstring")[0]
        assert row["component_id"] == "m.near" and row["content"] == "Near."

    def test_unknown_entity_rejected(self, live_store):
        with pytest.raises(ValueError):
            live_store.get_embedding_rows("widget")

    def test_hnsw_indexes_build(self, live_store, populated):
        live_store.build_vector_indexes()
        names = rows(live_store, "SELECT indexname FROM pg_indexes WHERE indexdef LIKE '%hnsw%' "
                                 "AND tablename LIKE 'test_tr_%'")
        assert len(names) == 3
