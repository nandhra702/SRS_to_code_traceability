"""
Tests for traceability.pipeline — Pipeline.

Unit tests mock Embedder and Store to verify orchestration logic without
touching any external service.

Integration tests skip unless sentence-transformers + PostgreSQL are available.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest

from traceability.config import TraceabilityConfig
from traceability.embedder import EmbeddingResult
from traceability.pipeline import Pipeline
from traceability.tests.conftest import FAKE_DIM, FAKE_VEC, requires_pg

FAKE_EMB = EmbeddingResult(vector=FAKE_VEC, model_version="test-v1", text_hash="abc")


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------

def _mock_embedder():
    inst = MagicMock()
    type(inst).dim = PropertyMock(return_value=FAKE_DIM)
    type(inst).model_name = PropertyMock(return_value="fake-model")
    type(inst).model_version = PropertyMock(return_value="test-v1")
    inst.embed_batch.return_value = {}
    inst.embed.return_value = FAKE_EMB
    mock_cls = MagicMock(return_value=inst)
    return mock_cls, inst


def _mock_store():
    inst = MagicMock()
    inst.init_schema = MagicMock()
    inst.record_migration = MagicMock()
    inst.upsert_document = MagicMock(return_value=1)
    inst.upsert_requirements = MagicMock(return_value=0)
    inst.upsert_components = MagicMock(return_value=0)
    inst.upsert_docstrings = MagicMock(return_value=0)
    inst.upsert_relationships = MagicMock(return_value=(0, 0))
    inst.build_vector_indexes = MagicMock()
    mock_cls = MagicMock()
    mock_cls.return_value.__enter__ = MagicMock(return_value=inst)
    mock_cls.return_value.__exit__ = MagicMock(return_value=False)
    return mock_cls, inst


def _cfg():
    return TraceabilityConfig(embed_dim=FAKE_DIM)


# ============================================================================
# load_only — no DB, no model
# ============================================================================

class TestLoadOnly:

    def test_returns_both_loaders(self, legacy_graph_file, srs_output_file):
        from traceability.loader import GraphLoader, SRSLoader
        g, s = Pipeline(_cfg()).load_only(legacy_graph_file, srs_output_file)
        assert isinstance(g, GraphLoader)
        assert isinstance(s, SRSLoader)

    def test_loaders_are_loaded(self, legacy_graph_file, srs_output_file):
        g, s = Pipeline(_cfg()).load_only(legacy_graph_file, srs_output_file)
        assert len(g.components) > 0
        assert len(s.requirements) > 0

    def test_v1_format_works(self, v1_graph_file, srs_output_file):
        g, _ = Pipeline(_cfg()).load_only(v1_graph_file, srs_output_file)
        assert len(g.components) > 0


# ============================================================================
# run() — mocked
# ============================================================================

class TestPipelineRunMocked:

    def test_embedder_instantiated_when_embed_true(
        self, legacy_graph_file, srs_output_file
    ):
        emb_cls, _ = _mock_embedder()
        store_cls, _ = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(legacy_graph_file, srs_output_file)
        emb_cls.assert_called_once()

    def test_dim_accessed(self, legacy_graph_file, srs_output_file):
        emb_cls, emb_inst = _mock_embedder()
        store_cls, _ = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(legacy_graph_file, srs_output_file)
        type(emb_inst).dim.fget.assert_called()

    def test_embed_batch_called_twice(self, legacy_graph_file, srs_output_file):
        emb_cls, emb_inst = _mock_embedder()
        store_cls, _ = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(legacy_graph_file, srs_output_file)
        assert emb_inst.embed_batch.call_count == 2

    def test_store_init_schema_called_with_dim(
        self, legacy_graph_file, srs_output_file
    ):
        emb_cls, _ = _mock_embedder()
        store_cls, store_inst = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(legacy_graph_file, srs_output_file)
        store_inst.init_schema.assert_called_once_with(FAKE_DIM)

    def test_migration_recorded(self, legacy_graph_file, srs_output_file):
        from traceability.schema import SCHEMA_VERSION
        emb_cls, _ = _mock_embedder()
        store_cls, store_inst = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(legacy_graph_file, srs_output_file)
        store_inst.record_migration.assert_called_once_with(
            SCHEMA_VERSION, "Phase 1 initial schema"
        )

    def test_upsert_document_called(self, legacy_graph_file, srs_output_file):
        emb_cls, _ = _mock_embedder()
        store_cls, store_inst = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(legacy_graph_file, srs_output_file)
        store_inst.upsert_document.assert_called_once()

    def test_upsert_requirements_called(self, legacy_graph_file, srs_output_file):
        emb_cls, _ = _mock_embedder()
        store_cls, store_inst = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(legacy_graph_file, srs_output_file)
        store_inst.upsert_requirements.assert_called_once()

    def test_upsert_components_called(self, legacy_graph_file, srs_output_file):
        emb_cls, _ = _mock_embedder()
        store_cls, store_inst = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(legacy_graph_file, srs_output_file)
        store_inst.upsert_components.assert_called_once()

    def test_upsert_relationships_called(self, legacy_graph_file, srs_output_file):
        emb_cls, _ = _mock_embedder()
        store_cls, store_inst = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(legacy_graph_file, srs_output_file)
        store_inst.upsert_relationships.assert_called_once()

    def test_repo_identity_forwarded(self, legacy_graph_file, srs_output_file):
        emb_cls, _ = _mock_embedder()
        store_cls, store_inst = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(
                legacy_graph_file, srs_output_file,
                repo_path="/my/repo",
                repo_snapshot="abc1234",
            )
        _, kw = store_inst.upsert_components.call_args
        assert kw.get("repo_path") == "/my/repo"
        assert kw.get("repo_snapshot") == "abc1234"

    def test_no_ivfflat_by_default(self, legacy_graph_file, srs_output_file):
        emb_cls, _ = _mock_embedder()
        store_cls, store_inst = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(legacy_graph_file, srs_output_file)
        store_inst.build_vector_indexes.assert_not_called()

    def test_ivfflat_when_requested(self, legacy_graph_file, srs_output_file):
        emb_cls, _ = _mock_embedder()
        store_cls, store_inst = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(
                legacy_graph_file, srs_output_file,
                build_vector_indexes=True,
            )
        store_inst.build_vector_indexes.assert_called_once()

    def test_embed_false_skips_embedder(self, legacy_graph_file, srs_output_file):
        emb_cls, _ = _mock_embedder()
        store_cls, _ = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(
                legacy_graph_file, srs_output_file, embed=False
            )
        emb_cls.assert_not_called()

    def test_embed_false_uses_config_dim(self, legacy_graph_file, srs_output_file):
        emb_cls, _ = _mock_embedder()
        store_cls, store_inst = _mock_store()
        cfg = TraceabilityConfig(embed_dim=1536)
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(cfg).run(legacy_graph_file, srs_output_file, embed=False)
        store_inst.init_schema.assert_called_once_with(1536)

    def test_embed_false_fallback_384(self, legacy_graph_file, srs_output_file):
        emb_cls, _ = _mock_embedder()
        store_cls, store_inst = _mock_store()
        cfg = TraceabilityConfig(embed_dim=None)
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(cfg).run(legacy_graph_file, srs_output_file, embed=False)
        store_inst.init_schema.assert_called_once_with(384)

    def test_v1_graph_runs_cleanly(self, v1_graph_file, srs_output_file):
        emb_cls, _ = _mock_embedder()
        store_cls, _ = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(v1_graph_file, srs_output_file)

    def test_original_docstrings_upserted(self, legacy_graph_file, srs_output_file):
        """Components with has_docstring=True must trigger upsert_docstrings."""
        emb_cls, _ = _mock_embedder()
        store_cls, store_inst = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(legacy_graph_file, srs_output_file)
        # helper.HelperClass has has_docstring=True in our fixtures
        assert store_inst.upsert_docstrings.call_count >= 1

    def test_embeddings_forwarded_to_upsert_requirements(
        self, legacy_graph_file, srs_output_file
    ):
        from traceability.loader import SRSLoader
        srs = SRSLoader(srs_output_file).load()
        req_embs = {r["requirement_id"]: FAKE_EMB for r in srs.requirements}

        emb_cls, emb_inst = _mock_embedder()
        emb_inst.embed_batch.side_effect = [{}, req_embs]
        store_cls, store_inst = _mock_store()
        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).run(legacy_graph_file, srs_output_file)

        positional = store_inst.upsert_requirements.call_args[0]
        # arg[2] is embeddings (document_pk=0, records=1, embeddings=2)
        assert positional[2] == req_embs


# ============================================================================
# rebuild_embeddings — mocked
# ============================================================================

class TestRebuildEmbeddingsMocked:

    def test_calls_get_stale_and_update(self, legacy_graph_file, srs_output_file):
        emb_cls, emb_inst = _mock_embedder()
        store_cls, store_inst = _mock_store()

        store_inst.get_stale_requirement_embeddings.return_value = [
            {"req_pk": 1, "retrieval_text": "login", "title": "Login",
             "normalized_text": ""},
        ]
        store_inst.get_stale_component_embeddings.return_value = [
            {"component_id": "mod.f", "signature": "f()", "docstring": "",
             "source_code": ""},
        ]

        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).rebuild_embeddings()

        store_inst.update_requirement_embedding.assert_called_once()
        store_inst.update_component_embedding.assert_called_once()

    def test_no_stale_rows_no_updates(self):
        emb_cls, emb_inst = _mock_embedder()
        store_cls, store_inst = _mock_store()

        store_inst.get_stale_requirement_embeddings.return_value = []
        store_inst.get_stale_component_embeddings.return_value = []

        with patch("traceability.pipeline.Embedder", emb_cls), \
             patch("traceability.pipeline.Store", store_cls):
            Pipeline(_cfg()).rebuild_embeddings()

        store_inst.update_requirement_embedding.assert_not_called()
        store_inst.update_component_embedding.assert_not_called()


# ============================================================================
# Integration — sentence-transformers + PostgreSQL required
# ============================================================================

def _integration_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import psycopg2               # noqa: F401
        from pgvector.psycopg2 import register_vector  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _integration_available(),
    reason=(
        "Integration test requires: "
        "(1) pip install sentence-transformers  "
        "(2) pip install psycopg2-binary pgvector  "
        "(3) PostgreSQL running with pgvector"
    ),
)
class TestPipelineIntegration:

    def test_full_run(self, real_graph_file, real_srs_file):
        from traceability.config import TraceabilityConfig
        cfg = TraceabilityConfig(
            pg_dbname="traceability_integration_test",
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            table_migrations="integ_migrations",
            table_srs_documents="integ_srs_docs",
            table_requirements="integ_requirements",
            table_components="integ_components",
            table_docstrings="integ_docstrings",
            table_relationships="integ_relationships",
            table_mappings="integ_mappings",
        )
        Pipeline(cfg).run(real_graph_file, real_srs_file)

    def test_search_after_run(self, real_graph_file, real_srs_file):
        from traceability.config import TraceabilityConfig
        from traceability.embedder import Embedder
        from traceability.store import Store

        cfg = TraceabilityConfig(
            pg_dbname="traceability_integration_test",
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            table_migrations="integ_migrations",
            table_srs_documents="integ_srs_docs",
            table_requirements="integ_requirements",
            table_components="integ_components",
            table_docstrings="integ_docstrings",
            table_relationships="integ_relationships",
            table_mappings="integ_mappings",
        )
        Pipeline(cfg).run(real_graph_file, real_srs_file)

        query_vec = Embedder(cfg).embed("user authentication login").vector
        with Store(cfg) as store:
            reqs  = store.search_requirements(query_vec, top_k=5)
            comps = store.search_components(query_vec, top_k=5)

        assert len(reqs) > 0
        assert len(comps) > 0
        if len(reqs) > 1:
            assert reqs[0]["distance"] <= reqs[-1]["distance"]

    def test_rebuild_embeddings(self, real_graph_file, real_srs_file):
        from traceability.config import TraceabilityConfig
        cfg = TraceabilityConfig(
            pg_dbname="traceability_integration_test",
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            table_migrations="integ_migrations",
            table_srs_documents="integ_srs_docs",
            table_requirements="integ_requirements",
            table_components="integ_components",
            table_docstrings="integ_docstrings",
            table_relationships="integ_relationships",
            table_mappings="integ_mappings",
        )
        p = Pipeline(cfg)
        p.run(real_graph_file, real_srs_file)
        p.rebuild_embeddings()   # must not raise
