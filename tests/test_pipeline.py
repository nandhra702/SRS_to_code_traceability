"""
Tests for traceability.pipeline — end-to-end ingestion into the test database.

Most tests use a deterministic FakeEmbedder (no model, dim 4) so they exercise
the real SQL paths quickly.  TestRealModel uses all-MiniLM-L6-v2 from the
local cache.  All DB tests skip with a reason when PostgreSQL is unavailable.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

import pytest

from traceability.embedder import Embedder, EmbeddingResult
from traceability.loader import GraphLoader, SRSLoader
from traceability.pipeline import Pipeline
from traceability.store import Store
from traceability.tests.conftest import (
    FAKE_DIM,
    make_cfg,
    requirement,
    requires_pg,
    reset_tables,
    write_json,
)


class FakeEmbedder:
    """Deterministic stand-in for Embedder with the same reporting surface."""

    def __init__(self, version: str = "fake-v1", dim: int = FAKE_DIM) -> None:
        self.model_version = version
        self.dim = dim
        self.skipped_empty: List = []
        self.truncated: List = []
        self.batches: List[Dict] = []

    def vector(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 + 0.01 for b in digest[: self.dim]]

    def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(self.vector(text), self.model_version, Embedder.text_hash(text))

    def embed_batch(self, texts):
        self.batches.append(dict(texts))
        self.skipped_empty = [k for k, t in texts.items() if not t.strip()]
        self.truncated = []
        return {k: self.embed(t) for k, t in texts.items() if t.strip()}


def pipeline(cfg, embedder=None) -> Pipeline:
    p = Pipeline(cfg)
    p._embedder = embedder or FakeEmbedder()
    return p


def q(cfg, sql, params=None):
    with Store(cfg) as s, s.transaction() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


@pytest.fixture()
def clean_cfg(cfg):
    reset_tables(cfg)
    return cfg


# ============================================================================
# No database
# ============================================================================


def test_load_only_returns_loaded_loaders(real_v1_graph, real_srs):
    g, s = Pipeline(make_cfg()).load_only(real_v1_graph, real_srs)
    assert isinstance(g, GraphLoader) and isinstance(s, SRSLoader)
    assert len(g.components) == 20 and len(s.requirements) == 62


# ============================================================================
# Graph ingestion
# ============================================================================


@requires_pg
class TestIngestGraph:

    def test_v1_report_and_rows(self, clean_cfg, real_v1_graph):
        r = pipeline(clean_cfg).ingest_graph(real_v1_graph, repo_id="simple", repo_snapshot="abc123")
        assert r["graph_format"] == "v1"
        assert (r["components"], r["code_embedded"], r["relationships"]) == (20, 20, 22)
        assert r["docstrings"] == {"existing": 18, "generated": 0}
        assert r["docstrings_embedded"] == 18
        repo = q(clean_cfg, f"SELECT * FROM {clean_cfg.table_repositories}")[0]
        assert (repo["repo_id"], repo["repo_snapshot"], repo["graph_format"]) == ("simple", "abc123", "v1")
        assert q(clean_cfg, f"SELECT count(*) AS n FROM {clean_cfg.table_components} "
                            f"WHERE embedding IS NOT NULL")[0]["n"] == 20

    def test_legacy_graph_skips_code_embeddings_with_warning(self, clean_cfg, real_legacy_graph):
        r = pipeline(clean_cfg).ingest_graph(real_legacy_graph)
        assert r["graph_format"] == "legacy" and "warning" in r
        assert r["code_embedded"] == 0 and len(r["code_skipped_empty_text"]) == 20
        # This real legacy graph was produced from a docstring-free copy of the repo
        assert r["docstrings"] == {"existing": 0, "generated": 0}
        assert r["relationships"] == 0
        assert r["repo_id"] == "simple_legacy_graph"

    def test_generated_docstring_recovered_from_source_root(self, clean_cfg, tmp_path):
        graph = {"components": {"mod.f": {
            "id": "mod.f", "component_type": "function", "file_path": "<repo>/mod.py",
            "relative_path": "mod.py", "depends_on": [], "source_code": "def f():\n    return 1",
            "start_line": 1, "end_line": 2, "has_docstring": False, "docstring": "",
            "signature": "f()", "parameters": [], "return_type": None,
            "containing_class": None, "relationships": []}}, "relationships": []}
        path = write_json(tmp_path / "g.json", graph)
        (tmp_path / "mod.py").write_text('def f():\n    """Returns one."""\n    return 1\n')
        r = pipeline(clean_cfg).ingest_graph(path, source_root=tmp_path)
        assert r["docstrings"] == {"existing": 0, "generated": 1}
        row = q(clean_cfg, f"SELECT docstring_kind, content, provenance, "
                           f"embedding IS NOT NULL AS has FROM {clean_cfg.table_docstrings}")[0]
        assert row["docstring_kind"] == "generated" and row["content"] == "Returns one."
        assert row["provenance"]["source"] == "post_run_source" and row["has"]

    def test_reingest_is_idempotent(self, clean_cfg, real_v1_graph):
        p = pipeline(clean_cfg)
        p.ingest_graph(real_v1_graph)
        p.ingest_graph(real_v1_graph)
        counts = {t: q(clean_cfg, f"SELECT count(*) AS n FROM {t}")[0]["n"] for t in (
            clean_cfg.table_repositories, clean_cfg.table_components,
            clean_cfg.table_docstrings, clean_cfg.table_relationships)}
        assert list(counts.values()) == [1, 20, 18, 22]

    def test_two_graphs_of_same_code_are_separate_repos(self, clean_cfg, real_v1_graph, real_legacy_graph):
        p = pipeline(clean_cfg)
        p.ingest_graph(real_v1_graph)
        p.ingest_graph(real_legacy_graph)
        assert q(clean_cfg, f"SELECT count(*) AS n FROM {clean_cfg.table_components}")[0]["n"] == 40


# ============================================================================
# SRS ingestion
# ============================================================================


@requires_pg
class TestIngestSRS:

    def test_real_output(self, clean_cfg, real_srs):
        r = pipeline(clean_cfg).ingest_srs(real_srs)
        assert (r["requirements"], r["embedded"], r["skipped_empty_text"]) == (62, 62, [])
        doc = q(clean_cfg, f"SELECT * FROM {clean_cfg.table_srs_documents}")[0]
        assert doc["file_name"] == "Basic_SRS_Document.docx" and doc["total_requirements"] == 62

    def test_failed_ingest_leaves_database_untouched(self, clean_cfg, tmp_path):
        bad = write_json(tmp_path / "bad.json",
                         [requirement("FR-1"), requirement("FR-2", type="mystery")])
        with pytest.raises(Exception):
            pipeline(clean_cfg).ingest_srs(bad)
        assert q(clean_cfg, f"SELECT count(*) AS n FROM {clean_cfg.table_srs_documents}")[0]["n"] == 0

    def test_no_embed_uses_configured_dim(self, clean_cfg, real_srs):
        r = pipeline(clean_cfg).ingest_srs(real_srs, embed=False)
        assert "embedded" not in r
        assert q(clean_cfg, f"SELECT count(*) AS n FROM {clean_cfg.table_requirements} "
                            f"WHERE embedding IS NULL")[0]["n"] == 62

    def test_no_embed_without_known_dim_refuses_to_guess(self, real_srs):
        cfg = make_cfg(embed_dim=None)
        reset_tables(cfg)
        with pytest.raises(ValueError, match="dimension unknown"):
            Pipeline(cfg).ingest_srs(real_srs, embed=False)

    def test_no_embed_learns_dim_from_existing_schema(self, clean_cfg, real_srs):
        pipeline(clean_cfg).ingest_srs(real_srs)
        cfg = make_cfg(embed_dim=None)
        assert Pipeline(cfg).ingest_srs(real_srs, embed=False)["requirements"] == 62

    def test_run_ingests_both(self, clean_cfg, real_v1_graph, real_srs):
        reports = pipeline(clean_cfg).run(real_v1_graph, real_srs, build_vector_indexes=True)
        assert reports["graph"]["components"] == 20 and reports["srs"]["requirements"] == 62


# ============================================================================
# Rebuild embeddings
# ============================================================================


@requires_pg
class TestRebuildEmbeddings:

    @pytest.fixture()
    def ingested(self, clean_cfg, real_v1_graph, real_srs):
        p = pipeline(clean_cfg)
        p.ingest_graph(real_v1_graph)
        p.ingest_srs(real_srs)
        return clean_cfg

    def test_fresh_database_needs_nothing(self, ingested):
        r = pipeline(ingested).rebuild_embeddings()
        assert r == {e: {"rebuilt": 0, "cleared": 0} for e in ("requirement", "component", "docstring")}

    def test_model_version_change_rebuilds_everything(self, ingested):
        r = pipeline(ingested, FakeEmbedder(version="fake-v2")).rebuild_embeddings()
        assert [r[e]["rebuilt"] for e in ("requirement", "component", "docstring")] == [62, 20, 18]
        versions = q(ingested, f"SELECT DISTINCT embed_model_version AS v FROM {ingested.table_components}")
        assert versions == [{"v": "fake-v2"}]

    def test_changed_text_rebuilds_only_that_row(self, ingested):
        q_cfg = ingested
        with Store(q_cfg) as s, s.transaction() as cur:
            cur.execute(f"UPDATE {q_cfg.table_requirements} SET title = 'Edited title' "
                        f"WHERE requirement_id = 'FR-01.1'")
        r = pipeline(q_cfg).rebuild_embeddings()
        assert r["requirement"]["rebuilt"] == 1 and r["component"]["rebuilt"] == 0

    def test_missing_vector_rebuilt(self, ingested):
        with Store(ingested) as s, s.transaction() as cur:
            cur.execute(f"UPDATE {ingested.table_docstrings} SET embedding = NULL "
                        f"WHERE docstring_pk = (SELECT min(docstring_pk) FROM {ingested.table_docstrings})")
        assert pipeline(ingested).rebuild_embeddings()["docstring"]["rebuilt"] == 1

    def test_now_empty_text_clears_vector(self, ingested):
        with Store(ingested) as s, s.transaction() as cur:
            cur.execute(f"UPDATE {ingested.table_docstrings} SET content = '   ' "
                        f"WHERE docstring_pk = (SELECT min(docstring_pk) FROM {ingested.table_docstrings})")
        assert pipeline(ingested).rebuild_embeddings()["docstring"] == {"rebuilt": 0, "cleared": 1}


# ============================================================================
# Real model (local cache, no network)
# ============================================================================


@requires_pg
class TestRealModel:

    def test_semantic_search_on_real_data(self, real_v1_graph, real_srs):
        cfg = make_cfg(embed_dim=None)  # dimension from the real model
        reset_tables(cfg)
        p = Pipeline(cfg)
        p.ingest_graph(real_v1_graph)
        p.ingest_srs(real_srs)
        embedder = p._get_embedder()
        assert embedder.dim == 384
        with Store(cfg) as s:
            hits = s.search_requirements(embedder.embed("validate user credentials").vector, 3)
            assert hits[0]["requirement_id"].startswith("FR-01")
            code = s.search_components(embedder.embed("process data and return the result").vector, 3)
            assert all(h["component_id"].count(".") >= 1 for h in code)
        assert p.rebuild_embeddings()["component"]["rebuilt"] == 0
        reset_tables(cfg)  # leave the test tables at the default dimension
