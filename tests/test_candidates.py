"""
Tests for traceability.candidates — the candidate search stage.

Runs on the real library dataset (graph + gold v2 fixtures) with the real
embedding model.  The "extracted SRS" is built from the gold text, i.e. a
perfect SRS Agent, so these numbers isolate the search stage.
"""

from __future__ import annotations

import pytest

from traceability.candidates import CandidateSearch
from traceability.evaluation import Evaluator, load_gold
from traceability.tests.conftest import FIXTURES, make_cfg, requires_pg, reset_tables, write_json

LIB = FIXTURES / "library"
pytestmark = requires_pg


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    from traceability.pipeline import Pipeline
    from traceability.query import QueryResolver
    from traceability.retrieval import Retriever
    from traceability.store import Store

    cfg = make_cfg(embed_dim=None)
    reset_tables(cfg)
    gold = load_gold(LIB / "gold_v2.json")
    srs = [{"requirement_id": r["requirement_id"], "id_source": "source", "raw_text": r["text"],
            "normalized_text": "", "retrieval_text": "", "title": r["title"], "type": r["type"]}
           for r in gold["requirements"]]
    p = Pipeline(cfg)
    p.ingest_graph(LIB / "library_v1_graph.json", repo_id="library")
    p.ingest_srs(write_json(tmp_path_factory.mktemp("srs") / "srs.json", srs), document_id="lib-srs")
    store = Store(cfg)
    resolver, retriever = QueryResolver(store), Retriever(store, p._get_embedder())
    yield CandidateSearch(store, resolver, retriever), Evaluator(resolver, retriever), store, gold
    store.close()
    reset_tables(cfg)


def links(store, **kw):
    return store.get_trace_links(pipeline_version=kw.pop("pv", "candidates-v1"), **kw)


def test_one_by_one_in_document_order(stage):
    search, _, store, gold = stage
    report = search.run("lib-srs", "library", top_n=10)
    gold_order = [r["requirement_id"] for r in gold["requirements"]]
    assert [r["requirement_id"] for r in report["per_requirement"]] == [
        rid for rid in sorted(gold_order, key=gold_order.index)]
    assert report["requirements"] == 39
    assert report["links_stored"] == 39 * 10
    stored = links(store)
    assert len(stored) == 390
    assert {l["status"] for l in stored} == {"candidate"}
    assert {l["link_type"] for l in stored} == {"requirement_to_component"}
    first = stored[0]
    assert first["evidence"]["rank"] == 1 and "channels" in first["scores"]


def test_candidate_recall_on_gold(stage):
    search, evaluator, _, gold = stage
    search.run("lib-srs", "library", top_n=10)
    report = evaluator.evaluate_candidates(gold, document_id="lib-srs", pipeline_version="candidates-v1")
    assert report["matching"] == {"by_id": 33, "by_text": 0, "unmatched": []}
    # Regression guard: code-only fused search finds a primary in the top 10 for >= 90%
    assert report["scores"]["hit@10"] >= 0.9


def test_rerun_is_idempotent_and_drops_stale_candidates(stage):
    search, _, store, _ = stage
    search.run("lib-srs", "library", top_n=10)
    search.run("lib-srs", "library", top_n=10)
    assert len(links(store)) == 390
    search.run("lib-srs", "library", top_n=3)
    assert len(links(store)) == 39 * 3


def test_reviewed_links_survive_reruns(stage):
    search, _, store, _ = stage
    search.run("lib-srs", "library", top_n=3)
    target = links(store)[-1]                       # a rank-3 candidate
    with store.transaction() as cur:
        cur.execute(f"UPDATE {store._cfg.table_trace_links} SET status = 'supported' "
                    f"WHERE link_pk = %s", (target["link_pk"],))
    search.run("lib-srs", "library", top_n=1)        # would drop rank 3
    remaining = {l["link_pk"]: l["status"] for l in links(store)}
    assert remaining[target["link_pk"]] == "supported"


def test_pipeline_versions_are_separate(stage):
    search, _, store, _ = stage
    search.run("lib-srs", "library", top_n=2, pipeline_version="candidates-exp")
    assert len(links(store, pv="candidates-exp")) == 39 * 2


def test_type_filter(stage):
    search, *_ = stage
    report = search.run("lib-srs", "library", top_n=1, types=["non-functional"],
                        pipeline_version="nfr-only")
    assert [r["requirement_id"] for r in report["per_requirement"]] == ["NFR-001", "NFR-002", "NFR-003"]


def test_signals_reported(stage):
    search, *_ = stage
    report = search.run("lib-srs", "library", top_n=5, pipeline_version="signals")
    sig = {r["requirement_id"]: r["signals"] for r in report["per_requirement"]}
    assert set(sig["FR-022"]) == {"best_code_similarity", "best_doc_similarity", "agreement", "lexical_hits"}
    assert sig["FR-022"]["best_doc_similarity"] is None      # no docstrings in this graph
    assert 0 < sig["FR-022"]["best_code_similarity"] <= 1


def test_keyword_query_cannot_create_links(stage):
    search, _, store, _ = stage
    from traceability.query import QueryResolver
    q = QueryResolver(store).resolve("overdue fine", repo_id="library")
    with pytest.raises(ValueError, match="stored requirement"):
        search.run_one(q)
