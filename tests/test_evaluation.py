"""
Tests for traceability.evaluation — metric maths, and an end-to-end run on the
real library dataset (graph + gold v2 in tests/fixtures/library) with the real
embedding model.
"""

from __future__ import annotations

import math

import pytest

from traceability.evaluation import (
    Evaluator,
    format_report,
    hit_at_k,
    load_gold,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from traceability.tests.conftest import FIXTURES, make_cfg, requires_pg, reset_tables, write_json

LIB = FIXTURES / "library"


# ============================================================================
# Metrics (pure)
# ============================================================================


def test_hit_recall_rr():
    ranked = ["a", "b", "c", "d"]
    assert hit_at_k(ranked, {"c"}, 2) == 0.0 and hit_at_k(ranked, {"c"}, 3) == 1.0
    assert recall_at_k(ranked, {"a", "d"}, 2) == 0.5
    assert recall_at_k(ranked, set(), 2) == 0.0
    assert reciprocal_rank(ranked, {"c", "d"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0


def test_ndcg_graded():
    gains = {"p": 2, "s": 1}
    assert ndcg_at_k(["p", "s"], gains, 10) == pytest.approx(1.0)
    swapped = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["s", "p"], gains, 10) == pytest.approx(swapped)
    assert ndcg_at_k(["x"], gains, 10) == 0.0


def test_load_gold_rejects_v1(tmp_path):
    with pytest.raises(ValueError, match="2.0"):
        load_gold(write_json(tmp_path / "g.json", {"dataset": "x", "requirements": []}))


# ============================================================================
# End-to-end on the library dataset (real model, code only — no docstrings)
# ============================================================================


@pytest.fixture(scope="module")
def library_eval():
    from traceability.pipeline import Pipeline
    from traceability.query import QueryResolver
    from traceability.retrieval import Retriever
    from traceability.store import Store

    cfg = make_cfg(embed_dim=None)
    reset_tables(cfg)
    p = Pipeline(cfg)
    p.ingest_graph(LIB / "library_v1_graph.json", repo_id="library")
    store = Store(cfg)
    yield Evaluator(QueryResolver(store), Retriever(store, p._get_embedder())), store, cfg, p
    store.close()
    reset_tables(cfg)  # leave the test tables free for the default dimension


@requires_pg
class TestLibraryEvaluation:

    def test_gold_text_report(self, library_eval):
        evaluator, *_ = library_eval
        report = evaluator.evaluate(load_gold(LIB / "gold_v2.json"), repo_id="library")
        assert report["requirements_scored"] == 33           # 30 implemented + 3 partial
        assert report["gold_components_missing_from_repo"] == []
        assert report["docstrings_present"] is False
        assert set(report["channels"]) == {"code_lexical", "code_vector", "code_hybrid", "all_hybrid"}
        # Regression guard on real data (code-only baseline measured at 0.97)
        assert report["channels"]["code_hybrid"]["hit@5"] >= 0.9
        neg = report["negatives"]
        assert set(neg["not_implemented_top1_code_similarity"]) == {"FR-031", "FR-032", "FR-033"}
        assert "code_hybrid" in format_report(report)

    def test_ingested_source_matches_by_id_then_text(self, library_eval, tmp_path):
        evaluator, store, cfg, pipeline = library_eval
        gold = load_gold(LIB / "gold_v2.json")
        by_id = {r["requirement_id"]: r for r in gold["requirements"]}
        # An "extracted" SRS: FR-006 keeps its ID, FR-007 got a generated ID, FR-005 is missing.
        extracted = [
            {"requirement_id": "FR-006", "id_source": "source", "raw_text": by_id["FR-006"]["text"],
             "normalized_text": "", "type": "functional", "title": by_id["FR-006"]["title"],
             "retrieval_text": ""},
            {"requirement_id": "REQ-007", "id_source": "generated", "raw_text": by_id["FR-007"]["text"],
             "normalized_text": "", "type": "functional", "title": by_id["FR-007"]["title"],
             "retrieval_text": ""},
        ]
        pipeline.ingest_srs(write_json(tmp_path / "srs.json", extracted), document_id="lib-srs")
        mini = dict(gold, requirements=[by_id["FR-005"], by_id["FR-006"], by_id["FR-007"]])
        report = evaluator.evaluate(mini, repo_id="library", document_id="lib-srs",
                                    query_source="ingested")
        assert report["matching"] == {"by_id": 1, "by_text": 1, "unmatched": ["FR-005"]}

    def test_ingested_requires_document(self, library_eval):
        evaluator, *_ = library_eval
        with pytest.raises(ValueError, match="document_id"):
            evaluator.evaluate(load_gold(LIB / "gold_v2.json"), repo_id="library",
                               query_source="ingested")
