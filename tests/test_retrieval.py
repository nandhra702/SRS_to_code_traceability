"""
Tests for traceability.retrieval.

Uses hand-placed 4-dim vectors and a FakeEmbedder that maps query strings to
chosen vectors, so lexical and vector behaviour can be checked independently.
"""

from __future__ import annotations

import pytest

from traceability.embedder import EmbeddingResult
from traceability.query import QueryResolver
from traceability.retrieval import Hit, Retriever, rrf_fuse
from traceability.store import DimensionMismatchError
from traceability.tests.conftest import requirement, requires_pg, v1_component

FINE, LOAN, BOOK, OTHER = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]


class FakeEmbedder:
    model_version = "test-v1"
    dim = 4

    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return EmbeddingResult(self.vectors.get(text, [0.5, 0.5, 0.5, 0.5]), self.model_version, "h")


def emb(vec, version="test-v1"):
    return EmbeddingResult(list(map(float, vec)), version, "h")


# ============================================================================
# RRF (pure)
# ============================================================================


def _hit(pk, channel, rank):
    return Hit("component", pk, f"c{pk}", {}, channels={channel: {"rank": rank, "score": 1.0}})


def test_rrf_rewards_agreement_and_keeps_channel_provenance():
    fused = rrf_fuse({
        "lex": [_hit(1, "lex", 1), _hit(2, "lex", 2)],
        "vec": [_hit(2, "vec", 1), _hit(3, "vec", 2)],
    }, rrf_k=60)
    assert [h.pk for h in fused] == [2, 1, 3]       # 2 appears in both lists
    assert set(fused[0].channels) == {"lex", "vec"}
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_ties_break_on_id_for_stable_order():
    fused = rrf_fuse({"a": [_hit(2, "a", 1)], "b": [_hit(1, "b", 1)]})
    assert [h.external_id for h in fused] == ["c1", "c2"]


# ============================================================================
# Search against the database
# ============================================================================


@pytest.fixture()
def world(live_store):
    """Two repos, two SRS documents, components/docstrings/requirements with known vectors."""
    s = live_store
    repo = s.upsert_repository({"repo_id": "lib", "graph_format": "v1"})
    other = s.upsert_repository({"repo_id": "other", "graph_format": "v1"})
    comps = s.upsert_components(repo, [
        v1_component("fines.calculate_fine", source_code="def calculate_fine(loan):\n    return overdue_days * RATE",
                     signature="calculate_fine(loan)"),
        v1_component("loan.Loan.renew", ctype="method", source_code="def renew(self):\n    self.renewals += 1",
                     signature="renew(self)"),
        v1_component("catalog.Catalog.search", ctype="method",
                     source_code="def search(self, keyword):\n    return [b for b in books if keyword in b.title]",
                     signature="search(self, keyword)"),
        v1_component("stale.old_func", source_code="def old_func():\n    return fine"),
    ], {"fines.calculate_fine": emb(FINE), "loan.Loan.renew": emb(LOAN),
        "catalog.Catalog.search": emb(BOOK), "stale.old_func": emb(FINE, version="old-model")})
    s.upsert_components(other, [v1_component("fines.calculate_fine", source_code="def calculate_fine():\n    pass")],
                        {"fines.calculate_fine": emb(FINE)})
    s.upsert_docstrings(comps, [
        {"component_id": "loan.Loan.renew", "kind": "generated",
         "content": "Extend the loan's due date unless the renewal limit was reached."},
    ], {("loan.Loan.renew", "generated"): emb(LOAN)})
    d1 = s.upsert_document({"document_id": "srs-a", "file_name": "a.docx", "source_type": "docx"})
    d2 = s.upsert_document({"document_id": "srs-b", "file_name": "b.docx", "source_type": "docx"})
    s.upsert_requirements(d1, [
        requirement("FR-022", title="Overdue fine", raw_text="The system shall calculate an overdue fine.",
                    normalized_text="", retrieval_text="overdue fine"),
        requirement("FR-020", title="Renew a loan", raw_text="The system shall renew a loan at most twice.",
                    normalized_text="", retrieval_text="renew loan"),
    ], {"FR-022": emb(FINE), "FR-020": emb(LOAN)})
    s.upsert_requirements(d2, [requirement("FR-022", title="Fine in other doc",
                                           raw_text="The system shall charge fines.")],
                          {"FR-022": emb(FINE)})
    embedder = FakeEmbedder({"overdue fine": FINE, "extend borrowing period": LOAN})
    return QueryResolver(s), Retriever(s, embedder), embedder


def ids(hits):
    return [h.external_id for h in hits]


@requires_pg
class TestSearch:

    def test_keyword_hybrid_finds_code_and_keeps_both_scores(self, world):
        resolver, retriever, _ = world
        r = retriever.search(resolver.resolve("overdue fine", repo_id="lib"), entities=("component",))
        top = r.results["component"][0]
        assert top.external_id == "fines.calculate_fine"
        assert set(top.channels) == {"component_lexical", "component_vector"}
        assert top.data["path"] == "fines.py"

    def test_lexical_stems_and_splits_identifiers(self, world):
        resolver, retriever, _ = world
        # 'fines' must match calculate_fine via stemming + identifier splitting
        r = retriever.search(resolver.resolve("fines", repo_id="lib"), method="lexical", entities=("component",))
        assert "fines.calculate_fine" in ids(r.results["component"])

    def test_vector_only_matches_meaning_without_shared_words(self, world):
        resolver, retriever, _ = world
        q = resolver.resolve("extend borrowing period", repo_id="lib")
        assert retriever.search(q, method="lexical", entities=("component",)).results["component"] == []
        vec = retriever.search(q, method="vector", entities=("component",)).results["component"]
        assert vec[0].external_id == "loan.Loan.renew"
        assert set(vec[0].channels) == {"component_vector"}

    def test_docstrings_searched_and_attributed_to_owner(self, world):
        resolver, retriever, _ = world
        r = retriever.search(resolver.resolve("renewal limit due date", repo_id="lib"))
        doc = r.results["docstring"][0]
        assert doc.data["component_id"] == "loan.Loan.renew"
        assert doc.data["docstring_kind"] == "generated"
        assert r.components[0].external_id == "loan.Loan.renew"

    def test_repository_scope(self, world):
        resolver, retriever, _ = world
        scoped = retriever.search(resolver.resolve("overdue fine", repo_id="lib"), entities=("component",))
        unscoped = retriever.search(resolver.resolve("overdue fine"), entities=("component",))
        assert {h.data["repo_pk"] for h in scoped.results["component"]} == {
            scoped.query.repo_pk}
        assert len({h.data["repo_pk"] for h in unscoped.results["component"]}) == 2

    def test_other_model_versions_never_compared(self, world):
        resolver, retriever, _ = world
        r = retriever.search(resolver.resolve("overdue fine", repo_id="lib"),
                             method="vector", entities=("component",))
        assert "stale.old_func" not in ids(r.results["component"])

    def test_requirement_id_query_uses_stored_vector_and_excludes_itself(self, world):
        resolver, retriever, embedder = world
        q = resolver.resolve("FR-022", document_id="srs-a", repo_id="lib")
        r = retriever.search(q)
        assert embedder.calls == []                      # stored requirement vector reused
        assert "FR-022" not in ids(r.results["requirement"])
        assert r.results["component"][0].external_id == "fines.calculate_fine"
        # requirements stay within the resolved requirement's document
        assert {h.data["document_pk"] for h in r.results["requirement"]} == {q.document_pk}

    def test_requirement_text_query_matches_stored_requirement(self, world):
        resolver, retriever, _ = world
        q = resolver.resolve("The system shall renew a loan at most twice.", repo_id="lib")
        assert q.requirement["requirement_id"] == "FR-020"
        assert retriever.search(q, entities=("component",)).results["component"][0].external_id == "loan.Loan.renew"

    def test_ambiguous_or_empty_query_returns_no_results(self, world):
        resolver, retriever, _ = world
        for q in (resolver.resolve("FR-022"), resolver.resolve("  ")):
            r = retriever.search(q)
            assert r.results == {} and r.components == []
            assert r.to_dict()["query"]["status"] in ("ambiguous", "empty")

    def test_top_k_and_stable_ranking(self, world):
        resolver, retriever, _ = world
        q = resolver.resolve("overdue fine", repo_id="lib")
        a = retriever.search(q, top_k=2)
        b = retriever.search(q, top_k=2)
        assert all(len(v) <= 2 for v in a.results.values())
        assert {e: ids(h) for e, h in a.results.items()} == {e: ids(h) for e, h in b.results.items()}

    def test_query_dimension_mismatch_rejected(self, world):
        resolver, _, _ = world
        retriever = Retriever(resolver._store, FakeEmbedder({"overdue fine": [1, 0, 0]}))
        with pytest.raises(DimensionMismatchError):
            retriever.search(resolver.resolve("overdue fine", repo_id="lib"), method="vector")

    def test_query_without_lexemes_skips_lexical(self, world):
        resolver, retriever, _ = world
        r = retriever.search(resolver.resolve("the and of", repo_id="lib"), method="lexical")
        assert all(v == [] for v in r.results.values())

    def test_invalid_arguments(self, world):
        resolver, retriever, _ = world
        q = resolver.resolve("x")
        with pytest.raises(ValueError):
            retriever.search(q, method="fuzzy")
        with pytest.raises(ValueError):
            retriever.search(q, entities=("widget",))

    def test_to_dict_is_json_ready(self, world):
        import json
        resolver, retriever, _ = world
        json.dumps(retriever.search(resolver.resolve("overdue fine", repo_id="lib")).to_dict())


@requires_pg
def test_missing_docstring_is_not_evidence_against_a_component(world):
    """
    calculate_fine has no docstring (like DocAgent's skipped __init__ methods);
    renew has one.  Both rank equally well where both can appear, so absence
    from the docstring channels must not push calculate_fine down.
    """
    resolver, retriever, _ = world
    with resolver._store.transaction() as cur:
        cur.execute(f"SELECT component_id, comp_pk FROM {resolver._cfg.table_components}")
        pk = {r["component_id"]: r["comp_pk"] for r in cur.fetchall()}

    def comp(cid, ch, rank):
        return Hit("component", pk[cid], cid, {"component_type": "function", "repo_pk": 1, "path": "p",
                   "start_line": 1, "end_line": 2}, channels={ch: {"rank": rank, "score": 1.0}})

    def doc(cid, ch):
        return Hit("docstring", 999, cid, {"comp_pk": pk[cid], "component_id": cid, "component_type": "method",
                   "repo_pk": 1, "path": "p", "start_line": 1, "end_line": 2, "docstring_kind": "generated"},
                   channels={ch: {"rank": 1, "score": 1.0}})

    channels = {
        "component_lexical": [comp("fines.calculate_fine", "component_lexical", 1), comp("loan.Loan.renew", "component_lexical", 2)],
        "component_vector":  [comp("fines.calculate_fine", "component_vector", 1), comp("loan.Loan.renew", "component_vector", 2)],
        "docstring_lexical": [doc("loan.Loan.renew", "docstring_lexical")],
        "docstring_vector":  [doc("loan.Loan.renew", "docstring_vector")],
    }
    fused = retriever.fuse_components(channels)
    # A plain RRF sum would rank renew first (4 votes vs 2).
    assert [h.external_id for h in fused] == ["fines.calculate_fine", "loan.Loan.renew"]
    assert fused[0].score == pytest.approx((2 / 61) / 2)
    assert fused[1].score == pytest.approx((2 / 62 + 2 / 61) / 4)
    assert fused[1].data["via_docstring"] == "generated"
