"""
Tests for traceability.mapping — the mapping agent.

Unit tests drive select_direct / expand_supporting / RepoGraph with a small
in-memory repository.  Database tests run the full stage on the real library
dataset (graph + gold v2 fixtures, real embedding model, code only).
"""

from __future__ import annotations

import pytest

from traceability.mapping import (
    Candidate,
    Mapper,
    MappingConfig,
    RepoGraph,
    expand_supporting,
    is_strong,
    select_direct,
)
from traceability.tests.conftest import FIXTURES, make_cfg, requires_pg, reset_tables, write_json

LIB = FIXTURES / "library"

# ---------------------------------------------------------------------------
# A small in-memory repository
# ---------------------------------------------------------------------------

SOURCES = {
    1: ("svc.Lending", "class", "class Lending:\n    pass", 1),
    2: ("svc.Lending.checkout", "method",
        "def checkout(self, member, isbn):\n"
        "    book = self.find(isbn)\n"
        "    if not member.can_borrow():\n"
        "        raise PermissionError('no')\n"
        "    book.take()\n"
        "    return Loan(book)", 10),
    3: ("svc.Lending.find", "method", "def find(self, isbn):\n    return isbn", 20),
    4: ("member.Member.can_borrow", "method", "def can_borrow(self):\n    return True", 5),
    5: ("book.Book.take", "method", "def take(self):\n    self.n -= 1", 7),
    6: ("loan.Loan", "class", "class Loan:\n    pass", 1),
    7: ("loan.Loan.__init__", "method", "def __init__(self, book):\n    self.book = book", 2),
    8: ("other.Thing.take", "method", "def take(self):\n    pass", 3),          # makes 'take' ambiguous
    9: ("report.audit", "function", "def audit(svc):\n    return svc.checkout(1, 2)", 1),
}
RELS = [(1, 2, "contains"), (1, 3, "contains"), (6, 7, "contains"), (2, 3, "calls")]


def graph(docstrings=None):
    comps = {pk: {"comp_pk": pk, "component_id": cid, "component_type": t, "relative_path": cid.split(".")[0] + ".py",
                  "start_line": start, "end_line": start + src.count("\n"), "source_code": src, "has_vec": True}
             for pk, (cid, t, src, start) in SOURCES.items()}
    return RepoGraph(comps, RELS, docstrings or {})


def cand(pk, rank, channel_ranks, eligible=4):
    cid, t, _, _ = SOURCES[pk]
    names = ["component_lexical", "component_vector", "docstring_lexical", "docstring_vector"]
    return Candidate(pk, cid, t, rank, {n: {"rank": r, "score": 1.0} for n, r in zip(names, channel_ranks) if r},
                     eligible=eligible)


# ---------------------------------------------------------------------------
# Direct selection
# ---------------------------------------------------------------------------


def test_strength_is_relative_to_eligible_channels():
    cfg = MappingConfig()
    assert is_strong(cand(2, 2, [1, 2, 3, 9]), cfg)               # 3 of 4
    assert not is_strong(cand(2, 2, [1, 2, 9, 9]), cfg)           # 2 of 4
    assert is_strong(cand(7, 2, [1, 2], eligible=2), cfg)         # no docstring: 2 of 2 is enough
    assert not is_strong(cand(7, 2, [1, 9], eligible=2), cfg)


def test_rank_one_always_direct_and_weak_lower_ranks_dropped():
    g, cfg = graph(), MappingConfig()
    direct, containers = select_direct([cand(4, 1, [5, 5, 5, 5]), cand(5, 2, [1, 9, 9, 9]),
                                        cand(3, 3, [1, 1, 2, 9])], g, cfg)
    assert [c.comp_pk for c, _ in direct] == [4, 3]
    assert direct[0][1] == "rank 1" and "3/4" in direct[1][1]
    assert containers == []


def test_beyond_max_direct_rank_never_direct():
    direct, _ = select_direct([cand(4, 1, [1]), cand(5, 2, [9, 9]), cand(3, 3, [9, 9]),
                               cand(2, 4, [1, 1, 1, 1])], graph(), MappingConfig())
    assert [c.comp_pk for c, _ in direct] == [4]


def test_class_with_own_method_in_top_is_a_container_and_method_promoted():
    direct, containers = select_direct([cand(1, 1, [1, 1]), cand(2, 2, [2, 9, 9, 9])], graph(), MappingConfig())
    assert [c.comp_pk for c, _ in direct] == [2]
    assert "promoted" in direct[0][1]
    assert [c.comp_pk for c, _ in containers] == [1]


def test_class_without_own_method_in_top_stays_direct():
    direct, containers = select_direct([cand(6, 1, [1, 1]), cand(4, 2, [9, 9])], graph(), MappingConfig())
    assert [c.comp_pk for c, _ in direct] == [6] and containers == []


def test_no_candidates():
    assert select_direct([], graph(), MappingConfig()) == ([], [])


# ---------------------------------------------------------------------------
# Call references and expansion
# ---------------------------------------------------------------------------


def _refs(g, pk, prefer=()):
    return {(g.components[r.target]["component_id"], r.line) for r in g.references(pk, set(prefer))}


def test_call_references_resolve_unique_names_with_absolute_lines():
    refs = _refs(graph(), 2)
    assert ("svc.Lending.find", 11) in refs              # self.find -> own class
    assert ("member.Member.can_borrow", 12) in refs      # unique name
    assert ("loan.Loan", 15) in refs and ("loan.Loan.__init__", 15) in refs   # Loan(...) runs __init__
    assert not any(cid.endswith(".take") for cid, _ in refs)   # 'take' is ambiguous


def test_ambiguous_name_resolved_by_candidate_preference():
    assert ("book.Book.take", 14) in _refs(graph(), 2, prefer={5})


def test_unparseable_source_has_no_references():
    g = graph()
    g.components[2]["source_code"] = "def broken(:\n"
    assert g.references(2, set()) == []


def test_expansion_uses_docagent_edges_and_source_references():
    g, cfg = graph(), MappingConfig()
    cands = [cand(2, 1, [1, 1]), cand(9, 5, [5, 5])]
    sup = expand_supporting([cands[0]], cands, g, cfg)
    by_id = {g.components[pk]["component_id"]: edges for pk, edges in sup.items()}
    assert {"source": "docagent", "type": "contains", "from": "svc.Lending", "to": "svc.Lending.checkout"} in by_id["svc.Lending"]
    assert any(e["source"] == "docagent" and e["type"] == "calls" for e in by_id["svc.Lending.find"])
    assert any(e["type"] == "call_reference" and e["line"] == 12 for e in by_id["member.Member.can_borrow"])
    assert "loan.Loan.__init__" in by_id
    # report.audit calls checkout and is a candidate -> supporting as a caller
    assert any(e["from"] == "report.audit" for e in by_id["report.audit"])
    assert 2 not in sup                                      # direct never repeated as supporting


def test_expansion_cap_orders_candidates_first():
    g = graph()
    cands = [cand(2, 1, [1, 1]), cand(4, 2, [9, 9])]
    sup = expand_supporting([cands[0]], cands, g, MappingConfig(max_supporting=1))
    assert list(sup) == [4]                                  # the candidate wins the single slot


def test_expansion_can_require_candidates_and_skip_source_references():
    g = graph()
    cands = [cand(2, 1, [1, 1]), cand(4, 2, [9, 9])]
    only_cands = expand_supporting([cands[0]], cands, g, MappingConfig(expand_requires_candidate=True))
    assert set(only_cands) == {4}
    no_refs = expand_supporting([cands[0]], cands, g, MappingConfig(use_source_references=False))
    assert set(no_refs) == {1, 3}                            # contains + DocAgent 'calls' only


def test_cycles_do_not_loop():
    g = RepoGraph({1: {"component_id": "m.a", "component_type": "function", "relative_path": "m.py",
                       "start_line": 1, "end_line": 2, "source_code": "def a():\n    return b()"},
                   2: {"component_id": "m.b", "component_type": "function", "relative_path": "m.py",
                       "start_line": 4, "end_line": 5, "source_code": "def b():\n    return a()"}},
                  [(1, 2, "calls"), (2, 1, "calls")], {})
    sup = expand_supporting([Candidate(1, "m.a", "function", 1)], [Candidate(1, "m.a", "function", 1)], g,
                            MappingConfig())
    assert set(sup) == {2}


# ---------------------------------------------------------------------------
# Full stage on the library dataset (database)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    from traceability.candidates import CandidateSearch
    from traceability.evaluation import Evaluator, load_gold
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
    resolver = QueryResolver(store)
    CandidateSearch(store, resolver, Retriever(store, p._get_embedder())).run("lib-srs", "library")
    yield Mapper(store, resolver), Evaluator(resolver, None), store, gold, cfg
    store.close()
    reset_tables(cfg)


def _links(store, pv="mapping-v1"):
    return store.get_trace_links(pipeline_version=pv)


@requires_pg
class TestMappingStage:

    def test_links_pending_with_evidence(self, stage):
        mapper, _, store, _, _ = stage
        report = mapper.run("lib-srs", "library")
        assert report["requirements"] == 39 and report["without_candidates"] == []
        links = [l for l in _links(store) if l["link_type"] == "requirement_to_component"]
        assert links and {l["status"] for l in links} == {"pending_verification"}
        direct = [l for l in links if l["evidence"]["role"] == "direct"]
        assert all(l["evidence"]["reason"] and l["scores"]["candidate_rank"] for l in direct)
        graph_links = [l for l in links if l["retrieval_method"] == "graph"]
        assert all(l["evidence"]["edges"] for l in graph_links)

    def test_quality_against_gold(self, stage):
        mapper, evaluator, _, gold, _ = stage
        mapper.run("lib-srs", "library")
        report = evaluator.evaluate_mapping(gold, document_id="lib-srs", pipeline_version="mapping-v1")
        assert report["requirements_scored"] == 33
        # Regression guards. Code-only fixture measured 0.84 direct recall and
        # 0.84 recall over all links (0.92 / 0.91 on the real run with
        # docstrings): keep most primaries direct and most gold code inside
        # the subgraph.
        assert report["scores"]["direct"]["recall"] >= 0.8
        assert report["scores"]["all"]["recall"] >= 0.8
        assert set(report["requirements_without_code"]) == {"FR-031", "FR-032", "FR-033",
                                                            "NFR-001", "NFR-002", "NFR-003"}

    def test_rerun_idempotent_and_shrinks_cleanly(self, stage):
        mapper, _, store, _, _ = stage
        mapper.run("lib-srs", "library")
        before = len(_links(store))
        mapper.run("lib-srs", "library")
        assert len(_links(store)) == before
        Mapper(store, mapper._resolver, MappingConfig(max_supporting=1)).run("lib-srs", "library")
        per_req = {}
        for l in _links(store):
            if l["evidence"].get("role") == "supporting":
                per_req[l["requirement_id"]] = per_req.get(l["requirement_id"], 0) + 1
        assert max(per_req.values()) <= 1 + 1          # one expansion slot + at most one demoted class

    def test_verdict_links_survive_remapping(self, stage):
        mapper, _, store, _, _ = stage
        mapper.run("lib-srs", "library")
        target = next(l for l in _links(store) if l["evidence"].get("role") == "supporting")
        with store.transaction() as cur:
            cur.execute(f"UPDATE {store._cfg.table_trace_links} SET status = 'supported' WHERE link_pk = %s",
                        (target["link_pk"],))
        Mapper(store, mapper._resolver, MappingConfig(max_supporting=0)).run("lib-srs", "library")
        assert {l["link_pk"]: l["status"] for l in _links(store)}[target["link_pk"]] == "supported"

    def test_subgraph_view(self, stage):
        mapper, *_ = stage
        mapper.run("lib-srs", "library")
        sub = mapper.subgraph("lib-srs", "FR-016")
        roles = {n["component_id"].rsplit(".", 2)[-2] + "." + n["component_id"].rsplit(".", 1)[-1]: n["role"]
                 for n in sub["nodes"]}
        assert roles.get("LendingService.checkout") == "direct"
        assert any(e["source"] == "source" and e["type"] == "call_reference" for e in sub["edges"])
        assert all({"from", "to", "type", "source"} <= set(e) for e in sub["edges"])

    def test_requirement_without_candidates_gets_no_links(self, stage):
        mapper, _, store, _, cfg = stage
        report = mapper.run("lib-srs", "library", candidates_version="no-such-run", pipeline_version="empty")
        assert len(report["without_candidates"]) == 39
        assert _links(store, "empty") == []


@requires_pg
def test_docstring_links(live_store):
    """requirement->docstring (pending) and docstring->component (ownership) for docstring-matched direct code."""
    from traceability.embedder import EmbeddingResult
    from traceability.query import QueryResolver
    from traceability.tests.conftest import requirement, v1_component

    s = live_store
    repo = s.upsert_repository({"repo_id": "r", "graph_format": "v1"})
    pks = s.upsert_components(repo, [v1_component("fines.calculate_fine")])
    s.upsert_docstrings(pks, [{"component_id": "fines.calculate_fine", "kind": "generated", "content": "Fine."}])
    doc = s.upsert_document({"document_id": "d", "file_name": "d.docx", "source_type": "docx"})
    req_pk = s.upsert_requirements(doc, [requirement("FR-022")])["FR-022"]
    s.upsert_trace_links([{"link_type": "requirement_to_component", "req_pk": req_pk,
                           "comp_pk": pks["fines.calculate_fine"], "status": "candidate",
                           "scores": {"channels": {"component_vector": {"rank": 1, "score": 0.9},
                                                   "docstring_vector": {"rank": 1, "score": 0.9}},
                                      "eligible_channels": 3},
                           "evidence": {"rank": 1}, "pipeline_version": "candidates-v1"}])
    Mapper(s, QueryResolver(s)).run("d", "r")
    by_type = {}
    for l in s.get_trace_links(pipeline_version="mapping-v1"):
        by_type.setdefault(l["link_type"], []).append(l)
    assert by_type["requirement_to_docstring"][0]["status"] == "pending_verification"
    own = by_type["docstring_to_component"][0]
    assert own["status"] == "supported" and own["retrieval_method"] == "ownership"
    assert own["comp_pk"] == pks["fines.calculate_fine"]


@requires_pg
def test_delete_refuses_verdict_statuses(live_store):
    with pytest.raises(ValueError):
        live_store.delete_stale_candidates(1, "v", [], statuses=("supported",))
