"""
Tests for the verifier (traceability.verification).

The LLM is a scripted fake, so every path — supported, partial, contradicted,
invented citations, model failure, malformed output — is repeatable without a
real model.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from traceability.verification.claims import split_claims
from traceability.verification.context import build_context, quote_overlap, validate_reference
from traceability.verification.llm import LLMError, OllamaClient
from traceability.verification.verifier import Verifier, aggregate
from traceability.tests.conftest import requirement, requires_pg, v1_component


# ============================================================================
# Pure pieces
# ============================================================================


class TestClaims:

    def test_sentences_become_claims_and_label_is_removed(self):
        claims, incomplete = split_claims(
            "FR-020 — Renew a loan The system shall allow renewal. A loan shall be renewed at most 2 times.",
            "Renew a loan")
        assert claims == ["The system shall allow renewal.", "A loan shall be renewed at most 2 times."]
        assert incomplete is False

    def test_list_items_get_the_introduction(self):
        claims, _ = split_claims("The system shall reject the book when:\n- the ISBN is invalid;\n"
                                 "- copies are fewer than 1.")
        assert claims == ["The system shall reject the book when the ISBN is invalid.",
                          "The system shall reject the book when copies are fewer than 1."]

    def test_missing_list_is_flagged_and_not_verified_when_other_claims_exist(self):
        claims, incomplete = split_claims("The system shall add a book. The system shall reject the book when:")
        assert claims == ["The system shall add a book."] and incomplete

    def test_only_an_introduction_is_kept_as_best_effort(self):
        claims, incomplete = split_claims("The fine shall be calculated as follows:")
        assert claims == ["The fine shall be calculated as follows:"] and incomplete

    def test_decimals_do_not_split_and_cap_merges_tail(self):
        assert split_claims("The fine is 0.50 per day up to 20.00 total.")[0] == [
            "The fine is 0.50 per day up to 20.00 total."]
        text = " ".join(f"The system shall do thing number {i}." for i in range(12))
        assert len(split_claims(text, max_claims=8)[0]) == 8


@pytest.mark.parametrize("verdicts, status", [
    (["supported", "supported"], "supported"),
    (["supported", "not_supported"], "partially_supported"),
    (["supported", "insufficient_evidence"], "partially_supported"),
    (["not_supported", "insufficient_evidence"], "not_supported_by_candidate"),
    (["insufficient_evidence"], "insufficient_evidence"),
    ([], "insufficient_evidence"),
])
def test_aggregate(verdicts, status):
    assert aggregate(verdicts) == status


def _comps():
    return [
        {"comp_pk": 1, "link_pk": 11, "component_id": "fines.calculate_fine", "component_type": "function",
         "role": "direct", "relative_path": "fines.py", "start_line": 6, "end_line": 9,
         "source_code": "def calculate_fine(loan):\n    if late <= 2:\n        return 0.0\n    return late * 0.5",
         "docstring": "Always returns zero."},
        {"comp_pk": 2, "link_pk": 12, "component_id": "loan.Loan", "component_type": "class",
         "role": "supporting", "relative_path": "loan.py", "start_line": 1, "end_line": 6,
         "source_code": "class Loan:\n    PERIOD = 14\n\n    def renew(self):\n        pass\n        return 1"},
    ]


class TestContext:

    def test_real_line_numbers_and_labelled_docstring(self):
        ctx = build_context(_comps())
        assert "    6 | def calculate_fine(loan):" in ctx.text
        assert "    9 |     return late * 0.5" in ctx.text
        assert "NOT evidence of behaviour): Always returns zero." in ctx.text
        assert (ctx.shown["C1"].first_line, ctx.shown["C1"].last_line) == (6, 9)

    def test_supporting_class_shows_header_only(self):
        ctx = build_context(_comps())
        assert ctx.shown["C2"].last_line == 3 and ctx.shown["C2"].truncated
        assert "def renew" not in ctx.text and "lines 4-6 not shown" in ctx.text

    def test_budget_omits_components_explicitly(self):
        ctx = build_context(_comps(), max_chars=200)
        assert list(ctx.shown) == ["C1"] and ctx.omitted == ["loan.Loan"]
        assert "Omitted for length (cannot be cited): loan.Loan" in ctx.text

    def test_reference_validation(self):
        ctx = build_context(_comps())
        ok, why = validate_reference(ctx, {"component": "c1", "start_line": 7, "end_line": 8, "quote": "if late <= 2:"})
        assert ok["component_id"] == "fines.calculate_fine" and ok["link_pk"] == 11 and why is None
        assert validate_reference(ctx, {"component": "C9", "start_line": 6, "end_line": 6})[1].startswith("unknown")
        assert "outside" in validate_reference(ctx, {"component": "C1", "start_line": 5, "end_line": 7})[1]
        assert "outside" in validate_reference(ctx, {"component": "C2", "start_line": 4, "end_line": 4})[1]
        assert "quote" in validate_reference(ctx, {"component": "C1", "start_line": 6, "end_line": 6,
                                                   "quote": "send_email(member)"})[1]
        assert validate_reference(ctx, {"component": "C1", "start_line": "x", "end_line": 6})[0] is None

    def test_off_by_one_citation_is_realigned_to_the_quoted_line(self):
        ctx = build_context(_comps())
        ok, _ = validate_reference(ctx, {"component": "C1", "start_line": 8, "end_line": 8,
                                         "quote": "return late * 0.5"})
        assert (ok["start_line"], ok["end_line"], ok["realigned_from"]) == (9, 9, [8, 8])
        # but a quote that exists nowhere near the cited lines is still rejected
        assert validate_reference(ctx, {"component": "C1", "start_line": 6, "end_line": 6,
                                        "quote": "send_email(member)"})[0] is None

    def test_quote_overlap(self):
        assert quote_overlap("return late * 0.5", "    return late * 0.5") == 1.0
        assert quote_overlap("send_email()", "return late") == 0.0


def test_ollama_unreachable_raises_llm_error_quickly():
    client = OllamaClient("qwen2.5-coder:7b", host="http://127.0.0.1:9", timeout=2, retries=0)
    with pytest.raises(LLMError):
        client.chat_json("s", "u", {"type": "object"})


# ============================================================================
# Verifier with a scripted LLM (database)
# ============================================================================


class FakeLLM:
    model = "fake-llm"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def chat_json(self, system, user, schema):
        self.prompts.append(user)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def cite(tag, a, b, quote=""):
    return {"component": tag, "start_line": a, "end_line": b, "quote": quote}


def judgement(*claims, summary="ok"):
    return {"claims": [{"claim_index": i, "verdict": v, "evidence": ev, "rationale": "r"}
                       for i, (v, ev) in enumerate(claims, start=1)], "summary": summary}


@pytest.fixture()
def world(live_store):
    """FR-022 (2 claims) mapped to calculate_fine (direct) and Loan (supporting); FR-031 with no links."""
    s = live_store
    repo = s.upsert_repository({"repo_id": "r", "graph_format": "v1"})
    comps = s.upsert_components(repo, [
        v1_component("fines.calculate_fine", source_code="def calculate_fine(loan):\n    if late <= 2:\n"
                     "        return 0.0\n    return round(late * 0.5, 2)", start_line=6, end_line=9),
        v1_component("loan.Loan", ctype="class", source_code="class Loan:\n    PERIOD = 14\n", start_line=1, end_line=2),
    ])
    s.upsert_docstrings(comps, [{"component_id": "fines.calculate_fine", "kind": "generated",
                                 "content": "Always returns zero."}])
    doc = s.upsert_document({"document_id": "d", "file_name": "d.docx", "source_type": "docx"})
    reqs = s.upsert_requirements(doc, [
        requirement("FR-022", raw_text="FR-022 — Fine No fine within 2 days. The fine is rounded to two decimals.",
                    title="Fine"),
        requirement("FR-031", raw_text="The system shall email reminders before the due date.", title="Remind"),
    ])
    s.upsert_trace_links([
        {"link_type": "requirement_to_component", "req_pk": reqs["FR-022"], "comp_pk": comps["fines.calculate_fine"],
         "status": "pending_verification", "evidence": {"role": "direct", "rank": 1}, "pipeline_version": "mapping-v1"},
        {"link_type": "requirement_to_component", "req_pk": reqs["FR-022"], "comp_pk": comps["loan.Loan"],
         "status": "pending_verification", "evidence": {"role": "supporting", "rank": 2}, "pipeline_version": "mapping-v1"},
    ])
    from traceability.query import QueryResolver
    return s, QueryResolver(s), reqs


def links(store):
    return {l["component_id"]: l["status"] for l in store.get_trace_links(pipeline_version="mapping-v1")}


def rows(store):
    with store.transaction() as cur:
        cur.execute(f"SELECT * FROM {store._cfg.table_verifications} ORDER BY verification_pk")
        return [dict(r) for r in cur.fetchall()]


@requires_pg
class TestVerifier:

    def test_supported_with_valid_citations(self, world):
        s, res, _ = world
        llm = FakeLLM([judgement(("supported", [cite("C1", 7, 8, "if late <= 2:")]),
                                 ("supported", [cite("C1", 9, 9, "round(late * 0.5, 2)")]))])
        rep = Verifier(s, res, llm).run("d", requirement_ids=["FR-022"])
        r = rep["per_requirement"][0]
        assert r["status"] == "supported" and [c["text"] for c in r["claims"]] == [
            "No fine within 2 days.", "The fine is rounded to two decimals."]
        assert links(s) == {"fines.calculate_fine": "supported", "loan.Loan": "pending_verification"}
        req_row = next(x for x in rows(s) if x["link_pk"] is None)
        assert req_row["status"] == "supported" and req_row["model"] == "fake-llm"
        assert req_row["evidence_refs"][0]["component_id"] == "fines.calculate_fine"

    def test_prompt_labels_docstring_and_shows_numbered_code(self, world):
        s, res, _ = world
        llm = FakeLLM([judgement(("not_supported", []), ("not_supported", []))])
        Verifier(s, res, llm).run("d", requirement_ids=["FR-022"])
        assert "NOT evidence" in llm.prompts[0] and "    7 |     if late <= 2:" in llm.prompts[0]

    def test_misleading_docstring_cannot_override_code(self, world):
        # The model reports the code contradicts the claim; the verdict follows the code.
        s, res, _ = world
        llm = FakeLLM([judgement(("not_supported", [cite("C1", 7, 8, "if late <= 2:")]),
                                 ("not_supported", []))])
        rep = Verifier(s, res, llm).run("d", requirement_ids=["FR-022"])
        assert rep["per_requirement"][0]["status"] == "not_supported_by_candidate"
        assert links(s)["fines.calculate_fine"] == "not_supported_by_candidate"

    def test_partial(self, world):
        s, res, _ = world
        llm = FakeLLM([judgement(("supported", [cite("C1", 7, 8, "if late <= 2:")]), ("not_supported", []))])
        assert Verifier(s, res, llm).run("d", requirement_ids=["FR-022"])["per_requirement"][0]["status"] \
            == "partially_supported"

    @pytest.mark.parametrize("bad", [cite("C7", 7, 8), cite("C1", 1, 3), cite("C1", 7, 8, "send_email(member)")])
    def test_invented_citation_downgrades_support(self, world, bad):
        s, res, _ = world
        llm = FakeLLM([judgement(("supported", [bad]), ("supported", [bad]))])
        r = Verifier(s, res, llm).run("d", requirement_ids=["FR-022"])["per_requirement"][0]
        assert r["status"] == "insufficient_evidence"
        assert all(c["verdict"] == "insufficient_evidence" and c["model_verdict"] == "supported"
                   and c["rejected_evidence"] for c in r["claims"])

    def test_support_without_citation_is_not_support(self, world):
        s, res, _ = world
        llm = FakeLLM([judgement(("supported", []), ("supported", []))])
        assert Verifier(s, res, llm).run("d", requirement_ids=["FR-022"])["per_requirement"][0]["status"] \
            == "insufficient_evidence"

    def test_missing_claim_in_reply(self, world):
        s, res, _ = world
        llm = FakeLLM([judgement(("supported", [cite("C1", 7, 8, "if late <= 2:")]))])   # only claim 1
        r = Verifier(s, res, llm).run("d", requirement_ids=["FR-022"])["per_requirement"][0]
        assert r["status"] == "partially_supported" and r["claims"][1]["verdict"] == "insufficient_evidence"

    def test_model_failure_is_never_positive(self, world):
        s, res, _ = world
        llm = FakeLLM([LLMError("timeout")])
        r = Verifier(s, res, llm).run("d", requirement_ids=["FR-022"])["per_requirement"][0]
        assert r["status"] == "insufficient_evidence" and "timeout" in r["error"]
        assert links(s)["fines.calculate_fine"] == "insufficient_evidence"

    def test_no_links_is_not_found_without_calling_the_model(self, world):
        s, res, _ = world
        llm = FakeLLM([])
        r = Verifier(s, res, llm).run("d", requirement_ids=["FR-031"])["per_requirement"][0]
        assert r["status"] == "not_found" and llm.prompts == []
        assert next(x for x in rows(s) if x["link_pk"] is None)["status"] == "not_found"

    def test_history_is_append_only_and_latest_wins(self, world):
        s, res, reqs = world
        Verifier(s, res, FakeLLM([judgement(("not_supported", []), ("not_supported", []))])).run("d", requirement_ids=["FR-022"])
        Verifier(s, res, FakeLLM([judgement(("supported", [cite("C1", 7, 8, "if late")]),
                                            ("supported", [cite("C1", 9, 9, "round")]))])).run("d", requirement_ids=["FR-022"])
        req_rows = [x for x in rows(s) if x["link_pk"] is None]
        assert [x["status"] for x in req_rows] == ["not_supported_by_candidate", "supported"]
        latest = s.latest_verifications(document_pk=res.document_pk("d"), pipeline_version="verification-v1")
        assert latest["FR-022"]["status"] == "supported"

    def test_uncited_context_returns_to_pending(self, world):
        s, res, _ = world
        both = judgement(("supported", [cite("C1", 7, 8, "if late"), cite("C2", 1, 2, "PERIOD = 14")]),
                         ("supported", [cite("C1", 9, 9, "round")]))
        Verifier(s, res, FakeLLM([both])).run("d", requirement_ids=["FR-022"])
        assert links(s)["loan.Loan"] == "supported"
        only_c1 = judgement(("supported", [cite("C1", 7, 8, "if late")]), ("supported", [cite("C1", 9, 9, "round")]))
        Verifier(s, res, FakeLLM([only_c1])).run("d", requirement_ids=["FR-022"])
        assert links(s)["loan.Loan"] == "pending_verification"

    def test_gold_claims_mode(self, world):
        s, res, _ = world
        llm = FakeLLM([judgement(("supported", [cite("C1", 9, 9, "round")]))])
        rep = Verifier(s, res, llm).run("d", gold_claims={"FR-022": [{"claim_id": "FR-022.4", "text": "Rounded."}]})
        assert rep["claims_source"] == "gold" and rep["requirements"] == 1
        assert rep["per_requirement"][0]["claims"][0]["claim_id"] == "FR-022.4"

    def test_incomplete_requirement_text_flagged(self, live_store):
        from traceability.query import QueryResolver
        s = live_store
        repo = s.upsert_repository({"repo_id": "r", "graph_format": "v1"})
        comp = s.upsert_components(repo, [v1_component("m.f")])["m.f"]
        doc = s.upsert_document({"document_id": "d", "file_name": "d.docx", "source_type": "docx"})
        req = s.upsert_requirements(doc, [requirement("FR-002", raw_text="The system shall reject the book when:",
                                                      title="X")])["FR-002"]
        s.upsert_trace_links([{"link_type": "requirement_to_component", "req_pk": req, "comp_pk": comp,
                               "status": "pending_verification", "evidence": {"role": "direct", "rank": 1},
                               "pipeline_version": "mapping-v1"}])
        r = Verifier(s, QueryResolver(s), FakeLLM([judgement(("insufficient_evidence", []))])).run("d")
        assert r["per_requirement"][0]["incomplete_requirement_text"] is True
        assert "incomplete" in s.latest_verifications(document_pk=QueryResolver(s).document_pk("d"),
                                                      pipeline_version="verification-v1")["FR-002"]["rationale"]


@requires_pg
def test_evaluate_verification_scores(world):
    from traceability.evaluation import Evaluator
    s, res, _ = world
    Verifier(s, res, FakeLLM([judgement(("supported", [cite("C1", 7, 8, "if late")]),
                                        ("supported", [cite("C1", 9, 9, "round"), cite("C9", 1, 1)]))])).run("d")
    gold = {"dataset": "x", "requirements": [
        {"requirement_id": "FR-022", "text": "", "status": "partial",
         "links": [{"component_id": "fines.calculate_fine", "relevance": "primary"}], "claims": []},
        {"requirement_id": "FR-031", "text": "", "status": "not_implemented", "links": [], "claims": []},
    ]}
    rep = Evaluator(res, None).evaluate_verification(gold, document_id="d", pipeline_version="verification-v1")
    assert rep["scores"]["false_support_rate"] == 0.5           # partial FR-022 declared supported
    assert rep["scores"]["exact_accuracy"] == 0.5               # FR-031 not_found is correct
    assert rep["counts"]["citations_rejected"] == 1
    assert rep["confusion"]["not_implemented"] == {"not_found": 1}


@requires_pg
def test_cli_verify_evaluate_and_show(world, capsys):
    from traceability import cli
    s, _, _ = world
    fake = FakeLLM([judgement(("supported", [cite("C1", 7, 8, "if late <= 2:")]), ("not_supported", []))])
    with patch.object(cli, "TraceabilityConfig", return_value=s._cfg), \
         patch("traceability.verification.OllamaClient", return_value=fake):
        assert cli.main(["verify", "--document-id", "d", "--requirements", "FR-022"]) == 0
        assert "FR-022     partially_supported" in capsys.readouterr().out
        assert cli.main(["show-verification", "FR-022", "--document-id", "d"]) == 0
        out = capsys.readouterr().out
        assert "partially_supported" in out and "cites fines.calculate_fine" in out


# ============================================================================
# Clause splitting of compound claims (v2.2)
# ============================================================================

from traceability.verification.claims import validate_split
from traceability.verification.verifier import VerifierConfig

COMPOUND = "A loan shall be renewed only when it has been renewed fewer than 2 times and the member has no fines."
GOOD_SPLIT = ["A loan shall be renewed only when it has been renewed fewer than 2 times.",
              "A loan shall be renewed only when the member has no fines."]


@pytest.mark.parametrize("parts, reason", [
    (GOOD_SPLIT, None),
    ([COMPOUND], "not split"),
    (GOOD_SPLIT[:1] + ["A loan shall be renewed only when the member has paid interest."], "adds words"),
    (["A loan shall be renewed only when it has been renewed fewer times.", GOOD_SPLIT[1]], "drops values"),
    ([GOOD_SPLIT[0], "and the member has no fines."], "not a complete requirement sentence"),
])
def test_validate_split(parts, reason):
    result = validate_split(COMPOUND, parts)
    assert (result is None) if reason is None else (reason in result)


def test_alternatives_are_never_split():
    claim = "The system shall reject the book when the ISBN does not contain exactly 10 or 13 characters."
    parts = [claim.replace("10 or 13", "10"), claim.replace("10 or 13", "13")]
    assert "alternative" in validate_split(claim, parts)


def test_repeated_condition_from_elsewhere_in_the_requirement_is_allowed():
    context = "When a member pays a fine, the system shall return the balance and shall record the payment."
    claim = "the system shall return the balance and shall record the payment."
    parts = ["When a member pays a fine, the system shall return the balance.",
             "When a member pays a fine, the system shall record the payment."]
    assert validate_split(claim, parts, context=context) is None


@pytest.fixture()
def compound_world(world):
    s, res, reqs = world
    from traceability.tests.conftest import requirement as req_rec
    doc = res.document_pk("d")
    pk = s.upsert_requirements(doc, [req_rec("FR-034", raw_text="FR-034 — Renew " + COMPOUND, title="Renew")])["FR-034"]
    comp = s.get_trace_links(pipeline_version="mapping-v1")[0]["comp_pk"]
    s.upsert_trace_links([{"link_type": "requirement_to_component", "req_pk": pk, "comp_pk": comp,
                           "status": "pending_verification", "evidence": {"role": "direct", "rank": 1},
                           "pipeline_version": "mapping-v1"}])
    return s, res


@requires_pg
def test_clauses_mode_splits_compound_claim_and_detects_partial(compound_world):
    s, res = compound_world
    llm = FakeLLM([{"parts": GOOD_SPLIT},
                   judgement(("supported", [cite("C1", 7, 8, "if late <= 2:")]), ("not_supported", []))])
    r = Verifier(s, res, llm, VerifierConfig(claim_splitting="clauses")).run("d", requirement_ids=["FR-034"])
    r = r["per_requirement"][0]
    assert [c["text"] for c in r["claims"]] == GOOD_SPLIT
    assert r["status"] == "partially_supported"
    assert r["claim_splits"][0]["parts"] == GOOD_SPLIT


@requires_pg
def test_clauses_mode_keeps_original_when_split_is_unfaithful(compound_world):
    s, res = compound_world
    bad = [GOOD_SPLIT[0], "A loan shall be renewed only when interest is paid."]
    llm = FakeLLM([{"parts": bad}, judgement(("not_supported", []))])
    r = Verifier(s, res, llm, VerifierConfig(claim_splitting="clauses")).run("d", requirement_ids=["FR-034"])
    r = r["per_requirement"][0]
    assert [c["text"] for c in r["claims"]] == [COMPOUND]
    assert "adds words" in r["claim_splits"][0]["reason"]


@requires_pg
def test_clauses_mode_survives_split_failure(compound_world):
    s, res = compound_world
    llm = FakeLLM([LLMError("timeout"), judgement(("not_supported", []))])
    r = Verifier(s, res, llm, VerifierConfig(claim_splitting="clauses")).run("d", requirement_ids=["FR-034"])
    assert [c["text"] for c in r["per_requirement"][0]["claims"]] == [COMPOUND]
    assert r["per_requirement"][0]["error"] is None


@requires_pg
def test_rules_mode_makes_no_split_call(compound_world):
    s, res = compound_world
    llm = FakeLLM([judgement(("not_supported", []))])
    Verifier(s, res, llm, VerifierConfig(claim_splitting="rules")).run("d", requirement_ids=["FR-034"])
    assert len(llm.prompts) == 1
