"""Tests for traceability.query — mode detection and resolution."""

from __future__ import annotations

import pytest

from traceability.query import QueryResolver, detect_mode, normalise_text
from traceability.tests.conftest import requirement, requires_pg


# ============================================================================
# Pure helpers
# ============================================================================


@pytest.mark.parametrize("text, mode", [
    ("FR-018", "id"), ("NFR-002", "id"), ("REQ-001", "id"), ("FR-01.2", "id"),
    ("  fr-018 ", "id"),
    ("overdue fine", "keyword"), ("ISBN", "keyword"), ("FR-018 renewal", "keyword"),
    ("The system shall refuse a checkout for duplicates", "text"),
])
def test_detect_mode(text, mode):
    assert detect_mode(text) == mode


def test_normalise_text_strips_label_case_and_spacing():
    assert normalise_text("FR-018: The  System\nshall X.") == "the system shall x."
    assert normalise_text("FR-018 — The system shall X.") == "the system shall x."
    assert normalise_text("The system shall X.") == "the system shall x."


# ============================================================================
# Resolution against the database
# ============================================================================


@pytest.fixture()
def resolver(live_store):
    d1 = live_store.upsert_document({"document_id": "srs-a", "file_name": "a.docx", "source_type": "docx"})
    d2 = live_store.upsert_document({"document_id": "srs-b", "file_name": "b.docx", "source_type": "docx"})
    live_store.upsert_requirements(d1, [
        requirement("FR-018", raw_text="FR-018: The system shall refuse a duplicate checkout of a title.",
                    normalized_text=""),
        requirement("FR-020", raw_text="The system shall allow a loan to be renewed at most twice."),
    ])
    live_store.upsert_requirements(d2, [requirement("FR-018")])
    live_store.upsert_repository({"repo_id": "lib", "graph_format": "v1"})
    return QueryResolver(live_store)


@requires_pg
class TestResolve:

    def test_id_scoped_to_document(self, resolver):
        q = resolver.resolve("FR-018", document_id="srs-a")
        assert (q.mode, q.status) == ("requirement_id", "ok")
        assert q.requirement["requirement_id"] == "FR-018"
        assert q.is_formal_requirement
        assert "refuse a duplicate checkout" in q.query_text

    def test_id_in_two_documents_without_scope_is_ambiguous(self, resolver):
        q = resolver.resolve("FR-018")
        assert q.status == "ambiguous"
        assert {c["document_id"] for c in q.candidates} == {"srs-a", "srs-b"}
        assert q.requirement is None

    def test_id_is_case_insensitive(self, resolver):
        assert resolver.resolve("fr-020").status == "ok"

    def test_unknown_id_not_found(self, resolver):
        q = resolver.resolve("FR-999", document_id="srs-a")
        assert q.status == "not_found" and q.requirement is None

    def test_full_text_matching_one_requirement(self, resolver):
        q = resolver.resolve("The system shall allow a loan to be renewed at most twice.")
        assert (q.mode, q.status) == ("requirement_text", "ok")
        assert q.requirement["requirement_id"] == "FR-020"

    def test_full_text_ignores_id_label_and_spacing(self, resolver):
        q = resolver.resolve("the system shall refuse a duplicate   checkout of a title.",
                             document_id="srs-a")
        assert q.requirement["requirement_id"] == "FR-018"

    def test_unmatched_text_is_adhoc_and_preserved(self, resolver):
        text = "The system shall print a receipt for every payment made"
        q = resolver.resolve(text)
        assert (q.mode, q.status) == ("adhoc_text", "ok")
        assert q.query_text == text and q.original == text
        assert not q.is_formal_requirement

    def test_adhoc_mode_never_matches_stored_requirement(self, resolver):
        q = resolver.resolve("The system shall allow a loan to be renewed at most twice.", mode="adhoc")
        assert q.mode == "adhoc_text" and q.requirement is None

    def test_keyword_never_creates_requirement(self, resolver, live_store):
        before = _count(live_store)
        q = resolver.resolve("overdue fine")
        assert (q.mode, q.status, q.requirement) == ("keyword", "ok", None)
        assert _count(live_store) == before

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_query(self, resolver, text):
        assert resolver.resolve(text).status == "empty"

    def test_unknown_scope_rejected(self, resolver):
        with pytest.raises(ValueError, match="repository"):
            resolver.resolve("x", repo_id="nope")
        with pytest.raises(ValueError, match="SRS document"):
            resolver.resolve("x", document_id="nope")

    def test_invalid_mode_rejected(self, resolver):
        with pytest.raises(ValueError):
            resolver.resolve("x", mode="fuzzy")

    def test_resolved_requirement_scopes_to_its_document(self, resolver):
        q = resolver.resolve("FR-020")
        assert q.document_pk == q.requirement["document_pk"]


def _count(store):
    with store.transaction() as cur:
        cur.execute(f"SELECT count(*) AS n FROM {store._cfg.table_requirements}")
        return cur.fetchone()["n"]
