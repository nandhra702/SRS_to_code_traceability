"""
Query normalisation and resolution (Phase 2).

A user query is one of:

  requirement ID     "FR-018"           → the stored requirement, resolved within
                                          an SRS document; ambiguous when the ID
                                          exists in several documents and no
                                          document scope is given
  requirement text   a full sentence    → the stored requirement when the text
                                          matches exactly one, otherwise an
                                          ad-hoc requirement-like query
  keyword / phrase   "overdue fine"     → searched directly; never creates or
                                          implies a formal Requirement

The original input is always preserved on the ResolvedQuery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .store import Store
from .texts import requirement_text

# FR-018, NFR-002, REQ-001, FR-01.2
REQUIREMENT_ID_RE = re.compile(r"^[A-Za-z]{1,10}-\d+(?:\.\d+)*$")

# Inputs with at least this many words are treated as requirement text
# rather than keywords when the mode is "auto".
TEXT_MIN_WORDS = 6

MODES = ("auto", "id", "text", "adhoc", "keyword")

_REQ_COLUMNS = (
    "req_pk, document_pk, requirement_id, title, raw_text, normalized_text, retrieval_text, "
    "actions, entities, domain_terms, synonyms, sub_requirements, "
    "embedding, embed_model_version"
)


@dataclass
class ResolvedQuery:
    """What a user query resolved to, plus everything retrieval needs."""

    original: str
    mode: str                 # requirement_id | requirement_text | adhoc_text | keyword
    status: str               # ok | ambiguous | not_found | empty
    query_text: str = ""      # text used for lexical matching and (if needed) embedding
    requirement: Optional[Dict[str, Any]] = None   # resolved stored requirement row
    document_pk: Optional[int] = None
    repo_pk: Optional[int] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)  # for ambiguous
    message: str = ""

    @property
    def is_formal_requirement(self) -> bool:
        """True only when the query is a stored SRS requirement."""
        return self.requirement is not None

    def describe(self) -> Dict[str, Any]:
        out = {"original": self.original, "mode": self.mode, "status": self.status}
        if self.requirement:
            out["requirement"] = {k: self.requirement[k] for k in
                                  ("req_pk", "requirement_id", "document_pk", "title")}
        if self.candidates:
            out["candidates"] = self.candidates
        if self.message:
            out["message"] = self.message
        return out


def normalise_text(text: str) -> str:
    """Lower-case, collapse whitespace, drop a leading 'FR-018:' / 'FR-018 —' label."""
    text = " ".join(text.split()).lower()
    return re.sub(r"^[a-z]{1,10}-\d+(?:\.\d+)*\s*[:—–-]\s*", "", text)


def detect_mode(text: str) -> str:
    """Classify an input for mode='auto' (no database access)."""
    stripped = text.strip()
    if REQUIREMENT_ID_RE.match(stripped):
        return "id"
    if len(stripped.split()) >= TEXT_MIN_WORDS:
        return "text"
    return "keyword"


class QueryResolver:
    """Resolves user input against the requirements stored in a Store."""

    def __init__(self, store: Store) -> None:
        self._store = store
        self._cfg = store._cfg

    # ------------------------------------------------------------------
    # Scope lookup
    # ------------------------------------------------------------------

    def document_pk(self, document_id: Optional[str]) -> Optional[int]:
        if document_id is None:
            return None
        return self._lookup_pk(self._cfg.table_srs_documents, "document_pk",
                               "document_id", document_id, "SRS document")

    def repo_pk(self, repo_id: Optional[str]) -> Optional[int]:
        if repo_id is None:
            return None
        return self._lookup_pk(self._cfg.table_repositories, "repo_pk",
                               "repo_id", repo_id, "repository")

    def _lookup_pk(self, table, pk_col, id_col, value, label) -> int:
        with self._store.transaction() as cur:
            cur.execute(f"SELECT {pk_col} FROM {table} WHERE {id_col} = %s;", (value,))
            row = cur.fetchone()
        if row is None:
            raise ValueError(f"Unknown {label} {value!r}.")
        return row[pk_col]

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        text: str,
        *,
        mode: str = "auto",
        document_id: Optional[str] = None,
        repo_id: Optional[str] = None,
    ) -> ResolvedQuery:
        """
        Resolve *text*.  mode: auto | id | text (match a stored requirement, else
        ad-hoc) | adhoc (never match a stored requirement) | keyword.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        doc_pk = self.document_pk(document_id)
        repo_pk = self.repo_pk(repo_id)
        base = dict(original=text, document_pk=doc_pk, repo_pk=repo_pk)

        if not text or not text.strip():
            return ResolvedQuery(mode="keyword", status="empty",
                                 message="Empty query.", **base)
        if mode == "auto":
            mode = detect_mode(text)

        if mode == "id":
            return self._resolve_id(text.strip(), base)
        if mode == "text":
            return self._resolve_text(text, base)
        if mode == "adhoc":
            return ResolvedQuery(mode="adhoc_text", status="ok", query_text=text.strip(), **base)
        return ResolvedQuery(mode="keyword", status="ok", query_text=text.strip(), **base)

    def document_requirements(
        self,
        document_id: str,
        *,
        repo_id: Optional[str] = None,
        types: Optional[List[str]] = None,
    ) -> List[ResolvedQuery]:
        """
        Every current requirement of a document as a resolved query, in
        document (ingestion) order — the input of the candidate search stage.
        """
        base = dict(original="", document_pk=self.document_pk(document_id),
                    repo_pk=self.repo_pk(repo_id))
        where, params = "r.document_pk = %(doc)s", {"doc": base["document_pk"]}
        if types:
            where += " AND r.type = ANY(%(types)s)"
            params["types"] = list(types)
        out = []
        for row in self._requirements(where, params, order="r.req_pk"):
            q = self._requirement_query("requirement_id", row, dict(base, original=row["requirement_id"]))
            out.append(q)
        return out

    def _requirements(self, where: str, params: Dict[str, Any],
                      order: str = "d.document_id, r.requirement_id") -> List[Dict[str, Any]]:
        cfg = self._cfg
        sql = (f"SELECT {_REQ_COLUMNS}, r.type, d.document_id "
               f"FROM {cfg.table_requirements} r "
               f"JOIN {cfg.table_srs_documents} d USING (document_pk) "
               f"WHERE r.is_current AND {where} ORDER BY {order};")
        with self._store.transaction() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def _resolve_id(self, rid: str, base: Dict[str, Any]) -> ResolvedQuery:
        where = "upper(r.requirement_id) = upper(%(rid)s)"
        params: Dict[str, Any] = {"rid": rid}
        if base["document_pk"] is not None:
            where += " AND r.document_pk = %(doc)s"
            params["doc"] = base["document_pk"]
        rows = self._requirements(where, params)
        if not rows:
            return ResolvedQuery(mode="requirement_id", status="not_found",
                                 message=f"No requirement {rid!r} in scope.", **base)
        if len(rows) > 1:
            return ResolvedQuery(
                mode="requirement_id", status="ambiguous",
                candidates=[{"document_id": r["document_id"], "req_pk": r["req_pk"]} for r in rows],
                message=f"{rid!r} exists in {len(rows)} SRS documents; specify a document.",
                **base)
        return self._requirement_query("requirement_id", rows[0], base)

    def _resolve_text(self, text: str, base: Dict[str, Any]) -> ResolvedQuery:
        where, params = "TRUE", {}
        if base["document_pk"] is not None:
            where, params = "r.document_pk = %(doc)s", {"doc": base["document_pk"]}
        target = normalise_text(text)
        matches = [
            r for r in self._requirements(where, params)
            if target in (normalise_text(r["raw_text"]), normalise_text(r["normalized_text"] or ""))
        ]
        if len(matches) == 1:
            return self._requirement_query("requirement_text", matches[0], base)
        if len(matches) > 1:
            return ResolvedQuery(
                mode="requirement_text", status="ambiguous",
                candidates=[{"document_id": r["document_id"], "req_pk": r["req_pk"],
                             "requirement_id": r["requirement_id"]} for r in matches],
                message="Text matches several stored requirements; specify a document.",
                **base)
        # No stored match: keep the exact text as an ad-hoc requirement-like query.
        return ResolvedQuery(mode="adhoc_text", status="ok", query_text=text.strip(), **base)

    @staticmethod
    def _requirement_query(mode: str, row: Dict[str, Any], base: Dict[str, Any]) -> ResolvedQuery:
        base = dict(base)
        # A resolved requirement scopes later lookups to its own document.
        base["document_pk"] = row["document_pk"]
        return ResolvedQuery(mode=mode, status="ok", requirement=row,
                             query_text=requirement_text(row), **base)
