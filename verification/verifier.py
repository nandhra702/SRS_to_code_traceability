"""
Verifier (Phase 4).

For each requirement of a document:

1. Load its mapped subgraph (Phase 3 links) and the code of those components.
   No links → status 'not_found' without calling the model (the pipeline found
   no candidate; this does not prove the feature is absent from the repo).
2. Claims: split the requirement into atomic claims with the LLM (or use the
   gold-standard claims, to evaluate verification in isolation).
3. Ask the LLM for a verdict per claim, citing line ranges of the shown code.
4. Validate every citation against the code actually shown.  A 'supported'
   claim left with no valid citation is downgraded to
   'insufficient_evidence' — support must be grounded in code, never in a
   docstring, a similarity score or an invented reference.
5. Aggregate claims into a requirement status:
     all supported                         → supported
     some supported                        → partially_supported
     none supported, some not_supported    → not_supported_by_candidate
     otherwise / model failure             → insufficient_evidence
6. Persist (append-only) one requirement-level row plus one row per judged
   link, and set link statuses: a link cited by a supported claim is
   'supported'; an uncited direct link is 'not_supported_by_candidate'
   (or 'insufficient_evidence' when the requirement could not be decided).
   Uncited supporting links stay 'pending_verification' — they were context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..query import QueryResolver
from ..store import Store
from .claims import split_claims
from .context import EvidenceContext, build_context, validate_reference
from .llm import LLMClient, LLMError
from .prompts import (CLAIMS_SCHEMA, CLAIMS_SYSTEM, PROMPT_VERSION, VERIFY_SCHEMA, VERIFY_SYSTEM,
                      claims_prompt, verify_prompt)

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "verification-v1"
MAPPING_VERSION = "mapping-v1"
_REQ_FIELDS = ("req_pk", "requirement_id", "title", "raw_text", "type",
               "conditions", "constraints", "error_conditions", "sub_requirements")


@dataclass
class VerifierConfig:
    max_context_chars: int = 14000
    max_lines_direct: int = 120
    max_lines_supporting: int = 40
    max_claims: int = 8
    # "rules": the requirement's own sentences/list items (claims.py) — nothing
    # can be invented.  "llm": ask the model to split (may paraphrase/invent).
    claim_splitting: str = "rules"


def aggregate(verdicts: Sequence[str]) -> str:
    supported = sum(v == "supported" for v in verdicts)
    if verdicts and supported == len(verdicts):
        return "supported"
    if supported:
        return "partially_supported"
    if any(v == "not_supported" for v in verdicts):
        return "not_supported_by_candidate"
    return "insufficient_evidence"


class Verifier:
    def __init__(self, store: Store, resolver: QueryResolver, llm: LLMClient,
                 config: Optional[VerifierConfig] = None) -> None:
        self._store = store
        self._resolver = resolver
        self._llm = llm
        self.config = config or VerifierConfig()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, document_id: str, *, requirement_ids: Optional[Sequence[str]] = None,
            gold_claims: Optional[Dict[str, List[Dict[str, Any]]]] = None,
            mapping_version: str = MAPPING_VERSION, pipeline_version: str = PIPELINE_VERSION,
            progress=None) -> Dict[str, Any]:
        """
        Verify every requirement of a document (or the given IDs).

        gold_claims: {requirement_id: [{"claim_id", "text"}, ...]} — when given,
        these claims are verified instead of LLM-generated ones (requirements
        without gold claims are skipped).
        """
        doc_pk = self._resolver.document_pk(document_id)
        reqs = self._resolver._requirements("r.document_pk = %(doc)s", {"doc": doc_pk}, order="r.req_pk")
        if requirement_ids:
            wanted = set(requirement_ids)
            reqs = [r for r in reqs if r["requirement_id"] in wanted]
        if gold_claims is not None:
            reqs = [r for r in reqs if gold_claims.get(r["requirement_id"])]
        links = self._store.get_trace_links(document_pk=doc_pk, pipeline_version=mapping_version,
                                            link_type="requirement_to_component")
        results = []
        for i, req in enumerate(reqs, start=1):
            full = self._full_requirement(req["req_pk"])
            req_links = [l for l in links if l["req_pk"] == req["req_pk"]]
            claims = None
            if gold_claims is not None:
                claims = gold_claims[req["requirement_id"]]
            result = self.verify_requirement(full, req_links, claims=claims,
                                             pipeline_version=pipeline_version)
            results.append(result)
            if progress:
                progress(i, len(reqs), result)
        counts: Dict[str, int] = {}
        for r in results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return {"document_id": document_id, "pipeline_version": pipeline_version,
                "mapping_version": mapping_version, "model": self._llm.model,
                "prompt_version": PROMPT_VERSION,
                "claims_source": "gold" if gold_claims is not None else "llm",
                "requirements": len(results), "status_counts": counts, "per_requirement": results}

    def verify_requirement(self, req: Dict[str, Any], links: List[Dict[str, Any]], *,
                           claims: Optional[List[Dict[str, Any]]] = None,
                           pipeline_version: str = PIPELINE_VERSION) -> Dict[str, Any]:
        base = {"requirement_id": req["requirement_id"], "req_pk": req["req_pk"]}
        self._summary = ""
        self._incomplete = False
        if not links:
            row = {"req_pk": req["req_pk"], "status": "not_found",
                   "rationale": "The mapping stage proposed no code for this requirement.",
                   "claims": [], "evidence_refs": []}
            self._persist(row, [], pipeline_version)
            return {**base, "status": "not_found", "claims": [], "error": None}

        ctx = self._context(links)
        error = None
        try:
            claim_list = claims if claims is not None else self._split_claims(req)
            judged = self._judge(req, claim_list, ctx)
        except LLMError as exc:
            error = str(exc)
            logger.warning("Verification of %s failed: %s", req["requirement_id"], exc)
            claim_list = claims or [{"claim_id": None, "text": req["raw_text"], "source": "requirement"}]
            judged = [{**c, "verdict": "insufficient_evidence", "evidence": [], "rejected_evidence": [],
                       "rationale": "model failure", "model_verdict": None} for c in claim_list]

        status = aggregate([c["verdict"] for c in judged])
        refs = [r for c in judged if c["verdict"] == "supported" for r in c["evidence"]]
        rationale = ("Model failure: " + error) if error else self._summary
        if self._incomplete:
            rationale = ("[requirement text looks incomplete: it introduces a list that is missing] "
                         + rationale)
        link_rows = self._link_verdicts(links, ctx, judged, status, error is not None)
        row = {"req_pk": req["req_pk"], "status": status, "rationale": rationale, "claims": judged,
               "evidence_refs": refs}
        self._persist(row, link_rows, pipeline_version)
        return {**base, "status": status, "claims": judged, "error": error,
                "omitted_components": ctx.omitted, "incomplete_requirement_text": self._incomplete}

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _split_claims(self, req: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._incomplete = False
        if self.config.claim_splitting == "rules":
            texts, self._incomplete = split_claims(req["raw_text"], req.get("title"),
                                                   self.config.max_claims)
            source = "requirement_text"
        else:
            reply = self._llm.chat_json(CLAIMS_SYSTEM, claims_prompt(req), CLAIMS_SCHEMA)
            texts = [str(c).strip() for c in reply.get("claims", []) if str(c).strip()]
            texts = list(dict.fromkeys(texts))[: self.config.max_claims]
            source = "llm"
            if not texts:                              # deterministic fallback
                texts, source = [req["raw_text"]], "requirement_text"
        return [{"claim_id": f"{req['requirement_id']}.c{i}", "text": t, "source": source}
                for i, t in enumerate(texts, start=1)]

    def _judge(self, req: Dict[str, Any], claims: List[Dict[str, Any]],
               ctx: EvidenceContext) -> List[Dict[str, Any]]:
        texts = [c["text"] for c in claims]
        reply = self._llm.chat_json(VERIFY_SYSTEM, verify_prompt(req, texts, ctx.text), VERIFY_SCHEMA)
        self._summary = str(reply.get("summary") or "")
        by_index: Dict[int, Dict[str, Any]] = {}
        for item in reply.get("claims", []):
            try:
                idx = int(item.get("claim_index"))
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(claims) and idx not in by_index:
                by_index[idx] = item

        judged = []
        for i, claim in enumerate(claims, start=1):
            item = by_index.get(i)
            if item is None:
                judged.append({**claim, "verdict": "insufficient_evidence", "model_verdict": None,
                               "evidence": [], "rejected_evidence": [],
                               "rationale": "The model returned no verdict for this claim."})
                continue
            model_verdict = item.get("verdict")
            if model_verdict not in ("supported", "not_supported", "insufficient_evidence"):
                model_verdict = "insufficient_evidence"
            valid, rejected = [], []
            for ref in item.get("evidence") or []:
                ok, reason = validate_reference(ctx, ref if isinstance(ref, dict) else {})
                if ok:
                    valid.append(ok)
                else:
                    rejected.append({"reference": ref, "reason": reason})
            verdict = model_verdict
            rationale = str(item.get("rationale") or "")
            if verdict == "supported" and not valid:
                verdict = "insufficient_evidence"
                rationale = "[downgraded: no valid code citation] " + rationale
            judged.append({**claim, "verdict": verdict, "model_verdict": model_verdict,
                           "evidence": valid, "rejected_evidence": rejected, "rationale": rationale})
        return judged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _summary = ""
    _incomplete = False

    def _context(self, links: List[Dict[str, Any]]) -> EvidenceContext:
        cfg = self._store._cfg
        pks = [l["comp_pk"] for l in links]
        with self._store.transaction() as cur:
            cur.execute(
                f"SELECT c.comp_pk, c.component_id, c.component_type, c.relative_path, c.start_line, "
                f"c.end_line, c.source_code, "
                f"(SELECT d.content FROM {cfg.table_docstrings} d WHERE d.comp_pk = c.comp_pk "
                f" AND d.is_current ORDER BY (d.docstring_kind = 'generated') DESC LIMIT 1) AS docstring "
                f"FROM {cfg.table_components} c WHERE c.comp_pk = ANY(%s);", (pks,))
            comps = {r["comp_pk"]: dict(r) for r in cur.fetchall()}
        ordered = sorted(links, key=lambda l: (l["evidence"].get("role") != "direct",
                                               l["evidence"].get("rank") or 10**6, l["link_pk"]))
        items = []
        for l in ordered:
            comp = comps.get(l["comp_pk"])
            if comp:
                items.append({**comp, "link_pk": l["link_pk"], "role": l["evidence"].get("role", "direct")})
        return build_context(items, max_chars=self.config.max_context_chars,
                             max_lines_direct=self.config.max_lines_direct,
                             max_lines_supporting=self.config.max_lines_supporting)

    def _full_requirement(self, req_pk: int) -> Dict[str, Any]:
        cfg = self._store._cfg
        with self._store.transaction() as cur:
            cur.execute(f"SELECT {', '.join(_REQ_FIELDS)} FROM {cfg.table_requirements} WHERE req_pk = %s;",
                        (req_pk,))
            return dict(cur.fetchone())

    @staticmethod
    def _link_verdicts(links, ctx, judged, status, failed) -> List[Dict[str, Any]]:
        cited: Dict[int, List[Dict[str, Any]]] = {}
        for c in judged:
            if c["verdict"] == "supported":
                for ref in c["evidence"]:
                    if ref.get("link_pk") is not None:
                        cited.setdefault(ref["link_pk"], []).append(ref)
        rows = []
        for l in links:
            role = l["evidence"].get("role", "direct")
            if l["link_pk"] in cited:
                rows.append({"link_pk": l["link_pk"], "status": "supported", "evidence_refs": cited[l["link_pk"]]})
            elif role == "direct":
                rows.append({"link_pk": l["link_pk"],
                             "status": "insufficient_evidence" if failed or status == "insufficient_evidence"
                             else "not_supported_by_candidate", "evidence_refs": []})
            elif l["status"] != "pending_verification":
                # Uncited context: back to pending (e.g. cited by an earlier run), no verdict row.
                rows.append({"link_pk": l["link_pk"], "status": "pending_verification", "record": False})
        return rows

    def _persist(self, row: Dict[str, Any], link_rows: List[Dict[str, Any]], pipeline_version: str) -> None:
        meta = {"model": self._llm.model, "prompt_version": PROMPT_VERSION, "pipeline_version": pipeline_version}
        with self._store.transaction():
            self._store.add_verification({**row, **meta})
            for lr in link_rows:
                if lr.get("record", True):
                    self._store.add_verification({"link_pk": lr["link_pk"], "req_pk": row["req_pk"],
                                                  "status": lr["status"], "evidence_refs": lr["evidence_refs"],
                                                  "rationale": f"requirement verdict: {row['status']}", **meta})
                self._store.set_link_status(lr["link_pk"], lr["status"])
