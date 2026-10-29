"""
Candidate search stage — the "search agent".

Takes the requirements the SRS Agent produced for one document, one by one in
document order, and finds the top-N implementation candidates for each with
the Phase 2 retriever (code + docstring channels, lexical + vector, fused).

Each candidate is stored as a requirement_to_component trace link with
status 'candidate', its fused rank, and every channel's rank and score.
Candidates are *not* claims of implementation: the mapping stage turns them
into proposed links and the verifier judges them.

Deliberately no found / not-found decision here.  The stage records signals
that later stages (and calibration against the gold) can use:

  best_code_similarity   cosine similarity of the best code-vector hit
  best_doc_similarity    same for docstrings (None when there are none)
  agreement              top candidate is in the top 3 of both its lexical
                         and vector channels
  lexical_hits           number of components matching any query term

No LLM is involved.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from .query import QueryResolver, ResolvedQuery
from .retrieval import Hit, Retriever
from .store import Store

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "candidates-v1"
CHANNELS = ("component_lexical", "component_vector", "docstring_lexical", "docstring_vector")


class CandidateSearch:
    def __init__(self, store: Store, resolver: QueryResolver, retriever: Retriever) -> None:
        self._store = store
        self._resolver = resolver
        self._retriever = retriever

    def run(
        self,
        document_id: str,
        repo_id: str,
        *,
        top_n: int = 10,
        candidate_k: int = 50,
        types: Optional[Sequence[str]] = None,
        pipeline_version: str = PIPELINE_VERSION,
    ) -> Dict[str, Any]:
        """
        Find and store candidates for every requirement of *document_id*
        against repository *repo_id*.  Re-running replaces the previous
        unreviewed candidates of the same pipeline_version.
        """
        queries = self._resolver.document_requirements(
            document_id, repo_id=repo_id, types=list(types) if types else None)
        summary = []
        for q in queries:
            summary.append(self.run_one(q, top_n=top_n, candidate_k=candidate_k,
                                        pipeline_version=pipeline_version))
        logger.info("Candidate search: %d requirement(s) processed.", len(summary))
        return {
            "document_id": document_id,
            "repo_id": repo_id,
            "pipeline_version": pipeline_version,
            "top_n": top_n,
            "requirements": len(summary),
            "links_stored": sum(len(s["candidates"]) for s in summary),
            "per_requirement": summary,
        }

    def run_one(
        self,
        query: ResolvedQuery,
        *,
        top_n: int = 10,
        candidate_k: int = 50,
        pipeline_version: str = PIPELINE_VERSION,
    ) -> Dict[str, Any]:
        """Candidates for one resolved requirement; stored in one transaction."""
        req = query.requirement
        if req is None:
            raise ValueError("Candidate search needs a stored requirement, not a keyword query.")
        channels = self._retriever.channels(query, method="hybrid",
                                            entities=("docstring", "component"),
                                            candidate_k=candidate_k)
        ranked = self._retriever.fuse_components(channels, CHANNELS)[:top_n]
        signals = self._signals(channels, ranked)

        records = [{
            "link_type": "requirement_to_component",
            "req_pk": req["req_pk"],
            "comp_pk": hit.pk,
            "status": "candidate",
            "retrieval_method": "hybrid",
            "scores": {"fused": round(hit.score, 6), "channels": hit.channels},
            "evidence": {"rank": rank, "component_id": hit.external_id,
                         "path": hit.data.get("path"),
                         "start_line": hit.data.get("start_line"),
                         "end_line": hit.data.get("end_line"),
                         "via_docstring": hit.data.get("via_docstring"),
                         "query_mode": query.mode},
            "pipeline_version": pipeline_version,
        } for rank, hit in enumerate(ranked, start=1)]

        with self._store.transaction():
            link_pks = self._store.upsert_trace_links(records)
            removed = self._store.delete_stale_candidates(
                req["req_pk"], pipeline_version, link_pks.values())
        return {
            "requirement_id": req["requirement_id"],
            "req_pk": req["req_pk"],
            "candidates": [h.external_id for h in ranked],
            "signals": signals,
            "stale_removed": removed,
        }

    @staticmethod
    def _signals(channels: Dict[str, List[Hit]], ranked: List[Hit]) -> Dict[str, Any]:
        def best(name):
            hits = channels.get(name) or []
            return round(hits[0].score, 4) if hits else None

        top = ranked[0] if ranked else None
        agreement = bool(top) and all(
            any(top.channels.get(f"{e}_{m}", {}).get("rank", 99) <= 3 for e in ("component", "docstring"))
            for m in ("lexical", "vector"))
        return {
            "best_code_similarity": best("component_vector"),
            "best_doc_similarity": best("docstring_vector"),
            "agreement": agreement,
            "lexical_hits": len(channels.get("component_lexical") or []),
        }
