"""
Retrieval evaluation against a gold-standard mapping (format v2).

For each gold requirement with status implemented / partial, a query is run
and every channel's component ranking is scored against the gold links:

  Hit@k      a primary component appears in the top k
  Recall@k   fraction of primary components in the top k
  MRR        mean reciprocal rank of the first primary component
  nDCG@10    graded: primary = 2, supporting = 1

Channels (all rank *components*; docstring hits count for their owner):
  code_lexical, code_vector, code_hybrid,
  doc_lexical,  doc_vector,  doc_hybrid,  all_hybrid

Query sources:
  gold_text  the gold requirement text as an ad-hoc query — isolates search
             quality from SRS extraction
  ingested   the requirement the SRS Agent extracted, matched to the gold by
             ID within the document, else by text similarity — end-to-end
"""

from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .query import QueryResolver, ResolvedQuery, normalise_text
from .retrieval import Hit, Retriever

KS = (1, 3, 5, 10)
CHANNELS = ("code_lexical", "code_vector", "code_hybrid",
            "doc_lexical", "doc_vector", "doc_hybrid", "all_hybrid")
GAIN = {"primary": 2, "supporting": 1}
TEXT_MATCH_THRESHOLD = 0.6

_CHANNEL_SOURCES = {
    "code_lexical": ["component_lexical"],
    "code_vector":  ["component_vector"],
    "code_hybrid":  ["component_lexical", "component_vector"],
    "doc_lexical":  ["docstring_lexical"],
    "doc_vector":   ["docstring_vector"],
    "doc_hybrid":   ["docstring_lexical", "docstring_vector"],
    "all_hybrid":   ["component_lexical", "component_vector",
                     "docstring_lexical", "docstring_vector"],
}


# ---------------------------------------------------------------------------
# Metrics (pure)
# ---------------------------------------------------------------------------


def hit_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
    return float(any(r in relevant for r in ranked[:k]))


def recall_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gains: Dict[str, int], k: int) -> float:
    dcg = sum(gains.get(r, 0) / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def score_ranking(ranked: Sequence[str], primary: set, gains: Dict[str, int],
                  ks: Sequence[int] = KS) -> Dict[str, float]:
    out = {f"hit@{k}": hit_at_k(ranked, primary, k) for k in ks}
    out.update({f"recall@{k}": recall_at_k(ranked, primary, k) for k in ks})
    out["mrr"] = reciprocal_rank(ranked, primary)
    out["ndcg@10"] = ndcg_at_k(ranked, gains, 10)
    return out


def mean_scores(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def load_gold(path: str | Path) -> Dict[str, Any]:
    gold = json.loads(Path(path).read_text(encoding="utf-8"))
    if gold.get("schema_version") != "2.0":
        raise ValueError(f"Expected gold schema_version 2.0, got {gold.get('schema_version')!r}")
    return gold


class Evaluator:
    def __init__(self, resolver: QueryResolver, retriever: Retriever) -> None:
        self._resolver = resolver
        self._retriever = retriever

    def evaluate(
        self,
        gold: Dict[str, Any],
        *,
        repo_id: str,
        document_id: Optional[str] = None,
        query_source: str = "gold_text",
        candidate_k: int = 50,
        ks: Sequence[int] = KS,
    ) -> Dict[str, Any]:
        if query_source not in ("gold_text", "ingested"):
            raise ValueError("query_source must be 'gold_text' or 'ingested'")
        if query_source == "ingested" and document_id is None:
            raise ValueError("query_source='ingested' needs document_id")

        repo_pk = self._resolver.repo_pk(repo_id)
        known = self._repo_components(repo_pk)
        has_docstrings = self._repo_has_docstrings(repo_pk)
        channels = [c for c in CHANNELS if has_docstrings or not c.startswith("doc_")]

        scored = [r for r in gold["requirements"] if r["status"] in ("implemented", "partial")]
        negatives = [r for r in gold["requirements"] if r["status"] == "not_implemented"]
        missing = sorted({l["component_id"] for r in scored for l in r["links"]} - known)

        per_channel: Dict[str, List[Dict[str, float]]] = {c: [] for c in channels}
        details, matching = [], {"by_id": 0, "by_text": 0, "unmatched": []}
        for req in scored:
            primary = {l["component_id"] for l in req["links"] if l["relevance"] == "primary"}
            gains = {l["component_id"]: GAIN[l["relevance"]] for l in req["links"]}
            query, how = self._query_for(req, query_source, repo_id, document_id)
            if how in ("by_id", "by_text"):
                matching[how] += 1
            elif query_source == "ingested":
                matching["unmatched"].append(req["requirement_id"])
            rankings = self._rankings(query, channels, candidate_k) if query else {c: [] for c in channels}
            detail = {"requirement_id": req["requirement_id"], "status": req["status"],
                      "query": how, "primary_ranks": {}}
            for channel in channels:
                per_channel[channel].append(score_ranking(rankings[channel], primary, gains, ks))
                detail["primary_ranks"][channel] = {
                    p: (rankings[channel].index(p) + 1 if p in rankings[channel] else None)
                    for p in sorted(primary)}
            detail["top5_all"] = rankings.get("all_hybrid", rankings.get("code_hybrid", []))[:5]
            details.append(detail)

        report = {
            "dataset": gold.get("dataset"),
            "repo_id": repo_id,
            "document_id": document_id,
            "query_source": query_source,
            "requirements_scored": len(scored),
            "docstrings_present": has_docstrings,
            "gold_components_missing_from_repo": missing,
            "channels": {c: mean_scores(per_channel[c]) for c in channels},
            "negatives": self._negatives(negatives, query_source, repo_id, document_id,
                                         scored, candidate_k),
            "per_requirement": details,
        }
        if query_source == "ingested":
            report["matching"] = matching
        return report

    def evaluate_candidates(
        self,
        gold: Dict[str, Any],
        *,
        document_id: str,
        pipeline_version: str,
        ks: Sequence[int] = KS,
    ) -> Dict[str, Any]:
        """
        Score the *stored* candidate links of a document (candidate search
        stage output) against the gold.  Gold requirements are matched to
        ingested requirements by ID, else by text.
        """
        store = self._resolver._store
        doc_pk = self._resolver.document_pk(document_id)
        links = store.get_trace_links(document_pk=doc_pk, pipeline_version=pipeline_version,
                                      link_type="requirement_to_component")
        by_req: Dict[str, List[str]] = {}
        for link in links:
            by_req.setdefault(link["requirement_id"], []).append(link["component_id"])

        rows, details = [], []
        matching = {"by_id": 0, "by_text": 0, "unmatched": []}
        ingested_ids = {r["requirement_id"] for r in self._resolver._requirements(
            "r.document_pk = %(doc)s", {"doc": doc_pk})}
        for req in gold["requirements"]:
            if req["status"] not in ("implemented", "partial"):
                continue
            rid = req["requirement_id"]
            if rid in ingested_ids:
                matched, how = rid, "by_id"
            else:
                matched = self._best_text_match(req["text"], document_id)
                how = "by_text" if matched else "unmatched"
            if how == "unmatched":
                matching["unmatched"].append(rid)
            else:
                matching[how] += 1
            ranked = by_req.get(matched, []) if matched else []
            primary = {l["component_id"] for l in req["links"] if l["relevance"] == "primary"}
            gains = {l["component_id"]: GAIN[l["relevance"]] for l in req["links"]}
            rows.append(score_ranking(ranked, primary, gains, ks))
            details.append({"requirement_id": rid, "matched_to": matched,
                            "candidates": len(ranked),
                            "primary_ranks": {p: (ranked.index(p) + 1 if p in ranked else None)
                                              for p in sorted(primary)}})
        return {
            "dataset": gold.get("dataset"),
            "document_id": document_id,
            "pipeline_version": pipeline_version,
            "requirements_scored": len(rows),
            "matching": matching,
            "scores": mean_scores(rows),
            "missed": [d["requirement_id"] for d in details
                       if not any(r is not None for r in d["primary_ranks"].values())],
            "per_requirement": details,
        }

    # ------------------------------------------------------------------

    def _repo_components(self, repo_pk: int) -> set:
        cfg = self._resolver._cfg
        with self._resolver._store.transaction() as cur:
            cur.execute(f"SELECT component_id FROM {cfg.table_components} "
                        f"WHERE repo_pk = %s AND is_current;", (repo_pk,))
            return {r["component_id"] for r in cur.fetchall()}

    def _repo_has_docstrings(self, repo_pk: int) -> bool:
        cfg = self._resolver._cfg
        with self._resolver._store.transaction() as cur:
            cur.execute(f"SELECT EXISTS (SELECT 1 FROM {cfg.table_docstrings} d "
                        f"JOIN {cfg.table_components} c ON c.comp_pk = d.comp_pk "
                        f"WHERE c.repo_pk = %s AND d.is_current) AS e;", (repo_pk,))
            return cur.fetchone()["e"]

    def _rankings(self, query: ResolvedQuery, channels: Sequence[str],
                  candidate_k: int) -> Dict[str, List[str]]:
        raw = self._retriever.channels(query, method="hybrid",
                                       entities=("docstring", "component"),
                                       candidate_k=candidate_k)
        out = {}
        for channel in channels:
            hits: List[Hit] = self._retriever.fuse_components(raw, _CHANNEL_SOURCES[channel])
            out[channel] = [h.external_id for h in hits]
        return out

    def _query_for(self, req, query_source, repo_id, document_id):
        if query_source == "gold_text":
            return self._resolver.resolve(req["text"], mode="adhoc", repo_id=repo_id), "gold_text"
        rq = self._resolver.resolve(req["requirement_id"], mode="id",
                                    document_id=document_id, repo_id=repo_id)
        if rq.status == "ok":
            return rq, "by_id"
        best = self._best_text_match(req["text"], document_id)
        if best is None:
            return None, "unmatched"
        return self._resolver.resolve(best, mode="id", document_id=document_id,
                                      repo_id=repo_id), "by_text"

    def _best_text_match(self, text: str, document_id: str) -> Optional[str]:
        """requirement_id of the most similar ingested requirement, if close enough."""
        doc_pk = self._resolver.document_pk(document_id)
        rows = self._resolver._requirements("r.document_pk = %(doc)s", {"doc": doc_pk})
        target = normalise_text(text)
        scored = [(difflib.SequenceMatcher(None, target, normalise_text(r["raw_text"])).ratio(),
                   r["requirement_id"]) for r in rows]
        if not scored:
            return None
        ratio, rid = max(scored)
        return rid if ratio >= TEXT_MATCH_THRESHOLD else None

    def _negatives(self, negatives, query_source, repo_id, document_id, scored, candidate_k):
        """
        Diagnostic: best code-vector similarity for not-implemented requirements
        vs implemented ones.  A lower value for negatives means the search
        signal can help separate 'found' from 'not found' in later phases.
        """
        def top_similarity(req):
            query, _ = self._query_for(req, query_source, repo_id, document_id)
            if query is None:
                return None
            hits = self._retriever.channels(query, method="vector", entities=("component",),
                                            candidate_k=1)["component_vector"]
            return hits[0].score if hits else None

        def mean(vals):
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        neg = {r["requirement_id"]: top_similarity(r) for r in negatives}
        return {
            "not_implemented_top1_code_similarity": {k: (round(v, 4) if v is not None else None)
                                                     for k, v in neg.items()},
            "mean_not_implemented": mean(neg.values()),
            "mean_implemented": mean(top_similarity(r) for r in scored),
        }


def format_report(report: Dict[str, Any]) -> str:
    """Human-readable summary table."""
    cols = ["hit@1", "hit@5", "recall@5", "recall@10", "mrr", "ndcg@10"]
    lines = [
        f"Dataset {report['dataset']} | repo {report['repo_id']} | query source "
        f"{report['query_source']} | {report['requirements_scored']} requirements scored",
        "",
        f"{'channel':14s}" + "".join(f"{c:>11s}" for c in cols),
    ]
    for channel, scores in report["channels"].items():
        lines.append(f"{channel:14s}" + "".join(f"{scores.get(c, 0):>11.3f}" for c in cols))
    if not report["docstrings_present"]:
        lines.append("(no docstrings in this repository: doc_* channels skipped)")
    if report["gold_components_missing_from_repo"]:
        lines.append(f"WARNING gold components missing from repo: {report['gold_components_missing_from_repo']}")
    neg = report["negatives"]
    lines += ["", f"Top-1 code similarity: implemented mean {neg['mean_implemented']}, "
                  f"not-implemented mean {neg['mean_not_implemented']} "
                  f"{neg['not_implemented_top1_code_similarity']}"]
    if "matching" in report:
        m = report["matching"]
        lines.append(f"Requirement matching: {m['by_id']} by ID, {m['by_text']} by text, "
                     f"unmatched {m['unmatched']}")
    misses = [d["requirement_id"] for d in report["per_requirement"]
              if not any(r == 1 for r in d["primary_ranks"].get("all_hybrid",
                         d["primary_ranks"].get("code_hybrid", {})).values())]
    lines.append(f"Requirements whose primary component is not ranked #1 (best fused channel): {misses}")
    return "\n".join(lines)
