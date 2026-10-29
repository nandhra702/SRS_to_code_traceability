"""
Mapping stage (Phase 3) — the "mapping agent".

Turns each requirement's stored candidates (candidate search stage) into an
explicit requirement → code subgraph of proposed trace links.  Deterministic:
no LLM is involved, and no component ID or graph edge is invented.

1. Direct links (the code believed to implement the requirement)
   - The rank-1 candidate is always direct.
   - A rank 2..max_direct_rank candidate is direct only when it is *strong*:
     in the top strong_rank of at least strong_fraction of the channels it is
     eligible for (a component without a docstring is judged on its code
     channels alone).
   - Classes are containers: a direct class whose own method is also in the
     top candidates becomes a supporting 'container' link, and (when
     promote_methods) that method becomes direct instead.

2. Supporting links (context needed by the direct code), one hop from each
   direct component, from two clearly separated evidence sources:
   - docagent:  DocAgent's typed relationships — the containing class
                ('contains'), callees ('calls'), instantiated classes and
                their __init__ ('instantiates'), and callers.
   - source:    call sites in the direct component's own source text, e.g.
                `member.can_borrow()` → Member.can_borrow.  A call name is
                resolved only when it names exactly one component, or exactly
                one among the requirement's candidates; ambiguous names are
                skipped.  Each reference cites its line.  DocAgent cannot
                resolve calls on variables, which hides most such links.

3. Docstring links: requirement → docstring for direct components that were
   matched through a docstring channel, plus docstring → owning component
   (a database fact, stored as 'supported' with basis 'ownership').

Requirement links are stored as status 'pending_verification' with their full
evidence.  Re-running replaces a requirement's unverified links; links with a
verdict are never touched.
"""

from __future__ import annotations

import ast
import logging
import math
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .query import QueryResolver
from .store import Store

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "mapping-v1"
CANDIDATES_VERSION = "candidates-v1"
_DOC_CHANNELS = ("docstring_lexical", "docstring_vector")


@dataclass
class MappingConfig:
    max_direct_rank: int = 3
    strong_rank: int = 3
    strong_fraction: float = 0.75
    promote_methods: bool = True
    # On the library dataset recall over all links is 0.91 at 6 and 0.93 at 8;
    # supporting precision stays ~0.2 at any cap (see TEST_RUN results).
    max_supporting: int = 6
    use_source_references: bool = True
    expand_requires_candidate: bool = False


@dataclass
class Candidate:
    comp_pk: int
    component_id: str
    component_type: str
    rank: int
    channels: Dict[str, Dict[str, float]] = field(default_factory=dict)
    eligible: int = 4
    fused: Optional[float] = None

    def top_channel_count(self, strong_rank: int) -> int:
        return sum(1 for v in self.channels.values() if v.get("rank", 10**6) <= strong_rank)

    @property
    def via_docstring(self) -> bool:
        return any(c in self.channels for c in _DOC_CHANNELS)


# ---------------------------------------------------------------------------
# Repository graph
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    name: str
    line: int                     # absolute line in the file
    target: int                   # resolved comp_pk


class RepoGraph:
    """In-memory view of one repository: components, typed edges, call sites."""

    def __init__(self, components: Dict[int, Dict[str, Any]],
                 relationships: List[Tuple[int, int, str]],
                 docstrings: Dict[int, Tuple[int, str]]) -> None:
        self.components = components
        self.docstrings = docstrings                       # comp_pk -> (docstring_pk, kind)
        self.by_id = {c["component_id"]: pk for pk, c in components.items()}
        self.by_short: Dict[str, List[int]] = {}
        for pk, c in components.items():
            self.by_short.setdefault(c["component_id"].rsplit(".", 1)[-1], []).append(pk)
        self.out_edges: Dict[int, List[Tuple[int, str]]] = {}
        self.in_edges: Dict[int, List[Tuple[int, str]]] = {}
        for src, tgt, rtype in relationships:
            self.out_edges.setdefault(src, []).append((tgt, rtype))
            self.in_edges.setdefault(tgt, []).append((src, rtype))
        self._raw_refs: Dict[int, List[Tuple[str, int, Optional[str]]]] = {}

    @classmethod
    def load(cls, store: Store, repo_pk: int) -> "RepoGraph":
        cfg = store._cfg
        with store.transaction() as cur:
            cur.execute(f"SELECT comp_pk, component_id, component_type, relative_path, start_line, "
                        f"end_line, source_code, containing_class, embedding IS NOT NULL AS has_vec "
                        f"FROM {cfg.table_components} WHERE repo_pk = %s AND is_current;", (repo_pk,))
            comps = {r["comp_pk"]: dict(r) for r in cur.fetchall()}
            cur.execute(f"SELECT source_comp_pk, target_comp_pk, rel_type FROM {cfg.table_relationships} "
                        f"WHERE repo_pk = %s;", (repo_pk,))
            rels = [(r["source_comp_pk"], r["target_comp_pk"], r["rel_type"]) for r in cur.fetchall()]
            cur.execute(f"SELECT DISTINCT ON (d.comp_pk) d.comp_pk, d.docstring_pk, d.docstring_kind "
                        f"FROM {cfg.table_docstrings} d JOIN {cfg.table_components} c USING (comp_pk) "
                        f"WHERE c.repo_pk = %s AND d.is_current "
                        f"ORDER BY d.comp_pk, (d.docstring_kind = 'generated') DESC;", (repo_pk,))
            docs = {r["comp_pk"]: (r["docstring_pk"], r["docstring_kind"]) for r in cur.fetchall()}
        rels = [(s, t, k) for s, t, k in rels if s in comps and t in comps]
        return cls(comps, rels, docs)

    # -- structure -----------------------------------------------------------

    def parent_class(self, pk: int) -> Optional[int]:
        for src, rtype in self.in_edges.get(pk, []):
            if rtype == "contains":
                return src
        return None

    def methods_of(self, class_pk: int) -> Set[int]:
        return {t for t, rtype in self.out_edges.get(class_pk, []) if rtype == "contains"}

    def init_of(self, class_pk: int) -> Optional[int]:
        cid = self.components[class_pk]["component_id"]
        return self.by_id.get(f"{cid}.__init__")

    def eligible_channels(self, pk: int) -> int:
        """Channels a component can appear in (mirrors Retriever eligibility)."""
        has_doc = pk in self.docstrings
        return 1 + int(bool(self.components[pk].get("has_vec"))) + 2 * int(has_doc)

    # -- call sites in source text ------------------------------------------

    def _call_sites(self, pk: int) -> List[Tuple[str, int, Optional[str]]]:
        """(name, absolute line, receiver) for every call in the component's own body."""
        if pk in self._raw_refs:
            return self._raw_refs[pk]
        comp = self.components[pk]
        out: List[Tuple[str, int, Optional[str]]] = []
        source = comp.get("source_code") or ""
        try:
            tree = ast.parse(textwrap.dedent(source))
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                line = comp["start_line"] + node.lineno - 1
                if isinstance(node.func, ast.Name):
                    out.append((node.func.id, line, None))
                elif isinstance(node.func, ast.Attribute):
                    recv = node.func.value
                    receiver = recv.id if isinstance(recv, ast.Name) else "?"
                    out.append((node.func.attr, line, receiver))
        self._raw_refs[pk] = out
        return out

    def references(self, pk: int, prefer: Set[int]) -> List[Reference]:
        """Call sites of a component resolved to components (see module docstring)."""
        refs: List[Reference] = []
        own_class = self.parent_class(pk)
        for name, line, receiver in self._call_sites(pk):
            if receiver == "self" and own_class is not None:
                cid = f"{self.components[own_class]['component_id']}.{name}"
                target = self.by_id.get(cid)
            else:
                target = self._resolve(name, prefer)
            if target is None or target == pk:
                continue
            refs.append(Reference(name, line, target))
            if self.components[target]["component_type"] == "class":
                init = self.init_of(target)              # Class(...) runs Class.__init__
                if init is not None:
                    refs.append(Reference(f"{name}.__init__", line, init))
        return refs

    def _resolve(self, name: str, prefer: Set[int]) -> Optional[int]:
        if name.startswith("__"):
            return None
        matches = self.by_short.get(name, [])
        if len(matches) == 1:
            return matches[0]
        preferred = [m for m in matches if m in prefer]
        return preferred[0] if len(preferred) == 1 else None


# ---------------------------------------------------------------------------
# Selection (pure)
# ---------------------------------------------------------------------------


def is_strong(c: Candidate, cfg: MappingConfig) -> bool:
    needed = max(1, math.ceil(cfg.strong_fraction * max(c.eligible, 1)))
    return c.top_channel_count(cfg.strong_rank) >= needed


def select_direct(cands: List[Candidate], graph: RepoGraph, cfg: MappingConfig
                  ) -> Tuple[List[Tuple[Candidate, str]], List[Tuple[Candidate, str]]]:
    """
    Returns (direct, containers): lists of (candidate, reason).  Containers are
    demoted classes that become supporting links.
    """
    if not cands:
        return [], []
    direct = [(cands[0], "rank 1")]
    for c in cands[1:cfg.max_direct_rank]:
        if is_strong(c, cfg):
            direct.append((c, f"rank {c.rank}; top-{cfg.strong_rank} in "
                              f"{c.top_channel_count(cfg.strong_rank)}/{c.eligible} eligible channels"))

    top = {c.comp_pk: c for c in cands[:cfg.strong_rank]}
    chosen = {c.comp_pk for c, _ in direct}
    containers, kept = [], []
    for c, reason in direct:
        if c.component_type == "class":
            own = [top[m] for m in graph.methods_of(c.comp_pk) if m in top]
            own.sort(key=lambda x: x.rank)
            if own:
                containers.append((c, f"class containing direct candidate {own[0].component_id}"))
                if cfg.promote_methods and own[0].comp_pk not in chosen:
                    kept.append((own[0], f"promoted: method of top-ranked class {c.component_id}"))
                    chosen.add(own[0].comp_pk)
                continue
        kept.append((c, reason))
    if not kept:                                   # never leave a requirement without a direct link
        kept, containers = [direct[0]], []
    kept.sort(key=lambda x: x[0].rank)
    return kept, containers


def expand_supporting(direct: List[Candidate], cands: List[Candidate], graph: RepoGraph,
                      cfg: MappingConfig) -> Dict[int, List[Dict[str, Any]]]:
    """
    One-hop supporting components for the direct set:
    {comp_pk: [edge evidence, ...]}.  Direct components are excluded.
    """
    cand_rank = {c.comp_pk: c.rank for c in cands}
    direct_pks = {c.comp_pk for c in direct}
    found: Dict[int, List[Dict[str, Any]]] = {}

    def cid(pk):
        return graph.components[pk]["component_id"]

    def add(pk, edge):
        if pk in direct_pks or pk not in graph.components:
            return
        found.setdefault(pk, [])
        if edge not in found[pk]:
            found[pk].append(edge)

    for d in direct:
        pk = d.comp_pk
        # DocAgent typed edges
        parent = graph.parent_class(pk)
        if parent is not None:
            add(parent, {"source": "docagent", "type": "contains", "from": cid(parent), "to": cid(pk)})
        for tgt, rtype in graph.out_edges.get(pk, []):
            if rtype in ("calls", "instantiates"):
                add(tgt, {"source": "docagent", "type": rtype, "from": cid(pk), "to": cid(tgt)})
                if rtype == "instantiates":
                    init = graph.init_of(tgt)
                    if init is not None:
                        add(init, {"source": "docagent", "type": "instantiates", "from": cid(pk),
                                   "to": cid(tgt), "then": "contains", "target": cid(init)})
        for src, rtype in graph.in_edges.get(pk, []):
            if rtype == "calls":
                add(src, {"source": "docagent", "type": "calls", "from": cid(src), "to": cid(pk)})
        # Call sites in source text
        if cfg.use_source_references:
            prefer = set(cand_rank)
            for ref in graph.references(pk, prefer):
                add(ref.target, {"source": "source", "type": "call_reference", "from": cid(pk),
                                 "to": cid(ref.target), "name": ref.name, "line": ref.line,
                                 "path": graph.components[pk]["relative_path"]})
            for other in cands:                    # callers among the candidates
                if other.comp_pk in direct_pks:
                    continue
                for ref in graph.references(other.comp_pk, prefer):
                    if ref.target == pk:
                        add(other.comp_pk, {"source": "source", "type": "call_reference",
                                            "from": cid(other.comp_pk), "to": cid(pk), "name": ref.name,
                                            "line": ref.line,
                                            "path": graph.components[other.comp_pk]["relative_path"]})

    if cfg.expand_requires_candidate:
        found = {pk: e for pk, e in found.items() if pk in cand_rank}
    ordered = sorted(found, key=lambda pk: (pk not in cand_rank, cand_rank.get(pk, 10**6),
                                            -len(found[pk]), cid(pk)))
    return {pk: found[pk] for pk in ordered[:cfg.max_supporting]}


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class Mapper:
    def __init__(self, store: Store, resolver: QueryResolver,
                 config: Optional[MappingConfig] = None) -> None:
        self._store = store
        self._resolver = resolver
        self.config = config or MappingConfig()

    def run(self, document_id: str, repo_id: str, *,
            candidates_version: str = CANDIDATES_VERSION,
            pipeline_version: str = PIPELINE_VERSION) -> Dict[str, Any]:
        doc_pk = self._resolver.document_pk(document_id)
        repo_pk = self._resolver.repo_pk(repo_id)
        graph = RepoGraph.load(self._store, repo_pk)
        by_req = self._load_candidates(doc_pk, candidates_version, graph)
        reqs = self._resolver._requirements("r.document_pk = %(doc)s", {"doc": doc_pk}, order="r.req_pk")
        summary = []
        for req in reqs:
            cands = by_req.get(req["req_pk"], [])
            summary.append(self.map_requirement(req, cands, graph, pipeline_version))
        return {
            "document_id": document_id, "repo_id": repo_id,
            "candidates_version": candidates_version, "pipeline_version": pipeline_version,
            "config": self.config.__dict__,
            "requirements": len(summary),
            "without_candidates": [s["requirement_id"] for s in summary if not s["direct"]],
            "links": {k: sum(len(s[k]) for s in summary) for k in ("direct", "supporting", "docstrings")},
            "per_requirement": summary,
        }

    def map_requirement(self, req: Dict[str, Any], cands: List[Candidate], graph: RepoGraph,
                        pipeline_version: str = PIPELINE_VERSION) -> Dict[str, Any]:
        cfg = self.config
        direct, containers = select_direct(cands, graph, cfg)
        direct_cands = [c for c, _ in direct]
        supporting = expand_supporting(direct_cands, cands, graph, cfg)
        cand_by_pk = {c.comp_pk: c for c in cands}
        for c, reason in containers:                       # demoted classes are supporting context
            supporting.setdefault(c.comp_pk, []).insert(0, {"source": "ranking", "type": "container",
                                                            "reason": reason})

        records = []
        for c, reason in direct:
            records.append(self._component_link(req, c.comp_pk, graph, pipeline_version,
                                                role="direct", reason=reason, cand=c))
        for pk, edges in supporting.items():
            records.append(self._component_link(req, pk, graph, pipeline_version, role="supporting",
                                                edges=edges, cand=cand_by_pk.get(pk)))
        doc_links = []
        for c in direct_cands:
            if c.via_docstring and c.comp_pk in graph.docstrings:
                ds_pk, kind = graph.docstrings[c.comp_pk]
                doc_links.append((c, ds_pk, kind))
                records.append({
                    "link_type": "requirement_to_docstring", "req_pk": req["req_pk"], "docstring_pk": ds_pk,
                    "status": "pending_verification", "retrieval_method": "hybrid",
                    "scores": {"channels": {k: v for k, v in c.channels.items() if k in _DOC_CHANNELS}},
                    "evidence": {"role": "direct", "component_id": c.component_id, "docstring_kind": kind},
                    "pipeline_version": pipeline_version})
                records.append({
                    "link_type": "docstring_to_component", "docstring_pk": ds_pk, "comp_pk": c.comp_pk,
                    "status": "supported", "retrieval_method": "ownership",
                    "evidence": {"basis": "ownership", "component_id": c.component_id},
                    "pipeline_version": pipeline_version})

        with self._store.transaction():
            link_pks = self._store.upsert_trace_links(records)
            removed = self._store.delete_stale_candidates(
                req["req_pk"], pipeline_version,
                [pk for key, pk in link_pks.items() if key[1] == req["req_pk"]],
                statuses=("pending_verification",))
        return {
            "requirement_id": req["requirement_id"], "req_pk": req["req_pk"],
            "direct": [c.component_id for c, _ in direct],
            "supporting": [graph.components[pk]["component_id"] for pk in supporting],
            "docstrings": [c.component_id for c, _, _ in doc_links],
            "stale_removed": removed,
        }

    def subgraph(self, document_id: str, requirement_id: str,
                 pipeline_version: str = PIPELINE_VERSION) -> Dict[str, Any]:
        """Nodes (linked components with roles) and edges among them."""
        doc_pk = self._resolver.document_pk(document_id)
        links = [l for l in self._store.get_trace_links(document_pk=doc_pk, pipeline_version=pipeline_version,
                                                        link_type="requirement_to_component")
                 if l["requirement_id"] == requirement_id]
        nodes = [{"component_id": l["component_id"], "role": l["evidence"].get("role"),
                  "status": l["status"], "path": l["evidence"].get("path"),
                  "lines": [l["evidence"].get("start_line"), l["evidence"].get("end_line")],
                  "reason": l["evidence"].get("reason"), "rank": l["evidence"].get("rank")} for l in links]
        pks = [l["comp_pk"] for l in links]
        edges: List[Dict[str, Any]] = []
        if pks:
            cfg = self._store._cfg
            with self._store.transaction() as cur:
                cur.execute(f"SELECT a.component_id AS src, b.component_id AS tgt, r.rel_type "
                            f"FROM {cfg.table_relationships} r "
                            f"JOIN {cfg.table_components} a ON a.comp_pk = r.source_comp_pk "
                            f"JOIN {cfg.table_components} b ON b.comp_pk = r.target_comp_pk "
                            f"WHERE r.source_comp_pk = ANY(%(p)s) AND r.target_comp_pk = ANY(%(p)s) "
                            f"ORDER BY 1, 2, 3;", {"p": pks})
                edges = [{"from": r["src"], "to": r["tgt"], "type": r["rel_type"], "source": "docagent"}
                         for r in cur.fetchall()]
        for l in links:
            for e in l["evidence"].get("edges", []):
                if e.get("source") == "source" and e not in edges:
                    edges.append(e)
        return {"requirement_id": requirement_id, "pipeline_version": pipeline_version,
                "nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------

    def _component_link(self, req, pk, graph, pipeline_version, *, role, reason=None,
                        edges=None, cand: Optional[Candidate] = None) -> Dict[str, Any]:
        comp = graph.components[pk]
        evidence = {"role": role, "component_id": comp["component_id"],
                    "component_type": comp["component_type"], "path": comp["relative_path"],
                    "start_line": comp["start_line"], "end_line": comp["end_line"],
                    "rank": cand.rank if cand else None}
        if reason:
            evidence["reason"] = reason
        if edges:
            evidence["edges"] = edges
        scores = {"candidate_rank": cand.rank if cand else None}
        if cand:
            scores.update(fused=cand.fused, channels=cand.channels, eligible_channels=cand.eligible)
        return {"link_type": "requirement_to_component", "req_pk": req["req_pk"], "comp_pk": pk,
                "status": "pending_verification",
                "retrieval_method": "hybrid" if role == "direct" else "graph",
                "scores": scores, "evidence": evidence, "pipeline_version": pipeline_version}

    def _load_candidates(self, doc_pk: int, version: str, graph: RepoGraph) -> Dict[int, List[Candidate]]:
        out: Dict[int, List[Candidate]] = {}
        for l in self._store.get_trace_links(document_pk=doc_pk, pipeline_version=version,
                                             link_type="requirement_to_component"):
            if l["comp_pk"] not in graph.components:       # other repo or no longer current
                continue
            comp = graph.components[l["comp_pk"]]
            scores, ev = l["scores"] or {}, l["evidence"] or {}
            out.setdefault(l["req_pk"], []).append(Candidate(
                comp_pk=l["comp_pk"], component_id=comp["component_id"],
                component_type=comp["component_type"], rank=ev.get("rank") or 10**6,
                channels=scores.get("channels") or {},
                eligible=scores.get("eligible_channels") or graph.eligible_channels(l["comp_pk"]),
                fused=scores.get("fused")))
        for cands in out.values():
            cands.sort(key=lambda c: c.rank)
        return out
