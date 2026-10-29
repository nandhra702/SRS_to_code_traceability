"""
Cross-entity retrieval (Phase 2).

Six independent channels — {requirement, docstring, component} × {lexical, vector}:

  lexical   PostgreSQL full-text search over each entity's search_tsv column.
            The query is turned into an OR of its stemmed lexemes (identifiers
            split exactly as at index time) and ranked with ts_rank.
  vector    pgvector cosine similarity over the entity's own embedding, only
            against rows embedded by the configured model version.

Raw lexical and vector scores are not comparable, so channels are combined
with Reciprocal Rank Fusion: fused = Σ 1 / (rrf_k + rank).  Every hit keeps
each channel's rank and raw score, so contributions stay observable.

Results are typed per entity.  `components` is the implementation-candidate
view: code hits and docstring hits (attributed to their owning component)
fused per component — the ranking Phase 3 mapping starts from.

No LLM is involved anywhere in retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .embedder import Embedder
from .query import ResolvedQuery
from .schema import TS_CONFIG, split_identifiers_sql
from .store import DimensionMismatchError, Store

ENTITIES = ("requirement", "docstring", "component")
METHODS = ("hybrid", "lexical", "vector")
RRF_K = 60
# ts_rank normalisation 1: divide by 1 + log(document length) so long code
# bodies do not win just by containing more words.
TS_RANK_NORMALISATION = 1


@dataclass
class Hit:
    """One retrieved entity with per-channel provenance."""

    entity: str                    # requirement | docstring | component
    pk: int
    external_id: str               # requirement_id or component_id
    data: Dict[str, Any]           # entity-specific fields (path, lines, snippet, …)
    channels: Dict[str, Dict[str, float]] = field(default_factory=dict)  # channel → {rank, score}
    score: float = 0.0             # fused score (or the single channel's score)

    def to_dict(self) -> Dict[str, Any]:
        return {"entity": self.entity, "pk": self.pk, "id": self.external_id,
                "score": round(self.score, 6), "channels": self.channels, **self.data}


@dataclass
class SearchResponse:
    query: ResolvedQuery
    method: str
    results: Dict[str, List[Hit]] = field(default_factory=dict)
    components: List[Hit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query.describe(),
            "method": self.method,
            "results": {e: [h.to_dict() for h in hits] for e, hits in self.results.items()},
            "components": [h.to_dict() for h in self.components],
        }


def rrf_fuse(channel_lists: Dict[str, List[Hit]], *, rrf_k: int = RRF_K,
             key=lambda h: (h.entity, h.pk)) -> List[Hit]:
    """
    Fuse ranked lists with Reciprocal Rank Fusion.  Each input hit's own
    channels are carried over; ties break on external_id for stable output.
    """
    fused: Dict[Any, Hit] = {}
    for channel, hits in channel_lists.items():
        for rank, hit in enumerate(hits, start=1):
            k = key(hit)
            if k not in fused:
                fused[k] = Hit(hit.entity, hit.pk, hit.external_id, dict(hit.data))
            target = fused[k]
            for field_name, value in hit.data.items():   # keep fields from every channel
                target.data.setdefault(field_name, value)
            target.channels.update(hit.channels)
            target.score += 1.0 / (rrf_k + rank)
    return sorted(fused.values(), key=lambda h: (-h.score, h.external_id))


class Retriever:
    """Runs lexical / vector / hybrid search for a ResolvedQuery."""

    def __init__(self, store: Store, embedder: Optional[Embedder] = None, *,
                 rrf_k: int = RRF_K) -> None:
        self._store = store
        self._cfg = store._cfg
        self._embedder = embedder
        self._rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def search(
        self,
        query: ResolvedQuery,
        *,
        method: str = "hybrid",
        entities: Sequence[str] = ENTITIES,
        top_k: int = 10,
        candidate_k: int = 50,
    ) -> SearchResponse:
        """
        Search every requested entity type.  Each channel returns up to
        candidate_k hits; fused / final lists are cut to top_k.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        bad = set(entities) - set(ENTITIES)
        if bad:
            raise ValueError(f"Unknown entity type(s) {sorted(bad)}")
        response = SearchResponse(query=query, method=method)
        if query.status != "ok":
            return response

        channels = self.channels(query, method=method, entities=entities, candidate_k=candidate_k)
        for entity in entities:
            per_entity = {name: hits for name, hits in channels.items()
                          if name.startswith(entity + "_")}
            response.results[entity] = self._combine(per_entity)[:top_k]
        if {"docstring", "component"} & set(entities):
            response.components = self.fuse_components(channels)[:top_k]
        return response

    def channels(
        self,
        query: ResolvedQuery,
        *,
        method: str = "hybrid",
        entities: Sequence[str] = ENTITIES,
        candidate_k: int = 50,
    ) -> Dict[str, List[Hit]]:
        """Raw ranked lists per channel, e.g. {'component_lexical': [...], ...}."""
        out: Dict[str, List[Hit]] = {}
        lexemes = self._tsquery(query.query_text) if method in ("hybrid", "lexical") else ""
        vector = self._query_vector(query) if method in ("hybrid", "vector") else None
        for entity in entities:
            if method in ("hybrid", "lexical"):
                out[f"{entity}_lexical"] = (
                    self._lexical(entity, lexemes, query, candidate_k) if lexemes else [])
            if method in ("hybrid", "vector"):
                out[f"{entity}_vector"] = self._vector(entity, vector, query, candidate_k)
        return out

    def fuse_components(self, channels: Dict[str, List[Hit]],
                        names: Optional[Iterable[str]] = None) -> List[Hit]:
        """
        Fuse component and docstring channels into one ranking of components.
        A docstring hit counts for its owning component (best rank per
        component per channel).

        With several channels the fused score is the RRF sum divided by the
        number of channels the component is *eligible* for: a component with
        no docstring (DocAgent skips __init__) or no code vector cannot appear
        in those channels, and that absence must not count against it.  On the
        library dataset a plain RRF sum dropped every __init__ out of the top
        10; averaging over eligible channels fixed that (MRR 0.864 → 0.908).
        """
        names = list(names) if names is not None else [
            n for n in channels if n.startswith(("component_", "docstring_"))]
        lists: Dict[str, List[Hit]] = {}
        for name in names:
            seen, ranked = set(), []
            for hit in channels.get(name, []):
                comp = self._as_component(hit)
                if comp.pk not in seen:
                    seen.add(comp.pk)
                    ranked.append(comp)
            lists[name] = ranked
        fused = self._combine(lists, key=lambda h: h.pk)
        if len(lists) <= 1:
            return fused
        eligible = self.eligible_channel_counts([h.pk for h in fused], names)
        for hit in fused:
            hit.data["eligible_channels"] = eligible.get(hit.pk, len(names))
            hit.score /= max(hit.data["eligible_channels"], 1)
        return sorted(fused, key=lambda h: (-h.score, h.external_id))

    def eligible_channel_counts(self, comp_pks: List[int], names: Sequence[str]) -> Dict[int, int]:
        """How many of the fused channels each component could appear in."""
        if not comp_pks:
            return {}
        cfg = self._cfg
        mv = self._get_embedder().model_version
        sql = (f"SELECT c.comp_pk, "
               f"  (c.embedding IS NOT NULL AND c.embed_model_version = %(mv)s) AS code_vec, "
               f"  EXISTS (SELECT 1 FROM {cfg.table_docstrings} d "
               f"          WHERE d.comp_pk = c.comp_pk AND d.is_current) AS doc_lex, "
               f"  EXISTS (SELECT 1 FROM {cfg.table_docstrings} d WHERE d.comp_pk = c.comp_pk "
               f"          AND d.is_current AND d.embedding IS NOT NULL "
               f"          AND d.embed_model_version = %(mv)s) AS doc_vec "
               f"FROM {cfg.table_components} c WHERE c.comp_pk = ANY(%(pks)s);")
        with self._store.transaction() as cur:
            cur.execute(sql, {"mv": mv, "pks": comp_pks})
            rows = cur.fetchall()
        flags = {"component_lexical": lambda r: True, "component_vector": lambda r: r["code_vec"],
                 "docstring_lexical": lambda r: r["doc_lex"], "docstring_vector": lambda r: r["doc_vec"]}
        return {r["comp_pk"]: sum(1 for n in names if flags.get(n, lambda _: True)(r)) for r in rows}

    # ------------------------------------------------------------------
    # Fusion helpers
    # ------------------------------------------------------------------

    def _combine(self, lists: Dict[str, List[Hit]], key=None) -> List[Hit]:
        """One list: return it as-is (raw score kept); several: RRF."""
        nonempty = {k: v for k, v in lists.items()}
        if len(nonempty) == 1:
            return list(next(iter(nonempty.values())))
        kwargs = {"rrf_k": self._rrf_k}
        if key is not None:
            kwargs["key"] = key
        return rrf_fuse(nonempty, **kwargs)

    @staticmethod
    def _as_component(hit: Hit) -> Hit:
        if hit.entity == "component":
            return hit
        d = hit.data
        return Hit("component", d["comp_pk"], d["component_id"], {
            k: d[k] for k in ("component_type", "repo_pk", "path", "start_line", "end_line")
        } | {"via_docstring": d["docstring_kind"]}, channels=dict(hit.channels), score=hit.score)

    # ------------------------------------------------------------------
    # Query preparation
    # ------------------------------------------------------------------

    def _tsquery(self, text: str) -> str:
        """OR-query of the text's stemmed lexemes, or '' when it has none."""
        if not text or not text.strip():
            return ""
        sql = (
            "SELECT array_to_string(ARRAY("
            "  SELECT quote_literal(lex) FROM unnest(tsvector_to_array("
            f"    to_tsvector('{TS_CONFIG}', {split_identifiers_sql('%(text)s')}))) AS lex"
            "  ORDER BY lex), ' | ') AS q;"
        )
        with self._store.transaction() as cur:
            cur.execute(sql, {"text": text})
            return cur.fetchone()["q"] or ""

    def _query_vector(self, query: ResolvedQuery) -> List[float]:
        embedder = self._get_embedder()
        req = query.requirement
        # Reuse the stored requirement vector when it came from this model.
        if req is not None and req.get("embedding") is not None \
                and req.get("embed_model_version") == embedder.model_version:
            stored = req["embedding"]   # pgvector returns a Vector (or ndarray) object
            vec = [float(x) for x in (stored.to_list() if hasattr(stored, "to_list") else stored)]
        else:
            vec = embedder.embed(query.query_text).vector
        dim = self._store.get_schema_embed_dim()
        if dim is not None and len(vec) != dim:
            raise DimensionMismatchError(
                f"Query vector length {len(vec)} does not match schema dimension {dim}.")
        return vec

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self._cfg)
        return self._embedder

    # ------------------------------------------------------------------
    # Channel SQL
    # ------------------------------------------------------------------

    def _entity_sql(self, entity: str, query: ResolvedQuery) -> Tuple[str, str, List[str], Dict[str, Any]]:
        """(select-columns, from-clause, where-conditions, params) for an entity."""
        cfg = self._cfg
        params: Dict[str, Any] = {}
        if entity == "requirement":
            cols = ("t.req_pk AS pk, t.requirement_id AS external_id, t.document_pk, "
                    "t.title, t.type, t.subtype, left(t.raw_text, 300) AS snippet")
            frm = f"{cfg.table_requirements} t"
            where = ["t.is_current"]
            if query.document_pk is not None:
                where.append("t.document_pk = %(document_pk)s")
                params["document_pk"] = query.document_pk
            if query.requirement is not None:          # don't return the query itself
                where.append("t.req_pk <> %(self_pk)s")
                params["self_pk"] = query.requirement["req_pk"]
        elif entity == "docstring":
            cols = ("t.docstring_pk AS pk, c.component_id AS external_id, t.docstring_kind, "
                    "c.comp_pk, c.component_id, c.component_type, c.repo_pk, "
                    "c.relative_path AS path, c.start_line, c.end_line, "
                    "left(t.content, 300) AS snippet")
            frm = f"{cfg.table_docstrings} t JOIN {cfg.table_components} c ON c.comp_pk = t.comp_pk"
            where = ["t.is_current", "c.is_current"]
            if query.repo_pk is not None:
                where.append("c.repo_pk = %(repo_pk)s")
                params["repo_pk"] = query.repo_pk
        elif entity == "component":
            cols = ("t.comp_pk AS pk, t.component_id AS external_id, t.component_type, t.repo_pk, "
                    "t.relative_path AS path, t.start_line, t.end_line, t.signature")
            frm = f"{cfg.table_components} t"
            where = ["t.is_current"]
            if query.repo_pk is not None:
                where.append("t.repo_pk = %(repo_pk)s")
                params["repo_pk"] = query.repo_pk
        else:
            raise ValueError(f"Unknown entity {entity!r}")
        return cols, frm, where, params

    def _lexical(self, entity: str, lexemes: str, query: ResolvedQuery, k: int) -> List[Hit]:
        cols, frm, where, params = self._entity_sql(entity, query)
        sql = (f"SELECT {cols}, ts_rank(t.search_tsv, q, {TS_RANK_NORMALISATION}) AS score "
               f"FROM {frm}, to_tsquery('{TS_CONFIG}', %(tsq)s) AS q "
               f"WHERE {' AND '.join(where + ['t.search_tsv @@ q'])} "
               f"ORDER BY score DESC, external_id, pk LIMIT %(k)s;")
        return self._run(entity, "lexical", sql, {**params, "tsq": lexemes, "k": k})

    def _vector(self, entity: str, vec: List[float], query: ResolvedQuery, k: int) -> List[Hit]:
        cols, frm, where, params = self._entity_sql(entity, query)
        where = where + ["t.embedding IS NOT NULL", "t.embed_model_version = %(mv)s"]
        sql = (f"SELECT {cols}, 1 - (t.embedding <=> %(vec)s::vector) AS score "
               f"FROM {frm} WHERE {' AND '.join(where)} "
               f"ORDER BY t.embedding <=> %(vec)s::vector, external_id, pk LIMIT %(k)s;")
        return self._run(entity, "vector", sql, {
            **params, "vec": vec, "mv": self._get_embedder().model_version, "k": k})

    def _run(self, entity: str, method: str, sql: str, params: Dict[str, Any]) -> List[Hit]:
        channel = f"{entity}_{method}"
        with self._store.transaction() as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
        hits = []
        for rank, row in enumerate(rows, start=1):
            pk, ext, score = row.pop("pk"), row.pop("external_id"), float(row.pop("score"))
            hits.append(Hit(entity, pk, ext, row,
                            channels={channel: {"rank": rank, "score": round(score, 6)}},
                            score=score))
        return hits
