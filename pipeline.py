"""
Pipeline — orchestrates load → embed → store.

ingest_srs()           one SRS output file  → document + requirements
ingest_graph()         one DocAgent graph   → repository + components +
                       docstrings + typed relationships
run()                  both of the above
rebuild_embeddings()   re-embed rows whose vector is missing, was made by a
                       different model version, or whose text has changed

Each ingest is a single database transaction: it either fully succeeds or
leaves the database untouched.  Re-running on the same inputs is idempotent.
Every ingest returns a report dict (counts, skipped empty texts, truncated
texts, skipped relationships) so nothing is dropped silently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import TraceabilityConfig
from .embedder import Embedder
from .loader import GraphLoader, SRSLoader, read_post_run_docstrings
from .store import Store
from .texts import component_code_text, docstring_text, requirement_text

logger = logging.getLogger(__name__)

Report = Dict[str, Any]


class Pipeline:
    """
    Usage::

        cfg = TraceabilityConfig()          # reads env vars
        p = Pipeline(cfg)
        p.ingest_srs("…/Basic_SRS_Document_requirements.txt")
        p.ingest_graph("…/raw_test_repo_simple_dependency_graph.json",
                       source_root="…/raw_test_repo_simple")
    """

    def __init__(self, config: Optional[TraceabilityConfig] = None) -> None:
        self._cfg = config or TraceabilityConfig()
        self._embedder: Optional[Embedder] = None

    # ------------------------------------------------------------------
    # SRS
    # ------------------------------------------------------------------

    def ingest_srs(
        self,
        srs_path: str | Path,
        *,
        document_id: Optional[str] = None,
        embed: bool = True,
    ) -> Report:
        """Load an SRS Agent output file into the database."""
        srs = SRSLoader(srs_path).load()
        doc_record = srs.document_record(document_id)
        logger.info("SRS loaded: %d requirements (%s).",
                    len(srs.requirements), doc_record["document_id"])

        report: Report = {"document_id": doc_record["document_id"],
                          "requirements": len(srs.requirements)}
        embeddings = {}
        if embed:
            embedder = self._get_embedder()
            embeddings = embedder.embed_batch(
                {r["requirement_id"]: requirement_text(r) for r in srs.requirements}
            )
            report["embedded"] = len(embeddings)
            report["skipped_empty_text"] = list(embedder.skipped_empty)
            report["truncated_text"] = list(embedder.truncated)

        with Store(self._cfg) as store:
            store.init_schema(self._resolve_dim(store, embed))
            with store.transaction():
                doc_pk = store.upsert_document(doc_record)
                store.upsert_requirements(doc_pk, srs.requirements, embeddings or None)
        report["document_pk"] = doc_pk
        logger.info("SRS ingest report: %s", report)
        return report

    # ------------------------------------------------------------------
    # DocAgent graph
    # ------------------------------------------------------------------

    def ingest_graph(
        self,
        graph_path: str | Path,
        *,
        repo_id: Optional[str] = None,
        repo_path: Optional[str] = None,
        repo_snapshot: Optional[str] = None,
        source_root: Optional[str | Path] = None,
        embed: bool = True,
    ) -> Report:
        """
        Load a DocAgent dependency graph into the database.

        Args:
            repo_id:       Logical repository name (default: graph file stem).
            repo_path:     Repository path, stored for identity.
            repo_snapshot: Git commit SHA / tag at parse time.
            source_root:   The repository *after* DocAgent's generation run.
                           When given, docstrings DocAgent wrote are recovered
                           as 'generated' records (the graph JSON is saved
                           before generation and never contains them).
            embed:         Compute embeddings (False: schema/data only).
        """
        graph = GraphLoader(graph_path).load()
        comps = graph.components
        post_run = read_post_run_docstrings(comps, source_root) if source_root else None
        docstrings = graph.docstring_records(post_run)
        logger.info("%r; %d docstring record(s).", graph, len(docstrings))

        report: Report = {
            "repo_id": repo_id or graph.default_repo_id(),
            "graph_format": graph.format,
            "components": len(comps),
            "docstrings": {
                "existing":  sum(d["kind"] == "existing" for d in docstrings),
                "generated": sum(d["kind"] == "generated" for d in docstrings),
            },
            "malformed_relationships": graph.malformed_relationships,
        }
        if graph.format == "legacy":
            report["warning"] = ("Legacy graph: no source code, signatures or typed "
                                 "relationships; code embeddings are skipped.")
        if source_root is not None:
            report["post_run_components_located"] = len(post_run)

        code_emb, doc_emb = {}, {}
        if embed:
            embedder = self._get_embedder()
            max_chars = self._cfg.embed_code_max_chars
            code_emb = embedder.embed_batch(
                {cid: component_code_text(c, max_chars) for cid, c in comps.items()}
            )
            report["code_embedded"] = len(code_emb)
            report["code_skipped_empty_text"] = list(embedder.skipped_empty)
            report["code_truncated_text"] = list(embedder.truncated)
            doc_emb = embedder.embed_batch({
                (d["component_id"], d["kind"]): docstring_text(comps[d["component_id"]], d["content"])
                for d in docstrings
            })
            report["docstrings_embedded"] = len(doc_emb)
            report["docstring_truncated_text"] = [list(k) for k in embedder.truncated]

        with Store(self._cfg) as store:
            store.init_schema(self._resolve_dim(store, embed))
            with store.transaction():
                repo_pk = store.upsert_repository({
                    "repo_id":       report["repo_id"],
                    "repo_path":     repo_path,
                    "repo_snapshot": repo_snapshot,
                    "graph_path":    str(Path(graph_path).resolve()),
                    "graph_format":  graph.format,
                })
                comp_pks = store.upsert_components(
                    repo_pk, list(comps.values()), code_emb or None
                )
                store.upsert_docstrings(
                    comp_pks, docstrings, doc_emb or None,
                    observed_kinds=("existing", "generated") if source_root else ("existing",),
                )
                inserted, skipped = store.replace_relationships(
                    repo_pk, comp_pks, graph.relationships
                )
        report.update(repo_pk=repo_pk, relationships=inserted, relationships_skipped=skipped)
        logger.info("Graph ingest report: %s", report)
        return report

    # ------------------------------------------------------------------
    # Both
    # ------------------------------------------------------------------

    def run(
        self,
        graph_path: str | Path,
        srs_path: str | Path,
        *,
        embed: bool = True,
        repo_id: Optional[str] = None,
        repo_path: Optional[str] = None,
        repo_snapshot: Optional[str] = None,
        source_root: Optional[str | Path] = None,
        document_id: Optional[str] = None,
        build_vector_indexes: bool = False,
    ) -> Dict[str, Report]:
        """Ingest a graph and an SRS file (two independent transactions)."""
        reports = {
            "graph": self.ingest_graph(
                graph_path, repo_id=repo_id, repo_path=repo_path,
                repo_snapshot=repo_snapshot, source_root=source_root, embed=embed,
            ),
            "srs": self.ingest_srs(srs_path, document_id=document_id, embed=embed),
        }
        if build_vector_indexes:
            with Store(self._cfg) as store:
                store.build_vector_indexes()
        return reports

    # ------------------------------------------------------------------
    # Rebuild stale embeddings
    # ------------------------------------------------------------------

    def rebuild_embeddings(self) -> Report:
        """
        Re-embed every current row whose vector is missing, was produced by a
        different model version, or whose embedding text no longer hashes to
        the stored embed_text_hash.  Rows whose text is now empty have their
        vector cleared.  Safe to run repeatedly — fresh rows are untouched.
        """
        embedder = self._get_embedder()
        version = embedder.model_version
        max_chars = self._cfg.embed_code_max_chars
        builders = {
            "requirement": ("req_pk", requirement_text),
            "component":   ("comp_pk", lambda r: component_code_text(r, max_chars)),
            "docstring":   ("docstring_pk", lambda r: docstring_text(r, r["content"])),
        }
        report: Report = {}
        with Store(self._cfg) as store:
            store.init_schema(embedder.dim)
            for entity, (pk_col, build) in builders.items():
                stale: Dict[int, str] = {}
                cleared = 0
                for row in store.get_embedding_rows(entity):
                    text = build(row)
                    if not text.strip():
                        if row["has_embedding"]:
                            store.update_embedding(entity, row[pk_col], None)
                            cleared += 1
                        continue
                    if (not row["has_embedding"]
                            or row["embed_model_version"] != version
                            or row["embed_text_hash"] != Embedder.text_hash(text)):
                        stale[row[pk_col]] = text
                results = embedder.embed_batch(stale)
                with store.transaction():
                    for pk, result in results.items():
                        store.update_embedding(entity, pk, result)
                report[entity] = {"rebuilt": len(results), "cleared": cleared}
        logger.info("Embedding rebuild report: %s", report)
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def load_only(
        self, graph_path: str | Path, srs_path: str | Path
    ) -> Tuple[GraphLoader, SRSLoader]:
        """Return both loaders already .load()ed, without any DB interaction."""
        return GraphLoader(graph_path).load(), SRSLoader(srs_path).load()

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self._cfg)
        return self._embedder

    def _resolve_dim(self, store: Store, embed: bool) -> int:
        """
        The vector dimension to initialise/verify the schema with.

        With embedding: the model's real dimension.  Without: EMBED_DIM, else
        the dimension already recorded in the database — never a guess.
        """
        if embed:
            return self._get_embedder().dim
        dim = self._cfg.embed_dim or store.get_schema_embed_dim()
        if dim is None:
            raise ValueError(
                "Embedding dimension unknown: set EMBED_DIM, or run once with "
                "embeddings enabled (or init-schema) so the schema records it."
            )
        return dim
