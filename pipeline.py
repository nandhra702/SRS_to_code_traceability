"""
Pipeline — orchestrates the full load → embed → store flow.

Steps
-----
1. Load DocAgent dependency graph (legacy or V1 JSON)
2. Load SRS requirements JSON
3. Resolve embedding dimension from the local sentence-transformers model
4. Initialise schema (idempotent — safe to call repeatedly)
5. Record schema migration version
6. Upsert SRS document record, then requirements with embeddings
7. Upsert code components with repo identity and embeddings
8. Upsert docstrings (original only at ingest time; generated added later)
9. Upsert typed relationships
10. Optionally build IVFFlat ANN indexes

Re-running the pipeline on the same inputs is safe — all writes are
ON CONFLICT DO UPDATE (idempotent).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import TraceabilityConfig
from .embedder import Embedder
from .loader import GraphLoader, SRSLoader
from .schema import SCHEMA_VERSION
from .store import Store

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates the full traceability ingestion flow.

    Usage::

        from traceability import Pipeline, TraceabilityConfig

        cfg = TraceabilityConfig()          # reads env vars
        Pipeline(cfg).run(
            graph_path="…/raw_test_repo_simple_dependency_graph.json",
            srs_path="…/Basic_SRS_Document_requirements.txt",
        )
    """

    def __init__(self, config: Optional[TraceabilityConfig] = None) -> None:
        self._cfg = config or TraceabilityConfig()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        graph_path: str | Path,
        srs_path: str | Path,
        *,
        embed: bool = True,
        repo_path: Optional[str] = None,
        repo_snapshot: Optional[str] = None,
        build_vector_indexes: bool = False,
    ) -> None:
        """
        Execute the full ingestion pipeline.

        Args:
            graph_path:           DocAgent dependency graph JSON path.
            srs_path:             SRS Agent output JSON path.
            embed:                Compute and store embeddings (True by default).
                                  Set False for schema-only or dry-run modes.
            repo_path:            Absolute path to the analysed repository
                                  (stored for snapshot identity).
            repo_snapshot:        Git commit SHA or tag at parse time.
            build_vector_indexes: Build IVFFlat indexes after loading.
                                  Only useful once the table has enough rows.
        """
        # Step 1 — Load
        logger.info("Loading dependency graph: %s", graph_path)
        graph = GraphLoader(graph_path).load()
        logger.info(
            "Graph loaded: %d components, %d relationships.",
            len(graph.components), len(graph.relationships),
        )

        logger.info("Loading SRS requirements: %s", srs_path)
        srs = SRSLoader(srs_path).load()
        logger.info("SRS loaded: %d requirements.", len(srs.requirements))

        # Step 2 — Resolve embed dim and compute embeddings
        comp_embeddings = {}
        req_embeddings  = {}
        embed_dim: int

        if embed:
            embedder  = Embedder(self._cfg)
            embed_dim = embedder.dim
            logger.info("Embedding dimension: %d (model=%s)", embed_dim, embedder.model_name)

            logger.info("Embedding %d code components…", len(graph.components))
            comp_embeddings = embedder.embed_batch(graph.as_embedding_texts())

            logger.info("Embedding %d requirements…", len(srs.requirements))
            req_embeddings = embedder.embed_batch(srs.as_embedding_texts())
        else:
            embed_dim = self._cfg.embed_dim or 384
            logger.info("Embedding disabled; using embed_dim=%d for schema.", embed_dim)

        # Step 3 — Persist
        with Store(self._cfg) as store:
            store.init_schema(embed_dim)
            store.record_migration(SCHEMA_VERSION, "Phase 1 initial schema")

            # SRS document
            doc_record = srs.document_record()
            doc_pk = store.upsert_document(doc_record)

            # Requirements
            store.upsert_requirements(
                doc_pk,
                srs.requirements,
                req_embeddings or None,
            )

            # Components
            store.upsert_components(
                list(graph.components.values()),
                comp_embeddings or None,
                repo_path=repo_path,
                repo_snapshot=repo_snapshot,
            )

            # Docstrings — ingest original docstrings from graph at this stage
            for comp in graph.components.values():
                if comp.get("has_docstring") and comp.get("docstring", "").strip():
                    store.upsert_docstrings(
                        comp["id"],
                        original=comp["docstring"],
                    )

            # Relationships
            inserted, skipped = store.upsert_relationships(graph.relationships)
            logger.info(
                "Relationships: %d inserted, %d skipped.", inserted, skipped
            )

            if build_vector_indexes:
                store.build_vector_indexes()

        logger.info("Pipeline complete.")

    # ------------------------------------------------------------------
    # Rebuild stale embeddings
    # ------------------------------------------------------------------

    def rebuild_embeddings(self) -> None:
        """
        Re-embed all requirements and components whose embedding is NULL
        or was produced with a different model version.

        Safe to run repeatedly — only stale rows are touched.
        """
        embedder = Embedder(self._cfg)
        model_version = embedder.model_version

        with Store(self._cfg) as store:
            # Requirements
            stale_reqs = store.get_stale_requirement_embeddings(model_version)
            logger.info(
                "Rebuilding embeddings for %d stale requirement(s).", len(stale_reqs)
            )
            for row in stale_reqs:
                text = " ".join(filter(None, [
                    row.get("retrieval_text", ""),
                    row.get("title", ""),
                    row.get("normalized_text", ""),
                ]))
                result = embedder.embed(text)
                store.update_requirement_embedding(row["req_pk"], result)

            # Components
            stale_comps = store.get_stale_component_embeddings(model_version)
            logger.info(
                "Rebuilding embeddings for %d stale component(s).", len(stale_comps)
            )
            for row in stale_comps:
                parts = [
                    row.get("signature") or "",
                    row.get("docstring") or "",
                    (row.get("source_code") or "")[:2000],
                ]
                text = "\n\n".join(p for p in parts if p)
                result = embedder.embed(text)
                store.update_component_embedding(row["component_id"], result)

        logger.info("Embedding rebuild complete.")

    # ------------------------------------------------------------------
    # Convenience: load without touching the DB
    # ------------------------------------------------------------------

    def load_only(
        self,
        graph_path: str | Path,
        srs_path: str | Path,
    ) -> tuple[GraphLoader, SRSLoader]:
        """Return both loaders already .load()ed, without any DB interaction."""
        return GraphLoader(graph_path).load(), SRSLoader(srs_path).load()
