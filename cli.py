"""
traceability CLI — commands for schema management and data ingestion.

Commands
--------
init-schema          Create extension + all tables + btree indexes.
ingest-requirements  Load an SRS output JSON into PostgreSQL.
ingest-graph         Load a DocAgent dependency graph JSON into PostgreSQL.
rebuild-embeddings   Re-embed rows whose embedding is NULL or stale.

Usage examples
--------------
# Initialise the schema (idempotent):
python -m traceability.cli init-schema

# Ingest requirements from an SRS output file:
python -m traceability.cli ingest-requirements \\
    SRS_agent/SRSAgent_Major_Project/output/Basic_SRS_Document_requirements.txt

# Ingest a DocAgent graph (auto-detects legacy vs V1 format):
python -m traceability.cli ingest-graph \\
    DocAgent_Major_Project/output/dependency_graphs/raw_test_repo_simple_dependency_graph.json \\
    --repo-path DocAgent_Major_Project/data/raw_test_repo_simple \\
    --repo-snapshot HEAD

# Rebuild stale embeddings after a model change:
python -m traceability.cli rebuild-embeddings

All database credentials are read from environment variables:
  TRACEABILITY_PG_DSN        — full libpq DSN (takes precedence)
  TRACEABILITY_PG_HOST       — default: localhost
  TRACEABILITY_PG_PORT       — default: 5432
  TRACEABILITY_PG_DBNAME     — default: traceability_dev
  TRACEABILITY_PG_USER       — default: postgres
  TRACEABILITY_PG_PASSWORD   — default: (empty)

Embedding model:
  EMBED_MODEL                — default: sentence-transformers/all-MiniLM-L6-v2
  EMBED_MODEL_VERSION        — default: same as EMBED_MODEL
  EMBED_DIM                  — default: auto-detected from model
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import TraceabilityConfig
from .embedder import Embedder
from .loader import GraphLoader, SRSLoader
from .pipeline import Pipeline
from .schema import SCHEMA_VERSION
from .store import Store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("traceability.cli")


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_init_schema(args: argparse.Namespace, cfg: TraceabilityConfig) -> int:
    """Create extension, tables, and btree indexes.  Never drops anything."""
    embedder = Embedder(cfg)
    embed_dim = embedder.dim
    with Store(cfg) as store:
        store.init_schema(embed_dim)
        store.record_migration(SCHEMA_VERSION, "Phase 1 initial schema")
    logger.info("Schema ready (embed_dim=%d, version=%s).", embed_dim, SCHEMA_VERSION)
    return 0


def cmd_ingest_requirements(
    args: argparse.Namespace, cfg: TraceabilityConfig
) -> int:
    """Ingest an SRS Agent output JSON file."""
    path = args.srs_path
    logger.info("Ingesting requirements from: %s", path)

    loader = SRSLoader(path).load()
    logger.info("Loaded %d requirements.", len(loader.requirements))

    embed = not args.no_embed
    embeddings = {}
    embed_dim: int

    if embed:
        embedder  = Embedder(cfg)
        embed_dim = embedder.dim
        embeddings = embedder.embed_batch(loader.as_embedding_texts())
        logger.info(
            "Computed %d embeddings (model=%s, dim=%d).",
            len(embeddings), embedder.model_name, embed_dim,
        )
    else:
        embed_dim = cfg.embed_dim or 384

    with Store(cfg) as store:
        store.init_schema(embed_dim)
        doc_pk = store.upsert_document(loader.document_record())
        n = store.upsert_requirements(
            doc_pk, loader.requirements, embeddings or None
        )
    logger.info("Done. %d requirement(s) upserted (document_pk=%d).", n, doc_pk)
    return 0


def cmd_ingest_graph(args: argparse.Namespace, cfg: TraceabilityConfig) -> int:
    """Ingest a DocAgent dependency graph JSON file."""
    path = args.graph_path
    logger.info("Ingesting graph from: %s", path)

    loader = GraphLoader(path).load()
    logger.info(
        "Loaded %d components, %d relationships.",
        len(loader.components), len(loader.relationships),
    )

    embed = not args.no_embed
    embeddings = {}
    embed_dim: int

    if embed:
        embedder  = Embedder(cfg)
        embed_dim = embedder.dim
        embeddings = embedder.embed_batch(loader.as_embedding_texts())
        logger.info(
            "Computed %d embeddings (model=%s, dim=%d).",
            len(embeddings), embedder.model_name, embed_dim,
        )
    else:
        embed_dim = cfg.embed_dim or 384

    with Store(cfg) as store:
        store.init_schema(embed_dim)

        nc = store.upsert_components(
            list(loader.components.values()),
            embeddings or None,
            repo_path=args.repo_path,
            repo_snapshot=args.repo_snapshot,
        )

        # Ingest original docstrings present in the graph
        nd = 0
        for comp in loader.components.values():
            if comp.get("has_docstring") and comp.get("docstring", "").strip():
                nd += store.upsert_docstrings(comp["id"], original=comp["docstring"])

        nr_ins, nr_skip = store.upsert_relationships(loader.relationships)

    logger.info(
        "Done. %d component(s), %d docstring(s), "
        "%d relationship(s) inserted (%d skipped).",
        nc, nd, nr_ins, nr_skip,
    )
    return 0


def cmd_rebuild_embeddings(
    args: argparse.Namespace, cfg: TraceabilityConfig
) -> int:
    """Re-embed all rows whose embedding is NULL or was made with a different model."""
    Pipeline(cfg).rebuild_embeddings()
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m traceability.cli",
        description="Traceability package — schema and ingestion commands.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init-schema
    sub.add_parser(
        "init-schema",
        help="Create extension, tables, and indexes (idempotent).",
    )

    # ingest-requirements
    p_req = sub.add_parser(
        "ingest-requirements",
        help="Ingest an SRS Agent output JSON file.",
    )
    p_req.add_argument("srs_path", metavar="SRS_PATH",
                       help="Path to the SRS output JSON file.")
    p_req.add_argument("--no-embed", action="store_true",
                       help="Skip embedding computation.")

    # ingest-graph
    p_graph = sub.add_parser(
        "ingest-graph",
        help="Ingest a DocAgent dependency graph JSON file.",
    )
    p_graph.add_argument("graph_path", metavar="GRAPH_PATH",
                         help="Path to the dependency graph JSON file.")
    p_graph.add_argument("--repo-path", default=None,
                         help="Absolute path to the analysed repository.")
    p_graph.add_argument("--repo-snapshot", default=None,
                         help="Git commit SHA or tag at parse time.")
    p_graph.add_argument("--no-embed", action="store_true",
                         help="Skip embedding computation.")

    # rebuild-embeddings
    sub.add_parser(
        "rebuild-embeddings",
        help="Re-embed stale rows (NULL or wrong model version).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    cfg    = TraceabilityConfig()

    dispatch = {
        "init-schema":         cmd_init_schema,
        "ingest-requirements": cmd_ingest_requirements,
        "ingest-graph":        cmd_ingest_graph,
        "rebuild-embeddings":  cmd_rebuild_embeddings,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args, cfg))


if __name__ == "__main__":
    main()
