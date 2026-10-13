"""
traceability CLI — schema management and data ingestion.

Commands
--------
init-schema            Create extension + tables + indexes (idempotent).
ingest-requirements    Load an SRS Agent output file.
ingest-graph           Load a DocAgent dependency graph (legacy or V1).
rebuild-embeddings     Re-embed rows that are missing/stale (model or text changed).
build-vector-indexes   Build optional HNSW indexes (exact search is used otherwise).

Each ingest prints a JSON report (counts, skipped/truncated texts, skipped
relationships) to stdout; progress logs go to stderr.

Examples (run from the directory containing this package)
--------
python -m traceability.cli init-schema
python -m traceability.cli ingest-requirements ../SRSAgent/output/Basic_SRS_Document_requirements.txt
python -m traceability.cli ingest-graph \\
    ../DocAgent/output/dependency_graphs/raw_test_repo_simple_dependency_graph.json \\
    --repo-path ../DocAgent/data/raw_test_repo_simple \\
    --source-root ../DocAgent/data/raw_test_repo_simple
python -m traceability.cli rebuild-embeddings

Database credentials come from environment variables only:
  TRACEABILITY_PG_DSN       full libpq DSN (takes precedence)
  TRACEABILITY_PG_HOST / _PORT / _DBNAME / _USER / _PASSWORD
Embedding model:
  EMBED_MODEL (default sentence-transformers/all-MiniLM-L6-v2),
  EMBED_MODEL_VERSION, EMBED_DIM, EMBED_CODE_MAX_CHARS
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import TraceabilityConfig
from .embedder import Embedder
from .pipeline import Pipeline
from .store import DimensionMismatchError, SchemaVersionError, Store

logger = logging.getLogger("traceability.cli")


def cmd_init_schema(args: argparse.Namespace, cfg: TraceabilityConfig) -> dict:
    embed_dim = Embedder(cfg).dim
    with Store(cfg) as store:
        store.init_schema(embed_dim)
    return {"embed_dim": embed_dim}


def cmd_ingest_requirements(args: argparse.Namespace, cfg: TraceabilityConfig) -> dict:
    return Pipeline(cfg).ingest_srs(
        args.srs_path, document_id=args.document_id, embed=not args.no_embed
    )


def cmd_ingest_graph(args: argparse.Namespace, cfg: TraceabilityConfig) -> dict:
    return Pipeline(cfg).ingest_graph(
        args.graph_path,
        repo_id=args.repo_id,
        repo_path=args.repo_path,
        repo_snapshot=args.repo_snapshot,
        source_root=args.source_root,
        embed=not args.no_embed,
    )


def cmd_rebuild_embeddings(args: argparse.Namespace, cfg: TraceabilityConfig) -> dict:
    return Pipeline(cfg).rebuild_embeddings()


def cmd_build_vector_indexes(args: argparse.Namespace, cfg: TraceabilityConfig) -> dict:
    with Store(cfg) as store:
        store.build_vector_indexes()
    return {"vector_indexes": "hnsw"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m traceability.cli",
        description="Traceability package — schema and ingestion commands.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-schema", help="Create extension, tables, and indexes (idempotent).")

    p_req = sub.add_parser("ingest-requirements", help="Ingest an SRS Agent output file.")
    p_req.add_argument("srs_path", metavar="SRS_PATH")
    p_req.add_argument("--document-id", default=None,
                       help="Logical document ID (default: SRSOutput document_id, else file stem).")
    p_req.add_argument("--no-embed", action="store_true", help="Skip embedding computation.")

    p_graph = sub.add_parser("ingest-graph", help="Ingest a DocAgent dependency graph JSON file.")
    p_graph.add_argument("graph_path", metavar="GRAPH_PATH")
    p_graph.add_argument("--repo-id", default=None,
                         help="Logical repository ID (default: graph file stem).")
    p_graph.add_argument("--repo-path", default=None, help="Path of the analysed repository.")
    p_graph.add_argument("--repo-snapshot", default=None, help="Git commit SHA or tag at parse time.")
    p_graph.add_argument("--source-root", default=None,
                         help="Repository after DocAgent generation; recovers generated docstrings.")
    p_graph.add_argument("--no-embed", action="store_true", help="Skip embedding computation.")

    sub.add_parser("rebuild-embeddings", help="Re-embed missing or stale rows.")
    sub.add_parser("build-vector-indexes", help="Build optional HNSW indexes.")
    return parser


_DISPATCH = {
    "init-schema":          cmd_init_schema,
    "ingest-requirements":  cmd_ingest_requirements,
    "ingest-graph":         cmd_ingest_graph,
    "rebuild-embeddings":   cmd_rebuild_embeddings,
    "build-vector-indexes": cmd_build_vector_indexes,
}


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    args = build_parser().parse_args(argv)
    try:
        result = _DISPATCH[args.command](args, TraceabilityConfig())
    except (SchemaVersionError, DimensionMismatchError, ValueError, FileNotFoundError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
