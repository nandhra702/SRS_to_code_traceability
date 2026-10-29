"""
traceability CLI — schema management and data ingestion.

Commands
--------
init-schema            Create extension + tables + indexes (idempotent).
ingest-requirements    Load an SRS Agent output file.
ingest-graph           Load a DocAgent dependency graph (legacy or V1).
rebuild-embeddings     Re-embed rows that are missing/stale (model or text changed).
build-vector-indexes   Build optional HNSW indexes (exact search is used otherwise).
search                 Search requirements, docstrings and code (lexical + vector, fused).
evaluate               Score retrieval against a gold-standard mapping (v2).
find-candidates        Search agent: top-N code candidates per requirement, stored as links.
evaluate-candidates    Score stored candidates against a gold-standard mapping (v2).

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


def cmd_search(args: argparse.Namespace, cfg: TraceabilityConfig):
    from .query import QueryResolver
    from .retrieval import Retriever

    entities = tuple(e.strip() for e in args.entities.split(",") if e.strip())
    with Store(cfg) as store:
        query = QueryResolver(store).resolve(
            args.query, mode=args.mode, document_id=args.document_id, repo_id=args.repo_id)
        response = Retriever(store).search(
            query, method=args.method, entities=entities,
            top_k=args.top_k, candidate_k=args.candidate_k)
    return response.to_dict() if args.json else format_search(response)


def format_search(response) -> str:
    q = response.query
    lines = [f"Query: {q.original!r}  ->  mode={q.mode}, status={q.status}, method={response.method}"]
    if q.requirement:
        lines.append(f"Resolved requirement {q.requirement['requirement_id']}: {q.requirement['title']}")
    if q.message:
        lines.append(q.message)
    for c in q.candidates:
        lines.append(f"  candidate: {c}")

    def chan(h):
        return ", ".join(f"{k.split('_', 1)[1]} #{v['rank']} ({v['score']:.3f})"
                         for k, v in sorted(h.channels.items()))

    sections = [(e, response.results[e]) for e in ("requirement", "docstring", "component")
                if e in response.results]
    if {"docstring", "component"} & set(response.results):
        sections.append(("implementation candidates (code + docstrings fused)", response.components))
    for title, hits in sections:
        lines.append(f"\n== {title} ==")
        if not hits:
            lines.append("  (no results)")
        for i, h in enumerate(hits, start=1):
            d = h.data
            where = f"  {d['path']}:{d['start_line']}-{d['end_line']}" if "path" in d else ""
            extra = f" [{d['docstring_kind']} docstring]" if "docstring_kind" in d else ""
            extra += f" [via {d['via_docstring']} docstring]" if "via_docstring" in d else ""
            label = f"{h.external_id}: {d['title']}" if h.entity == "requirement" else h.external_id
            lines.append(f"{i:3d}. {h.score:.4f}  {label}{extra}{where}")
            lines.append(f"       {chan(h)}")
    return "\n".join(lines)


def cmd_evaluate(args: argparse.Namespace, cfg: TraceabilityConfig):
    from .evaluation import Evaluator, format_report, load_gold
    from .query import QueryResolver
    from .retrieval import Retriever

    with Store(cfg) as store:
        report = Evaluator(QueryResolver(store), Retriever(store)).evaluate(
            load_gold(args.gold), repo_id=args.repo_id, document_id=args.document_id,
            query_source=args.query_source, candidate_k=args.candidate_k)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    return format_report(report) + (f"\n\nFull report written to {args.out}" if args.out else "")


def cmd_find_candidates(args: argparse.Namespace, cfg: TraceabilityConfig):
    from .candidates import CandidateSearch
    from .query import QueryResolver
    from .retrieval import Retriever

    types = [t.strip() for t in args.types.split(",")] if args.types else None
    with Store(cfg) as store:
        report = CandidateSearch(store, QueryResolver(store), Retriever(store)).run(
            args.document_id, args.repo_id, top_n=args.top_n,
            candidate_k=args.candidate_k, types=types, pipeline_version=args.pipeline_version)
    if args.json:
        return report
    lines = [f"Candidates for {report['requirements']} requirement(s) of {report['document_id']} "
             f"in {report['repo_id']} ({report['links_stored']} links, {report['pipeline_version']})", ""]
    for r in report["per_requirement"]:
        sig = r["signals"]
        top = ", ".join(c.split(".", 1)[-1] for c in r["candidates"][:3])
        lines.append(f"{r['requirement_id']:10s} sim={sig['best_code_similarity']} "
                     f"agree={'Y' if sig['agreement'] else 'n'}  top: {top}")
    return "\n".join(lines)


def cmd_evaluate_candidates(args: argparse.Namespace, cfg: TraceabilityConfig):
    from .evaluation import Evaluator, load_gold
    from .query import QueryResolver
    from .retrieval import Retriever

    with Store(cfg) as store:
        report = Evaluator(QueryResolver(store), Retriever(store)).evaluate_candidates(
            load_gold(args.gold), document_id=args.document_id,
            pipeline_version=args.pipeline_version)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    s, m = report["scores"], report["matching"]
    return "\n".join([
        f"Stored candidates ({report['pipeline_version']}) for {report['document_id']}: "
        f"{report['requirements_scored']} gold requirements scored",
        "  " + "  ".join(f"{k}={v:.3f}" for k, v in s.items()),
        f"  matching: {m['by_id']} by ID, {m['by_text']} by text, unmatched {m['unmatched']}",
        f"  primary component not among candidates: {report['missed']}",
    ])


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

    p_search = sub.add_parser("search", help="Search requirements, docstrings and code.")
    p_search.add_argument("query", help="Requirement ID, requirement text, or keywords.")
    p_search.add_argument("--repo-id", default=None, help="Restrict code/docstring results to a repository.")
    p_search.add_argument("--document-id", default=None, help="SRS document scope for requirement IDs.")
    p_search.add_argument("--mode", default="auto", choices=["auto", "id", "text", "adhoc", "keyword"])
    p_search.add_argument("--method", default="hybrid", choices=["hybrid", "lexical", "vector"])
    p_search.add_argument("--entities", default="requirement,docstring,component",
                          help="Comma-separated subset of requirement,docstring,component.")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--candidate-k", type=int, default=50, help="Hits per channel before fusion.")
    p_search.add_argument("--json", action="store_true", help="Print the full result as JSON.")

    p_eval = sub.add_parser("evaluate", help="Score retrieval against a gold-standard mapping (v2).")
    p_eval.add_argument("--gold", required=True, help="Path to gold_standard_mapping.json (v2).")
    p_eval.add_argument("--repo-id", required=True)
    p_eval.add_argument("--document-id", default=None, help="Needed for --query-source ingested.")
    p_eval.add_argument("--query-source", default="gold_text", choices=["gold_text", "ingested"])
    p_eval.add_argument("--candidate-k", type=int, default=50)
    p_eval.add_argument("--out", default=None, help="Write the full JSON report here.")

    p_cand = sub.add_parser("find-candidates",
                            help="Search agent: store top-N code candidates for every requirement of a document.")
    p_cand.add_argument("--document-id", required=True)
    p_cand.add_argument("--repo-id", required=True)
    p_cand.add_argument("--top-n", type=int, default=10)
    p_cand.add_argument("--candidate-k", type=int, default=50)
    p_cand.add_argument("--types", default=None, help="e.g. functional,non-functional (default: all)")
    p_cand.add_argument("--pipeline-version", default="candidates-v1")
    p_cand.add_argument("--json", action="store_true")

    p_ec = sub.add_parser("evaluate-candidates", help="Score stored candidates against a gold mapping (v2).")
    p_ec.add_argument("--gold", required=True)
    p_ec.add_argument("--document-id", required=True)
    p_ec.add_argument("--pipeline-version", default="candidates-v1")
    p_ec.add_argument("--out", default=None)
    return parser


_DISPATCH = {
    "init-schema":          cmd_init_schema,
    "ingest-requirements":  cmd_ingest_requirements,
    "ingest-graph":         cmd_ingest_graph,
    "rebuild-embeddings":   cmd_rebuild_embeddings,
    "build-vector-indexes": cmd_build_vector_indexes,
    "search":               cmd_search,
    "evaluate":             cmd_evaluate,
    "find-candidates":      cmd_find_candidates,
    "evaluate-candidates":  cmd_evaluate_candidates,
}


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    args = build_parser().parse_args(argv)
    if args.command in ("search", "evaluate", "find-candidates", "evaluate-candidates"):
        logging.getLogger().setLevel(logging.WARNING)  # keep result output readable
    try:
        result = _DISPATCH[args.command](args, TraceabilityConfig())
    except (SchemaVersionError, DimensionMismatchError, ValueError, FileNotFoundError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 2
    print(result if isinstance(result, str) else json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
