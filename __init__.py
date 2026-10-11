"""
traceability — integration boundary between DocAgent and SRS Agent.

Consumes their serialised JSON outputs.  Never imports from either sub-project.

Quick start::

    from traceability import Pipeline, TraceabilityConfig

    cfg = TraceabilityConfig()   # reads env vars
    Pipeline(cfg).run(
        graph_path="…/dependency_graph.json",
        srs_path="…/requirements.txt",
    )

CLI::

    python -m traceability.cli init-schema
    python -m traceability.cli ingest-requirements path/to/requirements.txt
    python -m traceability.cli ingest-graph path/to/graph.json
    python -m traceability.cli rebuild-embeddings
"""

from .config import TraceabilityConfig
from .embedder import Embedder, EmbeddingResult
from .loader import GraphLoader, SRSLoader
from .pipeline import Pipeline
from .store import DimensionMismatchError, Store

__all__ = [
    "TraceabilityConfig",
    "Embedder",
    "EmbeddingResult",
    "GraphLoader",
    "SRSLoader",
    "Pipeline",
    "Store",
    "DimensionMismatchError",
]
