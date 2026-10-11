"""
Central configuration for the traceability package.

All tunables are read from environment variables with safe defaults.
Credentials are never hard-coded and never logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TraceabilityConfig:
    # ------------------------------------------------------------------
    # Local sentence-transformers embedding (no AI service required)
    # ------------------------------------------------------------------

    # HuggingFace model name or local path passed to SentenceTransformer().
    # all-MiniLM-L6-v2: 384-dim, ~80 MB, CPU-friendly.
    # Downloaded once to ~/.cache/huggingface on first use; offline after that.
    embed_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # Human-readable version tag stored alongside every embedding row so that
    # stale embeddings can be detected when the model changes.
    # Defaults to the model name; override with a semver tag in production.
    embed_model_version: str = field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL_VERSION",
            os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        )
    )

    # Vector dimension.  None → resolved from model at first use via
    # SentenceTransformer.get_sentence_embedding_dimension().
    # Set explicitly (e.g. embed_dim=384) to skip the model-load probe.
    embed_dim: Optional[int] = field(
        default_factory=lambda: (
            int(v) if (v := os.getenv("EMBED_DIM")) else None
        )
    )

    # Batch size for sentence-transformers encode(); 64 is safe for CPU.
    embed_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBED_BATCH_SIZE", "64"))
    )

    # ------------------------------------------------------------------
    # PostgreSQL connection
    # ------------------------------------------------------------------

    # Full libpq DSN, e.g. "postgresql://user:pass@localhost:5432/traceability_dev".
    # When set, all individual pg_* fields below are ignored.
    pg_dsn: Optional[str] = field(
        default_factory=lambda: os.getenv("TRACEABILITY_PG_DSN")
    )

    pg_host: str = field(
        default_factory=lambda: os.getenv("TRACEABILITY_PG_HOST", "localhost")
    )
    pg_port: int = field(
        default_factory=lambda: int(os.getenv("TRACEABILITY_PG_PORT", "5432"))
    )
    pg_dbname: str = field(
        default_factory=lambda: os.getenv(
            "TRACEABILITY_PG_DBNAME", "traceability_dev"
        )
    )
    pg_user: str = field(
        default_factory=lambda: os.getenv("TRACEABILITY_PG_USER", "postgres")
    )
    # Password is read from the environment only — never stored in source.
    pg_password: str = field(
        default_factory=lambda: os.getenv("TRACEABILITY_PG_PASSWORD", "")
    )

    # ------------------------------------------------------------------
    # Schema / table names (configurable for multi-tenant isolation)
    # ------------------------------------------------------------------
    schema: str = "public"
    table_migrations: str = "tr_schema_migrations"
    table_srs_documents: str = "tr_srs_documents"
    table_requirements: str = "tr_requirements"
    table_components: str = "tr_code_components"
    table_docstrings: str = "tr_docstrings"
    table_relationships: str = "tr_relationships"
    table_mappings: str = "tr_mappings"

    # ------------------------------------------------------------------
    # ANN search
    # ------------------------------------------------------------------
    top_k: int = 10

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def build_dsn(self) -> str:
        """
        Return a psycopg2-compatible connection string.

        Prefers pg_dsn when set.  Assembles from parts otherwise.
        The password value is included but this string must never be logged.
        """
        if self.pg_dsn:
            return self.pg_dsn
        parts = [
            f"host={self.pg_host}",
            f"port={self.pg_port}",
            f"dbname={self.pg_dbname}",
            f"user={self.pg_user}",
        ]
        if self.pg_password:
            parts.append(f"password={self.pg_password}")
        return " ".join(parts)

    def safe_repr(self) -> str:
        """Return a credential-free string suitable for logging."""
        return (
            f"TraceabilityConfig(host={self.pg_host}, port={self.pg_port}, "
            f"dbname={self.pg_dbname}, user={self.pg_user}, "
            f"embed_model={self.embed_model!r})"
        )
