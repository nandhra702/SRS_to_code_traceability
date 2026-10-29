"""
Store — PostgreSQL persistence layer for the traceability package.

Uses psycopg2 directly (no ORM) with parameterized SQL throughout.

Transaction policy
------------------
Every public method runs inside ``Store.transaction()``.  Transactions nest:
an inner call joins the outer transaction instead of committing on its own.
The pipeline wraps each dataset (one SRS document, or one repository's
components + docstrings + relationships) in a single outer transaction, so a
failed ingestion rolls back completely — nothing is half-written.

Safety rules enforced here
--------------------------
- Parameterized queries for all values; table names come only from config.
- Credentials never appear in log output; only safe_repr() is logged.
- No DROP or TRUNCATE.  The only DELETE replaces a repository's derived
  relationship rows on re-ingest.
- Rows missing from a re-ingested source are flagged is_current = FALSE, not
  deleted, so trace links and verification history survive.
- The schema records its embedding dimension; init_schema() refuses a
  different one, and every vector is length-checked before it is written.
- Relationship endpoints must exist in the same repository and the type must
  be one of DocAgent's five; anything else is skipped and counted.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

from .config import TraceabilityConfig
from .embedder import EmbeddingResult
from .loader import content_hash
from .schema import (
    RELATIONSHIP_TYPES,
    SCHEMA_VERSION,
    all_create_ddl,
    create_extension_ddl,
    create_migrations_table_ddl,
    migration_steps,
    vector_index_ddl,
    version_key,
)

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
    from pgvector.psycopg2 import register_vector  # type: ignore[import]

    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

# Requirement keys stored in dedicated columns; every other key is kept in
# extraction_meta so no SRS Agent field is dropped.
_REQ_LIST_COLUMNS = (
    "actor", "actions", "entities", "inputs", "outputs",
    "preconditions", "postconditions", "conditions", "constraints",
    "error_conditions", "domain_terms", "synonyms", "sub_requirements",
)
_REQ_COLUMN_KEYS = frozenset(_REQ_LIST_COLUMNS) | {
    "requirement_id", "id_source", "raw_text", "normalized_text",
    "retrieval_text", "title", "type", "subtype", "review_required", "source",
}

# entity name → (table config attribute, primary key column)
_ENTITIES = {
    "requirement": ("table_requirements", "req_pk"),
    "component":   ("table_components", "comp_pk"),
    "docstring":   ("table_docstrings", "docstring_pk"),
}

# Keep an existing vector when an upsert carries none (e.g. --no-embed);
# rebuild-embeddings detects the resulting staleness through the text hash.
_KEEP_EMBEDDING_SQL = """
                embedding           = COALESCE(EXCLUDED.embedding, {t}.embedding),
                embed_model_version = CASE WHEN EXCLUDED.embedding IS NULL
                                           THEN {t}.embed_model_version
                                           ELSE EXCLUDED.embed_model_version END,
                embed_text_hash     = CASE WHEN EXCLUDED.embedding IS NULL
                                           THEN {t}.embed_text_hash
                                           ELSE EXCLUDED.embed_text_hash END"""


def _require_pg() -> None:
    if not _PG_AVAILABLE:
        raise ImportError(
            "psycopg2 and pgvector are required for Store. "
            "Install with: pip install psycopg2-binary pgvector"
        )


class DimensionMismatchError(ValueError):
    """An embedding vector length does not match the schema's vector dimension."""


class SchemaVersionError(RuntimeError):
    """The database was initialised with an incompatible schema version."""


class Store:
    """
    PostgreSQL persistence layer.

    Usage::

        with Store(cfg) as store:
            store.init_schema(embed_dim=384)
            with store.transaction():
                repo_pk = store.upsert_repository({...})
                comp_pks = store.upsert_components(repo_pk, comps, embeddings)
                store.upsert_docstrings(comp_pks, docstrings, ds_embeddings)
                store.replace_relationships(repo_pk, comp_pks, relationships)
    """

    def __init__(self, config: TraceabilityConfig) -> None:
        _require_pg()
        self._cfg = config
        # Never log the DSN — it may contain the password.
        logger.info("Connecting to PostgreSQL: %s", config.safe_repr())
        self._conn = psycopg2.connect(config.build_dsn())
        self._cur = None  # active transaction cursor, if any
        self._vector_registered = False
        self._register_vector()
        self._schema_embed_dim: Optional[int] = self.get_schema_embed_dim()

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Generator:
        """
        Yield a RealDictCursor inside one BEGIN/COMMIT block.

        Nested calls reuse the outer cursor and leave commit/rollback to the
        outermost block, so a multi-step ingestion is all-or-nothing.
        """
        if self._cur is not None:
            yield self._cur
            return
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        self._cur = cur
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._cur = None
            cur.close()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def init_schema(self, embed_dim: int) -> None:
        """
        Create the pgvector extension and all tables/indexes (idempotent).

        A database at an older schema version is migrated along
        schema.migration_steps() in the same transaction.

        Raises:
            SchemaVersionError:     the database holds a version with no
                                    migration path to SCHEMA_VERSION.
            DimensionMismatchError: the schema was created with another embed_dim.
        """
        t = self._cfg.table_migrations
        with self.transaction() as cur:
            cur.execute(create_extension_ddl())
            cur.execute(create_migrations_table_ddl(self._cfg))
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = %s AND column_name = 'embed_dim';",
                (t,),
            )
            if cur.fetchone() is None:
                raise SchemaVersionError(
                    f"{t} predates schema 2.0.0 (no embed_dim column). This is a "
                    "development database from the old schema; reset it before "
                    "initialising the new one."
                )
            cur.execute(f"SELECT version, embed_dim FROM {t};")
            recorded = {r["version"]: r["embed_dim"] for r in cur.fetchall()}
            if recorded:
                try:
                    current = max(recorded, key=version_key)
                except ValueError:
                    raise SchemaVersionError(f"Unrecognised schema versions {sorted(recorded)}.")
                if recorded[current] != embed_dim:
                    raise DimensionMismatchError(
                        f"Schema vector columns were created with dimension "
                        f"{recorded[current]}, but the embedding model produces "
                        f"{embed_dim}. Use the original model or a fresh database."
                    )
                self._migrate(cur, current, embed_dim)
            for stmt in all_create_ddl(self._cfg, embed_dim):
                cur.execute(stmt)
            cur.execute(
                f"INSERT INTO {t} (version, embed_dim, description) "
                f"VALUES (%s, %s, %s) ON CONFLICT (version) DO NOTHING;",
                (SCHEMA_VERSION, embed_dim, "traceability schema"),
            )
        self._register_vector()
        self._schema_embed_dim = embed_dim
        logger.info("Schema %s ready (embed_dim=%d).", SCHEMA_VERSION, embed_dim)

    def _migrate(self, cur, current: str, embed_dim: int) -> None:
        """Apply migration steps from *current* up to SCHEMA_VERSION."""
        if version_key(current) > version_key(SCHEMA_VERSION):
            raise SchemaVersionError(
                f"Database schema {current} is newer than this code ({SCHEMA_VERSION}).")
        steps = migration_steps(self._cfg)
        version = current
        while version != SCHEMA_VERSION:
            if version not in steps:
                raise SchemaVersionError(
                    f"Database has schema version {version}; this code expects "
                    f"{SCHEMA_VERSION} and has no migration from it.")
            target, stmts = steps[version]
            for stmt in stmts:
                cur.execute(stmt)
            cur.execute(
                f"INSERT INTO {self._cfg.table_migrations} (version, embed_dim, description) "
                f"VALUES (%s, %s, %s) ON CONFLICT (version) DO NOTHING;",
                (target, embed_dim, f"migrated from {version}"),
            )
            logger.info("Migrated schema %s -> %s.", version, target)
            version = target

    def get_schema_embed_dim(self) -> Optional[int]:
        """Embedding dimension recorded for the current schema, or None."""
        t = self._cfg.table_migrations
        with self.transaction() as cur:
            cur.execute("SELECT to_regclass(%s) AS reg;", (t,))
            if cur.fetchone()["reg"] is None:
                return None
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = %s AND column_name = 'embed_dim';",
                (t,),
            )
            if cur.fetchone() is None:
                return None
            # Every version row records the same dimension; read the latest so
            # this also works on a database that has not been migrated yet.
            cur.execute(f"SELECT embed_dim FROM {t} ORDER BY applied_at DESC LIMIT 1;")
            row = cur.fetchone()
        return row["embed_dim"] if row else None

    def build_vector_indexes(self) -> None:
        """Build HNSW cosine indexes on all embedding columns (optional)."""
        with self.transaction() as cur:
            for stmt in vector_index_ddl(self._cfg):
                cur.execute(stmt)
        logger.info("HNSW vector indexes built.")

    # ------------------------------------------------------------------
    # SRS documents + requirements
    # ------------------------------------------------------------------

    def upsert_document(self, record: Dict[str, Any]) -> int:
        """Insert or update an SRS document; returns document_pk."""
        t = self._cfg.table_srs_documents
        sql = f"""
            INSERT INTO {t} (document_id, file_name, source_type,
                             total_requirements, ingest_path)
            VALUES (%(document_id)s, %(file_name)s, %(source_type)s,
                    %(total_requirements)s, %(ingest_path)s)
            ON CONFLICT (document_id) DO UPDATE SET
                file_name          = EXCLUDED.file_name,
                source_type        = EXCLUDED.source_type,
                total_requirements = EXCLUDED.total_requirements,
                ingest_path        = EXCLUDED.ingest_path,
                updated_at         = NOW()
            RETURNING document_pk;
        """
        params = {
            "document_id":        record.get("document_id") or record.get("file_name", ""),
            "file_name":          record.get("file_name", ""),
            "source_type":        record.get("source_type", "unknown"),
            "total_requirements": record.get("total_requirements", 0),
            "ingest_path":        record.get("ingest_path"),
        }
        with self.transaction() as cur:
            cur.execute(sql, params)
            pk = cur.fetchone()["document_pk"]
        logger.info("Upserted document document_pk=%d (%s).", pk, params["document_id"])
        return pk

    def upsert_requirements(
        self,
        document_pk: int,
        records: List[Dict[str, Any]],
        embeddings: Optional[Dict[str, EmbeddingResult]] = None,
    ) -> Dict[str, int]:
        """
        Insert or update a document's requirements.

        Requirements of this document that are absent from *records* are
        flagged is_current = FALSE.

        Returns:
            {requirement_id: req_pk}
        """
        t = self._cfg.table_requirements
        list_cols = ", ".join(_REQ_LIST_COLUMNS)
        list_vals = ", ".join(f"%({c})s" for c in _REQ_LIST_COLUMNS)
        list_updates = ",\n                ".join(
            f"{c} = EXCLUDED.{c}" for c in _REQ_LIST_COLUMNS
        )
        sql = f"""
            INSERT INTO {t} (
                document_pk, requirement_id, id_source,
                raw_text, normalized_text, retrieval_text, title,
                type, subtype, review_required, {list_cols},
                source_location, extraction_meta, is_current,
                embedding, embed_model_version, embed_text_hash
            ) VALUES (
                %(document_pk)s, %(requirement_id)s, %(id_source)s,
                %(raw_text)s, %(normalized_text)s, %(retrieval_text)s, %(title)s,
                %(type)s, %(subtype)s, %(review_required)s, {list_vals},
                %(source_location)s, %(extraction_meta)s, TRUE,
                %(embedding)s, %(embed_model_version)s, %(embed_text_hash)s
            )
            ON CONFLICT (document_pk, requirement_id) DO UPDATE SET
                id_source        = EXCLUDED.id_source,
                raw_text         = EXCLUDED.raw_text,
                normalized_text  = EXCLUDED.normalized_text,
                retrieval_text   = EXCLUDED.retrieval_text,
                title            = EXCLUDED.title,
                type             = EXCLUDED.type,
                subtype          = EXCLUDED.subtype,
                review_required  = EXCLUDED.review_required,
                {list_updates},
                source_location  = EXCLUDED.source_location,
                extraction_meta  = EXCLUDED.extraction_meta,
                is_current       = TRUE,
                {_KEEP_EMBEDDING_SQL.format(t=t).strip()},
                updated_at       = NOW()
            RETURNING req_pk;
        """
        pks: Dict[str, int] = {}
        with self.transaction() as cur:
            for rec in records:
                emb = (embeddings or {}).get(rec["requirement_id"])
                self._validate_embedding_dim(emb)
                cur.execute(sql, self._req_params(document_pk, rec, emb))
                pks[rec["requirement_id"]] = cur.fetchone()["req_pk"]
            cur.execute(
                f"UPDATE {t} SET is_current = FALSE, updated_at = NOW() "
                f"WHERE document_pk = %s AND is_current "
                f"AND NOT (requirement_id = ANY(%s));",
                (document_pk, list(pks)),
            )
            if cur.rowcount:
                logger.info("Flagged %d requirement(s) no longer in the document.", cur.rowcount)
        logger.info("Upserted %d requirement(s) into document_pk=%d.", len(pks), document_pk)
        return pks

    # ------------------------------------------------------------------
    # Repositories, components, docstrings, relationships
    # ------------------------------------------------------------------

    def upsert_repository(self, record: Dict[str, Any]) -> int:
        """Insert or update a repository; returns repo_pk."""
        t = self._cfg.table_repositories
        sql = f"""
            INSERT INTO {t} (repo_id, repo_path, repo_snapshot, graph_path, graph_format)
            VALUES (%(repo_id)s, %(repo_path)s, %(repo_snapshot)s,
                    %(graph_path)s, %(graph_format)s)
            ON CONFLICT (repo_id) DO UPDATE SET
                repo_path     = EXCLUDED.repo_path,
                repo_snapshot = EXCLUDED.repo_snapshot,
                graph_path    = EXCLUDED.graph_path,
                graph_format  = EXCLUDED.graph_format,
                updated_at    = NOW()
            RETURNING repo_pk;
        """
        params = {
            "repo_id":       record["repo_id"],
            "repo_path":     record.get("repo_path"),
            "repo_snapshot": record.get("repo_snapshot"),
            "graph_path":    record.get("graph_path"),
            "graph_format":  record.get("graph_format", "legacy"),
        }
        with self.transaction() as cur:
            cur.execute(sql, params)
            pk = cur.fetchone()["repo_pk"]
        logger.info("Upserted repository repo_pk=%d (%s).", pk, params["repo_id"])
        return pk

    def upsert_components(
        self,
        repo_pk: int,
        records: List[Dict[str, Any]],
        embeddings: Optional[Dict[str, EmbeddingResult]] = None,
    ) -> Dict[str, int]:
        """
        Insert or update a repository's components.

        Components of this repository absent from *records* are flagged
        is_current = FALSE.

        Returns:
            {component_id: comp_pk}
        """
        t = self._cfg.table_components
        sql = f"""
            INSERT INTO {t} (
                repo_pk, component_id, component_type,
                file_path, relative_path, start_line, end_line,
                signature, parameters, return_type, containing_class, depends_on,
                source_code, source_hash, is_current,
                embedding, embed_model_version, embed_text_hash
            ) VALUES (
                %(repo_pk)s, %(component_id)s, %(component_type)s,
                %(file_path)s, %(relative_path)s, %(start_line)s, %(end_line)s,
                %(signature)s, %(parameters)s, %(return_type)s, %(containing_class)s,
                %(depends_on)s, %(source_code)s, %(source_hash)s, TRUE,
                %(embedding)s, %(embed_model_version)s, %(embed_text_hash)s
            )
            ON CONFLICT (repo_pk, component_id) DO UPDATE SET
                component_type   = EXCLUDED.component_type,
                file_path        = EXCLUDED.file_path,
                relative_path    = EXCLUDED.relative_path,
                start_line       = EXCLUDED.start_line,
                end_line         = EXCLUDED.end_line,
                signature        = EXCLUDED.signature,
                parameters       = EXCLUDED.parameters,
                return_type      = EXCLUDED.return_type,
                containing_class = EXCLUDED.containing_class,
                depends_on       = EXCLUDED.depends_on,
                source_code      = EXCLUDED.source_code,
                source_hash      = EXCLUDED.source_hash,
                is_current       = TRUE,
                {_KEEP_EMBEDDING_SQL.format(t=t).strip()},
                updated_at       = NOW()
            RETURNING comp_pk;
        """
        pks: Dict[str, int] = {}
        with self.transaction() as cur:
            for rec in records:
                emb = (embeddings or {}).get(rec["id"])
                self._validate_embedding_dim(emb)
                cur.execute(sql, self._comp_params(repo_pk, rec, emb))
                pks[rec["id"]] = cur.fetchone()["comp_pk"]
            cur.execute(
                f"UPDATE {t} SET is_current = FALSE, updated_at = NOW() "
                f"WHERE repo_pk = %s AND is_current AND NOT (component_id = ANY(%s));",
                (repo_pk, list(pks)),
            )
            if cur.rowcount:
                logger.info("Flagged %d component(s) no longer in the graph.", cur.rowcount)
        logger.info("Upserted %d component(s) into repo_pk=%d.", len(pks), repo_pk)
        return pks

    def upsert_docstrings(
        self,
        comp_pks: Dict[str, int],
        records: List[Dict[str, Any]],
        embeddings: Optional[Dict[Tuple[str, str], EmbeddingResult]] = None,
        *,
        observed_kinds: Iterable[str] = ("existing",),
    ) -> int:
        """
        Insert or update docstring records (from GraphLoader.docstring_records).

        Docstrings of the given components whose kind is in *observed_kinds*
        but which are absent from *records* are flagged is_current = FALSE.
        Kinds this ingestion did not look for (e.g. 'generated' when no
        post-run source tree was supplied) are left untouched.

        Args:
            comp_pks:   {component_id: comp_pk} from upsert_components().
            records:    [{"component_id", "kind", "content", "provenance"}].
            embeddings: {(component_id, kind): EmbeddingResult}.

        Returns:
            Number of docstrings upserted.
        """
        t = self._cfg.table_docstrings
        sql = f"""
            INSERT INTO {t} (comp_pk, docstring_kind, content, content_hash,
                             provenance, is_current,
                             embedding, embed_model_version, embed_text_hash)
            VALUES (%(comp_pk)s, %(kind)s, %(content)s, %(content_hash)s,
                    %(provenance)s, TRUE,
                    %(embedding)s, %(embed_model_version)s, %(embed_text_hash)s)
            ON CONFLICT (comp_pk, docstring_kind) DO UPDATE SET
                content      = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                provenance   = EXCLUDED.provenance,
                is_current   = TRUE,
                {_KEEP_EMBEDDING_SQL.format(t=t).strip()},
                updated_at   = NOW();
        """
        seen: List[Tuple[int, str]] = []
        with self.transaction() as cur:
            for rec in records:
                comp_pk = comp_pks.get(rec["component_id"])
                if comp_pk is None:
                    raise KeyError(
                        f"Docstring for unknown component {rec['component_id']!r}; "
                        "upsert its component first."
                    )
                emb = (embeddings or {}).get((rec["component_id"], rec["kind"]))
                self._validate_embedding_dim(emb)
                cur.execute(sql, {
                    "comp_pk":             comp_pk,
                    "kind":                rec["kind"],
                    "content":             rec["content"],
                    "content_hash":        content_hash(rec["content"]),
                    "provenance":          json.dumps(rec.get("provenance") or {}),
                    "embedding":           emb.vector if emb else None,
                    "embed_model_version": emb.model_version if emb else None,
                    "embed_text_hash":     emb.text_hash if emb else None,
                })
                seen.append((comp_pk, rec["kind"]))
            cur.execute(
                f"""
                UPDATE {t} SET is_current = FALSE, updated_at = NOW()
                WHERE is_current
                  AND comp_pk = ANY(%(comp_pks)s)
                  AND docstring_kind = ANY(%(kinds)s)
                  AND (comp_pk, docstring_kind) NOT IN (
                      SELECT * FROM unnest(%(seen_pks)s::bigint[], %(seen_kinds)s::text[]));
                """,
                {
                    "comp_pks":   list(comp_pks.values()),
                    "kinds":      list(observed_kinds),
                    "seen_pks":   [p for p, _ in seen],
                    "seen_kinds": [k for _, k in seen],
                },
            )
        logger.info("Upserted %d docstring(s).", len(seen))
        return len(seen)

    def replace_relationships(
        self,
        repo_pk: int,
        comp_pks: Dict[str, int],
        records: List[Dict[str, str]],
    ) -> Tuple[int, Dict[str, int]]:
        """
        Replace a repository's typed relationships with *records*.

        Relationships are derived data with nothing referencing them, so the
        repository's previous edges are deleted and the current set inserted.
        Edges whose endpoints are not components of this repository, or whose
        type is not one of DocAgent's five, are skipped and counted.

        Returns:
            (inserted, {"unknown_endpoint": n, "unknown_type": n})
        """
        t = self._cfg.table_relationships
        skipped = {"unknown_endpoint": 0, "unknown_type": 0}
        rows: List[Tuple[int, int, int, str]] = []
        for rec in records:
            src, tgt, rtype = rec["source"], rec["target"], rec["type"]
            if rtype not in RELATIONSHIP_TYPES:
                logger.warning("Unknown relationship type %r (%s -> %s); skipped.", rtype, src, tgt)
                skipped["unknown_type"] += 1
            elif src not in comp_pks or tgt not in comp_pks:
                logger.warning("Relationship endpoint not in repository (%s -> %s); skipped.", src, tgt)
                skipped["unknown_endpoint"] += 1
            else:
                rows.append((repo_pk, comp_pks[src], comp_pks[tgt], rtype))

        with self.transaction() as cur:
            cur.execute(f"DELETE FROM {t} WHERE repo_pk = %s;", (repo_pk,))
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO {t} (repo_pk, source_comp_pk, target_comp_pk, rel_type) "
                f"VALUES %s ON CONFLICT DO NOTHING;",
                rows,
            )
        logger.info("Relationships: %d stored, skipped %s.", len(rows), skipped)
        return len(rows), skipped

    # ------------------------------------------------------------------
    # Trace links
    # ------------------------------------------------------------------

    def upsert_trace_links(self, records: List[Dict[str, Any]]) -> Dict[Tuple, int]:
        """
        Insert or update trace links.

        Each record: link_type, req_pk / docstring_pk / comp_pk (as the link
        type requires), status, retrieval_method, scores, evidence,
        pipeline_version.  The natural key is (link_type, req_pk, docstring_pk,
        comp_pk, pipeline_version), so re-running a stage updates its links.
        A link that already has a verdict keeps its status.

        Returns:
            {(link_type, req_pk, docstring_pk, comp_pk): link_pk}
        """
        t = self._cfg.table_trace_links
        sql = f"""
            INSERT INTO {t} (link_type, req_pk, docstring_pk, comp_pk, status,
                             retrieval_method, scores, evidence, pipeline_version)
            VALUES (%(link_type)s, %(req_pk)s, %(docstring_pk)s, %(comp_pk)s, %(status)s,
                    %(retrieval_method)s, %(scores)s, %(evidence)s, %(pipeline_version)s)
            ON CONFLICT (link_type, req_pk, docstring_pk, comp_pk, pipeline_version) DO UPDATE SET
                status           = CASE WHEN {t}.status IN ('candidate', 'pending_verification')
                                        THEN EXCLUDED.status ELSE {t}.status END,
                retrieval_method = EXCLUDED.retrieval_method,
                scores           = EXCLUDED.scores,
                evidence         = EXCLUDED.evidence,
                updated_at       = NOW()
            RETURNING link_pk;
        """
        out: Dict[Tuple, int] = {}
        with self.transaction() as cur:
            for rec in records:
                params = {
                    "link_type":        rec["link_type"],
                    "req_pk":           rec.get("req_pk"),
                    "docstring_pk":     rec.get("docstring_pk"),
                    "comp_pk":          rec.get("comp_pk"),
                    "status":           rec.get("status", "pending_verification"),
                    "retrieval_method": rec.get("retrieval_method"),
                    "scores":           self._jsonb(rec.get("scores") or {}),
                    "evidence":         self._jsonb(rec.get("evidence") or {}),
                    "pipeline_version": rec["pipeline_version"],
                }
                cur.execute(sql, params)
                key = (params["link_type"], params["req_pk"], params["docstring_pk"], params["comp_pk"])
                out[key] = cur.fetchone()["link_pk"]
        return out

    def delete_stale_candidates(
        self, req_pk: int, pipeline_version: str, keep_link_pks: Iterable[int],
        *, statuses: Sequence[str] = ("candidate",),
    ) -> int:
        """
        Remove a requirement's links of *pipeline_version* that are not in
        keep_link_pks and whose status is still one of *statuses* (unreviewed:
        'candidate' for the search stage, 'pending_verification' for mapping).
        Links that carry a verdict are never deleted.
        """
        bad = set(statuses) - {"candidate", "pending_verification"}
        if bad:
            raise ValueError(f"Refusing to delete links with verdict statuses {sorted(bad)}")
        t = self._cfg.table_trace_links
        with self.transaction() as cur:
            cur.execute(
                f"DELETE FROM {t} WHERE req_pk = %s AND pipeline_version = %s "
                f"AND status = ANY(%s) AND NOT (link_pk = ANY(%s));",
                (req_pk, pipeline_version, list(statuses), list(keep_link_pks)),
            )
            return cur.rowcount

    def get_trace_links(
        self, *, document_pk: Optional[int] = None, pipeline_version: Optional[str] = None,
        link_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Links joined with requirement and component identity, for reports."""
        cfg = self._cfg
        where, params = ["TRUE"], {}
        if document_pk is not None:
            where.append("r.document_pk = %(doc)s")
            params["doc"] = document_pk
        if pipeline_version is not None:
            where.append("l.pipeline_version = %(pv)s")
            params["pv"] = pipeline_version
        if link_type is not None:
            where.append("l.link_type = %(lt)s")
            params["lt"] = link_type
        sql = (f"SELECT l.*, r.requirement_id, c.component_id "
               f"FROM {cfg.table_trace_links} l "
               f"LEFT JOIN {cfg.table_requirements} r ON r.req_pk = l.req_pk "
               f"LEFT JOIN {cfg.table_components} c ON c.comp_pk = l.comp_pk "
               f"WHERE {' AND '.join(where)} "
               f"ORDER BY r.req_pk, (l.evidence->>'rank')::int NULLS LAST, l.link_pk;")
        with self.transaction() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Embedding maintenance (rebuild-embeddings)
    # ------------------------------------------------------------------

    def get_embedding_rows(self, entity: str) -> List[Dict[str, Any]]:
        """
        All current rows of *entity* with the fields needed to rebuild their
        embedding text, plus the stored embed_model_version / embed_text_hash
        and a has_embedding flag.  The caller decides what is stale.
        """
        cfg = self._cfg
        if entity == "requirement":
            cols = ", ".join(("req_pk", "requirement_id", "title", "retrieval_text",
                              "normalized_text", "raw_text") + _REQ_LIST_COLUMNS)
            sql = f"SELECT {cols}, {{emb}} FROM {cfg.table_requirements} WHERE is_current;"
        elif entity == "component":
            sql = (f"SELECT comp_pk, component_id, component_type, signature, source_code, "
                   f"{{emb}} FROM {cfg.table_components} WHERE is_current;")
        elif entity == "docstring":
            sql = (f"SELECT d.docstring_pk, d.content, c.component_id, c.component_type, "
                   f"c.signature, {{emb}} FROM {cfg.table_docstrings} d "
                   f"JOIN {cfg.table_components} c ON c.comp_pk = d.comp_pk "
                   f"WHERE d.is_current;")
        else:
            raise ValueError(f"Unknown entity {entity!r}")
        prefix = "d." if entity == "docstring" else ""
        emb = (f"{prefix}embed_model_version, {prefix}embed_text_hash, "
               f"{prefix}embedding IS NOT NULL AS has_embedding")
        with self.transaction() as cur:
            cur.execute(sql.format(emb=emb))
            return [dict(r) for r in cur.fetchall()]

    def update_embedding(
        self, entity: str, pk: int, result: Optional[EmbeddingResult]
    ) -> None:
        """Set (or, with result=None, clear) one row's embedding."""
        table_attr, pk_col = _ENTITIES[entity]
        t = getattr(self._cfg, table_attr)
        self._validate_embedding_dim(result)
        with self.transaction() as cur:
            cur.execute(
                f"UPDATE {t} SET embedding = %(vec)s, embed_model_version = %(ver)s, "
                f"embed_text_hash = %(hash)s, updated_at = NOW() WHERE {pk_col} = %(pk)s;",
                {
                    "vec":  result.vector if result else None,
                    "ver":  result.model_version if result else None,
                    "hash": result.text_hash if result else None,
                    "pk":   pk,
                },
            )

    # ------------------------------------------------------------------
    # Vector search (exact; Phase 2 adds lexical + hybrid retrieval)
    # ------------------------------------------------------------------

    def search_requirements(
        self, query_vec: List[float], top_k: Optional[int] = None, *,
        document_pk: Optional[int] = None, model_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Cosine-distance search over current requirement embeddings."""
        t = self._cfg.table_requirements
        return self._vector_search(
            f"SELECT req_pk, requirement_id, document_pk, title, type, subtype, raw_text, "
            f"embedding <=> %(vec)s::vector AS distance FROM {t}",
            "document_pk", document_pk, query_vec, top_k, model_version,
        )

    def search_components(
        self, query_vec: List[float], top_k: Optional[int] = None, *,
        repo_pk: Optional[int] = None, model_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Cosine-distance search over current component (code) embeddings."""
        t = self._cfg.table_components
        return self._vector_search(
            f"SELECT comp_pk, repo_pk, component_id, component_type, relative_path, "
            f"start_line, end_line, signature, "
            f"embedding <=> %(vec)s::vector AS distance FROM {t}",
            "repo_pk", repo_pk, query_vec, top_k, model_version,
        )

    def search_docstrings(
        self, query_vec: List[float], top_k: Optional[int] = None, *,
        repo_pk: Optional[int] = None, model_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Cosine-distance search over current docstring embeddings."""
        ds, comp = self._cfg.table_docstrings, self._cfg.table_components
        return self._vector_search(
            f"SELECT * FROM (SELECT d.docstring_pk, d.docstring_kind, d.content, "
            f"d.is_current, d.embedding, d.embed_model_version, "
            f"c.comp_pk, c.repo_pk, c.component_id, c.relative_path, "
            f"d.embedding <=> %(vec)s::vector AS distance "
            f"FROM {ds} d JOIN {comp} c ON c.comp_pk = d.comp_pk) s",
            "repo_pk", repo_pk, query_vec, top_k, model_version,
        )

    def _vector_search(
        self, select_sql: str, scope_col: str, scope_val: Optional[int],
        query_vec: List[float], top_k: Optional[int], model_version: Optional[str],
    ) -> List[Dict[str, Any]]:
        if len(query_vec) != (self._schema_embed_dim or len(query_vec)):
            raise DimensionMismatchError(
                f"Query vector length {len(query_vec)} does not match schema "
                f"dimension {self._schema_embed_dim}."
            )
        where = ["embedding IS NOT NULL", "is_current"]
        if scope_val is not None:
            where.append(f"{scope_col} = %(scope)s")
        if model_version is not None:
            where.append("embed_model_version = %(mv)s")
        sql = (f"{select_sql} WHERE {' AND '.join(where)} "
               f"ORDER BY distance ASC LIMIT %(k)s;")
        with self.transaction() as cur:
            cur.execute(sql, {"vec": list(query_vec), "k": top_k or self._cfg.top_k,
                              "scope": scope_val, "mv": model_version})
            rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r.pop("embedding", None)  # only selected for the subquery filter
        return rows

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if not self._conn.closed:
            self._conn.close()
            logger.info("PostgreSQL connection closed.")

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_vector(self) -> None:
        """Register the pgvector type adapter once the extension exists."""
        if self._vector_registered:
            return
        try:
            register_vector(self._conn)
            self._vector_registered = True
        except psycopg2.ProgrammingError:
            # Extension not created yet — init_schema() will retry.
            logger.debug("pgvector type not found; deferring registration.")
        finally:
            if self._cur is None:
                self._conn.rollback()  # close the implicit read transaction

    def _validate_embedding_dim(self, result: Optional[EmbeddingResult]) -> None:
        if result is None or self._schema_embed_dim is None:
            return
        if len(result.vector) != self._schema_embed_dim:
            raise DimensionMismatchError(
                f"Embedding vector length {len(result.vector)} does not match "
                f"schema dimension {self._schema_embed_dim}."
            )

    @staticmethod
    def _jsonb(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _req_params(
        self, document_pk: int, rec: Dict[str, Any], emb: Optional[EmbeddingResult]
    ) -> Dict[str, Any]:
        params = {
            "document_pk":         document_pk,
            "requirement_id":      rec["requirement_id"],
            "id_source":           rec.get("id_source", "generated"),
            "raw_text":            rec["raw_text"],
            "normalized_text":     rec.get("normalized_text") or "",
            "retrieval_text":      rec.get("retrieval_text") or "",
            "title":               rec.get("title") or "",
            "type":                rec.get("type", "other"),
            "subtype":             rec.get("subtype") or "general",
            "review_required":     bool(rec.get("review_required", False)),
            "source_location":     self._jsonb(rec.get("source") or {}),
            "extraction_meta":     self._jsonb(
                {k: v for k, v in rec.items() if k not in _REQ_COLUMN_KEYS}
            ),
            "embedding":           emb.vector if emb else None,
            "embed_model_version": emb.model_version if emb else None,
            "embed_text_hash":     emb.text_hash if emb else None,
        }
        for col in _REQ_LIST_COLUMNS:
            params[col] = self._jsonb(rec.get(col) or [])
        return params

    def _comp_params(
        self, repo_pk: int, rec: Dict[str, Any], emb: Optional[EmbeddingResult]
    ) -> Dict[str, Any]:
        source = rec.get("source_code")
        return {
            "repo_pk":             repo_pk,
            "component_id":        rec["id"],
            "component_type":      rec["component_type"],
            "file_path":           rec.get("file_path") or rec["relative_path"],
            # Graphs made on Windows use backslashes; store POSIX paths.
            "relative_path":       rec["relative_path"].replace("\\", "/"),
            "start_line":          rec.get("start_line", 0),
            "end_line":            rec.get("end_line", 0),
            "signature":           rec.get("signature"),
            "parameters":          self._jsonb(rec.get("parameters") or []),
            "return_type":         rec.get("return_type"),
            "containing_class":    rec.get("containing_class"),
            "depends_on":          self._jsonb(sorted(rec.get("depends_on") or [])),
            "source_code":         source,
            "source_hash":         content_hash(source) if source else None,
            "embedding":           emb.vector if emb else None,
            "embed_model_version": emb.model_version if emb else None,
            "embed_text_hash":     emb.text_hash if emb else None,
        }
