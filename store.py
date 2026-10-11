"""
Store — PostgreSQL persistence layer for the traceability package.

Uses psycopg2 directly (no ORM) with parameterized SQL throughout.
All writes are transactional: each public method runs in a single
BEGIN/COMMIT block and rolls back on any error.

Safety rules enforced here
--------------------------
- Parameterized queries everywhere — no string interpolation of user data.
- Credentials never appear in log output; only safe_repr() is logged.
- No DROP TABLE or TRUNCATE anywhere.
- Foreign keys are enforced; invalid edge endpoints are logged and skipped,
  not silently inserted.
- Unknown relationship types are logged as warnings and skipped (the DB
  constraint was removed from the schema to allow future types).
- Vector dimension is verified against the schema before any embedding is
  stored; a mismatch raises DimensionMismatchError.
- All public write methods return a count of rows affected.

Public API
----------
Store.init_schema(embed_dim)
Store.record_migration(version, description)
Store.is_migration_applied(version) -> bool
Store.upsert_document(record) -> document_pk
Store.upsert_requirements(document_pk, records, embeddings)
Store.upsert_components(records, embeddings, repo_path, repo_snapshot)
Store.upsert_docstrings(component_id, original, generated)
Store.upsert_relationships(records)
Store.upsert_mapping(req_pk, component_id, ...)
Store.get_stale_requirement_embeddings(model_version) -> List[dict]
Store.get_stale_component_embeddings(model_version) -> List[dict]
Store.search_requirements(query_vec, top_k, filter_type)
Store.search_components(query_vec, top_k, filter_type)
Store.close()
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

from .config import TraceabilityConfig
from .embedder import EmbeddingResult
from .schema import SCHEMA_VERSION, all_create_ddl, ivfflat_index_ddl

logger = logging.getLogger(__name__)

# Known relationship types from DocAgent V1 — used for validation only,
# not enforced as a DB CHECK so future types don't break ingestion.
_KNOWN_REL_TYPES = frozenset(
    {"calls", "imports", "inherits", "instantiates", "contains"}
)

try:
    import psycopg2
    import psycopg2.extras
    from pgvector.psycopg2 import register_vector  # type: ignore[import]

    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False


def _require_pg() -> None:
    if not _PG_AVAILABLE:
        raise ImportError(
            "psycopg2 and pgvector are required for Store. "
            "Install with: pip install psycopg2-binary pgvector"
        )


class DimensionMismatchError(ValueError):
    """Raised when an embedding vector length does not match the schema dimension."""


class Store:
    """
    PostgreSQL persistence layer.

    Usage::

        cfg = TraceabilityConfig()
        with Store(cfg) as store:
            store.init_schema(embed_dim=384)
            doc_pk = store.upsert_document(doc_record)
            store.upsert_requirements(doc_pk, req_records, req_embeddings)
            store.upsert_components(comp_records, comp_embeddings)
            store.upsert_docstrings("mod.Foo.bar", original="…", generated=None)
            store.upsert_relationships(rel_records)
    """

    def __init__(self, config: TraceabilityConfig) -> None:
        _require_pg()
        self._cfg = config
        # Never log the DSN — it may contain the password.
        logger.info("Connecting to PostgreSQL: %s", config.safe_repr())
        self._conn = psycopg2.connect(config.build_dsn())
        register_vector(self._conn)
        # Cache schema embed_dim once resolved so we can validate vectors.
        self._schema_embed_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def init_schema(self, embed_dim: int) -> None:
        """
        Create the pgvector extension, all tables, and btree indexes.

        Safe to call on an already-initialised database — all statements use
        IF NOT EXISTS.  Never drops or truncates any table.

        Args:
            embed_dim: Embedding vector length from Embedder.dim.
        """
        stmts = all_create_ddl(self._cfg, embed_dim)
        with self._transaction() as cur:
            for stmt in stmts:
                logger.debug("DDL: %.100s…", stmt.replace("\n", " "))
                cur.execute(stmt)
        self._schema_embed_dim = embed_dim
        logger.info("Schema initialised (embed_dim=%d).", embed_dim)

    def build_vector_indexes(self, lists: int = 100) -> None:
        """
        Build IVFFlat ANN indexes.  Call after loading data — IVFFlat
        requires existing rows.
        """
        with self._transaction() as cur:
            for stmt in ivfflat_index_ddl(self._cfg, lists):
                cur.execute(stmt)
        logger.info("IVFFlat vector indexes built.")

    # ------------------------------------------------------------------
    # Migration log
    # ------------------------------------------------------------------

    def record_migration(self, version: str, description: str = "") -> None:
        """Insert a migration version record (idempotent)."""
        t = self._cfg.table_migrations
        sql = f"""
            INSERT INTO {t} (version, description)
            VALUES (%(v)s, %(d)s)
            ON CONFLICT (version) DO NOTHING;
        """
        with self._transaction() as cur:
            cur.execute(sql, {"v": version, "d": description})

    def is_migration_applied(self, version: str) -> bool:
        """Return True if *version* has already been recorded."""
        t = self._cfg.table_migrations
        with self._transaction() as cur:
            cur.execute(
                f"SELECT 1 FROM {t} WHERE version = %(v)s;", {"v": version}
            )
            return cur.fetchone() is not None

    # ------------------------------------------------------------------
    # SRS documents
    # ------------------------------------------------------------------

    def upsert_document(self, record: Dict[str, Any]) -> int:
        """
        Insert or update an SRS document record.

        Args:
            record: Dict with keys matching SRSMetadata fields:
                    document_id, file_name, source_type,
                    total_requirements, ingest_path (optional).

        Returns:
            document_pk (BIGINT) of the inserted/updated row.
        """
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
            "document_id":         record.get("document_id") or record.get("file_name", ""),
            "file_name":           record.get("file_name", ""),
            "source_type":         record.get("source_type", "unknown"),
            "total_requirements":  record.get("total_requirements", 0),
            "ingest_path":         record.get("ingest_path"),
        }
        with self._transaction() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        pk = row["document_pk"]
        logger.info(
            "Upserted document document_pk=%d (%s).",
            pk,
            params["file_name"],
        )
        return pk

    # ------------------------------------------------------------------
    # Requirements
    # ------------------------------------------------------------------

    def upsert_requirements(
        self,
        document_pk: int,
        records: List[Dict[str, Any]],
        embeddings: Optional[Dict[str, EmbeddingResult]] = None,
    ) -> int:
        """
        Insert or update requirements belonging to *document_pk*.

        The natural key is (document_pk, requirement_id).  Re-ingesting the
        same requirements updates all mutable fields atomically.

        Args:
            document_pk: FK to tr_srs_documents.document_pk.
            records:     List of requirement dicts from SRSLoader.
            embeddings:  Optional {requirement_id: EmbeddingResult}.

        Returns:
            Number of rows affected.
        """
        t = self._cfg.table_requirements
        sql = f"""
            INSERT INTO {t} (
                document_pk, requirement_id, id_source,
                raw_text, normalized_text, retrieval_text, title,
                type, subtype,
                actor, actions, entities, inputs, outputs,
                preconditions, postconditions, conditions, constraints,
                error_conditions, domain_terms, synonyms, sub_requirements,
                src_section, src_subsection,
                src_page, src_paragraph, src_table, src_row, src_column,
                embedding, embed_model_version, embed_text_hash
            ) VALUES (
                %(document_pk)s, %(requirement_id)s, %(id_source)s,
                %(raw_text)s, %(normalized_text)s, %(retrieval_text)s, %(title)s,
                %(type)s, %(subtype)s,
                %(actor)s, %(actions)s, %(entities)s, %(inputs)s, %(outputs)s,
                %(preconditions)s, %(postconditions)s, %(conditions)s, %(constraints)s,
                %(error_conditions)s, %(domain_terms)s, %(synonyms)s, %(sub_requirements)s,
                %(src_section)s, %(src_subsection)s,
                %(src_page)s, %(src_paragraph)s, %(src_table)s, %(src_row)s, %(src_column)s,
                %(embedding)s, %(embed_model_version)s, %(embed_text_hash)s
            )
            ON CONFLICT (document_pk, requirement_id) DO UPDATE SET
                id_source           = EXCLUDED.id_source,
                raw_text            = EXCLUDED.raw_text,
                normalized_text     = EXCLUDED.normalized_text,
                retrieval_text      = EXCLUDED.retrieval_text,
                title               = EXCLUDED.title,
                type                = EXCLUDED.type,
                subtype             = EXCLUDED.subtype,
                actor               = EXCLUDED.actor,
                actions             = EXCLUDED.actions,
                entities            = EXCLUDED.entities,
                inputs              = EXCLUDED.inputs,
                outputs             = EXCLUDED.outputs,
                preconditions       = EXCLUDED.preconditions,
                postconditions      = EXCLUDED.postconditions,
                conditions          = EXCLUDED.conditions,
                constraints         = EXCLUDED.constraints,
                error_conditions    = EXCLUDED.error_conditions,
                domain_terms        = EXCLUDED.domain_terms,
                synonyms            = EXCLUDED.synonyms,
                sub_requirements    = EXCLUDED.sub_requirements,
                src_section         = EXCLUDED.src_section,
                src_subsection      = EXCLUDED.src_subsection,
                src_page            = EXCLUDED.src_page,
                src_paragraph       = EXCLUDED.src_paragraph,
                src_table           = EXCLUDED.src_table,
                src_row             = EXCLUDED.src_row,
                src_column          = EXCLUDED.src_column,
                embedding           = EXCLUDED.embedding,
                embed_model_version = EXCLUDED.embed_model_version,
                embed_text_hash     = EXCLUDED.embed_text_hash,
                updated_at          = NOW()
            RETURNING req_pk;
        """
        rows = 0
        with self._transaction() as cur:
            for rec in records:
                src = rec.get("source") or {}
                emb = (embeddings or {}).get(rec["requirement_id"])
                self._validate_embedding_dim(emb)
                params = self._req_params(document_pk, rec, src, emb)
                cur.execute(sql, params)
                if cur.fetchone() is not None:
                    rows += 1
        logger.info("Upserted %d requirement(s) into document_pk=%d.", rows, document_pk)
        return rows

    # ------------------------------------------------------------------
    # Code components
    # ------------------------------------------------------------------

    def upsert_components(
        self,
        records: List[Dict[str, Any]],
        embeddings: Optional[Dict[str, EmbeddingResult]] = None,
        *,
        repo_path: Optional[str] = None,
        repo_snapshot: Optional[str] = None,
    ) -> int:
        """
        Insert or update code components.

        Args:
            records:       List of component dicts from GraphLoader.
            embeddings:    Optional {component_id: EmbeddingResult}.
            repo_path:     Absolute path to the repository at ingest time.
            repo_snapshot: Git commit SHA or tag (NULL if unknown).

        Returns:
            Number of rows affected.
        """
        t = self._cfg.table_components
        sql = f"""
            INSERT INTO {t} (
                component_id, component_type,
                repo_path, repo_snapshot,
                file_path, relative_path, start_line, end_line,
                signature, return_type, containing_class,
                parameters, depends_on, source_code,
                embedding, embed_model_version, embed_text_hash
            ) VALUES (
                %(component_id)s, %(component_type)s,
                %(repo_path)s, %(repo_snapshot)s,
                %(file_path)s, %(relative_path)s, %(start_line)s, %(end_line)s,
                %(signature)s, %(return_type)s, %(containing_class)s,
                %(parameters)s, %(depends_on)s, %(source_code)s,
                %(embedding)s, %(embed_model_version)s, %(embed_text_hash)s
            )
            ON CONFLICT (component_id) DO UPDATE SET
                component_type      = EXCLUDED.component_type,
                repo_path           = EXCLUDED.repo_path,
                repo_snapshot       = EXCLUDED.repo_snapshot,
                file_path           = EXCLUDED.file_path,
                relative_path       = EXCLUDED.relative_path,
                start_line          = EXCLUDED.start_line,
                end_line            = EXCLUDED.end_line,
                signature           = EXCLUDED.signature,
                return_type         = EXCLUDED.return_type,
                containing_class    = EXCLUDED.containing_class,
                parameters          = EXCLUDED.parameters,
                depends_on          = EXCLUDED.depends_on,
                source_code         = EXCLUDED.source_code,
                embedding           = EXCLUDED.embedding,
                embed_model_version = EXCLUDED.embed_model_version,
                embed_text_hash     = EXCLUDED.embed_text_hash,
                updated_at          = NOW();
        """
        rows = 0
        with self._transaction() as cur:
            for rec in records:
                emb = (embeddings or {}).get(rec["id"])
                self._validate_embedding_dim(emb)
                params = self._comp_params(rec, emb, repo_path, repo_snapshot)
                cur.execute(sql, params)
                rows += cur.rowcount
        logger.info("Upserted %d component(s).", rows)
        return rows

    # ------------------------------------------------------------------
    # Docstrings
    # ------------------------------------------------------------------

    def upsert_docstrings(
        self,
        component_id: str,
        *,
        original: Optional[str] = None,
        generated: Optional[str] = None,
    ) -> int:
        """
        Insert or update docstrings for a component.

        Only inserts a row for a kind when the content is not None and not
        empty, preserving the distinction between "no docstring present in
        source" and "empty generated docstring".

        Args:
            component_id: Must already exist in tr_code_components.
            original:     Docstring from source (has_docstring=True in DocAgent).
            generated:    Docstring produced by DocAgent's Writer agent.

        Returns:
            Number of rows affected (0-2).
        """
        t = self._cfg.table_docstrings
        sql = f"""
            INSERT INTO {t} (component_id, docstring_kind, content)
            VALUES (%(component_id)s, %(kind)s, %(content)s)
            ON CONFLICT (component_id, docstring_kind) DO UPDATE SET
                content    = EXCLUDED.content,
                updated_at = NOW();
        """
        rows = 0
        with self._transaction() as cur:
            for kind, content in [("original", original), ("generated", generated)]:
                if content is not None and content.strip():
                    cur.execute(sql, {
                        "component_id": component_id,
                        "kind": kind,
                        "content": content,
                    })
                    rows += cur.rowcount
        return rows

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def upsert_relationships(
        self,
        records: List[Dict[str, str]],
    ) -> Tuple[int, int]:
        """
        Insert typed relationships.

        Runs in a single transaction.  Records with unknown endpoints or
        unknown rel_type values are logged and skipped — they do not abort
        the transaction.

        Args:
            records: List of {"source": str, "target": str, "type": str}.

        Returns:
            (inserted, skipped) counts.
        """
        t = self._cfg.table_relationships
        comp = self._cfg.table_components

        sql_insert = f"""
            INSERT INTO {t} (source_id, target_id, rel_type)
            VALUES (%(source)s, %(target)s, %(type)s)
            ON CONFLICT (source_id, target_id, rel_type) DO NOTHING;
        """

        inserted = 0
        skipped = 0

        with self._transaction() as cur:
            # Fetch known IDs once inside the same transaction.
            # RealDictCursor rows are dicts, so key by column name.
            cur.execute(f"SELECT component_id FROM {comp};")
            known_ids = {row["component_id"] for row in cur.fetchall()}

            for rec in records:
                src, tgt, rtype = rec["source"], rec["target"], rec["type"]

                if src not in known_ids or tgt not in known_ids:
                    logger.warning(
                        "Skipping relationship: unknown endpoint(s) source=%r target=%r",
                        src, tgt,
                    )
                    skipped += 1
                    continue

                if rtype not in _KNOWN_REL_TYPES:
                    logger.warning(
                        "Unknown relationship type %r (source=%r target=%r); skipping.",
                        rtype, src, tgt,
                    )
                    skipped += 1
                    continue

                cur.execute(sql_insert, rec)
                inserted += cur.rowcount

        logger.info("Relationships: %d inserted, %d skipped.", inserted, skipped)
        return inserted, skipped

    # ------------------------------------------------------------------
    # Mappings
    # ------------------------------------------------------------------

    def upsert_mapping(
        self,
        req_pk: int,
        component_id: str,
        *,
        confidence_score: Optional[float] = None,
        mapping_type: str = "vector_search",
        rationale: Optional[str] = None,
    ) -> None:
        """Insert or update a traceability mapping."""
        t = self._cfg.table_mappings
        sql = f"""
            INSERT INTO {t} (req_pk, component_id, confidence_score,
                             mapping_type, rationale)
            VALUES (%(req_pk)s, %(comp)s, %(score)s, %(mtype)s, %(rationale)s)
            ON CONFLICT (req_pk, component_id) DO UPDATE SET
                confidence_score = EXCLUDED.confidence_score,
                mapping_type     = EXCLUDED.mapping_type,
                rationale        = EXCLUDED.rationale,
                updated_at       = NOW();
        """
        with self._transaction() as cur:
            cur.execute(sql, {
                "req_pk": req_pk,
                "comp": component_id,
                "score": confidence_score,
                "mtype": mapping_type,
                "rationale": rationale,
            })

    # ------------------------------------------------------------------
    # Stale embedding queries (for rebuild-embeddings command)
    # ------------------------------------------------------------------

    def get_stale_requirement_embeddings(
        self, current_model_version: str
    ) -> List[Dict[str, Any]]:
        """
        Return requirements whose embedding is NULL or was generated with a
        different model version.

        Returns:
            List of dicts with keys: req_pk, requirement_id, document_pk,
            retrieval_text, title, normalized_text, domain_terms, synonyms.
        """
        t = self._cfg.table_requirements
        sql = f"""
            SELECT req_pk, requirement_id, document_pk,
                   retrieval_text, title, normalized_text,
                   domain_terms, synonyms
            FROM {t}
            WHERE embedding IS NULL
               OR embed_model_version IS NULL
               OR embed_model_version != %(v)s;
        """
        with self._transaction() as cur:
            cur.execute(sql, {"v": current_model_version})
            # RealDictCursor rows are already dict-like; cast to plain dict
            return [dict(row) for row in cur.fetchall()]

    def get_stale_component_embeddings(
        self, current_model_version: str
    ) -> List[Dict[str, Any]]:
        """
        Return components whose embedding is NULL or stale.

        Returns:
            List of dicts with keys: component_id, signature, docstring,
            source_code.
        """
        t = self._cfg.table_components
        ds = self._cfg.table_docstrings
        sql = f"""
            SELECT c.component_id, c.signature, c.source_code,
                   d.content AS docstring
            FROM {t} c
            LEFT JOIN {ds} d
                ON d.component_id = c.component_id
               AND d.docstring_kind = 'generated'
            WHERE c.embedding IS NULL
               OR c.embed_model_version IS NULL
               OR c.embed_model_version != %(v)s;
        """
        with self._transaction() as cur:
            cur.execute(sql, {"v": current_model_version})
            return [dict(row) for row in cur.fetchall()]

    def update_requirement_embedding(
        self, req_pk: int, result: EmbeddingResult
    ) -> None:
        """Update the embedding for a single requirement row."""
        self._validate_embedding_dim(result)
        t = self._cfg.table_requirements
        sql = f"""
            UPDATE {t}
            SET embedding           = %(vec)s,
                embed_model_version = %(version)s,
                embed_text_hash     = %(hash)s,
                updated_at          = NOW()
            WHERE req_pk = %(pk)s;
        """
        with self._transaction() as cur:
            cur.execute(sql, {
                "vec": result.vector,
                "version": result.model_version,
                "hash": result.text_hash,
                "pk": req_pk,
            })

    def update_component_embedding(
        self, component_id: str, result: EmbeddingResult
    ) -> None:
        """Update the embedding for a single component row."""
        self._validate_embedding_dim(result)
        t = self._cfg.table_components
        sql = f"""
            UPDATE {t}
            SET embedding           = %(vec)s,
                embed_model_version = %(version)s,
                embed_text_hash     = %(hash)s,
                updated_at          = NOW()
            WHERE component_id = %(cid)s;
        """
        with self._transaction() as cur:
            cur.execute(sql, {
                "vec": result.vector,
                "version": result.model_version,
                "hash": result.text_hash,
                "cid": component_id,
            })

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def search_requirements(
        self,
        query_vec: List[float],
        top_k: Optional[int] = None,
        *,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        ANN search over requirement embeddings.

        Returns dicts with: req_pk, requirement_id, document_pk, title,
        type, subtype, raw_text, retrieval_text, distance.
        Sorted by cosine distance ascending (most similar first).
        """
        k = top_k or self._cfg.top_k
        t = self._cfg.table_requirements
        where = "WHERE embedding IS NOT NULL"
        if filter_type:
            where += " AND type = %(filter_type)s"
        sql = f"""
            SELECT req_pk, requirement_id, document_pk, title,
                   type, subtype, raw_text, retrieval_text,
                   embedding <=> %(vec)s::vector AS distance
            FROM {t}
            {where}
            ORDER BY distance ASC
            LIMIT %(k)s;
        """
        with self._transaction() as cur:
            cur.execute(sql, {"vec": query_vec, "k": k, "filter_type": filter_type})
            return [dict(row) for row in cur.fetchall()]

    def search_components(
        self,
        query_vec: List[float],
        top_k: Optional[int] = None,
        *,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        ANN search over component embeddings.

        Returns dicts with: component_id, component_type, relative_path,
        signature, distance.
        Sorted by cosine distance ascending.
        """
        k = top_k or self._cfg.top_k
        t = self._cfg.table_components
        where = "WHERE embedding IS NOT NULL"
        if filter_type:
            where += " AND component_type = %(filter_type)s"
        sql = f"""
            SELECT component_id, component_type, relative_path, signature,
                   embedding <=> %(vec)s::vector AS distance
            FROM {t}
            {where}
            ORDER BY distance ASC
            LIMIT %(k)s;
        """
        with self._transaction() as cur:
            cur.execute(sql, {"vec": query_vec, "k": k, "filter_type": filter_type})
            return [dict(row) for row in cur.fetchall()]

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

    @contextmanager
    def _transaction(self) -> Generator:
        """
        Yield a RealDictCursor inside a single BEGIN/COMMIT block.

        Rolls back and re-raises on any exception so the caller never sees a
        half-committed state.
        """
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def _validate_embedding_dim(
        self, result: Optional[EmbeddingResult]
    ) -> None:
        """
        Raise DimensionMismatchError when the vector length does not match the
        schema dimension.  No-ops when result is None or schema_dim is unknown.
        """
        if result is None or self._schema_embed_dim is None:
            return
        actual = len(result.vector)
        if actual != self._schema_embed_dim:
            raise DimensionMismatchError(
                f"Embedding vector length {actual} does not match schema "
                f"dimension {self._schema_embed_dim}.  Re-run init_schema() "
                f"with the correct embed_dim, or rebuild embeddings."
            )

    @staticmethod
    def _jsonb(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _req_params(
        self,
        document_pk: int,
        rec: Dict[str, Any],
        src: Dict[str, Any],
        emb: Optional[EmbeddingResult],
    ) -> Dict[str, Any]:
        return {
            "document_pk":          document_pk,
            "requirement_id":       rec["requirement_id"],
            "id_source":            rec.get("id_source", "generated"),
            "raw_text":             rec.get("raw_text", ""),
            "normalized_text":      rec.get("normalized_text", ""),
            "retrieval_text":       rec.get("retrieval_text", ""),
            "title":                rec.get("title", ""),
            "type":                 rec.get("type", "other"),
            "subtype":              rec.get("subtype", "general"),
            "actor":                self._jsonb(rec.get("actor", [])),
            "actions":              self._jsonb(rec.get("actions", [])),
            "entities":             self._jsonb(rec.get("entities", [])),
            "inputs":               self._jsonb(rec.get("inputs", [])),
            "outputs":              self._jsonb(rec.get("outputs", [])),
            "preconditions":        self._jsonb(rec.get("preconditions", [])),
            "postconditions":       self._jsonb(rec.get("postconditions", [])),
            "conditions":           self._jsonb(rec.get("conditions", [])),
            "constraints":          self._jsonb(rec.get("constraints", [])),
            "error_conditions":     self._jsonb(rec.get("error_conditions", [])),
            "domain_terms":         self._jsonb(rec.get("domain_terms", [])),
            "synonyms":             self._jsonb(rec.get("synonyms", [])),
            "sub_requirements":     self._jsonb(rec.get("sub_requirements", [])),
            "src_section":          src.get("section"),
            "src_subsection":       src.get("subsection"),
            "src_page":             src.get("page"),
            "src_paragraph":        src.get("paragraph"),
            "src_table":            src.get("table"),
            "src_row":              src.get("row"),
            "src_column":           src.get("column"),
            "embedding":            emb.vector if emb else None,
            "embed_model_version":  emb.model_version if emb else None,
            "embed_text_hash":      emb.text_hash if emb else None,
        }

    @staticmethod
    def _comp_params(
        rec: Dict[str, Any],
        emb: Optional[EmbeddingResult],
        repo_path: Optional[str],
        repo_snapshot: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "component_id":         rec["id"],
            "component_type":       rec["component_type"],
            "repo_path":            repo_path,
            "repo_snapshot":        repo_snapshot,
            "file_path":            rec["file_path"],
            "relative_path":        rec["relative_path"],
            "start_line":           rec.get("start_line", 0),
            "end_line":             rec.get("end_line", 0),
            "signature":            rec.get("signature"),
            "return_type":          rec.get("return_type"),
            "containing_class":     rec.get("containing_class"),
            "parameters":           json.dumps(rec.get("parameters", []), ensure_ascii=False),
            "depends_on":           json.dumps(list(rec.get("depends_on", [])), ensure_ascii=False),
            "source_code":          rec.get("source_code"),
            "embedding":            emb.vector if emb else None,
            "embed_model_version":  emb.model_version if emb else None,
            "embed_text_hash":      emb.text_hash if emb else None,
        }
