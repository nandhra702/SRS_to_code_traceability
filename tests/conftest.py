"""
Shared pytest fixtures for the traceability test suite.

Fixture categories
------------------
- cfg              : TraceabilityConfig pointing at the disposable test DB,
                     with test_-prefixed tables and a tiny embedding dimension
- Real fixtures    : tests/fixtures/ holds *real* outputs —
                       simple_v1_graph.json      DocAgent V1 parser run on raw_test_repo_simple
                       simple_legacy_graph.json  pre-V1 DocAgent graph of the same repo
                       basic_srs_requirements.json / text_input_requirements.json
                                                 SRS Agent CLI output
                       simple_repo/              the source tree the graphs describe
                     (absolute file paths were replaced with "<repo>/…")
- Small records    : a few records in the real V1 / SRS shapes for unit tests
- pg_available()   : runtime check used by the requires_pg marker
- live_store       : Store on clean test tables (requires_pg)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest

from traceability.config import TraceabilityConfig

FIXTURES = Path(__file__).parent / "fixtures"

FAKE_DIM = 4
FAKE_VEC = [0.1, 0.2, 0.3, 0.4]

TEST_TABLES = dict(
    table_migrations="test_tr_migrations",
    table_srs_documents="test_tr_srs_documents",
    table_requirements="test_tr_requirements",
    table_repositories="test_tr_repositories",
    table_components="test_tr_code_components",
    table_docstrings="test_tr_docstrings",
    table_relationships="test_tr_relationships",
    table_trace_links="test_tr_trace_links",
    table_verifications="test_tr_verifications",
)


def make_cfg(**overrides: Any) -> TraceabilityConfig:
    """Isolated test config — traceability_test DB, test_ tables, embed_dim=4."""
    params: Dict[str, Any] = dict(
        pg_dbname=os.getenv("TEST_PG_DBNAME", "traceability_test"),
        pg_user=os.getenv("TEST_PG_USER", "postgres"),
        pg_password=os.getenv("TEST_PG_PASSWORD", ""),
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        embed_model_version="test-v1",
        embed_dim=FAKE_DIM,
        **TEST_TABLES,
    )
    params.update(overrides)
    return TraceabilityConfig(**params)


@pytest.fixture()
def cfg() -> TraceabilityConfig:
    return make_cfg()


# ---------------------------------------------------------------------------
# Real fixture files
# ---------------------------------------------------------------------------


@pytest.fixture()
def real_v1_graph() -> Path:
    return FIXTURES / "simple_v1_graph.json"


@pytest.fixture()
def real_legacy_graph() -> Path:
    return FIXTURES / "simple_legacy_graph.json"


@pytest.fixture()
def real_srs() -> Path:
    return FIXTURES / "basic_srs_requirements.json"


@pytest.fixture()
def real_text_srs() -> Path:
    return FIXTURES / "text_input_requirements.json"


@pytest.fixture()
def repo_copy(tmp_path) -> Path:
    """A writable copy of the source tree the fixture graphs describe."""
    dst = tmp_path / "simple_repo"
    shutil.copytree(FIXTURES / "simple_repo", dst)
    return dst


# ---------------------------------------------------------------------------
# Small records in the real shapes
# ---------------------------------------------------------------------------


def v1_component(cid: str, ctype: str = "function", **fields: Any) -> Dict[str, Any]:
    """A component record shaped like DocAgent V1 CodeComponent.to_dict()."""
    module = cid.split(".")[0]
    name = cid.split(".")[-1]
    rec = {
        "id": cid,
        "component_type": ctype,
        "file_path": f"<repo>/{module}.py",
        "relative_path": f"{module}.py",
        "depends_on": [],
        "source_code": f"def {name}():\n    return 1",
        "start_line": 1,
        "end_line": 2,
        "has_docstring": False,
        "docstring": "",
        "signature": f"{name}()",
        "parameters": [],
        "return_type": None,
        "containing_class": None,
        "relationships": [],
    }
    rec.update(fields)
    return rec


def requirement(rid: str, **fields: Any) -> Dict[str, Any]:
    """A requirement record shaped like Requirement.model_dump(mode='json')."""
    rec = {
        "requirement_id": rid,
        "id_source": "source",
        "raw_text": f"{rid}: The system shall validate submitted credentials.",
        "normalized_text": "The system shall validate submitted credentials.",
        "type": "functional",
        "subtype": "security",
        "detection_confidence": "high",
        "detection_reason": "explicit obligation wording: shall",
        "candidate_id": "C-0001",
        "candidate_decision": "keep",
        "candidate_confidence": "high",
        "candidate_reason": "Fast path: strong requirement signal",
        "review_required": False,
        "title": "Validate Credentials",
        "actor": ["system"], "actions": ["validate"], "entities": ["credentials"],
        "inputs": [], "outputs": [], "preconditions": [], "postconditions": [],
        "conditions": [], "constraints": [], "error_conditions": [],
        "domain_terms": ["authentication"], "synonyms": ["authenticate"],
        "sub_requirements": [],
        "retrieval_text": "validate submitted credentials",
        "source": {
            "document_id": None, "file_name": "Basic_SRS_Document.docx",
            "section": None, "subsection": None, "page": None, "paragraph": 50,
            "table": None, "row": None, "column": None,
        },
    }
    rec.update(fields)
    return rec


def write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# PostgreSQL availability
# ---------------------------------------------------------------------------


def pg_available() -> bool:
    """True only when psycopg2, pgvector, and a reachable PG server exist."""
    try:
        import psycopg2
        from pgvector.psycopg2 import register_vector  # noqa: F401
        conn = psycopg2.connect(make_cfg().build_dsn())
        conn.close()
        return True
    except Exception:
        return False


_PG_SKIP_REASON = (
    "PostgreSQL/pgvector not available. Start the server and set "
    "TEST_PG_DBNAME / TEST_PG_USER / TEST_PG_PASSWORD to run DB tests."
)

requires_pg = pytest.mark.skipif(not pg_available(), reason=_PG_SKIP_REASON)


# ---------------------------------------------------------------------------
# Live DB fixture
# ---------------------------------------------------------------------------


def reset_tables(cfg: TraceabilityConfig) -> None:
    """Drop this config's (test-only) tables so a test starts clean."""
    import psycopg2

    tables = [getattr(cfg, attr) for attr in reversed(list(TEST_TABLES))]
    # Safety: never drop anything that is not a test table.
    assert all(t.startswith("test_") for t in tables), tables
    conn = psycopg2.connect(cfg.build_dsn())
    conn.autocommit = True
    with conn.cursor() as cur:
        for t in tables:
            cur.execute(f"DROP TABLE IF EXISTS {t} CASCADE;")
    conn.close()


@pytest.fixture()
def live_store(cfg):
    """A Store on freshly reset test tables, schema initialised at FAKE_DIM."""
    from traceability.store import Store

    reset_tables(cfg)
    store = Store(cfg)
    store.init_schema(FAKE_DIM)
    yield store
    store.close()
