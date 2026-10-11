"""
Shared pytest fixtures for the traceability test suite.

Fixture categories
------------------
- cfg             : TraceabilityConfig pointing at an isolated test DB
- Sample records  : verbatim from real DocAgent/SRS output files
- JSON temp files : write sample records to tmp_path and return paths
- Real file paths : point at actual on-disk output (skip if absent)
- pg_available()  : runtime check used by requires_pg marker
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pytest

from traceability.config import TraceabilityConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg() -> TraceabilityConfig:
    """Isolated test config — points at traceability_test DB, embed_dim preset."""
    return TraceabilityConfig(
        pg_dbname=os.getenv("TEST_PG_DBNAME", "traceability_test"),
        pg_user=os.getenv("TEST_PG_USER", "postgres"),
        pg_password=os.getenv("TEST_PG_PASSWORD", ""),
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        embed_model_version="test-v1",
        embed_dim=4,   # tiny dim so no real model needed in unit tests
        table_migrations="test_tr_migrations",
        table_srs_documents="test_tr_srs_documents",
        table_requirements="test_tr_requirements",
        table_components="test_tr_code_components",
        table_docstrings="test_tr_docstrings",
        table_relationships="test_tr_relationships",
        table_mappings="test_tr_mappings",
    )


# ---------------------------------------------------------------------------
# Sample component records (from DocAgent's raw_test_repo_simple)
# ---------------------------------------------------------------------------

COMPONENT_RECORDS: Dict[str, Dict[str, Any]] = {
    "main.main_function": {
        "id": "main.main_function",
        "component_type": "function",
        "file_path": "/project/data/raw_test_repo_simple/main.py",
        "relative_path": "main.py",
        "depends_on": ["main.utility_function", "helper.HelperClass"],
        "source_code": "def main_function():\n    h = HelperClass()\n    return h.process_data(utility_function())",
        "start_line": 4, "end_line": 9,
        "has_docstring": False, "docstring": "",
        "signature": "main_function()", "parameters": [], "return_type": None,
        "containing_class": None, "relationships": [],
    },
    "main.utility_function": {
        "id": "main.utility_function",
        "component_type": "function",
        "file_path": "/project/data/raw_test_repo_simple/main.py",
        "relative_path": "main.py",
        "depends_on": [],
        "source_code": "def utility_function():\n    return 42",
        "start_line": 11, "end_line": 12,
        "has_docstring": False, "docstring": "",
        "signature": "utility_function()", "parameters": [], "return_type": None,
        "containing_class": None, "relationships": [],
    },
    "helper.HelperClass": {
        "id": "helper.HelperClass",
        "component_type": "class",
        "file_path": "/project/data/raw_test_repo_simple/helper.py",
        "relative_path": "helper.py",
        "depends_on": ["helper.HelperClass.process_data"],
        "source_code": "class HelperClass:\n    pass",
        "start_line": 1, "end_line": 10,
        "has_docstring": True,
        "docstring": "A helper class for processing.",
        "signature": None, "parameters": [], "return_type": None,
        "containing_class": None,
        "relationships": [
            {"source": "helper.HelperClass",
             "target": "helper.HelperClass.process_data",
             "type": "contains"},
        ],
    },
    "helper.HelperClass.process_data": {
        "id": "helper.HelperClass.process_data",
        "component_type": "method",
        "file_path": "/project/data/raw_test_repo_simple/helper.py",
        "relative_path": "helper.py",
        "depends_on": [],
        "source_code": "def process_data(self, x):\n    return x * 2",
        "start_line": 2, "end_line": 3,
        "has_docstring": False, "docstring": "",
        "signature": "process_data(x)",
        "parameters": [{"name": "x", "annotation": None, "default": None}],
        "return_type": None,
        "containing_class": "helper.HelperClass",
        "relationships": [],
    },
}

# Legacy flat format — no top-level "components" key
LEGACY_GRAPH: Dict[str, Any] = {k: v for k, v in COMPONENT_RECORDS.items()}

# V1 format — explicit "components" + "relationships" keys
V1_GRAPH: Dict[str, Any] = {
    "components": {k: v for k, v in COMPONENT_RECORDS.items()},
    "relationships": [
        {"source": "helper.HelperClass",
         "target": "helper.HelperClass.process_data",
         "type": "contains"},
        {"source": "main.main_function",
         "target": "helper.HelperClass",
         "type": "instantiates"},
        {"source": "main.main_function",
         "target": "main.utility_function",
         "type": "calls"},
    ],
}

# ---------------------------------------------------------------------------
# Sample requirement records (verbatim from Basic_SRS_Document output)
# ---------------------------------------------------------------------------

REQUIREMENT_RECORDS: List[Dict[str, Any]] = [
    {
        "requirement_id": "REQ-001",
        "id_source": "generated",
        "raw_text": "The purpose of this SRS is to clearly describe what the system shall do.",
        "normalized_text": "The requirements are written so they can be tested independently.",
        "type": "non-functional",
        "subtype": "testability",
        "title": "Requirements Written for Testability",
        "actor": ["writer"], "actions": ["write requirements"],
        "entities": ["requirements"], "inputs": [], "outputs": [],
        "preconditions": [], "postconditions": [], "conditions": [],
        "constraints": [], "error_conditions": [],
        "domain_terms": ["requirements"], "synonyms": [],
        "sub_requirements": [],
        "retrieval_text": "requirements testing",
        "source": {
            "document_id": None, "file_name": "Basic_SRS_Document.docx",
            "section": None, "subsection": None,
            "page": None, "paragraph": 11,
            "table": None, "row": None, "column": None,
        },
    },
    {
        "requirement_id": "FR-01.1",
        "id_source": "source",
        "raw_text": "FR-01.1: The system shall provide a login interface.",
        "normalized_text": "The system shall provide a login interface.",
        "type": "functional",
        "subtype": "interface",
        "title": "Login Interface",
        "actor": ["user"], "actions": ["provide"],
        "entities": ["system", "login interface"], "inputs": [], "outputs": [],
        "preconditions": [], "postconditions": [], "conditions": [],
        "constraints": [], "error_conditions": [],
        "domain_terms": ["login", "interface", "authentication"],
        "synonyms": ["login page", "sign in"],
        "sub_requirements": [],
        "retrieval_text": "login interface user provide",
        "source": {
            "document_id": None, "file_name": "Basic_SRS_Document.docx",
            "section": None, "subsection": None,
            "page": None, "paragraph": 49,
            "table": None, "row": None, "column": None,
        },
    },
    {
        "requirement_id": "FR-01.2",
        "id_source": "source",
        "raw_text": "FR-01.2: The system shall validate submitted credentials.",
        "normalized_text": "The system shall validate submitted credentials.",
        "type": "functional",
        "subtype": "security",
        "title": "Validate Credentials",
        "actor": ["system", "user"], "actions": ["validate", "submit"],
        "entities": ["credentials", "system"],
        "inputs": ["submitted credentials"], "outputs": ["validation result"],
        "preconditions": ["credentials provided"], "postconditions": ["result returned"],
        "conditions": ["credentials valid"], "constraints": [],
        "error_conditions": ["invalid credentials"],
        "domain_terms": ["authentication", "authorization"],
        "synonyms": ["validate", "authenticate"],
        "sub_requirements": [],
        "retrieval_text": "system validate submitted credentials",
        "source": {
            "document_id": None, "file_name": "Basic_SRS_Document.docx",
            "section": None, "subsection": None,
            "page": None, "paragraph": 50,
            "table": None, "row": None, "column": None,
        },
    },
]

SRS_DOCUMENT_RECORD = {
    "document_id":        "Basic_SRS_Document",
    "file_name":          "Basic_SRS_Document.docx",
    "source_type":        "docx",
    "total_requirements": len(REQUIREMENT_RECORDS),
    "ingest_path":        "/data/input/Basic_SRS_Document.docx",
}

# A minimal fake EmbeddingResult for tests that need one
FAKE_DIM = 4
FAKE_VEC = [0.1, 0.2, 0.3, 0.4]


# ---------------------------------------------------------------------------
# JSON temp files
# ---------------------------------------------------------------------------


@pytest.fixture()
def legacy_graph_file(tmp_path) -> str:
    p = tmp_path / "legacy_graph.json"
    p.write_text(json.dumps(LEGACY_GRAPH), encoding="utf-8")
    return str(p)


@pytest.fixture()
def v1_graph_file(tmp_path) -> str:
    p = tmp_path / "v1_graph.json"
    p.write_text(json.dumps(V1_GRAPH), encoding="utf-8")
    return str(p)


@pytest.fixture()
def srs_output_file(tmp_path) -> str:
    p = tmp_path / "requirements.txt"
    p.write_text(json.dumps(REQUIREMENT_RECORDS), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Real on-disk files (skip if absent)
# ---------------------------------------------------------------------------


@pytest.fixture()
def real_graph_file() -> str:
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "../../DocAgent_Major_Project/output/dependency_graphs"
        "/raw_test_repo_simple_dependency_graph.json",
    ))
    if not os.path.isfile(path):
        pytest.skip("DocAgent graph not found. Run generate_docstrings.py first.")
    return path


@pytest.fixture()
def real_srs_file() -> str:
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "../../SRS_agent/SRSAgent_Major_Project/output"
        "/Basic_SRS_Document_requirements.txt",
    ))
    if not os.path.isfile(path):
        pytest.skip("SRS output not found. Run app.py first.")
    return path


# ---------------------------------------------------------------------------
# PostgreSQL availability
# ---------------------------------------------------------------------------


def pg_available() -> bool:
    """True only when psycopg2, pgvector, and a reachable PG server exist."""
    try:
        import psycopg2
        from pgvector.psycopg2 import register_vector  # noqa: F401
        cfg = TraceabilityConfig(
            pg_dbname=os.getenv("TEST_PG_DBNAME", "traceability_test"),
            pg_user=os.getenv("TEST_PG_USER", "postgres"),
            pg_password=os.getenv("TEST_PG_PASSWORD", ""),
        )
        conn = psycopg2.connect(cfg.build_dsn())
        conn.close()
        return True
    except Exception:
        return False


_PG_SKIP_REASON = (
    "PostgreSQL/pgvector not available. "
    "Start the server and set TEST_PG_DBNAME / TEST_PG_USER / "
    "TEST_PG_PASSWORD (or TRACEABILITY_PG_DSN) to run DB tests."
)

requires_pg = pytest.mark.skipif(not pg_available(), reason=_PG_SKIP_REASON)


# ---------------------------------------------------------------------------
# Live DB fixture — creates a fresh isolated schema per test class
# ---------------------------------------------------------------------------


@pytest.fixture()
def live_store(cfg):
    """
    Yield a Store connected to an isolated test schema.

    Drops all test tables before yielding so each test starts clean.
    Requires requires_pg to have passed.
    """
    import psycopg2
    from pgvector.psycopg2 import register_vector
    from traceability.store import Store

    conn = psycopg2.connect(cfg.build_dsn())
    register_vector(conn)
    conn.autocommit = True
    cur = conn.cursor()

    # Drop in reverse FK order
    for t in [
        cfg.table_mappings, cfg.table_relationships, cfg.table_docstrings,
        cfg.table_components, cfg.table_requirements,
        cfg.table_srs_documents, cfg.table_migrations,
    ]:
        cur.execute(f"DROP TABLE IF EXISTS {t} CASCADE;")
    cur.close()
    conn.close()

    store = Store(cfg)
    store.init_schema(FAKE_DIM)
    yield store
    store.close()
