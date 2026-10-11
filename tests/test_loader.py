"""
Tests for traceability.loader — GraphLoader and SRSLoader.

Pure Python — no DB, no model.
"""

from __future__ import annotations

import json
import pytest

from traceability.loader import GraphLoader, SRSLoader
from traceability.tests.conftest import (
    COMPONENT_RECORDS, LEGACY_GRAPH, V1_GRAPH, REQUIREMENT_RECORDS
)


# ============================================================================
# GraphLoader — legacy flat format
# ============================================================================

class TestGraphLoaderLegacy:

    def test_load_returns_self(self, legacy_graph_file):
        gl = GraphLoader(legacy_graph_file)
        assert gl.load() is gl

    def test_component_count(self, legacy_graph_file):
        gl = GraphLoader(legacy_graph_file).load()
        assert len(gl.components) == len(COMPONENT_RECORDS)

    def test_all_ids_present(self, legacy_graph_file):
        gl = GraphLoader(legacy_graph_file).load()
        for cid in COMPONENT_RECORDS:
            assert cid in gl.components

    def test_component_fields_intact(self, legacy_graph_file):
        gl = GraphLoader(legacy_graph_file).load()
        c = gl.components["helper.HelperClass.process_data"]
        assert c["component_type"] == "method"
        assert c["relative_path"] == "helper.py"
        assert c["start_line"] == 2

    def test_relationships_from_inline_list(self, legacy_graph_file):
        gl = GraphLoader(legacy_graph_file).load()
        types = {r["type"] for r in gl.relationships}
        assert "contains" in types

    def test_relationships_deduplicated(self, legacy_graph_file):
        # Write a graph where the same rel appears twice
        data = {
            "a.A": {
                "id": "a.A", "component_type": "class",
                "file_path": "/p.py", "relative_path": "p.py",
                "depends_on": [], "start_line": 1, "end_line": 2,
                "has_docstring": False, "docstring": "",
                "relationships": [
                    {"source": "a.A", "target": "a.A.m", "type": "contains"},
                    {"source": "a.A", "target": "a.A.m", "type": "contains"},
                ],
            },
            "a.A.m": {
                "id": "a.A.m", "component_type": "method",
                "file_path": "/p.py", "relative_path": "p.py",
                "depends_on": [], "start_line": 3, "end_line": 4,
                "has_docstring": False, "docstring": "", "relationships": [],
            },
        }
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name
        try:
            gl = GraphLoader(path).load()
            assert len(gl.relationships) == 1
        finally:
            os.unlink(path)

    def test_missing_relationships_key_ok(self, tmp_path):
        data = {
            "mod.f": {
                "id": "mod.f", "component_type": "function",
                "file_path": "/p.py", "relative_path": "p.py",
                "depends_on": [], "start_line": 1, "end_line": 2,
                "has_docstring": False, "docstring": "",
                # no "relationships" key
            }
        }
        p = tmp_path / "g.json"
        p.write_text(json.dumps(data))
        gl = GraphLoader(str(p)).load()
        assert "mod.f" in gl.components
        assert gl.relationships == []

    def test_access_before_load_raises(self, legacy_graph_file):
        with pytest.raises(RuntimeError):
            GraphLoader(legacy_graph_file).components

    def test_repr(self, legacy_graph_file):
        r = repr(GraphLoader(legacy_graph_file).load())
        assert "legacy_graph.json" in r


# ============================================================================
# GraphLoader — V1 format
# ============================================================================

class TestGraphLoaderV1:

    def test_component_count(self, v1_graph_file):
        gl = GraphLoader(v1_graph_file).load()
        assert len(gl.components) == len(V1_GRAPH["components"])

    def test_relationships_from_top_level(self, v1_graph_file):
        gl = GraphLoader(v1_graph_file).load()
        assert len(gl.relationships) == len(V1_GRAPH["relationships"])

    def test_relationship_fields(self, v1_graph_file):
        gl = GraphLoader(v1_graph_file).load()
        for rel in gl.relationships:
            assert "source" in rel and "target" in rel and "type" in rel

    def test_v1_dedup(self, tmp_path):
        data = {
            "components": {
                "a.A": {"id": "a.A", "component_type": "class",
                        "file_path": "/p.py", "relative_path": "p.py",
                        "depends_on": [], "start_line": 1, "end_line": 1,
                        "has_docstring": False, "docstring": ""},
                "a.A.m": {"id": "a.A.m", "component_type": "method",
                          "file_path": "/p.py", "relative_path": "p.py",
                          "depends_on": [], "start_line": 2, "end_line": 3,
                          "has_docstring": False, "docstring": ""},
            },
            "relationships": [
                {"source": "a.A", "target": "a.A.m", "type": "contains"},
                {"source": "a.A", "target": "a.A.m", "type": "contains"},
            ],
        }
        p = tmp_path / "v1.json"
        p.write_text(json.dumps(data))
        gl = GraphLoader(str(p)).load()
        assert len(gl.relationships) == 1


# ============================================================================
# GraphLoader — as_embedding_texts
# ============================================================================

class TestGraphEmbedTexts:

    def test_all_ids_present(self, legacy_graph_file):
        gl = GraphLoader(legacy_graph_file).load()
        texts = gl.as_embedding_texts()
        assert set(texts.keys()) == set(gl.components.keys())

    def test_signature_included(self, legacy_graph_file):
        gl = GraphLoader(legacy_graph_file).load()
        texts = gl.as_embedding_texts()
        assert "process_data" in texts["helper.HelperClass.process_data"]

    def test_docstring_included(self, legacy_graph_file):
        gl = GraphLoader(legacy_graph_file).load()
        texts = gl.as_embedding_texts()
        assert "helper class" in texts["helper.HelperClass"].lower()

    def test_source_code_truncated(self, tmp_path):
        long_code = "x = 1\n" * 1000
        data = {"mod.f": {
            "id": "mod.f", "component_type": "function",
            "file_path": "/p.py", "relative_path": "p.py",
            "depends_on": [], "start_line": 1, "end_line": 1000,
            "has_docstring": False, "docstring": "", "source_code": long_code,
        }}
        p = tmp_path / "g.json"
        p.write_text(json.dumps(data))
        gl = GraphLoader(str(p)).load()
        assert len(gl.as_embedding_texts()["mod.f"]) <= 2010

    def test_empty_component_gives_empty_string(self, tmp_path):
        data = {"mod.bare": {
            "id": "mod.bare", "component_type": "class",
            "file_path": "/p.py", "relative_path": "p.py",
            "depends_on": [], "start_line": 1, "end_line": 1,
            "has_docstring": False, "docstring": "",
            "signature": None, "source_code": None,
        }}
        p = tmp_path / "g.json"
        p.write_text(json.dumps(data))
        gl = GraphLoader(str(p)).load()
        assert gl.as_embedding_texts()["mod.bare"] == ""


# ============================================================================
# GraphLoader — real file on disk
# ============================================================================

class TestGraphLoaderRealFile:

    def test_loads(self, real_graph_file):
        gl = GraphLoader(real_graph_file).load()
        assert len(gl.components) > 0

    def test_ids_have_dots(self, real_graph_file):
        gl = GraphLoader(real_graph_file).load()
        for cid in gl.components:
            assert "." in cid

    def test_known_ids_present(self, real_graph_file):
        gl = GraphLoader(real_graph_file).load()
        for eid in ("main.main_function", "main.utility_function", "helper.HelperClass"):
            assert eid in gl.components


# ============================================================================
# SRSLoader
# ============================================================================

class TestSRSLoader:

    def test_load_returns_self(self, srs_output_file):
        sl = SRSLoader(srs_output_file)
        assert sl.load() is sl

    def test_requirement_count(self, srs_output_file):
        sl = SRSLoader(srs_output_file).load()
        assert len(sl.requirements) == len(REQUIREMENT_RECORDS)

    def test_ids_preserved(self, srs_output_file):
        sl = SRSLoader(srs_output_file).load()
        ids = {r["requirement_id"] for r in sl.requirements}
        assert {"REQ-001", "FR-01.1", "FR-01.2"} <= ids

    def test_required_fields_present(self, srs_output_file):
        sl = SRSLoader(srs_output_file).load()
        for rec in sl.requirements:
            for f in ("requirement_id", "raw_text", "type", "retrieval_text", "title"):
                assert f in rec

    def test_source_location_preserved(self, srs_output_file):
        sl = SRSLoader(srs_output_file).load()
        req = next(r for r in sl.requirements if r["requirement_id"] == "REQ-001")
        assert req["source"]["paragraph"] == 11
        assert req["source"]["file_name"] == "Basic_SRS_Document.docx"

    def test_list_fields_are_lists(self, srs_output_file):
        sl = SRSLoader(srs_output_file).load()
        for rec in sl.requirements:
            for f in ("actor", "actions", "entities", "domain_terms"):
                assert isinstance(rec.get(f, []), list)

    def test_access_before_load_raises(self, srs_output_file):
        with pytest.raises(RuntimeError):
            SRSLoader(srs_output_file).requirements

    def test_non_array_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"not": "array"}))
        with pytest.raises(ValueError, match="JSON array"):
            SRSLoader(str(p)).load()

    def test_missing_requirement_id_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps([{"raw_text": "no id"}]))
        with pytest.raises(ValueError, match="requirement_id"):
            SRSLoader(str(p)).load()

    def test_document_record_inference(self, srs_output_file):
        sl = SRSLoader(srs_output_file).load()
        dr = sl.document_record()
        assert dr["file_name"] == "Basic_SRS_Document.docx"
        assert dr["source_type"] == "docx"
        assert dr["total_requirements"] == len(REQUIREMENT_RECORDS)

    def test_repr(self, srs_output_file):
        r = repr(SRSLoader(srs_output_file).load())
        assert "requirements" in r


# ============================================================================
# SRSLoader — as_embedding_texts
# ============================================================================

class TestSRSEmbedTexts:

    def test_all_ids_present(self, srs_output_file):
        sl = SRSLoader(srs_output_file).load()
        texts = sl.as_embedding_texts()
        for rec in sl.requirements:
            assert rec["requirement_id"] in texts

    def test_retrieval_text_included(self, srs_output_file):
        sl = SRSLoader(srs_output_file).load()
        texts = sl.as_embedding_texts()
        assert "login" in texts["FR-01.1"]

    def test_domain_terms_included(self, srs_output_file):
        sl = SRSLoader(srs_output_file).load()
        texts = sl.as_embedding_texts()
        assert "authentication" in texts["FR-01.1"]

    def test_synonyms_included(self, srs_output_file):
        sl = SRSLoader(srs_output_file).load()
        texts = sl.as_embedding_texts()
        assert "sign in" in texts["FR-01.1"]


# ============================================================================
# SRSLoader — real file
# ============================================================================

class TestSRSLoaderRealFile:

    def test_loads(self, real_srs_file):
        sl = SRSLoader(real_srs_file).load()
        assert len(sl.requirements) > 0

    def test_ids_non_empty(self, real_srs_file):
        sl = SRSLoader(real_srs_file).load()
        for r in sl.requirements:
            assert r["requirement_id"].strip()

    def test_type_values_valid(self, real_srs_file):
        sl = SRSLoader(real_srs_file).load()
        valid = {"functional", "non-functional", "other"}
        for r in sl.requirements:
            assert r["type"] in valid
