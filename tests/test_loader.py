"""
Tests for traceability.loader — GraphLoader, read_post_run_docstrings, SRSLoader.

Uses the real DocAgent / SRS Agent outputs in tests/fixtures; no DB, no model.
"""

from __future__ import annotations

import json

import pytest

from traceability.loader import GraphLoader, SRSLoader, read_post_run_docstrings
from traceability.tests.conftest import requirement, v1_component, write_json


# ============================================================================
# GraphLoader — real files
# ============================================================================


class TestGraphLoaderRealFiles:

    def test_v1_detected_with_typed_relationships(self, real_v1_graph):
        g = GraphLoader(real_v1_graph).load()
        assert g.format == "v1"
        assert len(g.components) == 20
        assert len(g.relationships) == 22
        assert {r["type"] for r in g.relationships} <= {
            "calls", "imports", "inherits", "instantiates", "contains"}

    def test_v1_components_carry_source_and_signature(self, real_v1_graph):
        g = GraphLoader(real_v1_graph).load()
        methods = [c for c in g.components.values() if c["component_type"] == "method"]
        assert all(c["source_code"] for c in g.components.values())
        assert all(c["containing_class"] for c in methods)

    def test_legacy_detected_without_relationships(self, real_legacy_graph):
        g = GraphLoader(real_legacy_graph).load()
        assert g.format == "legacy"
        assert len(g.components) == 20
        assert g.relationships == []

    def test_legacy_and_v1_describe_same_components(self, real_v1_graph, real_legacy_graph):
        v1 = GraphLoader(real_v1_graph).load()
        legacy = GraphLoader(real_legacy_graph).load()
        assert set(v1.components) == set(legacy.components)

    def test_default_repo_id_strips_docagent_suffix(self, tmp_path):
        p = write_json(tmp_path / "my_repo_dependency_graph.json", {"components": {}})
        assert GraphLoader(p).load().default_repo_id() == "my_repo"


# ============================================================================
# GraphLoader — validation
# ============================================================================


class TestGraphLoaderValidation:

    def test_must_call_load_first(self, real_v1_graph):
        with pytest.raises(RuntimeError):
            GraphLoader(real_v1_graph).components

    def test_non_object_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            GraphLoader(write_json(tmp_path / "g.json", [1, 2])).load()

    def test_malformed_relationships_counted_and_dropped(self, tmp_path):
        graph = {
            "components": {"m.a": v1_component("m.a"), "m.b": v1_component("m.b")},
            "relationships": [
                {"source": "m.a", "target": "m.b", "type": "calls"},
                {"source": "m.a", "target": "m.b"},            # no type
                {"source": "m.a", "target": None, "type": "calls"},
                "not-a-dict",
            ],
        }
        g = GraphLoader(write_json(tmp_path / "g.json", graph)).load()
        assert g.relationships == [{"source": "m.a", "target": "m.b", "type": "calls"}]
        assert g.malformed_relationships == 3

    def test_duplicate_relationships_deduplicated(self, tmp_path):
        rel = {"source": "m.a", "target": "m.b", "type": "calls"}
        graph = {"components": {"m.a": v1_component("m.a"), "m.b": v1_component("m.b")},
                 "relationships": [rel, dict(rel)]}
        assert len(GraphLoader(write_json(tmp_path / "g.json", graph)).load().relationships) == 1

    def test_malformed_components_ignored(self, tmp_path):
        graph = {"components": {"m.a": v1_component("m.a"), "m.bad": {"id": "m.bad"}}}
        assert list(GraphLoader(write_json(tmp_path / "g.json", graph)).load().components) == ["m.a"]

    def test_legacy_inline_relationships_collected(self, tmp_path):
        rel = {"source": "m.a", "target": "m.b", "type": "calls"}
        graph = {"m.a": v1_component("m.a", relationships=[rel]), "m.b": v1_component("m.b")}
        g = GraphLoader(write_json(tmp_path / "g.json", graph)).load()
        assert g.format == "legacy" and g.relationships == [rel]


# ============================================================================
# Docstring provenance
# ============================================================================


class TestDocstringRecords:

    def test_existing_docstrings_from_graph(self, real_v1_graph):
        g = GraphLoader(real_v1_graph).load()
        recs = g.docstring_records()
        expected = sum(1 for c in g.components.values()
                       if c["has_docstring"] and c["docstring"].strip())
        assert len(recs) == expected == 18
        assert {r["kind"] for r in recs} == {"existing"}

    def test_unchanged_source_yields_no_generated(self, real_v1_graph, repo_copy):
        g = GraphLoader(real_v1_graph).load()
        post = read_post_run_docstrings(g.components, repo_copy)
        assert len(post) == 20  # every component located
        assert all(r["kind"] == "existing" for r in g.docstring_records(post))

    def test_docstring_added_by_generation_is_generated(self, real_v1_graph, repo_copy):
        g = GraphLoader(real_v1_graph).load()
        target = next(cid for cid, c in g.components.items()
                      if c["component_type"] == "method" and not c["has_docstring"])
        cls = target.rsplit(".", 1)[0].split(".")[-1]
        method = target.rsplit(".", 1)[1]
        path = repo_copy / g.components[target]["relative_path"]
        lines = path.read_text().splitlines()
        # Insert a docstring after the method's def line, like DocAgent does —
        # this shifts every line below, so lookup must be by name, not line.
        in_class = False
        for i, line in enumerate(lines):
            if line.startswith(f"class {cls}"):
                in_class = True
            elif in_class and line.strip().startswith(f"def {method}("):
                indent = line[: len(line) - len(line.lstrip())] + "    "
                lines.insert(i + 1, f'{indent}"""Generated by DocAgent."""')
                break
        path.write_text("\n".join(lines) + "\n")

        post = read_post_run_docstrings(g.components, repo_copy)
        gen = [r for r in g.docstring_records(post) if r["kind"] == "generated"]
        assert [r["component_id"] for r in gen] == [target]
        assert gen[0]["content"] == "Generated by DocAgent."
        assert gen[0]["provenance"]["replaced_existing"] is False

    def test_overwritten_docstring_keeps_both(self, tmp_path):
        comp = v1_component("mod.f", has_docstring=True, docstring="Old doc.")
        g = GraphLoader(write_json(tmp_path / "g.json", {"components": {"mod.f": comp}})).load()
        (tmp_path / "mod.py").write_text('def f():\n    """New doc."""\n    return 1\n')
        recs = g.docstring_records(read_post_run_docstrings(g.components, tmp_path))
        assert {(r["kind"], r["content"]) for r in recs} == {
            ("existing", "Old doc."), ("generated", "New doc.")}

    def test_missing_source_file_skipped(self, real_v1_graph, tmp_path):
        g = GraphLoader(real_v1_graph).load()
        assert read_post_run_docstrings(g.components, tmp_path) == {}


# ============================================================================
# SRSLoader
# ============================================================================


class TestSRSLoader:

    def test_real_docx_output(self, real_srs):
        s = SRSLoader(real_srs).load()
        assert len(s.requirements) == 62
        doc = s.document_record()
        assert doc["file_name"] == "Basic_SRS_Document.docx"
        assert doc["source_type"] == "docx"
        assert doc["document_id"] == real_srs.stem

    def test_real_text_output_has_unknown_source(self, real_text_srs):
        doc = SRSLoader(real_text_srs).load().document_record()
        assert doc["source_type"] == "unknown"

    def test_document_id_override(self, real_srs):
        assert SRSLoader(real_srs).load().document_record("my-srs")["document_id"] == "my-srs"

    def test_srsoutput_wrapper_accepted(self, tmp_path):
        wrapped = {"analysis_id": "a1",
                   "source": {"document_id": "doc-7", "file_name": "spec.pdf",
                              "source_type": "pdf", "total_requirements": 1},
                   "requirements": [requirement("FR-1")]}
        s = SRSLoader(write_json(tmp_path / "out.json", wrapped)).load()
        doc = s.document_record()
        assert len(s.requirements) == 1
        assert (doc["document_id"], doc["source_type"]) == ("doc-7", "pdf")

    def test_duplicate_ids_rejected(self, tmp_path):
        p = write_json(tmp_path / "r.json", [requirement("FR-1"), requirement("FR-1")])
        with pytest.raises(ValueError, match="Duplicate"):
            SRSLoader(p).load()

    @pytest.mark.parametrize("bad", [
        {"raw_text": "x"},                                 # no id
        {"requirement_id": "", "raw_text": "x"},           # empty id
        {"requirement_id": "FR-1", "raw_text": "  "},      # empty text
    ])
    def test_invalid_records_rejected(self, tmp_path, bad):
        with pytest.raises(ValueError):
            SRSLoader(write_json(tmp_path / "r.json", [bad])).load()

    def test_non_list_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            SRSLoader(write_json(tmp_path / "r.json", {"x": 1})).load()

    def test_empty_array_is_valid(self, tmp_path):
        assert SRSLoader(write_json(tmp_path / "r.json", [])).load().requirements == []
