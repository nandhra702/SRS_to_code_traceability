"""Tests for traceability.texts — embedding-text builders."""

from __future__ import annotations

import json

from traceability.texts import (
    component_code_text,
    docstring_text,
    requirement_text,
    strip_docstring,
)
from traceability.tests.conftest import FIXTURES, requirement, v1_component


# ---------------------------------------------------------------------------
# strip_docstring
# ---------------------------------------------------------------------------


def test_strip_docstring_removes_leading_docstring():
    src = 'def f(x):\n    """Docs here.\n\n    More."""\n    return x * 2'
    out = strip_docstring(src)
    assert "Docs here" not in out
    assert "return x * 2" in out


def test_strip_docstring_handles_indented_method():
    src = '    def m(self):\n        """Doc."""\n        # keep comment\n        return 1'
    out = strip_docstring(src)
    assert "Doc." not in out and "# keep comment" in out


def test_strip_docstring_no_docstring_unchanged():
    src = "def f():\n    return 1"
    assert strip_docstring(src) == src


def test_strip_docstring_unparseable_returned_as_is():
    src = "def broken(:\n    pass"
    assert strip_docstring(src) == src


# ---------------------------------------------------------------------------
# component_code_text
# ---------------------------------------------------------------------------


def test_code_text_has_identity_signature_and_body_without_docstring():
    comp = v1_component(
        "helper.process", signature="process(data: list)",
        source_code='def process(data: list):\n    """Doubles data."""\n    return [d * 2 for d in data]',
    )
    text = component_code_text(comp, 1500)
    assert text.startswith("function helper.process")
    assert "process(data: list)" in text
    assert "d * 2" in text
    assert "Doubles data" not in text


def test_code_text_empty_without_source_or_signature():
    # Legacy DocAgent graphs carry neither
    comp = v1_component("m.f", source_code=None, signature=None)
    assert component_code_text(comp, 1500) == ""


def test_code_text_body_truncated_to_max_chars():
    comp = v1_component("m.f", source_code="def f():\n" + "    x = 1\n" * 500)
    assert len(component_code_text(comp, 100)) < 200


def test_code_text_accepts_db_row_shape():
    row = {"component_id": "m.f", "component_type": "function",
           "signature": "f()", "source_code": "def f():\n    return 1"}
    assert "function m.f" in component_code_text(row, 1500)


def test_code_text_nonempty_for_every_real_v1_component():
    comps = json.loads((FIXTURES / "simple_v1_graph.json").read_text())["components"]
    assert all(component_code_text(c, 1500) for c in comps.values())


def test_code_text_empty_for_every_real_legacy_component():
    comps = json.loads((FIXTURES / "simple_legacy_graph.json").read_text())
    assert not any(component_code_text(c, 1500) for c in comps.values())


# ---------------------------------------------------------------------------
# docstring_text
# ---------------------------------------------------------------------------


def test_docstring_text_cleans_indentation_and_adds_symbol():
    comp = v1_component("helper.process", signature="process()")
    text = docstring_text(comp, "\n        Doubles the data.\n\n        Returns: list\n    ")
    assert text.splitlines()[0] == "function helper.process"
    assert "process()" in text
    assert "Doubles the data.\n\nReturns: list" in text


def test_docstring_text_blank_is_empty():
    assert docstring_text(v1_component("m.f"), "   \n  ") == ""


# ---------------------------------------------------------------------------
# requirement_text
# ---------------------------------------------------------------------------


def test_requirement_text_includes_semantic_fields():
    text = requirement_text(requirement("FR-1"))
    for expected in ("Validate Credentials", "validate submitted credentials",
                     "The system shall validate", "authentication", "authenticate"):
        assert expected in text


def test_requirement_text_falls_back_to_raw_text():
    # The SRS Agent sometimes returns an empty normalized_text (8/62 in the sample)
    rec = requirement("FR-1", normalized_text="", raw_text="FR-1: Raw wording only.")
    assert "Raw wording only." in requirement_text(rec)


def test_requirement_text_nonempty_for_every_real_requirement():
    reqs = json.loads((FIXTURES / "basic_srs_requirements.json").read_text(encoding="utf-8"))
    assert all(requirement_text(r) for r in reqs)
