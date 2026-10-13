"""
Embedding-text builders — one per entity type.

Requirements, docstrings and code are embedded *separately* so that a match
can be attributed to the representation that produced it (plan §4.3).  The
same builders are used at ingest time and by ``rebuild-embeddings`` so a text
hash computed from a database row matches the one computed from the source
file.

Every builder returns "" when the entity carries no meaningful text; callers
must skip embedding in that case rather than embed an empty/placeholder string.

Builders accept either loader records or database rows (component rows use
``component_id`` where loader records use ``id``).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Dict, Iterable, List

_REQ_LIST_FIELDS = ("actions", "entities", "domain_terms", "synonyms", "sub_requirements")


def _component_id(comp: Dict[str, Any]) -> str:
    return comp.get("id") or comp.get("component_id") or ""


def _join_list(values: Iterable[str]) -> str:
    return "; ".join(v.strip() for v in values if v and v.strip())


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------


def requirement_text(req: Dict[str, Any]) -> str:
    """
    Title, retrieval text, normalized text (raw text when normalization is
    empty, which the SRS Agent sometimes returns) and the semantic lists.
    """
    parts: List[str] = []
    for key in ("title", "retrieval_text"):
        val = (req.get(key) or "").strip()
        if val:
            parts.append(val)
    body = (req.get("normalized_text") or "").strip() or (req.get("raw_text") or "").strip()
    if body:
        parts.append(body)
    for key in _REQ_LIST_FIELDS:
        joined = _join_list(req.get(key) or [])
        if joined:
            parts.append(joined)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Code components
# ---------------------------------------------------------------------------


def strip_docstring(source: str) -> str:
    """
    Remove the leading docstring of the first def/class in *source*.

    Works on the (dedented) source text by line range so comments and
    formatting survive.  Returns *source* unchanged if it cannot be parsed.
    """
    text = textwrap.dedent(source)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return source
    node = next(
        (n for n in tree.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))),
        None,
    )
    if node is None or not node.body:
        return text
    first = node.body[0]
    if not (isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return text
    lines = text.splitlines()
    del lines[first.lineno - 1:first.end_lineno]
    return "\n".join(lines)


def component_code_text(comp: Dict[str, Any], max_chars: int) -> str:
    """
    Identity + signature + start of the body, with the docstring removed so the
    code vector reflects implementation rather than documentation.

    Returns "" when the record has neither source nor signature (legacy
    DocAgent graphs), since an ID alone is not meaningful code evidence.
    """
    source = comp.get("source_code") or ""
    signature = comp.get("signature") or ""
    if not source.strip() and not signature.strip():
        return ""
    parts = [f"{comp.get('component_type', '')} {_component_id(comp)}".strip()]
    if signature:
        parts.append(signature)
    body = strip_docstring(source).strip() if source else ""
    if body:
        parts.append(body[:max_chars])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Docstrings
# ---------------------------------------------------------------------------


def docstring_text(comp: Dict[str, Any], docstring: str) -> str:
    """Owning symbol + signature + cleaned docstring; "" for a blank docstring."""
    cleaned = inspect.cleandoc(docstring or "").strip()
    if not cleaned:
        return ""
    parts = [f"{comp.get('component_type', '')} {_component_id(comp)}".strip()]
    if comp.get("signature"):
        parts.append(comp["signature"])
    parts.append(cleaned)
    return "\n".join(parts)
