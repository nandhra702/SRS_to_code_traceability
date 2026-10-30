"""
Evidence context: the exact code the verifier is allowed to cite.

Each mapped component is shown with a tag (C1, C2, …) and its source lines
numbered with their real line numbers in the file.  Only the lines shown can
be cited; evidence_validation checks every citation against this record.

- Direct components are shown in full, up to max_lines_direct.
- Supporting functions and methods are shown up to max_lines_supporting.
- Supporting classes show only their header (class line and class-level
  statements up to the first method).  Their methods appear as their own
  components when they matter.
- Direct components' docstrings are included, labelled as DocAgent-generated
  documentation that is not evidence of behaviour.
- Components are added in priority order until max_chars is reached; the
  rest are listed as omitted, so the model knows the view is partial.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ShownComponent:
    tag: str
    comp_pk: int
    link_pk: Optional[int]
    component_id: str
    role: str
    path: str
    first_line: int
    last_line: int              # last line shown (may be < the component's end)
    truncated: bool
    lines: Dict[int, str] = field(default_factory=dict)   # line number -> source text shown


@dataclass
class EvidenceContext:
    text: str
    shown: Dict[str, ShownComponent] = field(default_factory=dict)
    omitted: List[str] = field(default_factory=list)

    def by_tag(self, tag: str) -> Optional[ShownComponent]:
        return self.shown.get(tag.strip().upper())


def _class_header_end(source: str, start_line: int, end_line: int) -> int:
    """Last line before the first method of a class (the class 'header')."""
    try:
        tree = ast.parse(textwrap.dedent(source))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        first_method = next((n for n in cls.body
                             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if first_method is None:
            return end_line
        first = first_method.lineno
        if first_method.decorator_list:
            first = min(d.lineno for d in first_method.decorator_list)
        return max(start_line, start_line + first - 2)
    except (SyntaxError, StopIteration):
        return start_line


def build_context(components: List[Dict[str, Any]], *, max_chars: int = 14000,
                  max_lines_direct: int = 120, max_lines_supporting: int = 40,
                  docstring_chars: int = 500) -> EvidenceContext:
    """
    components: dicts with comp_pk, link_pk, component_id, component_type, role
    ('direct' | 'supporting'), relative_path, start_line, end_line,
    source_code, docstring (optional) — already in priority order.
    """
    ctx = EvidenceContext(text="")
    blocks: List[str] = []
    used = 0
    for comp in components:
        tag = f"C{len(ctx.shown) + 1}"
        lines = (comp.get("source_code") or "").splitlines()
        start, end = comp["start_line"], comp["end_line"]
        if comp["role"] == "supporting" and comp["component_type"] == "class":
            last = _class_header_end(comp.get("source_code") or "", start, end)
        else:
            limit = max_lines_direct if comp["role"] == "direct" else max_lines_supporting
            last = min(end, start + limit - 1, start + len(lines) - 1)
        shown_lines = lines[: last - start + 1]
        truncated = last < end
        header = (f"[{tag}] {comp['component_id']} ({comp['component_type']}, {comp['role']}) "
                  f"-- {comp['relative_path']} lines {start}-{end}")
        body = "\n".join(f"{start + i:5d} | {ln}" for i, ln in enumerate(shown_lines))
        note = ""
        if truncated:
            note = (f"\n      ... lines {last + 1}-{end} not shown "
                    f"({'class body: its methods are listed separately' if comp['component_type'] == 'class' else 'truncated'})")
        doc = ""
        if comp["role"] == "direct" and comp.get("docstring"):
            text = " ".join(comp["docstring"].split())[:docstring_chars]
            doc = f"\n      Docstring (DocAgent-generated documentation, NOT evidence of behaviour): {text}"
        block = f"{header}\n{body}{note}{doc}"
        if used + len(block) > max_chars and ctx.shown:
            ctx.omitted.append(comp["component_id"])
            continue
        used += len(block)
        blocks.append(block)
        ctx.shown[tag] = ShownComponent(tag, comp["comp_pk"], comp.get("link_pk"), comp["component_id"],
                                        comp["role"], comp["relative_path"], start, last, truncated,
                                        {start + i: ln for i, ln in enumerate(shown_lines)})
    if ctx.omitted:
        blocks.append("Omitted for length (cannot be cited): " + ", ".join(ctx.omitted))
    ctx.text = "\n\n".join(blocks)
    return ctx


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")
QUOTE_MIN_OVERLAP = 0.6


def quote_overlap(quote: str, cited_text: str) -> float:
    """Share of the quote's identifiers/numbers that occur in the cited lines."""
    tokens = _TOKEN.findall(quote)
    if not tokens:
        return 0.0
    present = set(_TOKEN.findall(cited_text))
    return sum(t in present for t in tokens) / len(tokens)


def validate_reference(ctx: EvidenceContext, ref: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Check one model citation {component, start_line, end_line, quote} against
    what was shown.  The range must lie inside the shown lines, and a non-empty
    quote must match them (≥60% of its identifiers/numbers present) — a
    citation whose quote is not in the cited code is rejected.
    Returns (normalised reference, None) or (None, reason).
    """
    tag = str(ref.get("component", ""))
    shown = ctx.by_tag(tag)
    if shown is None:
        return None, f"unknown component tag {tag!r}"
    try:
        start, end = int(ref.get("start_line")), int(ref.get("end_line"))
    except (TypeError, ValueError):
        return None, f"{tag}: non-numeric line range"
    if start > end:
        start, end = end, start
    if start < shown.first_line or end > shown.last_line:
        return None, (f"{tag}: lines {start}-{end} outside the shown range "
                      f"{shown.first_line}-{shown.last_line}")
    quote = str(ref.get("quote") or "").strip()
    realigned_from = None
    if quote and quote_overlap(quote, _text(shown, start, end)) < QUOTE_MIN_OVERLAP:
        # Models often cite the line next to the code they quote.  Re-anchor to
        # where the quote really is, within REALIGN_WINDOW lines in the same
        # component; the citation still has to quote shown code.
        moved = _realign(shown, quote, start, end)
        if moved is None:
            return None, f"{tag}: quote {quote[:60]!r} not found in lines {start}-{end}"
        realigned_from, (start, end) = [start, end], moved
    out = {"component_id": shown.component_id, "comp_pk": shown.comp_pk, "link_pk": shown.link_pk,
           "path": shown.path, "start_line": start, "end_line": end, "tag": shown.tag,
           "quote": quote, "quote_checked": bool(quote)}
    if realigned_from:
        out["realigned_from"] = realigned_from
    return out, None


REALIGN_WINDOW = 3


def _text(shown: ShownComponent, start: int, end: int) -> str:
    return "\n".join(shown.lines.get(n, "") for n in range(start, end + 1))


def _realign(shown: ShownComponent, quote: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    """Nearest same-length range within the window whose lines contain the quote."""
    span = end - start
    best: Optional[Tuple[int, int]] = None
    for s in range(max(shown.first_line, start - REALIGN_WINDOW),
                   min(shown.last_line, start + REALIGN_WINDOW) + 1):
        e = min(shown.last_line, s + span)
        if quote_overlap(quote, _text(shown, s, e)) >= QUOTE_MIN_OVERLAP:
            if best is None or abs(s - start) < abs(best[0] - start):
                best = (s, e)
    return best
