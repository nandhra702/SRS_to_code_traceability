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
    what was shown.  Returns (normalised reference, None) or (None, reason).

    Models quote code more reliably than they number lines (they miscount, or
    number lines relative to the method instead of the file), so a citation
    is anchored by its **quote**:

    1. The cited lines are inside the shown range and contain the quote
       (≥60% of its identifiers/numbers) → accepted as cited.
    2. Otherwise, with a quote, find it verbatim (whitespace-insensitive) in
       the component's shown lines.  One occurrence → re-anchored there.
       Several → the single one within REALIGN_WINDOW lines of the cited start
       is used; otherwise rejected as ambiguous.
    3. No verbatim match → the nearest range within REALIGN_WINDOW lines whose
       identifiers/numbers match the quote (a fuzzy match never moves a
       citation further than that).
    4. No quote, or the quote is not in the shown code → rejected.

    A re-anchored citation records `realigned_from`.  Every accepted citation
    still points at code that was shown and that contains its quote.
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
    quote = str(ref.get("quote") or "").strip()
    in_range = shown.first_line <= start and end <= shown.last_line

    if in_range and (not quote or quote_overlap(quote, _text(shown, start, end)) >= QUOTE_MIN_OVERLAP):
        return _accepted(shown, start, end, quote), None
    if not quote:
        return None, (f"{tag}: lines {start}-{end} outside the shown range "
                      f"{shown.first_line}-{shown.last_line}")

    exact = _locate_exact(shown, quote)
    target: Optional[Tuple[int, int]] = None
    if len(exact) == 1:
        target = exact[0]
    elif len(exact) > 1:
        near = sorted((hit for hit in exact if abs(hit[0] - start) <= REALIGN_WINDOW),
                      key=lambda hit: abs(hit[0] - start))
        tie = len(near) > 1 and abs(near[0][0] - start) == abs(near[1][0] - start)
        if not near or tie:
            return None, (f"{tag}: quote {quote[:60]!r} occurs {len(exact)} times in the shown code "
                          f"and the cited lines {start}-{end} do not identify one")
        target = near[0]
    else:
        target = _realign(shown, quote, start, end)
    if target is None:
        where = "outside the shown range" if not in_range else "not found in those lines"
        return None, f"{tag}: quote {quote[:60]!r} {where} (lines {start}-{end}) and not found nearby"
    out = _accepted(shown, target[0], target[1], quote)
    out["realigned_from"] = [start, end]
    return out, None


def _accepted(shown: ShownComponent, start: int, end: int, quote: str) -> Dict[str, Any]:
    return {"component_id": shown.component_id, "comp_pk": shown.comp_pk, "link_pk": shown.link_pk,
            "path": shown.path, "start_line": start, "end_line": end, "tag": shown.tag,
            "quote": quote, "quote_checked": bool(quote)}


def _locate_exact(shown: ShownComponent, quote: str) -> List[Tuple[int, int]]:
    """
    Every place the quote occurs verbatim (whitespace-insensitive, possibly
    across lines) in the shown lines, as (first line, last line).  Quotes
    shorter than 4 characters are too generic to locate.
    """
    needle = " ".join(quote.split())
    if len(needle) < 4:
        return []
    text, line_of = "", []
    for n in range(shown.first_line, shown.last_line + 1):
        segment = " ".join(shown.lines.get(n, "").split())
        if not segment:
            continue
        if text:
            text += " "
            line_of.append(n)
        text += segment
        line_of.extend([n] * len(segment))
    hits: List[Tuple[int, int]] = []
    i = text.find(needle)
    while i != -1:
        hit = (line_of[i], line_of[i + len(needle) - 1])
        if hit not in hits:
            hits.append(hit)
        i = text.find(needle, i + 1)
    return hits


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
