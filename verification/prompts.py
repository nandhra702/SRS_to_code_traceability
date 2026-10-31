"""
Prompts and JSON schemas for the verifier.  Bump PROMPT_VERSION whenever a
prompt or schema changes; it is stored with every verification row.
"""

from __future__ import annotations

from typing import Any, Dict, List

PROMPT_VERSION = "verify-v2.3"  # v2.3: citations anchored by their quote anywhere in the shown component

# ---------------------------------------------------------------------------
# Step 1: split a requirement into atomic claims
# ---------------------------------------------------------------------------

CLAIMS_SYSTEM = """You split software requirements into atomic, independently checkable claims.

Rules:
- Each claim states ONE behaviour, rule, value or condition the software must have.
- Keep the requirement's own wording and numbers. Do not add anything the requirement does not say.
- A simple requirement is a single claim. Never return more than 8 claims.
- Write each claim as a full sentence starting with "The system"."""

CLAIMS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
    },
    "required": ["claims"],
}


def claims_prompt(req: Dict[str, Any]) -> str:
    parts = [f"Requirement {req['requirement_id']}: {req.get('title') or ''}".strip(),
             f"Text: {req['raw_text']}"]
    for key in ("conditions", "constraints", "error_conditions", "sub_requirements"):
        values = [v for v in (req.get(key) or []) if v]
        if values:
            parts.append(f"{key.replace('_', ' ').capitalize()} (from the SRS extraction): " + "; ".join(values))
    parts.append('Return JSON: {"claims": ["The system ...", ...]}')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Step 2: judge each claim against the shown code
# ---------------------------------------------------------------------------

VERIFY_SYSTEM = """You are a strict software verification agent. You decide whether source code implements requirement claims.

Rules:
- Judge ONLY from the source code shown. Each component has a tag like [C1] and numbered lines.
- Docstrings are documentation written by another tool. They are NOT evidence. If a docstring and the code disagree, the code wins.
- "supported": the shown code clearly performs the claimed behaviour, including any stated values, limits or conditions. Cite the exact lines.
- "not_supported": the relevant code is shown but does not do what the claim says (missing check, different value, behaviour absent).
- "insufficient_evidence": the code needed to decide is not shown, or is ambiguous.
- The cited code must act on the SAME thing the claim is about. Code doing a similar check on a different object or feature does not count (for example, a member's borrowing check is not evidence for a loan-renewal rule).
- If none of the shown components implements the claim, the verdict is "not_supported" (related code shown) or "insufficient_evidence" (nothing relevant shown) -- never "supported".
- Every "supported" verdict MUST cite at least one line range from a shown component and quote the key code from those lines exactly: {"component": "C2", "start_line": 41, "end_line": 43, "quote": "if fine > 0:"}. Only cite line numbers that appear in the listing.
- Search-ranking or similarity is not evidence. Do not guess. When unsure, use "insufficient_evidence"."""

VERIFY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_index": {"type": "integer"},
                    "verdict": {"type": "string",
                                "enum": ["supported", "not_supported", "insufficient_evidence"]},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"component": {"type": "string"},
                                           "start_line": {"type": "integer"},
                                           "end_line": {"type": "integer"},
                                           "quote": {"type": "string"}},
                            "required": ["component", "start_line", "end_line", "quote"],
                        },
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["claim_index", "verdict", "evidence", "rationale"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["claims", "summary"],
}


def verify_prompt(req: Dict[str, Any], claims: List[str], context_text: str) -> str:
    numbered = "\n".join(f"  {i}. {c}" for i, c in enumerate(claims, start=1))
    return (
        f"Requirement {req['requirement_id']}: {req.get('title') or ''}\n"
        f"Text: {req['raw_text']}\n\n"
        f"Claims to verify:\n{numbered}\n\n"
        f"Source code (the only evidence you may use):\n\n{context_text}\n\n"
        "For EVERY claim (claim_index 1.." f"{len(claims)}) return a verdict, the cited line ranges "
        "and a one-sentence rationale. Then a one-sentence summary.\n"
        'Return JSON: {"claims": [{"claim_index": 1, "verdict": "...", "evidence": '
        '[{"component": "C1", "start_line": 0, "end_line": 0, "quote": "..."}], "rationale": "..."}], "summary": "..."}'
    )


# ---------------------------------------------------------------------------
# Clause splitting of one compound claim (validated by claims.validate_split)
# ---------------------------------------------------------------------------

SPLIT_SYSTEM = """You split ONE requirement sentence into its separate parts, so each part can be checked on its own.

Rules:
- Split only where the sentence combines separate behaviours or separate conditions (joined by "and", "or", "as well as", "in addition to", "also").
- Do NOT split a simple list of attributes or values that describe one behaviour (e.g. "title, author and year").
- Each part MUST be a complete sentence with its own subject and its own "shall"/"must" verb. Repeat the subject, the verb and any leading condition ("When a member pays a fine, ...") in every part.
- Copy the sentence's own words otherwise.
- Never split alternatives inside one condition or value ("10 or 13 characters", "PDF or DOCX").
- Never add, explain, or rephrase anything. Keep every number and value exactly.
- If the sentence is a single behaviour, return it unchanged as the only part."""

SPLIT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"parts": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6}},
    "required": ["parts"],
}


def split_prompt(sentence: str) -> str:
    return f'Sentence: {sentence}\nReturn JSON: {{"parts": ["...", "..."]}}'
