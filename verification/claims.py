"""
Deterministic claim splitting.

A requirement is split into its own sentences and list items — nothing is
paraphrased or added, so a claim can never be invented.  (Asking a 7B model to
"split into claims" produced claims such as "applies the correct interest
rate" for a fine-calculation requirement.)

- A leading "FR-018 — Title" label is removed.
- Sentences split on . ! ? followed by a capital letter (so 0.50 or 20.00 do
  not split).
- An introductory sentence ending in ":" followed by list items becomes one
  claim per item, each prefixed with the introduction.
- An introduction ending in ":" with no items after it means the list was lost
  upstream (the SRS Agent sometimes drops bullets and tables); it is kept as a
  claim and the requirement is flagged incomplete.
- Very short fragments are merged into the previous claim; at most
  max_claims claims are returned (the tail is merged into the last one).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

_LABEL = re.compile(r"^\s*[A-Za-z]{1,10}-\d+(?:\.\d+)*\s*[—–:-]?\s*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _strip_label(text: str, title: Optional[str]) -> str:
    text = _LABEL.sub("", text.strip(), count=1)
    if title and text.lower().startswith(title.lower()):
        rest = text[len(title):].lstrip(" :—–-")
        if rest:
            text = rest
    return text.strip()


def split_claims(text: str, title: Optional[str] = None, max_claims: int = 8) -> Tuple[List[str], bool]:
    """
    Returns (claims, incomplete).  incomplete is True when the text ends with an
    introduction (":") whose list items are missing.
    """
    body = _strip_label(text, title)
    lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    claims: List[str] = []
    dangling: List[str] = []                     # introductions whose items are missing
    intro: Optional[str] = None
    pending_intro = False
    for line in lines:
        if _ITEM.match(line) and intro:
            item = _ITEM.sub("", line).rstrip(";,. ")
            claims.append(f"{intro.rstrip(':').strip()} {item}.")
            pending_intro = False
            continue
        for sentence in _SENTENCE.split(" ".join(line.split())):
            sentence = sentence.strip()
            if not sentence:
                continue
            if sentence.endswith(":"):
                if pending_intro:
                    dangling.append(intro)
                intro, pending_intro = sentence, True
                continue
            if pending_intro:                    # the intro had no items before this sentence
                dangling.append(intro)
                pending_intro = False
            if claims and len(sentence.split()) < 4:
                claims[-1] = f"{claims[-1]} {sentence}"
            else:
                claims.append(sentence)
    if pending_intro:
        dangling.append(intro)
    incomplete = bool(dangling)
    # A dangling introduction ("... shall reject the book when:") has lost its
    # content and cannot be verified either way: verify the remaining claims,
    # and only fall back to the introduction when it is all there is.
    if not claims:
        claims = dangling or [body or text.strip()]
    if len(claims) > max_claims:
        claims = claims[: max_claims - 1] + [" ".join(claims[max_claims - 1:])]
    return claims, incomplete


# ---------------------------------------------------------------------------
# Clause splitting (model-proposed, mechanically validated)
# ---------------------------------------------------------------------------

# A claim that combines several behaviours or conditions; worth splitting.
COMPOUND_CUE = re.compile(r"\b(and|or|as well as|in addition to|both|also)\b", re.I)

# Words a split may add without adding meaning (repeating the subject/modal).
_GLUE = {
    "the", "a", "an", "it", "its", "this", "that", "these", "those", "they", "their", "them",
    "system", "shall", "must", "should", "will", "may", "can", "be", "is", "are", "was", "been",
    "has", "have", "had", "do", "does", "when", "if", "only", "also", "not", "no", "of", "to",
    "in", "on", "for", "with", "by", "and", "or", "which", "who", "each", "all", "any", "s",
}
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def content_words(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _GLUE}


_MODAL = re.compile(r"\b(shall|must|should|will|may|can)\b", re.I)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Values offered together ("10 or 13", "3, 7 and 10") form one alternative.
_NUMBER_GROUP = re.compile(r"\d+(?:\.\d+)?(?:\s*(?:,|or|and|to|-)\s*\d+(?:\.\d+)?)+")


def validate_split(claim: str, parts: List[str], context: str = "") -> Optional[str]:
    """
    None if *parts* is a faithful split of *claim*, otherwise the reason.

    - nothing invented: every content word of every part occurs in the claim
      or the rest of the requirement (*context*), e.g. a repeated condition;
    - nothing lost: together the parts keep every number of the claim;
    - alternatives stay whole: numbers offered together ("10 or 13") must
      appear together in one part — splitting them changes the meaning;
    - complete sentences: if the claim has a modal verb (shall, must, ...),
      every part needs one too.
    """
    if len(parts) < 2:
        return "not split"
    allowed = content_words(f"{context} {claim}")
    for part in parts:
        new = content_words(part) - allowed
        if new:
            return f"adds words not in the requirement: {sorted(new)[:5]}"
    numbers = set(_NUMBER.findall(claim))
    kept = set(_NUMBER.findall(" ".join(parts)))
    if numbers - kept:
        return f"drops values {sorted(numbers - kept)}"
    for group in _NUMBER_GROUP.findall(claim):
        values = set(_NUMBER.findall(group))
        if not any(values <= set(_NUMBER.findall(part)) for part in parts):
            return f"splits the alternative {group!r}"
    if _MODAL.search(claim) and not all(_MODAL.search(part) for part in parts):
        return "a part is not a complete requirement sentence"
    return None
