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
