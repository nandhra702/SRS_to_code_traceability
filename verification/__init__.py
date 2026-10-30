"""
Verification stage (Phase 4) — the "verifier agent".

The only stage that uses an LLM.  It checks each requirement's proposed
mapping (Phase 3) against the actual source code and records claim-level,
evidence-cited verdicts.  Retrieval scores and docstrings are context, never
proof.
"""

from .llm import LLMClient, LLMError, OllamaClient
from .verifier import Verifier, VerifierConfig

__all__ = ["LLMClient", "LLMError", "OllamaClient", "Verifier", "VerifierConfig"]
