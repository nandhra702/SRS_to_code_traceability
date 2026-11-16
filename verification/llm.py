"""
LLM client for the verifier.

LLMClient is the interface the verifier depends on (tests use a scripted
fake).  OllamaClient talks to a local Ollama server over its HTTP API using
only the standard library:

- structured output: the JSON schema is passed as `format`, so the model is
  constrained to the expected shape;
- deterministic: temperature 0 and a fixed seed;
- bounded retries for transient failures (connection errors, timeouts,
  unparseable output); after that an LLMError is raised — the caller turns it
  into an 'insufficient_evidence' verdict, never a positive one.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5-coder:7b"


class LLMError(RuntimeError):
    """The model could not produce a valid response (after retries)."""


class LLMClient(Protocol):
    model: str

    def chat_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Return the model's reply parsed as a JSON object matching *schema*."""
        ...


class OllamaClient:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        *,
        num_ctx: int = 8192,
        temperature: float = 0.0,
        seed: int = 42,
        timeout: float = 300.0,
        retries: int = 2,
    ) -> None:
        self.model = model or os.getenv("VERIFIER_MODEL", DEFAULT_MODEL)
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.retries = retries

    def chat_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "options": {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx},
        }
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            try:
                req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    reply = json.loads(resp.read().decode("utf-8"))
                content = reply.get("message", {}).get("content", "")
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("model reply is not a JSON object")
                return parsed
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                if exc.code == 404:                      # model not pulled: retrying won't help
                    raise LLMError(f"Ollama has no model {self.model!r}: {detail}") from exc
                last_error = exc
            except (urllib.error.URLError, socket.timeout, TimeoutError,
                    json.JSONDecodeError, ValueError, ConnectionError) as exc:
                last_error = exc
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, self.retries + 1, last_error)
            if attempt <= self.retries:
                time.sleep(min(2 ** attempt, 10))
        raise LLMError(f"{self.model} failed after {self.retries + 1} attempts: {last_error}")
