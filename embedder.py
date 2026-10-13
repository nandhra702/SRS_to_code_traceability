"""
Embedder — local, deterministic text/code embedding via sentence-transformers.

No AI service.  No API calls.  No network dependency at runtime.
The model downloads once on first use (~80 MB) and then runs fully offline.

Default model: sentence-transformers/all-MiniLM-L6-v2  (384-dim)

The Embedder interface is deliberately thin so it can be replaced with a
mock in tests without patching internals:

    class FakeEmbedder:
        dim = 4
        model_name = "fake"
        def embed(self, text): return [0.0] * 4
        def embed_batch(self, texts): return {k: [0.0]*4 for k in texts}
        def text_hash(self, text): return "abc123"

EmbeddingResult bundles a vector with its provenance (model version + text
hash) so the store can record them alongside each row.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, NamedTuple, Optional

from .config import TraceabilityConfig

logger = logging.getLogger(__name__)


class EmbeddingResult(NamedTuple):
    """A vector paired with the metadata needed to detect stale embeddings."""
    vector: List[float]
    model_version: str
    text_hash: str       # SHA-256 hex of the text that produced this vector


class Embedder:
    """
    Wraps a sentence-transformers SentenceTransformer model.

    The model is loaded lazily on the first call to embed() / embed_batch()
    so that importing this module never triggers a download.
    """

    def __init__(self, config: TraceabilityConfig) -> None:
        self._config = config
        self._model_name = config.embed_model
        self._model_version = config.embed_model_version
        self._model = None
        self._dim: Optional[int] = config.embed_dim
        # Per-call reporting from embed_batch()
        self.skipped_empty: List[str] = []
        self.truncated: List[str] = []

    # ------------------------------------------------------------------
    # Dimension
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """
        Vector dimension — resolved from the model on first access.

        Raises:
            ImportError: if sentence-transformers is not installed.
        """
        if self._dim is None:
            self._ensure_model()
            # sentence-transformers >= 6 renamed the accessor; support both.
            get_dim = (
                getattr(self._model, "get_embedding_dimension", None)
                or self._model.get_sentence_embedding_dimension
            )
            self._dim = get_dim()
            logger.info(
                "Embedding dimension resolved: %d (model=%s)",
                self._dim,
                self._model_name,
            )
        return self._dim

    # ------------------------------------------------------------------
    # Text hash (deterministic, model-independent)
    # ------------------------------------------------------------------

    @staticmethod
    def text_hash(text: str) -> str:
        """Return the SHA-256 hex digest of *text* encoded as UTF-8."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Single embed
    # ------------------------------------------------------------------

    def embed(self, text: str) -> EmbeddingResult:
        """
        Embed a single string.

        Args:
            text: The text to embed.

        Returns:
            EmbeddingResult(vector, model_version, text_hash).

        Raises:
            ValueError: if *text* is empty or whitespace — an empty string has
                no meaning to embed, so callers must skip it instead.
        """
        if not text or not text.strip():
            raise ValueError("Refusing to embed empty text.")
        self._ensure_model()
        vec = self._model.encode(text, convert_to_numpy=True)
        return EmbeddingResult(
            vector=vec.tolist(),
            model_version=self._model_version,
            text_hash=self.text_hash(text),
        )

    # ------------------------------------------------------------------
    # Batch embed
    # ------------------------------------------------------------------

    def embed_batch(
        self,
        texts: Dict[str, str],
        *,
        batch_size: Optional[int] = None,
    ) -> Dict[str, EmbeddingResult]:
        """
        Embed multiple texts identified by arbitrary string keys.

        sentence-transformers encodes the whole list in one call, which is
        significantly faster than individual encodes (especially on GPU).

        Args:
            texts:      {identifier: text_to_embed}
            batch_size: Overrides config.embed_batch_size when set.

        Returns:
            {identifier: EmbeddingResult} — keys whose text is empty or
            whitespace are omitted (see ``skipped_empty``).

        After the call, ``skipped_empty`` lists keys omitted for empty text
        and ``truncated`` lists keys whose text exceeded the model's maximum
        sequence length (the model silently encodes only the first part).
        """
        self.skipped_empty: List[str] = [
            k for k, t in texts.items() if not t or not t.strip()
        ]
        self.truncated: List[str] = []
        if self.skipped_empty:
            logger.warning(
                "Skipping %d empty text(s); no embedding stored for them.",
                len(self.skipped_empty),
            )
        skipped = set(self.skipped_empty)
        keys = [k for k in texts if k not in skipped]
        if not keys:
            return {}

        self._ensure_model()
        bs = batch_size or self._config.embed_batch_size
        sentences = [texts[k] for k in keys]

        self.truncated = self._find_truncated(keys, sentences)
        if self.truncated:
            logger.warning(
                "%d text(s) exceed the model's max sequence length and will be "
                "truncated when embedded.",
                len(self.truncated),
            )

        logger.debug("Encoding %d texts (batch_size=%d)…", len(sentences), bs)
        vectors = self._model.encode(
            sentences,
            batch_size=bs,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return {
            key: EmbeddingResult(
                vector=vec.tolist(),
                model_version=self._model_version,
                text_hash=self.text_hash(text),
            )
            for key, vec, text in zip(keys, vectors, sentences)
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_truncated(self, keys: List[str], sentences: List[str]) -> List[str]:
        """Keys whose tokenized length exceeds the model's max_seq_length."""
        limit = getattr(self._model, "max_seq_length", None)
        tokenizer = getattr(self._model, "tokenizer", None)
        if not isinstance(limit, int) or tokenizer is None:
            return []
        try:
            ids = tokenizer(sentences, add_special_tokens=True)["input_ids"]
            return [k for k, toks in zip(keys, ids) if len(toks) > limit]
        except Exception:  # truncation reporting is best-effort, never fatal
            logger.debug("Could not compute token lengths.", exc_info=True)
            return []

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding. "
                "Install with: pip install sentence-transformers"
            ) from exc
        local_only = not self._config.embed_allow_download
        logger.info("Loading embedding model: %s (local_files_only=%s)",
                    self._model_name, local_only)
        try:
            self._model = SentenceTransformer(self._model_name, local_files_only=local_only)
        except OSError as exc:
            if not local_only:
                raise
            raise RuntimeError(
                f"Embedding model {self._model_name!r} is not in the local cache. "
                "Downloads are disabled by default; set EMBED_ALLOW_DOWNLOAD=1 "
                "once to fetch it."
            ) from exc
        logger.info("Embedding model loaded.")

    # ------------------------------------------------------------------
    # Properties / repr
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    def __repr__(self) -> str:
        dim_str = str(self._dim) if self._dim is not None else "unresolved"
        return f"Embedder(model={self._model_name!r}, version={self._model_version!r}, dim={dim_str})"
