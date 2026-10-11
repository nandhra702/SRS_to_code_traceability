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
            self._dim = self._model.get_sentence_embedding_dimension()
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
        """
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
            {identifier: EmbeddingResult}
        """
        if not texts:
            return {}

        self._ensure_model()
        bs = batch_size or self._config.embed_batch_size
        keys = list(texts.keys())
        sentences = [texts[k] for k in keys]

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
        logger.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
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
