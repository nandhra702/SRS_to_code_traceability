"""
Tests for traceability.embedder — Embedder and EmbeddingResult.

All sentence-transformers model calls are mocked so no download is needed.
Integration tests at the bottom skip if the package is not installed.
"""

from __future__ import annotations

import hashlib
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from traceability.config import TraceabilityConfig
from traceability.embedder import Embedder, EmbeddingResult

FAKE_DIM = 4
FAKE_VEC = [0.1, 0.2, 0.3, 0.4]
FAKE_VEC_NP = np.array(FAKE_VEC, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(dim: int = FAKE_DIM, vec: List[float] = None) -> MagicMock:
    arr = np.array(vec or FAKE_VEC, dtype=np.float32)
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim

    def encode_side(inp, **kw):
        if isinstance(inp, str):
            return arr
        return np.stack([arr] * len(inp))

    model.encode.side_effect = encode_side
    return model


def _embedder(dim: int = None, version: str = "test-v1") -> tuple[Embedder, MagicMock]:
    cfg = TraceabilityConfig(embed_dim=dim, embed_model_version=version)
    emb = Embedder(cfg)
    mock_model = _make_mock_model(dim or FAKE_DIM)
    emb._model = mock_model
    if dim is None:
        emb._dim = None
    return emb, mock_model


# ============================================================================
# EmbeddingResult
# ============================================================================

class TestEmbeddingResult:

    def test_is_named_tuple(self):
        er = EmbeddingResult(vector=[0.1], model_version="v1", text_hash="abc")
        assert er.vector == [0.1]
        assert er.model_version == "v1"
        assert er.text_hash == "abc"

    def test_unpacking(self):
        vec, ver, h = EmbeddingResult([0.1], "v1", "abc")
        assert vec == [0.1]


# ============================================================================
# Embedder.text_hash
# ============================================================================

class TestTextHash:

    def test_sha256_of_empty_string(self):
        expected = hashlib.sha256(b"").hexdigest()
        assert Embedder.text_hash("") == expected

    def test_sha256_of_known_string(self):
        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert Embedder.text_hash("hello") == expected

    def test_different_texts_different_hashes(self):
        assert Embedder.text_hash("a") != Embedder.text_hash("b")

    def test_static_method(self):
        # Must be callable without an instance
        result = Embedder.text_hash("test")
        assert len(result) == 64  # SHA-256 hex


# ============================================================================
# dim property
# ============================================================================

class TestDimProperty:

    def test_preset_dim_no_model_load(self):
        cfg = TraceabilityConfig(embed_dim=384)
        emb = Embedder(cfg)
        with patch("traceability.embedder.SentenceTransformer") as MockST:
            assert emb.dim == 384
        MockST.assert_not_called()

    def test_resolves_from_model_when_none(self):
        emb, mock_model = _embedder(dim=None)
        assert emb.dim == FAKE_DIM
        mock_model.get_sentence_embedding_dimension.assert_called_once()

    def test_dim_cached(self):
        emb, mock_model = _embedder(dim=None)
        _ = emb.dim
        _ = emb.dim
        mock_model.get_sentence_embedding_dimension.assert_called_once()

    def test_import_error_raises_helpful_message(self):
        cfg = TraceabilityConfig(embed_dim=None)
        emb = Embedder(cfg)
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                _ = emb.dim


# ============================================================================
# embed (single)
# ============================================================================

class TestEmbedSingle:

    def test_returns_embedding_result(self):
        emb, _ = _embedder()
        result = emb.embed("hello")
        assert isinstance(result, EmbeddingResult)

    def test_vector_is_float_list(self):
        emb, _ = _embedder()
        result = emb.embed("hello")
        assert isinstance(result.vector, list)
        assert all(isinstance(x, float) for x in result.vector)

    def test_vector_length(self):
        emb, _ = _embedder()
        assert len(emb.embed("hello").vector) == FAKE_DIM

    def test_model_version_in_result(self):
        emb, _ = _embedder(version="my-v2")
        result = emb.embed("test")
        assert result.model_version == "my-v2"

    def test_text_hash_in_result(self):
        emb, _ = _embedder()
        text = "user authentication"
        result = emb.embed(text)
        assert result.text_hash == Embedder.text_hash(text)

    def test_empty_string_no_crash(self):
        emb, _ = _embedder()
        result = emb.embed("")
        assert len(result.vector) == FAKE_DIM

    def test_encode_called_with_correct_text(self):
        emb, mock_model = _embedder()
        emb.embed("specific text")
        assert mock_model.encode.call_args[0][0] == "specific text"

    def test_convert_to_numpy_true(self):
        emb, mock_model = _embedder()
        emb.embed("x")
        _, kw = mock_model.encode.call_args
        assert kw.get("convert_to_numpy") is True


# ============================================================================
# embed_batch
# ============================================================================

class TestEmbedBatch:

    def test_all_keys_returned(self):
        emb, _ = _embedder()
        texts = {"a": "foo", "b": "bar", "c": "baz"}
        result = emb.embed_batch(texts)
        assert set(result.keys()) == {"a", "b", "c"}

    def test_each_value_is_embedding_result(self):
        emb, _ = _embedder()
        result = emb.embed_batch({"x": "hello"})
        assert isinstance(result["x"], EmbeddingResult)

    def test_text_hash_correct_per_key(self):
        emb, _ = _embedder()
        texts = {"a": "foo", "b": "bar"}
        result = emb.embed_batch(texts)
        for key, text in texts.items():
            assert result[key].text_hash == Embedder.text_hash(text)

    def test_model_version_propagated(self):
        emb, _ = _embedder(version="v99")
        result = emb.embed_batch({"x": "text"})
        assert result["x"].model_version == "v99"

    def test_empty_dict_no_encode_call(self):
        emb, mock_model = _embedder()
        result = emb.embed_batch({})
        assert result == {}
        mock_model.encode.assert_not_called()

    def test_single_encode_call_for_whole_batch(self):
        emb, mock_model = _embedder()
        emb.embed_batch({str(i): f"t{i}" for i in range(10)})
        assert mock_model.encode.call_count == 1

    def test_batch_size_forwarded(self):
        emb, mock_model = _embedder()
        emb.embed_batch({"a": "x"}, batch_size=16)
        _, kw = mock_model.encode.call_args
        assert kw.get("batch_size") == 16

    def test_key_order_preserved(self):
        emb, mock_model = _embedder()
        vec_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        mock_model.encode.side_effect = None
        mock_model.encode.return_value = np.stack([vec_a, vec_b])
        result = emb.embed_batch({"a": "t_a", "b": "t_b"})
        assert result["a"].vector == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-5)
        assert result["b"].vector == pytest.approx([0.0, 1.0, 0.0, 0.0], abs=1e-5)


# ============================================================================
# Lazy loading
# ============================================================================

class TestLazyLoading:

    def test_model_none_at_init(self):
        emb = Embedder(TraceabilityConfig())
        assert emb._model is None

    def test_model_loaded_on_first_embed(self):
        cfg = TraceabilityConfig(embed_dim=FAKE_DIM)
        emb = Embedder(cfg)
        mock = _make_mock_model()
        with patch("traceability.embedder.SentenceTransformer", return_value=mock):
            emb.embed("trigger")
        assert emb._model is mock

    def test_model_loaded_once(self):
        cfg = TraceabilityConfig(embed_dim=FAKE_DIM)
        emb = Embedder(cfg)
        mock = _make_mock_model()
        with patch(
            "traceability.embedder.SentenceTransformer", return_value=mock
        ) as MockST:
            emb.embed("first")
            emb.embed("second")
        MockST.assert_called_once()


# ============================================================================
# Properties / repr
# ============================================================================

class TestProperties:

    def test_model_name(self):
        cfg = TraceabilityConfig(embed_model="my-model")
        assert Embedder(cfg).model_name == "my-model"

    def test_model_version(self):
        cfg = TraceabilityConfig(embed_model_version="v3")
        assert Embedder(cfg).model_version == "v3"

    def test_repr_unresolved(self):
        cfg = TraceabilityConfig(embed_dim=None)
        assert "unresolved" in repr(Embedder(cfg))

    def test_repr_with_dim(self):
        cfg = TraceabilityConfig(embed_dim=384, embed_model_version="v1")
        r = repr(Embedder(cfg))
        assert "384" in r and "v1" in r


# ============================================================================
# Integration — real model
# ============================================================================

def _st_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _st_available(),
    reason="sentence-transformers not installed. Run: pip install sentence-transformers",
)
class TestEmbedderIntegration:

    @pytest.fixture(scope="class")
    def embedder(self):
        cfg = TraceabilityConfig(
            embed_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        return Embedder(cfg)

    def test_dim_is_384(self, embedder):
        assert embedder.dim == 384

    def test_embed_returns_correct_length(self, embedder):
        result = embedder.embed("user authentication login")
        assert len(result.vector) == 384

    def test_text_hash_matches(self, embedder):
        text = "user authentication"
        result = embedder.embed(text)
        assert result.text_hash == Embedder.text_hash(text)

    def test_different_texts_differ(self, embedder):
        r1 = embedder.embed("user authentication login")
        r2 = embedder.embed("database schema migration")
        assert r1.vector != r2.vector

    def test_batch_matches_individual(self, embedder):
        texts = {"a": "user login", "b": "database schema"}
        batch = embedder.embed_batch(texts)
        for key, text in texts.items():
            ind = embedder.embed(text)
            assert batch[key].vector == pytest.approx(ind.vector, abs=1e-5)

    def test_similar_texts_closer_than_dissimilar(self, embedder):
        import math

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x**2 for x in a))
            nb = math.sqrt(sum(x**2 for x in b))
            return dot / (na * nb) if na and nb else 0.0

        v_login = embedder.embed("user login authentication").vector
        v_auth  = embedder.embed("user authentication credentials").vector
        v_db    = embedder.embed("database table schema migration").vector

        assert cosine(v_login, v_auth) > cosine(v_login, v_db)
