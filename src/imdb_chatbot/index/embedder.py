"""Text embedders and L2 normalization.

Two implementations sit behind one ``Embedder`` protocol:

- ``StubEmbedder``: deterministic, dependency-free (hash -> seeded vector). Every
  unit test uses it, so tests never need torch or a network.
- ``SentenceTransformerEmbedder``: the real model. ``sentence_transformers`` is
  imported LAZILY inside ``__init__`` so merely importing this module never pulls
  torch; you only pay for it when you actually construct the real embedder.

Cosine similarity is done as inner product over L2-normalized vectors, so
``l2_normalize`` is applied to document vectors at build time AND to the query
vector at search time. Forgetting the query-side normalization is the classic
retrieval bug that a named regression test guards against.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Turns text into float32 vectors.

    ``encode`` returns a ``(len(texts), dim)`` float32 array. ``name`` is a
    filesystem-safe identifier that feeds the index version stamp, and it is also
    part of the embedding cache key, so different embedders never collide.
    """

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row so inner product equals cosine similarity.

    Zero vectors are left as zeros (their norm is replaced by 1 to avoid a
    divide-by-zero) rather than producing NaNs.
    """
    mat = np.asarray(vectors, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (mat / norms).astype(np.float32)


class StubEmbedder:
    """Deterministic, dependency-free embedder for tests.

    Each text is hashed with SHA-256; the digest seeds a NumPy generator that
    draws a ``dim``-length vector. Same text -> same vector, always, with no
    model download and no torch. The output is NOT normalized here; the build
    normalizes it (matching how the real embedder is treated).
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return f"stub{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big", signed=False)
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(self._dim).astype(np.float32)
        return out


class SentenceTransformerEmbedder:
    """The real embedder, backed by ``sentence-transformers`` (torch).

    ``sentence_transformers`` is imported lazily in ``__init__`` so importing this
    module stays torch-free. A clear error is raised if the ``embed`` extra is not
    installed.
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "SentenceTransformerEmbedder requires the 'embed' extra. "
                "Install it with: pip install -e '.[embed]'"
            ) from exc

        self._model_id = model
        self._model = SentenceTransformer(model)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        # Keep it filesystem-safe: drop any org prefix like "sentence-transformers/".
        return self._model_id.split("/")[-1]

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=False
        )
        return np.asarray(vectors, dtype=np.float32)
