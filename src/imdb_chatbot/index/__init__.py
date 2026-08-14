"""Versioned dense (FAISS) + sparse (BM25) index build with an embedding cache.

Ticket #15. The public surface:

- ``embedder``: the ``Embedder`` protocol plus a dependency-free ``StubEmbedder``
  (used by every unit test), a lazy ``SentenceTransformerEmbedder`` (the local
  model, behind the ``embed`` extra), an ``OpenRouterEmbedder`` (hosted OpenAI
  embeddings on the existing key), and ``build_embedder`` to pick one by config.
- ``cache``: an infinite-TTL SQLite embedding cache so re-runs only embed misses.
- ``build``: read a corpus TraceStore, compose per-movie text, embed (cached) and
  L2-normalize, build a FAISS ``IndexFlatIP`` + a BM25 index, save versioned
  artifacts, and flip ``config/live_index.json`` at the new version.

``faiss`` is imported lazily inside ``build`` so importing this package never
requires faiss (or torch).
"""

from __future__ import annotations

from .cache import EmbeddingCache, embed_cached
from .embedder import (
    Embedder,
    OpenRouterEmbedder,
    SentenceTransformerEmbedder,
    StubEmbedder,
    build_embedder,
    l2_normalize,
)

__all__ = [
    "Embedder",
    "EmbeddingCache",
    "OpenRouterEmbedder",
    "SentenceTransformerEmbedder",
    "StubEmbedder",
    "build_embedder",
    "embed_cached",
    "l2_normalize",
]
