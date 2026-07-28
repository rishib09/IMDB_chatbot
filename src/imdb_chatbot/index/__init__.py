"""Versioned dense (FAISS) + sparse (BM25) index build with an embedding cache.

Ticket #15. The public surface:

- ``embedder``: the ``Embedder`` protocol plus a dependency-free ``StubEmbedder``
  (used by every unit test) and a lazy ``SentenceTransformerEmbedder`` (the real
  one, behind the ``embed`` extra).
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
    SentenceTransformerEmbedder,
    StubEmbedder,
    l2_normalize,
)

__all__ = [
    "Embedder",
    "EmbeddingCache",
    "SentenceTransformerEmbedder",
    "StubEmbedder",
    "embed_cached",
    "l2_normalize",
]
