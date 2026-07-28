"""Hybrid retrieval (ticket #16).

Public surface:

- ``HybridRetriever``: dense (FAISS) + sparse (BM25) -> RRF fusion -> deterministic
  metadata filters -> top-k ``ScoredMovie``, with an in-process LRU result cache.
- ``rrf_fuse``: the hand-rolled Reciprocal Rank Fusion helper.
- ``build_movie_map``: load a ``tmdb_id -> MovieRecord`` map from a corpus store.
"""

from __future__ import annotations

from .retrieve import (
    DENSE_K,
    FINAL_K,
    RRF_K,
    SPARSE_K,
    HybridRetriever,
    build_movie_map,
    rrf_fuse,
)

__all__ = [
    "DENSE_K",
    "FINAL_K",
    "RRF_K",
    "SPARSE_K",
    "HybridRetriever",
    "build_movie_map",
    "rrf_fuse",
]
