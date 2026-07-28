"""Infinite-TTL SQLite embedding cache.

Embedding a text with a fixed model is a pure function, so its result never goes
stale: the cache has no TTL. Keys are ``sha256(f"{model_name}::{text}")`` and
values are the raw float32 vector bytes. On build we look up every text, embed
only the misses, and store them, so a warm re-run does zero model work.

The cache is a standalone SQLite file (not the corpus store) so it can be shared
across builds and wiped independently.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Self

import numpy as np

from .embedder import Embedder

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key   TEXT PRIMARY KEY,
    dim   INTEGER NOT NULL,
    vec   BLOB NOT NULL
);
"""


def cache_key(model_name: str, text: str) -> str:
    """Content-addressed key: same (model, text) -> same key, forever."""
    return hashlib.sha256(f"{model_name}::{text}".encode()).hexdigest()


class EmbeddingCache:
    """SQLite-backed store of ``key -> float32 vector``."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> np.ndarray | None:
        row = self._conn.execute(
            "SELECT vec FROM embeddings WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32)

    def put(self, key: str, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float32)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)",
            (key, int(vec.shape[-1]), vec.tobytes()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def embed_cached(
    embedder: Embedder,
    texts: list[str],
    cache: EmbeddingCache | None,
) -> np.ndarray:
    """Embed ``texts``, reusing cached vectors and only embedding cache misses.

    Returns a ``(len(texts), dim)`` float32 array in the same order as ``texts``.
    When ``cache`` is provided, a warm re-run embeds nothing: the embedder's
    ``encode`` is called only for texts not already stored (and never at all if
    every text is a hit).
    """
    if not texts:
        return np.empty((0, embedder.dim), dtype=np.float32)

    out = np.empty((len(texts), embedder.dim), dtype=np.float32)

    if cache is None:
        out[:] = embedder.encode(texts)
        return out

    keys = [cache_key(embedder.name, t) for t in texts]
    miss_idx: list[int] = []
    for i, key in enumerate(keys):
        hit = cache.get(key)
        if hit is None:
            miss_idx.append(i)
        else:
            out[i] = hit

    if miss_idx:
        miss_texts = [texts[i] for i in miss_idx]
        fresh = embedder.encode(miss_texts)
        for j, i in enumerate(miss_idx):
            out[i] = fresh[j]
            cache.put(keys[i], fresh[j])

    return out
