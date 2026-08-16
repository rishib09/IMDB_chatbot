"""Infinite-TTL SQLite embedding cache - the ONE embedding cache in this repo.

Embedding a text with a fixed model is a pure function, so its result never goes
stale: the cache has no TTL. Keys are ``sha256(f"{model_name}::{text}")`` and
values are the raw float32 vector bytes. On build we look up every text, embed
only the misses, and store them, so a warm re-run does zero model work.

The cache is a standalone SQLite file (not the corpus store) so it can be shared
across builds and wiped independently. Misses are written in batches - one
``executemany`` + one commit per ``batch_size`` texts (ticket #74) - so a cold
46k build costs ~46 fsyncs instead of 46k, and a killed build resumes from the
last committed batch.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Self

import numpy as np

from .embedder import Embedder

BATCH_SIZE = 1024

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
    """SQLite-backed store of ``key -> float32 vector``.

    Thread-safe (one connection + a lock) so the live query path can share a
    single instance across the UI's worker threads.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT vec FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32)

    def put_many(self, items: Iterable[tuple[str, np.ndarray]]) -> None:
        """Store many ``(key, vector)`` pairs in ONE transaction / ONE commit."""
        rows = [
            (key, int(vec.shape[-1]), vec.tobytes())
            for key, vec in ((k, np.asarray(v, dtype=np.float32)) for k, v in items)
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)", rows
            )
            self._conn.commit()

    def put(self, key: str, vector: np.ndarray) -> None:
        self.put_many([(key, vector)])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def embed_cached(
    embedder: Embedder,
    texts: list[str],
    cache: EmbeddingCache | None,
    *,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Embed ``texts``, reusing cached vectors and only embedding cache misses.

    Returns a ``(len(texts), dim)`` float32 array in the same order as ``texts``.
    When ``cache`` is provided, a warm re-run embeds nothing: the embedder's
    ``encode`` is called only for texts not already stored (and never at all if
    every text is a hit). Misses are embedded and committed ``batch_size`` at a
    time, so N misses cost ``ceil(N / batch_size)`` commits, not N.
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

    batch_size = max(1, batch_size)
    for start in range(0, len(miss_idx), batch_size):
        batch = miss_idx[start : start + batch_size]
        fresh = embedder.encode([texts[i] for i in batch])
        out[batch] = fresh
        cache.put_many((keys[i], fresh[j]) for j, i in enumerate(batch))

    return out
