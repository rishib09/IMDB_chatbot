"""Versioned FAISS + BM25 index build with a hot-swap pointer.

Pipeline:

1. Read every movie from a corpus ``TraceStore`` (read-only, WAL-safe).
2. Compose one text per movie under chunk policy ``v1_one_per_movie``.
3. Embed via the (cached) embedder and L2-normalize -> cosine == inner product.
4. Build a FAISS ``IndexFlatIP`` over the doc vectors.
5. Build a ``BM25Okapi`` over the tokenized composed text.
6. Save artifacts under ``{out_root}/{dataset_version}_{embedder}_{chunk_policy}``:
   ``index.faiss``, ``bm25.pkl``, and ``sidecar.json`` (row_index -> tmdb_id).
7. Flip ``config/live_index.json`` to the new version. Hot-swap is the pointer
   flip; rollback is reverting it.

The whole thing is idempotent and cache-hot: re-running against an unchanged
corpus re-embeds nothing and just rewrites the (identical) artifacts.

``faiss`` and ``rank_bm25`` are imported lazily inside the functions that need
them, so importing this module never requires faiss or torch.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..config import CONFIG_DIR
from ..schemas import MovieRecord
from ..store import TraceStore
from .cache import EmbeddingCache, embed_cached
from .embedder import Embedder, l2_normalize

CHUNK_POLICY_V1 = "v1_one_per_movie"

DEFAULT_INDEX_ROOT = Path("data") / "index"
DEFAULT_CACHE_PATH = Path("data") / "embedding_cache.sqlite"
LIVE_INDEX_PATH = CONFIG_DIR / "live_index.json"


def compose_text(movie: MovieRecord) -> str:
    """Chunk policy ``v1_one_per_movie``: one movie -> one chunk of text."""
    genres = ", ".join(movie.genres)
    cast = ", ".join(movie.cast)
    director = movie.director or ""
    plot = movie.plot or ""
    return (
        f"{movie.title} ({movie.year}). "
        f"Genre: {genres}. "
        f"Director: {director}. "
        f"Cast: {cast}. "
        f"Plot: {plot}"
    )


_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization shared by the BM25 index and its queries."""
    return _TOKEN_RE.findall(text.lower())


def version_stamp(dataset_version: str, embedder: Embedder, chunk_policy: str) -> str:
    return f"{dataset_version}_{embedder.name}_{chunk_policy}"


@dataclass
class BuildResult:
    """What a build produced and where it landed."""

    version: str
    out_dir: Path
    count: int
    dim: int
    tmdb_ids: list[int]


def embed_query(embedder: Embedder, text: str, cache: EmbeddingCache | None = None) -> np.ndarray:
    """Embed and L2-normalize a single query -> shape ``(1, dim)`` float32.

    This MUST normalize: cosine search over an ``IndexFlatIP`` only works if the
    query is normalized the same way the documents were. See the named regression
    test in ``tests/test_index.py``.
    """
    raw = embed_cached(embedder, [text], cache)
    return l2_normalize(raw)


def write_live_pointer(
    version: str,
    out_dir: Path,
    *,
    live_index_path: Path = LIVE_INDEX_PATH,
) -> None:
    """Flip the live-index pointer to ``version`` (the hot-swap).

    Preserves the human-readable ``note`` if the pointer file already has one.
    """
    note = (
        "Set by the index build (ticket #15) to "
        "{dataset_version}_{embedder}_{chunk_policy}. "
        "Hot-swap = flip this pointer; rollback = revert it."
    )
    if live_index_path.exists():
        try:
            existing = json.loads(live_index_path.read_text(encoding="utf-8"))
            note = existing.get("note", note)
        except (json.JSONDecodeError, OSError):
            pass

    payload = {
        "active": version,
        "path": out_dir.as_posix(),
        "updated_utc": datetime.now(UTC).isoformat(),
        "note": note,
    }
    live_index_path.parent.mkdir(parents=True, exist_ok=True)
    live_index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_index(
    store: TraceStore,
    embedder: Embedder,
    *,
    dataset_version: str = "dev",
    chunk_policy: str = CHUNK_POLICY_V1,
    out_root: Path | str = DEFAULT_INDEX_ROOT,
    cache: EmbeddingCache | None = None,
    flip_pointer: bool = True,
    live_index_path: Path = LIVE_INDEX_PATH,
) -> BuildResult:
    """Build versioned FAISS + BM25 artifacts from a corpus store.

    ``faiss`` and ``rank_bm25`` are imported here (lazily) so the module import
    stays light. When ``flip_pointer`` is true the live pointer is flipped to the
    freshly built version.
    """
    import faiss
    from rank_bm25 import BM25Okapi

    if chunk_policy != CHUNK_POLICY_V1:
        raise ValueError(f"Unsupported chunk_policy: {chunk_policy!r}")

    movies = store.read_all_movies()
    tmdb_ids = [m.tmdb_id for m in movies]
    texts = [compose_text(m) for m in movies]

    # Dense side: embed (cached) then L2-normalize so IP == cosine.
    raw = embed_cached(embedder, texts, cache)
    vectors = l2_normalize(raw)

    version = version_stamp(dataset_version, embedder, chunk_policy)
    out_dir = Path(out_root) / version
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss_index = faiss.IndexFlatIP(embedder.dim)
    if len(vectors) > 0:
        faiss_index.add(vectors)
    faiss.write_index(faiss_index, str(out_dir / "index.faiss"))

    # Sparse side: BM25 over tokenized composed text.
    tokenized = [tokenize(t) for t in texts]
    # BM25Okapi requires a non-empty corpus; guard the empty-store case.
    bm25 = BM25Okapi(tokenized) if tokenized else None
    with (out_dir / "bm25.pkl").open("wb") as fh:
        pickle.dump(bm25, fh)

    sidecar = {
        "version": version,
        "dataset_version": dataset_version,
        "embedder": embedder.name,
        "dim": embedder.dim,
        "chunk_policy": chunk_policy,
        "count": len(movies),
        "created_utc": datetime.now(UTC).isoformat(),
        # row_index -> tmdb_id (list index IS the FAISS/BM25 row index).
        "row_to_tmdb_id": tmdb_ids,
    }
    (out_dir / "sidecar.json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )

    if flip_pointer:
        write_live_pointer(version, out_dir, live_index_path=live_index_path)

    return BuildResult(
        version=version,
        out_dir=out_dir,
        count=len(movies),
        dim=embedder.dim,
        tmdb_ids=tmdb_ids,
    )


@dataclass
class LoadedIndex:
    """An index loaded back from disk, ready to search."""

    version: str
    dim: int
    row_to_tmdb_id: list[int]
    faiss_index: object  # faiss.IndexFlatIP
    bm25: object | None  # rank_bm25.BM25Okapi | None


def load_index(out_dir: Path | str) -> LoadedIndex:
    """Load FAISS + BM25 + sidecar artifacts from a version directory."""
    import faiss

    out_dir = Path(out_dir)
    sidecar = json.loads((out_dir / "sidecar.json").read_text(encoding="utf-8"))
    faiss_index = faiss.read_index(str(out_dir / "index.faiss"))
    with (out_dir / "bm25.pkl").open("rb") as fh:
        bm25 = pickle.load(fh)
    return LoadedIndex(
        version=sidecar["version"],
        dim=sidecar["dim"],
        row_to_tmdb_id=sidecar["row_to_tmdb_id"],
        faiss_index=faiss_index,
        bm25=bm25,
    )


def dense_search(
    loaded: LoadedIndex,
    embedder: Embedder,
    query: str,
    k: int = 5,
    cache: EmbeddingCache | None = None,
) -> list[tuple[int, float]]:
    """Cosine search: returns ``[(tmdb_id, score), ...]`` best-first.

    The query is L2-normalized via ``embed_query`` before it hits the FAISS
    ``IndexFlatIP`` - the whole point of the normalization contract.
    """
    qvec = embed_query(embedder, query, cache)
    scores, idxs = loaded.faiss_index.search(qvec, k)
    out: list[tuple[int, float]] = []
    for score, row in zip(scores[0], idxs[0], strict=True):
        if row < 0:
            continue
        out.append((loaded.row_to_tmdb_id[row], float(score)))
    return out
