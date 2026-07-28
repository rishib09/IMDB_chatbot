"""Tests for the versioned FAISS/BM25 index build + embedding cache (ticket #15).

All tests use the dependency-free ``StubEmbedder`` and a handful of synthetic
``MovieRecord``s. Tests that need ``faiss`` are skipped when it is not installed
(e.g. local Python where no faiss-cpu wheel exists); they run on CI (Python 3.12).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from imdb_chatbot.index.build import (
    CHUNK_POLICY_V1,
    build_index,
    compose_text,
    dense_search,
    embed_query,
    load_index,
    version_stamp,
    write_live_pointer,
)
from imdb_chatbot.index.cache import EmbeddingCache, cache_key, embed_cached
from imdb_chatbot.index.embedder import StubEmbedder, l2_normalize
from imdb_chatbot.schemas import MovieRecord
from imdb_chatbot.store import TraceStore

HAS_FAISS = importlib.util.find_spec("faiss") is not None
requires_faiss = pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")


# -- fixtures / helpers -------------------------------------------------------


class CountingEmbedder:
    """Wraps an embedder and counts how many texts it actually encodes.

    Used to prove a warm re-run does no re-embedding. ``name``/``dim`` delegate to
    the inner embedder so cache keys are identical across runs.
    """

    def __init__(self, inner: StubEmbedder) -> None:
        self.inner = inner
        self.encoded_texts: list[str] = []
        self.calls = 0

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def dim(self) -> int:
        return self.inner.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.encoded_texts.extend(texts)
        return self.inner.encode(texts)


def _movies() -> list[MovieRecord]:
    return [
        MovieRecord(
            tmdb_id=101,
            title="Inception",
            year=2010,
            genres=["Sci-Fi", "Thriller"],
            director="Christopher Nolan",
            cast=["Leonardo DiCaprio", "Joseph Gordon-Levitt"],
            plot="A thief enters dreams to plant an idea.",
        ),
        MovieRecord(
            tmdb_id=202,
            title="Parasite",
            year=2019,
            genres=["Drama", "Thriller"],
            director="Bong Joon-ho",
            cast=["Song Kang-ho"],
            plot="A poor family schemes to work for a wealthy household.",
        ),
        MovieRecord(
            tmdb_id=303,
            title="Toy Story",
            year=1995,
            genres=["Animation", "Family"],
            director="John Lasseter",
            cast=["Tom Hanks", "Tim Allen"],
            plot="Toys come to life when people are away.",
        ),
    ]


@pytest.fixture()
def store(tmp_path: Path) -> TraceStore:
    s = TraceStore(tmp_path / "corpus.sqlite")
    try:
        for movie in _movies():
            s.write_movie(movie)
        yield s
    finally:
        s.close()


# -- store read-only access ---------------------------------------------------


def test_read_all_movies_returns_all_ordered(store: TraceStore) -> None:
    movies = store.read_all_movies()
    assert [m.tmdb_id for m in movies] == [101, 202, 303]


def test_iter_movies_matches_read_all(store: TraceStore) -> None:
    assert [m.tmdb_id for m in store.iter_movies()] == [101, 202, 303]


# -- chunk policy / composition ----------------------------------------------


def test_compose_text_chunk_policy_v1() -> None:
    movie = _movies()[0]
    text = compose_text(movie)
    assert text == (
        "Inception (2010). "
        "Genre: Sci-Fi, Thriller. "
        "Director: Christopher Nolan. "
        "Cast: Leonardo DiCaprio, Joseph Gordon-Levitt. "
        "Plot: A thief enters dreams to plant an idea."
    )


# -- normalization ------------------------------------------------------------


def test_l2_normalize_unit_norm() -> None:
    embedder = StubEmbedder(dim=8)
    raw = embedder.encode(["a", "b", "c"])
    normed = l2_normalize(raw)
    norms = np.linalg.norm(normed, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_normalize_zero_vector_stays_zero() -> None:
    zero = np.zeros((1, 4), dtype=np.float32)
    out = l2_normalize(zero)
    assert not np.isnan(out).any()
    assert np.allclose(out, 0.0)


def test_query_vector_is_l2_normalized_regression() -> None:
    """Regression guard for the classic 'forgot to normalize the query' bug.

    Cosine search over an IndexFlatIP only works if the query vector is
    L2-normalized exactly like the document vectors. If ``embed_query`` ever stops
    normalizing, this fails.
    """
    embedder = StubEmbedder(dim=8)
    qvec = embed_query(embedder, "movies like Inception")
    assert qvec.shape == (1, 8)
    assert np.linalg.norm(qvec[0]) == pytest.approx(1.0, abs=1e-5)


# -- embedding cache ----------------------------------------------------------


def test_cache_key_is_model_scoped() -> None:
    assert cache_key("m1", "hello") != cache_key("m2", "hello")
    assert cache_key("m1", "hello") == cache_key("m1", "hello")


def test_cache_round_trips_vector(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        key = cache_key("stub8", "x")
        assert cache.get(key) is None
        cache.put(key, vec)
        got = cache.get(key)
        assert got is not None
        assert np.allclose(got, vec)
    finally:
        cache.close()


def test_rerun_is_cache_hot(tmp_path: Path) -> None:
    """A warm re-run must embed NOTHING: the embedder is not called on hits."""
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        embedder = CountingEmbedder(StubEmbedder(dim=8))
        texts = [compose_text(m) for m in _movies()]

        first = embed_cached(embedder, texts, cache)
        assert embedder.calls == 1
        assert len(embedder.encoded_texts) == len(texts)

        # Second pass: every text is a cache hit -> encode never called.
        embedder.calls = 0
        embedder.encoded_texts = []
        second = embed_cached(embedder, texts, cache)
        assert embedder.calls == 0
        assert embedder.encoded_texts == []
        assert np.allclose(first, second)
    finally:
        cache.close()


def test_cache_embeds_only_misses(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        embedder = CountingEmbedder(StubEmbedder(dim=8))
        embed_cached(embedder, ["a", "b"], cache)
        embedder.encoded_texts = []
        # "a" is cached, "c" is new -> only "c" is embedded.
        embed_cached(embedder, ["a", "c"], cache)
        assert embedder.encoded_texts == ["c"]
    finally:
        cache.close()


# -- version stamping ---------------------------------------------------------


def test_version_stamp_format() -> None:
    embedder = StubEmbedder(dim=8)
    assert version_stamp("ds1", embedder, CHUNK_POLICY_V1) == "ds1_stub8_v1_one_per_movie"


# -- live pointer (no faiss needed) ------------------------------------------


def test_write_live_pointer_flips(tmp_path: Path) -> None:
    ptr = tmp_path / "live_index.json"
    write_live_pointer("v1_stub8_v1_one_per_movie", tmp_path / "d1", live_index_path=ptr)
    data = json.loads(ptr.read_text(encoding="utf-8"))
    assert data["active"] == "v1_stub8_v1_one_per_movie"

    write_live_pointer("v2_stub8_v1_one_per_movie", tmp_path / "d2", live_index_path=ptr)
    flipped = json.loads(ptr.read_text(encoding="utf-8"))
    assert flipped["active"] == "v2_stub8_v1_one_per_movie"


# -- full build (requires faiss) ---------------------------------------------


@requires_faiss
def test_build_index_creates_artifacts(store: TraceStore, tmp_path: Path) -> None:
    embedder = StubEmbedder(dim=8)
    ptr = tmp_path / "live_index.json"
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        result = build_index(
            store,
            embedder,
            dataset_version="ds1",
            out_root=tmp_path / "index",
            cache=cache,
            live_index_path=ptr,
        )
    finally:
        cache.close()

    assert result.version == "ds1_stub8_v1_one_per_movie"
    assert result.count == 3
    assert (result.out_dir / "index.faiss").exists()
    assert (result.out_dir / "bm25.pkl").exists()
    assert (result.out_dir / "sidecar.json").exists()

    # FAISS index has N vectors of the right dim.
    loaded = load_index(result.out_dir)
    assert loaded.faiss_index.ntotal == 3
    assert loaded.dim == 8
    # BM25 was built.
    assert loaded.bm25 is not None
    # Sidecar maps row_index -> tmdb_id.
    assert loaded.row_to_tmdb_id == [101, 202, 303]

    # Pointer resolves to the built version.
    ptr_data = json.loads(ptr.read_text(encoding="utf-8"))
    assert ptr_data["active"] == result.version


@requires_faiss
def test_build_vectors_are_l2_normalized(store: TraceStore, tmp_path: Path) -> None:
    import faiss

    embedder = StubEmbedder(dim=8)
    result = build_index(
        store,
        embedder,
        dataset_version="ds1",
        out_root=tmp_path / "index",
        cache=None,
        flip_pointer=False,
    )
    index = faiss.read_index(str(result.out_dir / "index.faiss"))
    stored = index.reconstruct_n(0, index.ntotal)
    norms = np.linalg.norm(stored, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@requires_faiss
def test_dense_search_returns_tmdb_ids(store: TraceStore, tmp_path: Path) -> None:
    embedder = StubEmbedder(dim=8)
    result = build_index(
        store,
        embedder,
        dataset_version="ds1",
        out_root=tmp_path / "index",
        cache=None,
        flip_pointer=False,
    )
    loaded = load_index(result.out_dir)
    hits = dense_search(loaded, embedder, "some query", k=3)
    assert len(hits) == 3
    ids = {tmdb_id for tmdb_id, _ in hits}
    assert ids == {101, 202, 303}


@requires_faiss
def test_build_flip_changes_pointer(store: TraceStore, tmp_path: Path) -> None:
    ptr = tmp_path / "live_index.json"
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        build_index(
            store,
            StubEmbedder(dim=8),
            dataset_version="ds1",
            out_root=tmp_path / "index",
            cache=cache,
            live_index_path=ptr,
        )
        first = json.loads(ptr.read_text(encoding="utf-8"))["active"]

        build_index(
            store,
            StubEmbedder(dim=16),
            dataset_version="ds1",
            out_root=tmp_path / "index",
            cache=cache,
            live_index_path=ptr,
        )
        second = json.loads(ptr.read_text(encoding="utf-8"))["active"]
    finally:
        cache.close()

    assert first == "ds1_stub8_v1_one_per_movie"
    assert second == "ds1_stub16_v1_one_per_movie"
    assert first != second


@requires_faiss
def test_build_rerun_is_cache_hot(store: TraceStore, tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        embedder = CountingEmbedder(StubEmbedder(dim=8))
        build_index(
            store, embedder, dataset_version="ds1",
            out_root=tmp_path / "index", cache=cache, flip_pointer=False,
        )
        assert embedder.calls == 1
        embedder.calls = 0
        embedder.encoded_texts = []
        build_index(
            store, embedder, dataset_version="ds1",
            out_root=tmp_path / "index", cache=cache, flip_pointer=False,
        )
        assert embedder.calls == 0
    finally:
        cache.close()
