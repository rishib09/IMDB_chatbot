"""Tests for the versioned FAISS/BM25 index build + embedding cache (ticket #15).

All tests use the dependency-free ``StubEmbedder`` and a handful of synthetic
``MovieRecord``s. Tests that need ``faiss`` are skipped when it is not installed
(e.g. local Python where no faiss-cpu wheel exists); they run on CI (Python 3.12).
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

from imdb_chatbot.index.build import (
    CAST_IN_TEXT,
    CHUNK_POLICY_V1,
    build_index,
    compose_text,
    dense_search,
    embed_query,
    load_index,
    over_budget,
    version_stamp,
    write_live_pointer,
)
from imdb_chatbot.index.cache import EmbeddingCache, cache_key, embed_cached
from imdb_chatbot.index.embedder import StubEmbedder, l2_normalize
from imdb_chatbot.retrieval.retrieve import _passes_filters
from imdb_chatbot.schemas import MovieRecord, ParsedQuery
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
        "Plot: A thief enters dreams to plant an idea. "
        "Cast: Leonardo DiCaprio, Joseph Gordon-Levitt"
    )


def test_plot_offset_is_independent_of_cast_size() -> None:
    """Attacks: 'the plot sits at a bounded offset in the embedded text.'

    This is the exact mechanism of #83. Cast used to precede the plot and was
    unbounded, so one 396-name cast pushed 'Plot:' past token 1,750 of a
    256-token window and the synopsis never reached the vector. Attack it with a
    cast far larger than anything in the corpus: the plot's offset must not move.
    """
    base = _movies()[0]
    small = base.model_copy(update={"cast": ["A. Actor"]})
    huge = base.model_copy(update={"cast": [f"Extra {i:04d}" for i in range(5000)]})

    small_text, huge_text = compose_text(small), compose_text(huge)
    assert small_text.index("Plot: ") == huge_text.index("Plot: ")
    # And the whole plot survives verbatim regardless of cast size.
    assert base.plot in huge_text
    # Cast trails the plot, so overflow eats actor names and never the synopsis.
    assert huge_text.index("Plot: ") < huge_text.index("Cast: ")


def test_embedded_cast_is_capped_but_the_record_is_not() -> None:
    """Attacks: 'capping the cast in the embedded text weakens exclusion.'

    #83 caps cast INSIDE compose_text only. Exclusion filtering and the actor
    index read ``MovieRecord.cast``, which must stay complete - otherwise a film
    stops being excluded once its excluded actor is billed 40th, silently
    breaking the exclusion-precision guarantee the pipeline advertises.
    """
    cast = [f"Star {i:02d}" for i in range(40)] + ["Buried Villain"]
    movie = _movies()[0].model_copy(update={"cast": cast})

    text = compose_text(movie)
    assert "Buried Villain" not in text  # capped out of the embedded text
    assert text.count("Star ") == CAST_IN_TEXT  # exactly the top-billed ten
    assert movie.cast == cast  # ...but the record is untouched

    parsed = ParsedQuery(exclude_actors=["Buried Villain"])
    assert not _passes_filters(movie, parsed, frozenset())
    # The actor-index path must find him too, not just the exclusion path.
    parsed_actor = ParsedQuery(actor="Buried Villain")
    assert _passes_filters(movie, parsed_actor, frozenset())


class _WindowedEmbedder(StubEmbedder):
    """StubEmbedder that declares a word-counted context window."""

    max_tokens = 5

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(t.split()) for t in texts]


def test_over_budget_reports_every_clipped_row_and_no_others() -> None:
    """Attacks: 'the build's clipped-row report is complete and exact.'

    An off-by-one (``>=`` instead of ``>``) would flag a text that fits exactly,
    and a text one token over would be dropped from the report - which is how a
    silent truncation returns wearing a report's clothes. Sit inputs on both
    sides of the boundary.
    """
    texts = ["a b c d", "a b c d e", "a b c d e f", "x"]  # 4, 5, 6, 1 words
    assert over_budget(_WindowedEmbedder(dim=8), texts) == [2]
    # An embedder that declares no window must report nothing, not crash.
    assert over_budget(StubEmbedder(dim=8), texts) == []


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


def _count_commits(cache: EmbeddingCache) -> list[str]:
    """Attach a SQLite trace callback; the returned list grows by one per COMMIT."""
    commits: list[str] = []
    cache._conn.set_trace_callback(
        lambda sql: commits.append(sql) if sql.strip().upper().startswith("COMMIT") else None
    )
    return commits


def test_a_thousand_misses_cost_at_most_one_commit(tmp_path: Path) -> None:
    """Attacks: N cache misses in one embed call cost <= 1 commit (ticket #74).

    Before #74 ``put`` committed per row, so a cold 46k build fsynced 46k times.
    Count COMMITs via the connection's trace callback while embedding 1,000
    distinct short texts through the deterministic local StubEmbedder.
    """
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        commits = _count_commits(cache)
        embedder = CountingEmbedder(StubEmbedder(dim=8))
        texts = [f"t{i}" for i in range(1000)]
        embed_cached(embedder, texts, cache)
        assert len(commits) <= 1
        assert embedder.calls == 1
        # Warm re-run: every text hits -> no encode, no write, no commit.
        commits.clear()
        embed_cached(embedder, texts, cache)
        assert embedder.calls == 1
        assert commits == []
    finally:
        cache.close()


def test_mixed_hits_and_ragged_batches_commit_once_per_batch(tmp_path: Path) -> None:
    """Attacks: commits == ceil(misses / batch_size), and batching keeps order.

    A batch mixing hits and misses whose miss count is NOT a multiple of
    ``batch_size`` (7 misses, batch 4) must commit exactly twice, encode exactly
    the misses, and return vectors identical to an uncached embed - a wrong
    index map in the batched write path would silently scramble rows.
    """
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        embedder = CountingEmbedder(StubEmbedder(dim=8))
        texts = [f"t{i}" for i in range(10)]
        embed_cached(embedder, texts[:3], cache)  # 3 hits pre-seeded
        commits = _count_commits(cache)
        embedder.calls, embedder.encoded_texts = 0, []
        got = embed_cached(embedder, texts, cache, batch_size=4)
        assert len(commits) == 2 and embedder.calls == 2
        assert embedder.encoded_texts == texts[3:]
        assert np.allclose(got, StubEmbedder(dim=8).encode(texts))
        assert np.allclose(embed_cached(embedder, texts, cache), got)
        assert embedder.calls == 2
    finally:
        cache.close()


@requires_faiss
def test_embedding_cache_is_the_single_source_of_truth(store: TraceStore, tmp_path: Path) -> None:
    """Attacks: one cache holds every vector - build path AND query path.

    Ticket #74 found the same cache implemented twice (``EmbeddingCache`` and a
    ``cache`` table in ``TraceStore``). Embed through both entry points, then
    assert the vectors live in exactly one SQLite file/table, and that the
    corpus store no longer carries a cache table of its own.
    """
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        embedder = CountingEmbedder(StubEmbedder(dim=8))
        result = build_index(
            store, embedder, dataset_version="ds1",
            out_root=tmp_path / "index", cache=cache, flip_pointer=False,
        )
        loaded = load_index(result.out_dir)
        dense_search(loaded, embedder, "dream heist", cache=cache)
        dense_search(loaded, embedder, "dream heist", cache=cache)  # warm query
        assert embedder.encoded_texts.count("dream heist") == 1
        (n_rows,) = cache._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        assert n_rows == len(set(embedder.encoded_texts)) == len(_movies()) + 1
    finally:
        cache.close()
    tables = {
        row["name"] for row in store._read_conn.execute("SELECT name FROM sqlite_master")
    }
    assert "cache" not in tables and not hasattr(store, "cache_put")


@pytest.mark.live
@requires_faiss
def test_warm_rebuild_over_real_corpus_embeds_nothing(
    live_corpus_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Live: a warm rebuild over the real 46k corpus + real cache embeds NOTHING.

    The embedder is a name/dim-matched shim that RAISES on ``encode``: a single
    miss aborts the build before anything is written to the shared cache. Also
    records the warm wall-clock (the cold number is in the #74 PR).
    """
    cache_path = live_corpus_path.parent / "embedding_cache.sqlite"
    if not cache_path.is_file():
        pytest.skip(f"real embedding cache not found beside the corpus: {cache_path}")

    class NeverEmbed:
        name, dim = "all-MiniLM-L6-v2", 384

        def encode(self, texts: list[str]) -> np.ndarray:
            raise AssertionError(f"warm rebuild tried to embed {len(texts)} texts")

    store = TraceStore(live_corpus_path)
    cache = EmbeddingCache(cache_path)
    try:
        t0 = time.perf_counter()
        result = build_index(
            store, NeverEmbed(), out_root=tmp_path / "index", cache=cache, flip_pointer=False,
        )
        print(f"warm rebuild: {result.count} movies, 0 embedder calls, {time.perf_counter()-t0:.1f}s")
    finally:
        cache.close()
        store.close()
    assert result.count > 40_000


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


# -- the real corpus, the real tokenizer (live tier) --------------------------


@pytest.mark.live
def test_no_movie_in_the_corpus_loses_its_plot_to_the_window(
    live_corpus_path: Path,
) -> None:
    """Attacks: 'every movie's plot reaches the vector' - on all 46,364 real rows.

    The synthetic guards above fix the mechanism; only the corpus proves the
    fleet is clean. Before #83 this failed for 2,779 movies whose plot began
    past the 256-token window, plus 3,929 more cut mid-sentence. Residual
    clipping (a plot that alone exceeds the window) must stay a reported
    handful, not a silent majority.
    """
    from imdb_chatbot.index.embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    store = TraceStore(live_corpus_path)
    try:
        movies = store.read_all_movies()
    finally:
        store.close()

    texts = [compose_text(m) for m in movies]
    # Where does the plot START? Everything before it is fixed-size overhead now.
    prefixes = [t[: t.index("Plot: ")] for t in texts]
    prefix_tokens = embedder.count_tokens(prefixes)

    lost = [
        m.tmdb_id
        for m, n in zip(movies, prefix_tokens, strict=True)
        if n >= embedder.max_tokens
    ]
    assert lost == [], f"{len(lost)} movies still lose their whole plot"

    # Residual: plots too long to fit even when they lead. Reported, not silent.
    clipped = over_budget(embedder, texts)
    assert len(clipped) < len(movies) * 0.02, (
        f"{len(clipped)}/{len(movies)} texts exceed the window - the cap regressed"
    )
