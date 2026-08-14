"""Adversary tests for the configurable / hosted embedder (#76).

Two assumptions the rest of the system leans on, each attacked here:

1. *"A failed query embedding degrades to sparse-only, never raises."*
   Dense retrieval became a NETWORK call the moment an embedder profile could be
   a hosted API. BM25 is local, so L2 ("LLM-free deterministic retrieval") is
   supposed to survive an outage. The attack simulates the embedder failing
   mid-query against a real, already-built FAISS + BM25 index.

2. *"The embedding cache key distinguishes any two different embedders."*
   Broken by ``dimensions``: two OpenRouter profiles on the SAME model differing
   only in output width share every cache key unless ``name`` carries the width -
   and then a 512-float vector is read back into a 1024-float slot.

Neither test needs a network. The live round-trip that confirms OpenRouter
actually honours ``dimensions`` is gated behind ``OPENROUTER_LIVE_TEST=1``,
following the ``TMDB_LIVE_TEST`` pattern.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from imdb_chatbot.index.build import build_index, load_index
from imdb_chatbot.index.cache import EmbeddingCache, cache_key, embed_cached
from imdb_chatbot.index.embedder import (
    OpenRouterEmbedder,
    StubEmbedder,
    build_embedder,
    l2_normalize,
)
from imdb_chatbot.retrieval.retrieve import HybridRetriever
from imdb_chatbot.schemas import MovieRecord
from imdb_chatbot.store import TraceStore

HAS_FAISS = importlib.util.find_spec("faiss") is not None
requires_faiss = pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")

LIVE = pytest.mark.skipif(
    os.environ.get("OPENROUTER_LIVE_TEST") != "1",
    reason="live OpenRouter call disabled (set OPENROUTER_LIVE_TEST=1 to enable)",
)


class NetworkOutage(RuntimeError):
    """What a hosted embedder raises when the network is gone."""


class FlakyEmbedder:
    """A real embedder that stops working part-way through the session.

    Identity (``name``/``dim``) delegates to the inner stub, so an index built
    with the stub stays valid: only the *query-time* call fails, which is exactly
    the outage shape - documents already embedded, queries no longer embeddable.
    """

    def __init__(self, inner: StubEmbedder, *, fail_after: int = 0) -> None:
        self.inner = inner
        self.fail_after = fail_after
        self.calls = 0

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def dim(self) -> int:
        return self.inner.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        if self.calls > self.fail_after:
            raise NetworkOutage("connection reset by peer")
        return self.inner.encode(texts)


def _corpus() -> list[MovieRecord]:
    return [
        MovieRecord(
            tmdb_id=1,
            title="Midnight Detective",
            year=2001,
            genres=["Thriller"],
            director="Dir A",
            cast=["Alice Kim"],
            plot="A detective hunts a killer through a rainy city at midnight.",
            regions=["US"],
            vote_count=500,
        ),
        MovieRecord(
            tmdb_id=2,
            title="Haunted Manor",
            year=2015,
            genres=["Horror"],
            director="Dir B",
            cast=["Bob Stone"],
            plot="A family is terrorized by ghosts in a haunted manor.",
            regions=["US"],
            vote_count=400,
        ),
        MovieRecord(
            tmdb_id=3,
            title="Delhi Nights",
            year=2012,
            genres=["Romance"],
            director="Dir C",
            cast=["Priya Nair"],
            plot="Two strangers fall in love over one long night in Delhi.",
            regions=["IN"],
            vote_count=300,
        ),
    ]


@pytest.fixture()
def built(tmp_path: Path):
    """A real FAISS + BM25 index over the synthetic corpus, plus its store."""
    store = TraceStore(tmp_path / "corpus.sqlite")
    for movie in _corpus():
        store.write_movie(movie)
    result = build_index(
        store,
        StubEmbedder(dim=8),
        dataset_version="ds1",
        out_root=tmp_path / "index",
        cache=None,
        flip_pointer=False,
    )
    try:
        yield load_index(result.out_dir), store
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Invariant 1: a failed query embedding degrades to sparse-only, never raises.
# ---------------------------------------------------------------------------


@requires_faiss
def test_dense_failure_degrades_to_sparse_only_instead_of_raising(built) -> None:
    """The embedder dies at query time; the turn must still answer from BM25."""
    loaded, store = built
    embedder = FlakyEmbedder(StubEmbedder(dim=8), fail_after=0)
    retriever = HybridRetriever.from_store(loaded, embedder, store)

    results = retriever.retrieve("haunted manor ghosts")

    # It answered, it recorded the degradation, and every dense score is absent.
    assert results, "sparse-only retrieval returned nothing"
    assert retriever.dense_failures == 1
    assert all(r.dense_score == 0.0 for r in results)
    # BM25 alone still finds the right film.
    assert "Haunted Manor" in {r.title for r in results}


@requires_faiss
def test_degraded_turn_still_honours_exclusion_filters(built) -> None:
    """Degradation must not quietly drop the deterministic filters with it."""
    from imdb_chatbot.schemas import ParsedQuery

    loaded, store = built
    retriever = HybridRetriever.from_store(
        loaded, FlakyEmbedder(StubEmbedder(dim=8), fail_after=0), store
    )

    results = retriever.retrieve(
        "haunted manor ghosts", ParsedQuery(exclude_genres=["Horror"])
    )

    assert retriever.dense_failures == 1
    assert "Haunted Manor" not in {r.title for r in results}


@requires_faiss
def test_outage_mid_session_does_not_poison_later_turns(built) -> None:
    """A recovered embedder must go straight back to full hybrid retrieval."""
    loaded, store = built
    embedder = FlakyEmbedder(StubEmbedder(dim=8), fail_after=0)
    retriever = HybridRetriever.from_store(loaded, embedder, store)

    retriever.retrieve("haunted manor ghosts")
    assert retriever.dense_failures == 1

    # Network comes back.
    embedder.fail_after = 999
    healthy = retriever.retrieve("detective in a rainy city")

    assert retriever.dense_failures == 1  # no new failure
    assert any(r.dense_score != 0.0 for r in healthy), "dense side did not recover"


# ---------------------------------------------------------------------------
# Invariant 2: the cache key distinguishes any two different embedders.
# ---------------------------------------------------------------------------


def test_same_model_different_dimensions_are_different_cache_namespaces() -> None:
    """Two profiles differing ONLY in `dimensions` must not share cache keys.

    If ``name`` ignored the width, a warm 512-d cache would be replayed into a
    1024-d index and every vector would be garbage (or a broadcast error).
    """
    small = OpenRouterEmbedder(
        "openai/text-embedding-3-large", dimensions=512, api_key="not-used", client=object()
    )
    large = OpenRouterEmbedder(
        "openai/text-embedding-3-large", dimensions=1024, api_key="not-used", client=object()
    )

    assert small.name != large.name
    assert cache_key(small.name, "Inception") != cache_key(large.name, "Inception")


def test_a_warm_cache_from_a_narrower_model_is_not_reused(tmp_path: Path) -> None:
    """End-to-end version of the same attack, through the real cache.

    Fill the cache at dim 4, then embed the same texts at dim 16. If the two
    embedders shared a namespace the second pass would read 4-float vectors into
    16-float slots; here it must simply miss and re-embed at the right width.
    """
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        texts = ["Inception (2010).", "Parasite (2019)."]
        narrow = embed_cached(StubEmbedder(dim=4), texts, cache)
        wide = embed_cached(StubEmbedder(dim=16), texts, cache)
        assert narrow.shape == (2, 4)
        assert wide.shape == (2, 16)
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# Config-driven selection (offline: no key, no network).
# ---------------------------------------------------------------------------


def test_unknown_profile_falls_back_to_the_local_model() -> None:
    """An unusable profile name degrades, it does not take the app down."""
    cfg = {
        "embedder": {
            "default": "local",
            "fallback": "local",
            "profiles": {
                "local": {"kind": "stub", "dimensions": 8},
                "hosted": {"kind": "openrouter", "model": "openai/text-embedding-3-large"},
            },
        }
    }
    assert build_embedder("no-such-profile", cfg).name == "stub8"


def test_hosted_profile_without_a_key_falls_back_to_the_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No OPENROUTER_API_KEY must mean 'local embedder', not a crash."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = {
        "embedder": {
            "default": "hosted",
            "fallback": "local",
            "profiles": {
                "local": {"kind": "stub", "dimensions": 8},
                "hosted": {
                    "kind": "openrouter",
                    "model": "openai/text-embedding-3-large",
                    "dimensions": 1024,
                },
            },
        }
    }
    assert build_embedder(None, cfg).name == "stub8"


def test_asking_for_more_dimensions_than_the_model_has_is_rejected() -> None:
    """4096 from a 3072-wide model would build an index of padded garbage."""
    with pytest.raises(ValueError, match="at most 3072"):
        OpenRouterEmbedder(
            "openai/text-embedding-3-large", dimensions=4096, api_key="x", client=object()
        )


def test_unknown_model_without_explicit_dimensions_is_rejected() -> None:
    """``dim`` is needed BEFORE any call (FAISS is sized from it), so guess-free."""
    with pytest.raises(RuntimeError, match="dimensions"):
        OpenRouterEmbedder("some/new-embedding-model", api_key="x", client=object())


def test_repo_config_keeps_a_local_default_and_fallback() -> None:
    """The shipped config must not make the app depend on a network embedder."""
    from imdb_chatbot.config import load_models_config

    section = load_models_config()["embedder"]
    assert section["profiles"][section["fallback"]]["kind"] == "sentence_transformer"
    assert section["profiles"][section["default"]]["kind"] == "sentence_transformer"


# ---------------------------------------------------------------------------
# Live round-trip. Opt-in: it spends money.
# ---------------------------------------------------------------------------


@LIVE
def test_live_openrouter_honours_the_requested_dimensions() -> None:
    """The whole config surface rests on OpenRouter passing `dimensions` through.

    Nothing offline can establish that; this is the only test that proves the
    hosted profile produces vectors of the width the index was sized for.
    """
    embedder = OpenRouterEmbedder("openai/text-embedding-3-large", dimensions=256)
    try:
        vectors = embedder.encode(["a gritty korean revenge thriller", "a pixar cartoon"])
    finally:
        embedder.close()

    assert vectors.shape == (2, 256)
    assert vectors.dtype == np.float32
    assert embedder.input_tokens > 0
    # Distinct texts must not collapse to the same vector.
    normed = l2_normalize(vectors)
    assert float(normed[0] @ normed[1]) < 0.99
