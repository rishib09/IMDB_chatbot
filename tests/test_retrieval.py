"""Tests for hybrid retrieval (ticket #16).

Everything runs on a small SYNTHETIC corpus and the dependency-free
``StubEmbedder``: a tiny FAISS + BM25 index is built with #15's ``build_index``,
then a ``HybridRetriever`` runs the dense + sparse + RRF + filter pipeline over it.

The load-bearing property is exclusion precision = 1.00: for any query carrying
``exclude_actors`` / ``exclude_genres``, NO returned candidate contains the
excluded actor or genre. Those tests are deliberately thorough and also assert
the filter is non-vacuous (the excluded thing appears when the filter is off).

Tests that need ``faiss`` are skipped when it is not installed; they run on CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from imdb_chatbot.index.build import build_index, load_index
from imdb_chatbot.index.embedder import StubEmbedder
from imdb_chatbot.retrieval.retrieve import (
    HybridRetriever,
    _effective_rating,
    _normalize_name,
    rrf_fuse,
)
from imdb_chatbot.schemas import MovieRecord, ParsedQuery, ScoredMovie
from imdb_chatbot.store import TraceStore

HAS_FAISS = importlib.util.find_spec("faiss") is not None
requires_faiss = pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")


# -- synthetic corpus ---------------------------------------------------------

# A shared actor and a shared genre appear across several films so that the
# exclusion tests are meaningful (the excluded thing really shows up otherwise).
REPEAT_STAR = "Repeat Star"

# vote_count per tmdb_id, applied only when a test asks for a "voted" corpus.
# Chosen so the top-5 order (8, 2, 12, 7, 4) differs from any RRF order, making
# the vote_count re-rank (#52) observable.
_VOTES = {1: 500, 2: 900, 3: 100, 4: 700, 5: 300, 6: 50, 7: 800, 8: 950, 9: 200, 10: 600, 11: 400, 12: 850}


def _corpus(with_votes: bool = False) -> list[MovieRecord]:
    movies = [
        MovieRecord(
            tmdb_id=1,
            title="Midnight Detective",
            year=2001,
            genres=["Thriller", "Mystery"],
            director="Dir A",
            cast=[REPEAT_STAR, "Alice Kim"],
            plot="A detective hunts a killer through a rainy city at midnight.",
            regions=["US"],
            rating_raw=7.5,
        ),
        MovieRecord(
            tmdb_id=2,
            title="Seoul Mystery",
            year=2019,
            genres=["Mystery", "Drama"],
            director="Dir B",
            cast=["Song Lee", "Alice Kim"],
            plot="A quiet detective story about a missing sister in Seoul.",
            regions=["KR"],
            rating_raw=8.2,
        ),
        MovieRecord(
            tmdb_id=3,
            title="Haunted Manor",
            year=2015,
            genres=["Horror", "Thriller"],
            director="Dir C",
            cast=[REPEAT_STAR, "Bob Stone"],
            plot="A family is terrorized by ghosts in a haunted manor.",
            regions=["US"],
            rating_raw=6.1,
        ),
        MovieRecord(
            tmdb_id=4,
            title="Delhi Nights",
            year=2012,
            genres=["Drama", "Romance"],
            director="Dir D",
            cast=["Ravi Menon", "Priya Nair"],
            plot="A romance blooms across the night markets of Delhi.",
            regions=["IN"],
            rating_raw=7.0,
        ),
        MovieRecord(
            tmdb_id=5,
            title="Cold Case Files",
            year=2008,
            genres=["Thriller", "Crime"],
            director="Dir E",
            cast=[REPEAT_STAR, "Nina West"],
            plot="Detectives reopen a cold murder case from decades past.",
            regions=["US", "IN"],
            rating_raw=6.8,
        ),
        MovieRecord(
            tmdb_id=6,
            title="Mumbai Horror",
            year=2020,
            genres=["Horror"],
            director="Dir F",
            cast=["Ravi Menon", "Bob Stone"],
            plot="A cursed film reel unleashes terror in a Mumbai cinema.",
            regions=["IN"],
            rating_raw=5.4,
        ),
        MovieRecord(
            tmdb_id=7,
            title="The Silent Witness",
            year=1998,
            genres=["Mystery", "Thriller"],
            director="Dir G",
            cast=["Alice Kim", "Nina West"],
            plot="A mute witness holds the key to a baffling murder mystery.",
            regions=["US"],
            rating_raw=7.9,
        ),
        MovieRecord(
            tmdb_id=8,
            title="Busan Ghost",
            year=2016,
            genres=["Horror", "Drama"],
            director="Dir H",
            cast=["Song Lee", REPEAT_STAR],
            plot="A grieving mother is haunted by a ghost on a train to Busan.",
            regions=["KR"],
            rating_raw=8.0,
        ),
        MovieRecord(
            tmdb_id=9,
            title="Sunny Comedy",
            year=2011,
            genres=["Comedy"],
            director="Dir I",
            cast=["Priya Nair", "Bob Stone"],
            plot="A sunny feel-good comedy about a dysfunctional wedding.",
            regions=["IN"],
            rating_raw=6.5,
        ),
        MovieRecord(
            tmdb_id=10,
            title="Detective Dawn",
            year=2022,
            genres=["Crime", "Mystery"],
            director="Dir J",
            cast=["Nina West", "Song Lee"],
            plot="At dawn a rookie detective cracks a citywide smuggling ring.",
            regions=["US"],
            rating_raw=7.2,
        ),
        MovieRecord(
            tmdb_id=11,
            title="Quiet Drama",
            year=2005,
            genres=["Drama"],
            director="Dir K",
            cast=["Alice Kim", "Ravi Menon"],
            plot="A slow, quiet drama about two estranged brothers reuniting.",
            regions=["IN"],
            rating_raw=7.6,
        ),
        MovieRecord(
            tmdb_id=12,
            title="Night Terror",
            year=2018,
            genres=["Horror", "Mystery"],
            director="Dir L",
            cast=[REPEAT_STAR, "Priya Nair"],
            plot="A detective investigates murders that only happen at night.",
            regions=["US", "KR"],
            rating_raw=6.9,
        ),
    ]
    if with_votes:
        for movie in movies:
            movie.vote_count = _VOTES[movie.tmdb_id]
    return movies


def _voted_retriever(tmp_path: Path, *, rerank_pool: int = 20) -> HybridRetriever:
    """A retriever over the corpus WITH vote_counts, to exercise the #52 re-rank."""
    store = TraceStore(tmp_path / "voted.sqlite")
    for movie in _corpus(with_votes=True):
        store.write_movie(movie)
    embedder = StubEmbedder(dim=16)
    result = build_index(
        store, embedder, dataset_version="t52", out_root=tmp_path / "vindex",
        cache=None, flip_pointer=False,
    )
    loaded = load_index(result.out_dir)
    r = HybridRetriever.from_store(loaded, embedder, store, rerank_pool=rerank_pool)
    store.close()
    return r


@pytest.fixture()
def retriever(tmp_path: Path):
    """A HybridRetriever over a freshly built stub index of the synthetic corpus."""
    store = TraceStore(tmp_path / "corpus.sqlite")
    try:
        for movie in _corpus():
            store.write_movie(movie)
        embedder = StubEmbedder(dim=16)
        result = build_index(
            store,
            embedder,
            dataset_version="t16",
            out_root=tmp_path / "index",
            cache=None,
            flip_pointer=False,
        )
        loaded = load_index(result.out_dir)
        yield HybridRetriever.from_store(loaded, embedder, store)
    finally:
        store.close()


def _new_retriever(tmp_path: Path) -> HybridRetriever:
    """Build a second, independent retriever (fresh cache) over the same corpus."""
    store = TraceStore(tmp_path / "corpus2.sqlite")
    for movie in _corpus():
        store.write_movie(movie)
    embedder = StubEmbedder(dim=16)
    result = build_index(
        store,
        embedder,
        dataset_version="t16",
        out_root=tmp_path / "index2",
        cache=None,
        flip_pointer=False,
    )
    loaded = load_index(result.out_dir)
    r = HybridRetriever.from_store(loaded, embedder, store)
    store.close()
    return r


QUERY = "detective murder mystery thriller at night"


# -- RRF math (no faiss needed) ----------------------------------------------


def test_rrf_fuse_math() -> None:
    # doc 1: rank 1 in first ranker, rank 2 in second; doc 2 rank 2 in first
    # only; doc 3 rank 1 in second only.
    fused = rrf_fuse([[1, 2], [3, 1]], k=60)
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2] == pytest.approx(1 / 62)
    assert fused[3] == pytest.approx(1 / 61)
    # A doc that tops both rankers must outrank docs that appear in only one.
    assert fused[1] > fused[2]
    assert fused[1] > fused[3]


def test_rrf_fuse_empty() -> None:
    assert rrf_fuse([[], []]) == {}


def test_effective_rating_prefers_region_zscore() -> None:
    movie = MovieRecord(
        tmdb_id=99,
        title="Z",
        year=2000,
        rating_raw=1.0,
        rating_z={"IN": 2.5},
    )
    # With a region whose z-score is populated, the z-score wins.
    assert _effective_rating(movie, "IN") == 2.5
    # No region context -> fall back to rating_raw.
    assert _effective_rating(movie, None) == 1.0
    # Region present but not in rating_z -> fall back to rating_raw.
    assert _effective_rating(movie, "US") == 1.0


def test_effective_rating_empty_z_falls_back() -> None:
    movie = MovieRecord(tmdb_id=99, title="Z", year=2000, rating_raw=6.4, rating_z={})
    assert _effective_rating(movie, "IN") == 6.4


# -- shape / fused ranking ----------------------------------------------------


@requires_faiss
def test_returns_scored_movies_top5(retriever: HybridRetriever) -> None:
    out = retriever.retrieve(QUERY, ParsedQuery())
    assert 0 < len(out) <= 5
    assert all(isinstance(s, ScoredMovie) for s in out)
    # Ranks are dense 1..n in order.
    assert [s.rank for s in out] == list(range(1, len(out) + 1))
    # RRF scores are positive and non-increasing (already sorted best-first).
    assert all(s.rrf_score > 0 for s in out)
    assert out == sorted(out, key=lambda s: (-s.rrf_score, s.tmdb_id))
    # Metadata carried through.
    for s in out:
        assert s.title
        assert s.year > 0


@requires_faiss
def test_dense_and_sparse_both_contribute(retriever: HybridRetriever) -> None:
    out = retriever.retrieve(QUERY, ParsedQuery())
    # At least one candidate should have a non-zero sparse (BM25) score for a
    # query full of corpus terms, proving the sparse side is wired in.
    assert any(s.sparse_score > 0 for s in out)
    # And every candidate has a dense score attached (cosine over the stub index).
    assert all(isinstance(s.dense_score, float) for s in out)


# -- exclusion precision = 1.00 (load-bearing) --------------------------------


@requires_faiss
def test_exclusion_precision_actor(retriever: HybridRetriever) -> None:
    # Non-vacuous: without the exclusion the star shows up among candidates.
    baseline = retriever.retrieve(QUERY, ParsedQuery())
    star_ids = {m.tmdb_id for m in _corpus() if REPEAT_STAR in m.cast}
    assert star_ids & {s.tmdb_id for s in baseline}, "excluded actor must appear first"

    out = retriever.retrieve(QUERY, ParsedQuery(exclude_actors=[REPEAT_STAR]))
    corpus = {m.tmdb_id: m for m in _corpus()}
    for s in out:
        assert REPEAT_STAR not in corpus[s.tmdb_id].cast


@requires_faiss
@pytest.mark.parametrize(
    "query",
    [
        "detective murder mystery thriller at night",
        "horror ghost haunted terror",
        "quiet drama romance",
        "cold case crime detective",
    ],
)
def test_exclusion_precision_actor_many_queries(
    retriever: HybridRetriever, query: str
) -> None:
    out = retriever.retrieve(query, ParsedQuery(exclude_actors=[REPEAT_STAR]))
    corpus = {m.tmdb_id: m for m in _corpus()}
    for s in out:
        assert REPEAT_STAR not in corpus[s.tmdb_id].cast


@requires_faiss
def test_exclusion_precision_multiple_actors(retriever: HybridRetriever) -> None:
    excluded = ["Repeat Star", "Alice Kim", "Nina West"]
    out = retriever.retrieve(QUERY, ParsedQuery(exclude_actors=excluded))
    corpus = {m.tmdb_id: m for m in _corpus()}
    for s in out:
        assert not (set(corpus[s.tmdb_id].cast) & set(excluded))


@requires_faiss
def test_exclusion_precision_genre(retriever: HybridRetriever) -> None:
    baseline = retriever.retrieve("horror ghost haunted", ParsedQuery())
    horror_ids = {m.tmdb_id for m in _corpus() if "Horror" in m.genres}
    assert horror_ids & {s.tmdb_id for s in baseline}, "excluded genre must appear first"

    out = retriever.retrieve("horror ghost haunted", ParsedQuery(exclude_genres=["Horror"]))
    corpus = {m.tmdb_id: m for m in _corpus()}
    for s in out:
        assert "Horror" not in corpus[s.tmdb_id].genres


@requires_faiss
def test_exclusion_precision_actor_and_genre(retriever: HybridRetriever) -> None:
    out = retriever.retrieve(
        QUERY, ParsedQuery(exclude_actors=[REPEAT_STAR], exclude_genres=["Horror"])
    )
    corpus = {m.tmdb_id: m for m in _corpus()}
    for s in out:
        assert REPEAT_STAR not in corpus[s.tmdb_id].cast
        assert "Horror" not in corpus[s.tmdb_id].genres


# -- metadata filters ---------------------------------------------------------


@requires_faiss
def test_region_filter(retriever: HybridRetriever) -> None:
    out = retriever.retrieve(QUERY, ParsedQuery(region="IN"))
    corpus = {m.tmdb_id: m for m in _corpus()}
    assert out, "expected some IN-region matches"
    for s in out:
        assert "IN" in corpus[s.tmdb_id].regions


@requires_faiss
def test_min_and_max_year(retriever: HybridRetriever) -> None:
    out = retriever.retrieve(QUERY, ParsedQuery(min_year=2010, max_year=2019))
    corpus = {m.tmdb_id: m for m in _corpus()}
    assert out
    for s in out:
        assert 2010 <= corpus[s.tmdb_id].year <= 2019


@requires_faiss
def test_min_rating_uses_rating_raw(retriever: HybridRetriever) -> None:
    out = retriever.retrieve(QUERY, ParsedQuery(min_rating=7.5))
    corpus = {m.tmdb_id: m for m in _corpus()}
    assert out
    for s in out:
        assert corpus[s.tmdb_id].rating_raw is not None
        assert corpus[s.tmdb_id].rating_raw >= 7.5


@requires_faiss
def test_shown_movies_exclusion(retriever: HybridRetriever) -> None:
    first = retriever.retrieve(QUERY, ParsedQuery())
    shown = {s.tmdb_id for s in first}
    out = retriever.retrieve(QUERY, ParsedQuery(), shown_movies=shown)
    returned = {s.tmdb_id for s in out}
    assert not (returned & shown), "shown movies must not repeat"


# -- vote_count re-rank + top-5 cap (ticket #52) -----------------------------


@requires_faiss
def test_reranks_relevant_pool_by_vote_count(tmp_path: Path) -> None:
    # No filter -> all 12 are the eligible pool; top 5 by vote_count (ties by
    # relevance): 8 (950), 2 (900), 12 (850), 7 (800), 4 (700).
    out = _voted_retriever(tmp_path).retrieve(QUERY, ParsedQuery())
    assert [s.tmdb_id for s in out] == [8, 2, 12, 7, 4]


@requires_faiss
def test_caps_at_five(tmp_path: Path) -> None:
    out = _voted_retriever(tmp_path).retrieve(QUERY, ParsedQuery())
    assert len(out) == 5


@requires_faiss
def test_fewer_than_five_returns_all(tmp_path: Path) -> None:
    # Only 3 KR films exist -> all 3 returned, vote_count-ordered, not padded.
    out = _voted_retriever(tmp_path).retrieve(QUERY, ParsedQuery(region="KR"))
    assert [s.tmdb_id for s in out] == [8, 2, 12]


@requires_faiss
def test_rerank_pool_gates_eligibility_before_vote_count(tmp_path: Path) -> None:
    # A tiny relevance pool caps eligibility BEFORE vote_count is consulted, so
    # only 2 come back even though all 12 pass the filters - relevance gates first.
    out = _voted_retriever(tmp_path, rerank_pool=2).retrieve(QUERY, ParsedQuery())
    assert len(out) == 2


@requires_faiss
def test_graph_excludes_shown_before_the_cap(tmp_path: Path) -> None:
    """Attacks: the graph's retrieve node drops shown titles only AFTER the
    retriever's top-5 cap (ticket #88). Adversary pool: the top-5 by vote_count
    are all already shown, so a post-cap filter would leave zero candidates and
    strand the refine in fallback. Exclusion inside the retriever refills the
    five slots with unseen films."""
    from imdb_chatbot.graph.build import _make_retrieve
    from imdb_chatbot.graph.tracing import TraceCollector
    from imdb_chatbot.schemas import TurnState

    hybrid = _voted_retriever(tmp_path)
    shown = [s.tmdb_id for s in hybrid.retrieve(QUERY, ParsedQuery())]
    assert shown == [8, 2, 12, 7, 4]  # the whole uncapped top-5 is already shown

    node = _make_retrieve(hybrid.retrieve, TraceCollector())
    state = TurnState(
        trace_id="t", session_id="s", raw_query=QUERY, parsed=ParsedQuery(), shown_movies=shown
    )
    got = [m.tmdb_id for m in node(state)["retrieved"]]

    assert len(got) == 5, got
    assert not set(got) & set(shown), got
    assert got == [10, 1, 11, 5, 9]  # next five by vote_count, unseen


# -- person-aware retrieval (ticket #50) -------------------------------------


def test_normalize_name_folds_punctuation() -> None:
    # Hyphen/space/period variants normalize the same, so "Bong Joon-ho" matches
    # a corpus that stores "Bong Joon Ho".
    assert _normalize_name("Bong Joon-ho") == _normalize_name("Bong Joon Ho") == "bong joon ho"
    assert _normalize_name("J.J. Abrams") == "j j abrams"


@requires_faiss
def test_director_returns_only_that_directors_films(retriever: HybridRetriever) -> None:
    # "Dir B" directs only Seoul Mystery (tmdb 2).
    out = retriever.retrieve(QUERY, ParsedQuery(director="Dir B"))
    assert [s.tmdb_id for s in out] == [2]


@requires_faiss
def test_director_match_is_case_insensitive_substring(retriever: HybridRetriever) -> None:
    # "b" is contained in "dir b" (normalized) -> matches that director only.
    out = retriever.retrieve(QUERY, ParsedQuery(director="b"))
    assert [s.tmdb_id for s in out] == [2]


@requires_faiss
def test_actor_returns_only_films_with_that_actor(retriever: HybridRetriever) -> None:
    # Song Lee is in the cast of tmdb 2, 8, 10.
    out = retriever.retrieve(QUERY, ParsedQuery(actor="Song Lee"))
    assert {s.tmdb_id for s in out} == {2, 8, 10}


@requires_faiss
def test_unknown_person_returns_empty(retriever: HybridRetriever) -> None:
    assert retriever.retrieve(QUERY, ParsedQuery(director="Christopher Nolan")) == []
    assert retriever.retrieve(QUERY, ParsedQuery(actor="Tom Hanks")) == []


@requires_faiss
def test_person_films_surface_even_when_query_is_unrelated(
    retriever: HybridRetriever,
) -> None:
    # A query unrelated to the film still returns the director's movie, because
    # the person index injects it into the candidate pool.
    out = retriever.retrieve("sunny feel-good comedy wedding", ParsedQuery(director="Dir B"))
    assert [s.tmdb_id for s in out] == [2]


# -- cache --------------------------------------------------------------------


@requires_faiss
def test_cache_serves_repeat_call(retriever: HybridRetriever) -> None:
    parsed = ParsedQuery(exclude_actors=[REPEAT_STAR])
    first = retriever.retrieve(QUERY, parsed)
    assert retriever.search_calls == 1
    second = retriever.retrieve(QUERY, parsed)
    # Second identical call served from cache: no new pipeline run.
    assert retriever.search_calls == 1
    assert [s.tmdb_id for s in first] == [s.tmdb_id for s in second]


@requires_faiss
def test_cache_key_distinguishes_filters(retriever: HybridRetriever) -> None:
    retriever.retrieve(QUERY, ParsedQuery())
    assert retriever.search_calls == 1
    # A different filter set is a genuine miss -> pipeline runs again.
    retriever.retrieve(QUERY, ParsedQuery(region="IN"))
    assert retriever.search_calls == 2


# -- determinism --------------------------------------------------------------


@requires_faiss
def test_determinism_same_inputs(retriever: HybridRetriever, tmp_path: Path) -> None:
    parsed = ParsedQuery(exclude_actors=[REPEAT_STAR])
    a = retriever.retrieve(QUERY, parsed)
    # A second, independent retriever (fresh cache, freshly built index) must
    # produce byte-identical ordering and scores.
    other = _new_retriever(tmp_path)
    b = other.retrieve(QUERY, parsed)
    assert [s.tmdb_id for s in a] == [s.tmdb_id for s in b]
    assert [s.rrf_score for s in a] == [s.rrf_score for s in b]
    assert [s.rank for s in a] == [s.rank for s in b]
