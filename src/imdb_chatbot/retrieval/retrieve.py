"""Hybrid retrieval: dense + sparse -> RRF fusion -> deterministic filters -> cache.

Ticket #16. Given a rewritten query plus a ``ParsedQuery`` of filters, return the
top ``final_k`` candidate movies through the full hybrid path:

1. Dense: embed the query, L2-normalize it (reuse #15's ``embed_query``), FAISS
   cosine search -> top ``dense_k``.
2. Sparse: BM25 scores over the tokenized query -> top ``sparse_k``.
3. RRF fusion (hand-rolled, no dependency):
   ``score(d) = sum over rankers of 1 / (rrf_k + rank_r(d))`` with ``rrf_k=60``.
4. Deterministic metadata filters (pure Python, cheap -> expensive), driven by a
   ``ParsedQuery`` plus a ``shown_movies`` exclusion set.
5. Emit the top ``final_k`` survivors as ``ScoredMovie`` records.

Determinism: for a fixed index + query + filters the output is identical across
calls, with stable tie-breaking by ``tmdb_id``. In particular the exclusion
filters give an exclusion precision of 1.00 - a returned candidate NEVER contains
an excluded actor or genre.

An in-process LRU cache keyed by ``(hash(rewritten_query), index_version,
hash(frozen_filters))`` short-circuits repeated identical calls. There is no TTL:
the index version is part of the key, so a new index invalidates the cache
naturally.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from ..index.build import LoadedIndex, dense_search, tokenize
from ..index.cache import EmbeddingCache
from ..index.embedder import Embedder
from ..schemas import MovieRecord, ParsedQuery, ScoredMovie
from ..store import TraceStore

DENSE_K = 20
SPARSE_K = 20
RRF_K = 60
FINAL_K = 5
CACHE_SIZE = 128


def rrf_fuse(rankings: Iterable[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion over one or more ranked id lists.

    Each ranking is an ordered list of ``tmdb_id`` best-first (rank 1 = best). A
    document's fused score is the sum, over the rankers it appears in, of
    ``1 / (k + rank)``. Documents missing from a ranker simply contribute nothing
    for that ranker. Hand-rolled on purpose - no external dependency.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def build_movie_map(store: TraceStore) -> dict[int, MovieRecord]:
    """Load a ``tmdb_id -> MovieRecord`` map from the corpus store (read-only)."""
    return {m.tmdb_id: m for m in store.read_all_movies()}


def _freeze_filters(parsed: ParsedQuery, shown_movies: frozenset[int]) -> tuple:
    """A hashable, order-stable snapshot of everything that changes the result.

    Lists are sorted into tuples so that filter sets that differ only in ordering
    map to the same cache entry.
    """
    return (
        tuple(sorted(parsed.genres)),
        parsed.similar_to,
        tuple(sorted(parsed.exclude_actors)),
        tuple(sorted(parsed.exclude_genres)),
        parsed.min_year,
        parsed.max_year,
        parsed.min_rating,
        parsed.region,
        tuple(sorted(shown_movies)),
    )


def _effective_rating(movie: MovieRecord, region: str | None) -> float | None:
    """Pick the rating value the ``min_rating`` filter compares against.

    Prefer the region z-score when the query carries a region and ``rating_z`` is
    populated for it; otherwise fall back to ``rating_raw``. ``rating_z`` is often
    an empty ``{}`` today, so the fall-back is the common path.
    """
    if region is not None and movie.rating_z:
        z = movie.rating_z.get(region)
        if z is not None:
            return z
    return movie.rating_raw


def _passes_filters(
    movie: MovieRecord,
    parsed: ParsedQuery,
    shown_movies: frozenset[int],
) -> bool:
    """Apply the deterministic metadata filters, cheap checks first.

    The actor/genre exclusions are the load-bearing ones: if any excluded actor
    appears in ``cast`` or any excluded genre appears in ``genres`` the movie is
    dropped, guaranteeing exclusion precision = 1.00.
    """
    # Cheapest: set-membership / identity checks.
    if movie.tmdb_id in shown_movies:
        return False

    if parsed.region is not None and parsed.region not in movie.regions:
        return False

    # Exclusions (load-bearing). Any overlap -> drop.
    if parsed.exclude_genres:
        genres = set(movie.genres)
        if any(g in genres for g in parsed.exclude_genres):
            return False

    if parsed.exclude_actors:
        cast = set(movie.cast)
        if any(a in cast for a in parsed.exclude_actors):
            return False

    # Year window.
    if parsed.min_year is not None and movie.year < parsed.min_year:
        return False
    if parsed.max_year is not None and movie.year > parsed.max_year:
        return False

    # Rating threshold (most involved: consults rating_z then rating_raw).
    if parsed.min_rating is not None:
        rating = _effective_rating(movie, parsed.region)
        if rating is None or rating < parsed.min_rating:
            return False

    return True


class HybridRetriever:
    """Runs the hybrid dense+sparse+RRF+filter pipeline with an LRU result cache.

    Construct once per loaded index; the ``tmdb_id -> MovieRecord`` map is read
    from the corpus store at init time and reused for every filter pass. The
    ``search_calls`` counter increments once per genuine (cache-miss) pipeline run
    so tests can prove the cache serves repeats.
    """

    def __init__(
        self,
        loaded: LoadedIndex,
        embedder: Embedder,
        movies: dict[int, MovieRecord] | Iterable[MovieRecord],
        *,
        dense_k: int = DENSE_K,
        sparse_k: int = SPARSE_K,
        rrf_k: int = RRF_K,
        final_k: int = FINAL_K,
        cache_size: int = CACHE_SIZE,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        self.loaded = loaded
        self.embedder = embedder
        if isinstance(movies, dict):
            self.movies_by_id = dict(movies)
        else:
            self.movies_by_id = {m.tmdb_id: m for m in movies}
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rrf_k = rrf_k
        self.final_k = final_k
        self.cache_size = cache_size
        self.embedding_cache = embedding_cache

        self.version = loaded.version
        self._cache: OrderedDict[tuple, list[ScoredMovie]] = OrderedDict()
        # Number of genuine pipeline runs (cache misses). Test hook.
        self.search_calls = 0

    @classmethod
    def from_store(
        cls,
        loaded: LoadedIndex,
        embedder: Embedder,
        store: TraceStore,
        **kwargs: object,
    ) -> HybridRetriever:
        """Build a retriever, sourcing the metadata map from a corpus store."""
        return cls(loaded, embedder, build_movie_map(store), **kwargs)

    # -- ranking sides --------------------------------------------------------

    def _dense_ranking(self, query: str) -> tuple[list[int], dict[int, float]]:
        """FAISS cosine search -> (ordered tmdb_ids, tmdb_id -> dense score)."""
        hits = dense_search(
            self.loaded, self.embedder, query, k=self.dense_k, cache=self.embedding_cache
        )
        ranking = [tmdb_id for tmdb_id, _ in hits]
        scores = {tmdb_id: score for tmdb_id, score in hits}
        return ranking, scores

    def _sparse_ranking(self, query: str) -> tuple[list[int], dict[int, float]]:
        """BM25 search -> (ordered tmdb_ids, tmdb_id -> sparse score).

        Ties are broken by ``tmdb_id`` so the ranking is fully deterministic.
        """
        bm25 = self.loaded.bm25
        if bm25 is None:
            return [], {}
        row_to_id = self.loaded.row_to_tmdb_id
        raw = bm25.get_scores(tokenize(query))
        scored = [
            (row_to_id[row], float(score))
            for row, score in enumerate(raw)
            if row < len(row_to_id)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        top = scored[: self.sparse_k]
        ranking = [tmdb_id for tmdb_id, _ in top]
        scores = {tmdb_id: score for tmdb_id, score in top}
        return ranking, scores

    # -- public entry point ---------------------------------------------------

    def retrieve(
        self,
        rewritten_query: str,
        parsed: ParsedQuery | None = None,
        shown_movies: Iterable[int] | None = None,
    ) -> list[ScoredMovie]:
        """Return up to ``final_k`` ``ScoredMovie`` candidates for the query.

        Deterministic for a fixed index + query + filters. Repeated identical
        calls are served from the in-process LRU cache without re-running search.
        """
        parsed = parsed or ParsedQuery()
        shown = frozenset(shown_movies or ())

        key = (
            hash(rewritten_query),
            self.version,
            hash(_freeze_filters(parsed, shown)),
        )
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            # Return copies so callers cannot mutate cached state.
            return [c.model_copy() for c in cached]

        result = self._run(rewritten_query, parsed, shown)

        self._cache[key] = [c.model_copy() for c in result]
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return result

    def _run(
        self,
        rewritten_query: str,
        parsed: ParsedQuery,
        shown: frozenset[int],
    ) -> list[ScoredMovie]:
        """The genuine (cache-miss) pipeline. Increments ``search_calls``."""
        self.search_calls += 1

        dense_ranking, dense_scores = self._dense_ranking(rewritten_query)
        sparse_ranking, sparse_scores = self._sparse_ranking(rewritten_query)

        fused = rrf_fuse([dense_ranking, sparse_ranking], k=self.rrf_k)

        survivors: list[tuple[int, float]] = []
        for tmdb_id, rrf_score in fused.items():
            movie = self.movies_by_id.get(tmdb_id)
            if movie is None:
                continue
            if not _passes_filters(movie, parsed, shown):
                continue
            survivors.append((tmdb_id, rrf_score))

        # Deterministic ordering: highest RRF first, ties broken by tmdb_id.
        survivors.sort(key=lambda pair: (-pair[1], pair[0]))

        out: list[ScoredMovie] = []
        for rank, (tmdb_id, rrf_score) in enumerate(survivors[: self.final_k], start=1):
            movie = self.movies_by_id[tmdb_id]
            out.append(
                ScoredMovie(
                    tmdb_id=tmdb_id,
                    title=movie.title,
                    year=movie.year,
                    regions=list(movie.regions),
                    dense_score=dense_scores.get(tmdb_id, 0.0),
                    sparse_score=sparse_scores.get(tmdb_id, 0.0),
                    rrf_score=rrf_score,
                    rank=rank,
                )
            )
        return out
