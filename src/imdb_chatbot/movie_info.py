"""Answer factual questions about a specific film from the corpus (ticket #58).

When a turn is classified ``MOVIE_QUESTION`` ("who is the cast of X", "what is X
about", "have you heard of X"), we extract the title, look it up in the corpus,
and answer from the stored fields (year, genres, director, cast, plot) - grounded
in our own data rather than an LLM's memory. An unknown title gets an honest
"couldn't find it" plus an offer to recommend, so Maya stays a recommender that
can also answer basics about films she knows.

The title extractor (an LLM) and the lookup are both injectable, so the answer
logic is fully unit-testable without a network or an index.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .text import normalize_text as normalize_title

# question -> the movie title mentioned in it (an LLM at runtime, a fake in tests).
TitleExtractor = Callable[[str], str]
# a normalized title -> the best-matching MovieRecord (or None).
TitleLookup = Callable[[str], Any]


def build_title_lookup(movies: Iterable[Any]) -> TitleLookup:
    """Build a tolerant title -> MovieRecord lookup over the corpus.

    Exact normalized match first; otherwise a substring match (either direction,
    minimum 4 chars to avoid noise), breaking ties by vote_count so the canonical
    film wins. Returns ``None`` when nothing plausible matches.
    """
    index: dict[str, list[Any]] = {}
    for movie in movies:
        index.setdefault(normalize_title(movie.title), []).append(movie)

    def _vote(movie: Any) -> int:
        return int(getattr(movie, "vote_count", 0) or 0)

    def lookup(title: str) -> Any:
        key = normalize_title(title)
        if not key:
            return None
        if key in index:
            return max(index[key], key=_vote)
        best: Any = None
        best_vote = -1
        for corpus_key, movies_for_key in index.items():
            if (len(key) >= 4 and key in corpus_key) or (
                len(corpus_key) >= 4 and corpus_key in key
            ):
                for movie in movies_for_key:
                    if _vote(movie) > best_vote:
                        best, best_vote = movie, _vote(movie)
        return best

    return lookup


def format_movie_facts(movie: Any) -> str:
    """A one-paragraph fact summary from the stored MovieRecord fields."""
    lead = f"**{movie.title}** ({movie.year})"
    genres = ", ".join(movie.genres[:3]) if movie.genres else ""
    if genres:
        lead += f" is a {genres} film"
    if movie.director:
        lead += f" directed by {movie.director}"
    parts = [lead + "."]
    if movie.cast:
        parts.append("Starring " + ", ".join(movie.cast[:4]) + ".")
    if movie.plot:
        plot = movie.plot.strip()
        parts.append(plot if len(plot) <= 300 else plot[:297].rstrip() + "...")
    parts.append("Want me to recommend something similar?")
    return " ".join(parts)


def answer_movie_question(
    query: str,
    lookup: TitleLookup,
    extractor: TitleExtractor,
) -> str:
    """Answer a factual movie question from the corpus, or redirect if unknown."""
    title = (extractor(query) or "").strip()
    movie = lookup(title) if title else None
    if movie is None:
        name = f'"{title}"' if title else "that film"
        return (
            f"I couldn't find {name} in my catalog, so I can't pull up its details. "
            "Want me to recommend something instead?"
        )
    return format_movie_facts(movie)
