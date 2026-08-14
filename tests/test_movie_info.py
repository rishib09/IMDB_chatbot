"""Corpus-grounded movie-question answering (ticket #58)."""

from __future__ import annotations

from imdb_chatbot.movie_info import (
    answer_movie_question,
    build_title_lookup,
    format_movie_facts,
    normalize_title,
)
from imdb_chatbot.schemas import MovieRecord


def _movies() -> list[MovieRecord]:
    return [
        MovieRecord(
            tmdb_id=1,
            title="Parasite",
            year=2019,
            genres=["Drama", "Thriller"],
            director="Bong Joon Ho",
            cast=["Song Kang-ho", "Lee Sun-kyun"],
            plot="A poor family schemes their way into a wealthy household.",
            vote_count=20000,
        ),
        MovieRecord(
            tmdb_id=2,
            title="The Dark Knight",
            year=2008,
            genres=["Action", "Crime"],
            director="Christopher Nolan",
            cast=["Christian Bale", "Heath Ledger"],
            plot="Batman faces the Joker.",
            vote_count=36000,
        ),
    ]


def test_normalize_title_folds_punctuation() -> None:
    assert normalize_title("Spider-Man: No Way Home") == "spider man no way home"


def test_lookup_exact_and_substring() -> None:
    lookup = build_title_lookup(_movies())
    assert lookup("parasite").tmdb_id == 1
    assert lookup("dark knight").tmdb_id == 2  # substring of "The Dark Knight"


def test_lookup_unknown_returns_none() -> None:
    assert build_title_lookup(_movies())("movie obsession") is None


def test_fuzzy_match_tolerates_typos_without_inventing_a_match() -> None:
    """The assumption fuzzy matching puts at risk: an unknown title stays unknown.

    Typo tolerance is only worth having if it cannot turn "a film we do not
    have" into a confident wrong answer - the user would be told facts about the
    wrong movie. So both halves are pinned: a one-character slip resolves, and
    titles that merely share letters with the corpus still return None.
    """
    lookup = build_title_lookup(_movies())
    assert lookup("parasit").tmdb_id == 1  # dropped letter
    assert lookup("The Dark Knght").tmdb_id == 2  # transposed/missing letter
    for unknown in ("movie obsession", "the dark tower", "paradise", "xyzzy"):
        assert lookup(unknown) is None, unknown


def test_format_movie_facts_uses_stored_fields() -> None:
    facts = format_movie_facts(_movies()[0])
    assert "Parasite" in facts
    assert "Bong Joon Ho" in facts
    assert "Song Kang-ho" in facts
    assert "recommend" in facts.lower()


def test_answer_known_movie_from_corpus() -> None:
    lookup = build_title_lookup(_movies())
    text = answer_movie_question("who is the cast of parasite", lookup, lambda q: "Parasite")
    assert "Song Kang-ho" in text


def test_answer_unknown_movie_redirects() -> None:
    lookup = build_title_lookup(_movies())
    text = answer_movie_question(
        "have you heard of movie obsession", lookup, lambda q: "movie obsession"
    )
    assert "couldn't find" in text.lower()
    assert "recommend" in text.lower()
