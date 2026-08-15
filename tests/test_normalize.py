"""Adversary tests for extractor-output normalization (ticket #84).

Invariant under attack: *a constraint the corpus cannot satisfy is never applied
as a filter*. The cheap extractor emits ``region='korean'``, ``genres=['revenge']``
and ``similar_to='gritty korean'``; each of those, applied verbatim, empties the
candidate pool. These tests run the real metadata filter over a small corpus so
a regression shows up as an empty result, not as a mismatched constant.
"""

from __future__ import annotations

from imdb_chatbot.graph import GraphModels, run_turn
from imdb_chatbot.graph.normalize import CorpusVocab, normalize_parsed
from imdb_chatbot.retrieval.retrieve import _passes_filters
from imdb_chatbot.schemas import (
    MovieRecommendation,
    MovieRecord,
    ParsedQuery,
    RecommendationSet,
    ScoredMovie,
    TurnState,
)

CORPUS = [
    MovieRecord(tmdb_id=1, title="Oldboy", year=2003, genres=["Thriller"], regions=["KR"]),
    MovieRecord(tmdb_id=2, title="Parasite", year=2019, genres=["Thriller", "Drama"], regions=["KR"]),
    MovieRecord(tmdb_id=3, title="Heat", year=1995, genres=["Crime"], regions=["US"]),
    # A title that is a SUBSTRING of the extractor's junk phrase 'gritty korean'.
    MovieRecord(tmdb_id=4, title="KORE", year=2023, genres=["Drama"], regions=["KR"]),
]
VOCAB = CorpusVocab.from_movies(CORPUS)

# The exact mis-parse traced live in #84 for "a gritty korean revenge thriller".
BAD_PARSE = ParsedQuery(genres=["thriller", "revenge"], similar_to="gritty korean", region="korean")


def _survivors(parsed: ParsedQuery) -> set[str]:
    return {m.title for m in CORPUS if _passes_filters(m, parsed, frozenset())}


def test_unsatisfiable_constraints_are_dropped_not_applied() -> None:
    """Attacks: every normalized field is a value the corpus can satisfy (or None)."""
    out = normalize_parsed(BAD_PARSE, VOCAB)
    assert out.region == "KR"
    assert out.genres == ["Thriller"]  # 'revenge' is not a genre; casing fixed
    assert out.similar_to is None  # not a title, even though 'KORE' is a substring of it
    assert _survivors(BAD_PARSE) == set()  # the raw parse really does empty the pool
    assert _survivors(out) == {"Oldboy", "Parasite", "KORE"}


def test_unmappable_region_becomes_none_never_a_string() -> None:
    """Attacks: 'unknown region -> None' for a word no demonym table could contain."""
    for junk in ("Klingon", "korean-ish", "", "ZZ"):
        out = normalize_parsed(ParsedQuery(region=junk), VOCAB)
        assert out.region is None, junk
        assert _survivors(out) == {m.title for m in CORPUS}


def test_valid_inputs_pass_through_intact() -> None:
    """Attacks: normalization must not eat constraints the corpus CAN satisfy."""
    out = normalize_parsed(
        ParsedQuery(region="kr", genres=["Drama", "sci-fi"], similar_to="parasite"), VOCAB
    )
    assert out.region == "KR"
    assert out.genres == ["Drama"]  # 'sci-fi' maps to Science Fiction, absent from THIS corpus
    assert out.similar_to == "parasite"


def test_graph_turn_with_demonym_region_reaches_generate() -> None:
    """Attacks: the guard runs inside the extract node, before the region default."""

    def retriever(query: str, parsed: ParsedQuery) -> list[ScoredMovie]:
        return [
            ScoredMovie(tmdb_id=m.tmdb_id, title=m.title, year=m.year, regions=m.regions)
            for m in CORPUS
            if _passes_filters(m, parsed, frozenset())
        ]

    models = GraphModels(
        rewrite=lambda q, h: q,
        extract=lambda q: BAD_PARSE,
        generate=lambda q, c: RecommendationSet(
            picks=[MovieRecommendation(title=c[0].title, year=c[0].year, reason="ok")],
            prose="",
        ),
    )
    state = TurnState(trace_id="t", session_id="s", raw_query="a gritty korean revenge thriller")
    result = run_turn(state, retriever=retriever, models=models, vocab=VOCAB)
    assert "fallback" not in result.state.path_taken, result.state.path_taken
    assert result.state.parsed is not None and result.state.parsed.region == "KR"
    assert {c.title for c in result.state.candidates} == {"Oldboy", "Parasite", "KORE"}
