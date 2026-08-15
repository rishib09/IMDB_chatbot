"""Tests for Gate-4 output validation (ticket #19).

Two layers, both with ZERO network I/O and no live LLM call:

- PURE unit tests exercise ``run_gate4`` / the individual checks directly on
  hand-built ``RecommendationSet`` + candidate objects.
- One INTEGRATION test drives the whole graph with FAKE models to reproduce the
  PRD section 7.5 worked example: a first generation that leaks an excluded actor
  is caught by Gate-4, the regen edge fires, and a clean second generation ends
  the turn.
"""

from __future__ import annotations

from imdb_chatbot.graph import GraphModels, run_gate4, run_turn
from imdb_chatbot.graph.gate4 import (
    check_excluded_actors,
    check_fact_grounding,
    check_titles_exist,
)
from imdb_chatbot.schemas import (
    MovieRecommendation,
    MovieRecord,
    ParsedQuery,
    RecommendationSet,
    ScoredMovie,
    TurnState,
)

# -- builders -----------------------------------------------------------------


def _candidate(tmdb_id: int = 1, title: str = "John Wick", year: int = 2014) -> ScoredMovie:
    return ScoredMovie(tmdb_id=tmdb_id, title=title, year=year, rrf_score=0.5, rank=1)


def _record(
    title: str = "John Wick",
    year: int = 2014,
    director: str = "Chad Stahelski",
    cast: list[str] | None = None,
) -> MovieRecord:
    return MovieRecord(
        tmdb_id=1,
        title=title,
        year=year,
        director=director,
        cast=cast if cast is not None else ["Keanu Reeves", "Michael Nyqvist"],
    )


def _rec(picks: list[MovieRecommendation], prose: str) -> RecommendationSet:
    return RecommendationSet(picks=picks, prose=prose)


def _pick(title: str = "John Wick", year: int = 2014, reason: str = "lean action") -> MovieRecommendation:
    return MovieRecommendation(title=title, year=year, reason=reason)


# -- (a) excluded actors ------------------------------------------------------


def test_excluded_actor_in_prose_fails() -> None:
    response = _rec([_pick()], prose="A slick pick starring Tom Cruise as the lead.")
    result = run_gate4(response, [_candidate()], exclude_actors=["Tom Cruise"])
    assert result.ok is False
    assert result.reason == "excluded_actor:tom cruise"


def test_excluded_actor_in_pick_fails() -> None:
    # The excluded name leaks into a pick's reason rather than the prose.
    response = _rec(
        [_pick(reason="great chemistry with Tom Cruise")],
        prose="Here is a lean action pick.",
    )
    reason = check_excluded_actors(response, ["Tom Cruise"])
    assert reason == "excluded_actor:tom cruise"


def test_excluded_actor_case_insensitive() -> None:
    response = _rec([_pick()], prose="features TOM CRUISE prominently")
    assert check_excluded_actors(response, ["tom cruise"]) == "excluded_actor:tom cruise"


def test_no_excluded_actor_passes() -> None:
    response = _rec([_pick()], prose="Here is a lean action pick.")
    assert check_excluded_actors(response, ["Tom Cruise"]) is None
    assert run_gate4(response, [_candidate()], exclude_actors=["Tom Cruise"]).ok is True


# -- (b) title existence / G1 anti-hallucination ------------------------------


def test_hallucinated_title_fails() -> None:
    # The model recommends a film that was never a candidate.
    response = _rec([_pick(title="John Wick 5")], prose="Try this one.")
    result = run_gate4(response, [_candidate()])
    assert result.ok is False
    assert result.reason == "hallucinated_title:john wick 5"


def test_recommended_title_exists_passes() -> None:
    response = _rec([_pick(title="john wick")], prose="Try this one.")
    assert check_titles_exist(response, [_candidate()]) is None


# -- (c) prose fact-grounding -------------------------------------------------


def test_wrong_year_in_prose_fails() -> None:
    # Candidate is 2014; prose asserts 2011 (training-knowledge leakage).
    response = _rec([_pick()], prose="John Wick, released in 2011, is a lean thriller.")
    result = run_gate4(response, [_candidate()])
    assert result.ok is False
    assert result.reason == "ungrounded_year:2011"


def test_correct_year_in_prose_passes() -> None:
    response = _rec([_pick()], prose="John Wick, released in 2014, is a lean thriller.")
    assert check_fact_grounding(response, [_candidate()]) is None
    assert run_gate4(response, [_candidate()]).ok is True


def test_wrong_director_in_prose_fails() -> None:
    # MovieRecord carries the director, so "directed by X" is grounded.
    response = _rec([_pick()], prose="John Wick, directed by Christopher Nolan.")
    result = run_gate4(response, [_record()])
    assert result.ok is False
    assert result.reason == "ungrounded_director:christopher nolan"


def test_correct_director_in_prose_passes() -> None:
    response = _rec([_pick()], prose="John Wick, directed by Chad Stahelski, delivers.")
    assert check_fact_grounding(response, [_record()]) is None


def test_wrong_cast_in_prose_fails() -> None:
    response = _rec([_pick()], prose="John Wick, starring Brad Pitt, is relentless.")
    result = run_gate4(response, [_record()])
    assert result.ok is False
    assert result.reason == "ungrounded_cast:brad pitt"


def test_correct_cast_in_prose_passes() -> None:
    response = _rec([_pick()], prose="John Wick, starring Keanu Reeves, is relentless.")
    assert check_fact_grounding(response, [_record()]) is None


def test_director_grounding_skipped_without_metadata() -> None:
    # ScoredMovie carries no director, so name grounding is a documented no-op:
    # only the year check has signal here.
    response = _rec([_pick()], prose="John Wick, directed by Anyone At All (2014).")
    assert check_fact_grounding(response, [_candidate()]) is None


# -- INTEGRATION: PRD section 7.5 worked example ------------------------------


def _worked_example_models(gen_calls: dict[str, int]) -> GraphModels:
    """Fake models where the FIRST generation leaks an excluded actor.

    Turn-1 recommends John Wick. The generator's first attempt writes prose that
    mentions the excluded actor Tom Cruise (Gate-4 (a) fails); the second attempt
    is clean. The extractor supplies ``exclude_actors=["Tom Cruise"]``.
    """

    def rewrite(raw_query: str, history) -> str:
        return f"standalone: {raw_query}"

    def extract(query: str) -> ParsedQuery:
        return ParsedQuery(genres=["Action"], exclude_actors=["Tom Cruise"])

    def generate(query: str, candidates) -> RecommendationSet:
        gen_calls["n"] += 1
        if gen_calls["n"] == 1:
            # Leaks the excluded actor -> Gate-4 rejects this.
            return _rec(
                [_pick()],
                prose="A relentless action pick, and fans of Tom Cruise will love it.",
            )
        # Clean regeneration: no excluded actor, facts grounded.
        return _rec([_pick()], prose="Here is a lean action pick from 2014.")

    return GraphModels(rewrite=rewrite, extract=extract, generate=generate)


def test_section_7_5_worked_example_regen_then_clean() -> None:
    gen_calls = {"n": 0}
    models = _worked_example_models(gen_calls)

    def retriever(query: str, parsed: ParsedQuery, shown_movies=()):
        return [_candidate()]

    result = run_turn(
        TurnState(
            trace_id="trace-7-5",
            session_id="sess-1",
            raw_query="something like John Wick but without Tom Cruise",
        ),
        retriever=retriever,
        models=models,
    )

    # The regen edge fired: a double generate -> validate cycle is visible in the
    # path, and the generator was invoked exactly twice.
    assert result.state.path_taken == [
        "rewrite",
        "extract",
        "retrieve",
        "filter",
        "generate",
        "validate",
        "generate",
        "validate",
    ]
    assert gen_calls["n"] == 2
    assert result.state.gen_retries == 1

    # Final answer is clean: Gate-4 passes and the excluded actor is gone.
    assert result.state.validation_failed is False
    assert result.state.validation_reason is None
    assert result.state.response is not None
    assert "tom cruise" not in result.state.response.prose.casefold()
    assert len(result.state.response.picks) == 1
    assert result.state.response.picks[0].title == "John Wick"
    # The turn ENDed after the clean validate (no fallback).
    assert "fallback" not in result.state.path_taken
    assert result.state.path_taken[-1] == "validate"


def test_gate4_exhausts_retries_then_fallback() -> None:
    # A generator that ALWAYS leaks the excluded actor exhausts the 2 regen
    # attempts and drops to the deterministic fallback.
    def rewrite(raw_query: str, history) -> str:
        return raw_query

    def extract(query: str) -> ParsedQuery:
        return ParsedQuery(exclude_actors=["Tom Cruise"])

    def generate(query: str, candidates) -> RecommendationSet:
        return _rec([_pick()], prose="You'll love this if you like Tom Cruise.")

    models = GraphModels(rewrite=rewrite, extract=extract, generate=generate)

    result = run_turn(
        TurnState(trace_id="t", session_id="s", raw_query="no cruise please"),
        retriever=lambda q, p, shown=(): [_candidate()],
        models=models,
    )

    assert result.state.path_taken.count("generate") == 2
    assert result.state.path_taken[-1] == "fallback"
    assert result.state.gen_retries == 2
    assert "fallback" in result.state.degradation
