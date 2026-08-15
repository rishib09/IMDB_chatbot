"""Tests for the single-turn LangGraph (ticket #18).

Everything runs on FAKE models and a stub retriever - there is NO network I/O and
no live LLM call. The point of the graph is that the control flow (retries,
branches, fallback) is deterministic and inspectable, so these tests assert on
``path_taken`` and the persisted ``TurnTrace`` rather than on model output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from imdb_chatbot.graph import GraphModels, build_graph, run_turn
from imdb_chatbot.graph.build import (
    apply_region_default,
    correct_person_role,
    regex_extract,
)
from imdb_chatbot.schemas import (
    MovieRecommendation,
    ParsedQuery,
    RecommendationSet,
    ScoredMovie,
    TurnState,
)
from imdb_chatbot.store import TraceStore

# -- fakes --------------------------------------------------------------------


def _candidate(tmdb_id: int = 1, title: str = "John Wick", year: int = 2014) -> ScoredMovie:
    return ScoredMovie(tmdb_id=tmdb_id, title=title, year=year, rrf_score=0.5, rank=1)


def _recommendation() -> RecommendationSet:
    return RecommendationSet(
        picks=[MovieRecommendation(title="John Wick", year=2014, reason="fast action")],
        prose="Here is a lean action pick.",
    )


def _stub_retriever(candidates: list[ScoredMovie]):
    """A retriever callable that ignores the query and returns fixed candidates."""

    def retrieve(query: str, parsed: ParsedQuery, shown_movies=()) -> list[ScoredMovie]:
        return list(candidates)

    return retrieve


def _good_models() -> GraphModels:
    """Fake models where every slot succeeds on the first attempt."""

    def rewrite(raw_query: str, history) -> str:
        return f"standalone: {raw_query}"

    def extract(query: str) -> ParsedQuery:
        return ParsedQuery(genres=["Action"], exclude_actors=["Tom Cruise"])

    def generate(query: str, candidates) -> RecommendationSet:
        return _recommendation()

    return GraphModels(rewrite=rewrite, extract=extract, generate=generate)


def _initial_state(query: str = "something like that but without Tom Cruise") -> TurnState:
    return TurnState(
        trace_id="trace-1",
        session_id="sess-1",
        user_id="user-1",
        raw_query=query,
    )


# -- happy path ---------------------------------------------------------------


def test_single_turn_end_to_end_persists_trace(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    try:
        result = run_turn(
            _initial_state(),
            retriever=_stub_retriever([_candidate()]),
            models=_good_models(),
            store=store,
        )

        assert result.state.path_taken == [
            "rewrite",
            "extract",
            "retrieve",
            "filter",
            "generate",
            "validate",
        ]
        # rewriter ran (query resolved), extractor produced a ParsedQuery.
        assert result.state.rewritten_query is not None
        assert result.state.rewritten_query.startswith("standalone:")
        assert isinstance(result.state.parsed, ParsedQuery)
        assert result.state.response is not None
        assert len(result.state.response.picks) == 1
        assert result.state.validation_failed is False

        # The trace was persisted and is the system-of-record for the turn.
        persisted = store.read_trace("trace-1")
        assert persisted is not None
        assert persisted.path_taken == result.state.path_taken
        assert persisted.extract_retries == 0
        assert persisted.gen_retries == 0
        # timings recorded for every node that ran; placeholder versions stamped.
        assert set(persisted.timings_ms) == set(result.state.path_taken)
        assert persisted.prompt_version == "dev"
        assert persisted.index_version == "dev"
    finally:
        store.close()


def test_structured_outputs_enforced() -> None:
    result = run_turn(
        _initial_state(),
        retriever=_stub_retriever([_candidate()]),
        models=_good_models(),
    )
    # extract yields a valid ParsedQuery; generate a valid RecommendationSet.
    assert isinstance(result.state.parsed, ParsedQuery)
    assert result.state.parsed.exclude_actors == ["Tom Cruise"]
    assert isinstance(result.state.response, RecommendationSet)
    assert all(isinstance(p, MovieRecommendation) for p in result.state.response.picks)


# -- extract retry cycle ------------------------------------------------------


def test_extract_retry_then_success() -> None:
    calls = {"n": 0}

    def flaky_extract(query: str) -> ParsedQuery:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("model emitted trailing prose - unparseable JSON")
        return ParsedQuery(genres=["Action"])

    models = _good_models()
    models.extract = flaky_extract

    result = run_turn(
        _initial_state(),
        retriever=_stub_retriever([_candidate()]),
        models=models,
    )

    # The retry is an EDGE, visible in the path as a repeated extract node.
    assert result.state.path_taken == [
        "rewrite",
        "extract",
        "extract",
        "retrieve",
        "filter",
        "generate",
        "validate",
    ]
    assert result.state.extract_retries == 1
    assert calls["n"] == 2
    # region defaults to US (ticket #53): no region/person named in the query.
    assert result.state.parsed == ParsedQuery(genres=["Action"], region="US")


# -- default region = US (ticket #53) ----------------------------------------


def test_region_defaults_to_us_for_generic_query() -> None:
    out = apply_region_default(ParsedQuery(genres=["Thriller"]), "a good thriller")
    assert out.region == "US"


def test_explicit_region_is_not_overridden() -> None:
    out = apply_region_default(ParsedQuery(region="KR"), "a korean thriller")
    assert out.region == "KR"


def test_no_us_default_when_person_named() -> None:
    # A person query should span all their films regardless of country.
    assert apply_region_default(ParsedQuery(director="Nolan"), "nolan movies").region is None
    assert apply_region_default(ParsedQuery(actor="Tom Hanks"), "tom hanks films").region is None


def test_any_region_phrase_clears_the_default() -> None:
    out = apply_region_default(ParsedQuery(), "a great thriller from anywhere")
    assert out.region is None


# -- director/actor role correction (ticket #50) -----------------------------


def test_actor_cue_moves_name_from_director_to_actor() -> None:
    # Extractor wrongly filed a star under director; "starring" fixes it.
    out = correct_person_role(ParsedQuery(director="Meryl Streep"), "films starring meryl streep")
    assert out.actor == "Meryl Streep" and out.director is None


def test_director_cue_moves_name_from_actor_to_director() -> None:
    out = correct_person_role(ParsedQuery(actor="Nolan"), "movies directed by nolan")
    assert out.director == "Nolan" and out.actor is None


def test_no_person_no_change() -> None:
    # "with a twist" has no extracted person -> untouched.
    out = correct_person_role(ParsedQuery(genres=["Thriller"]), "a thriller with a twist")
    assert out.director is None and out.actor is None


def test_correct_role_leaves_correct_tagging_alone() -> None:
    out = correct_person_role(ParsedQuery(actor="Tom Hanks"), "movies with tom hanks")
    assert out.actor == "Tom Hanks" and out.director is None


def test_extract_regex_fallback_after_two_failures() -> None:
    def always_fail(query: str) -> ParsedQuery:
        raise ValueError("never valid JSON")

    models = _good_models()
    models.extract = always_fail

    result = run_turn(
        _initial_state("a horror movie after 2015"),
        retriever=_stub_retriever([_candidate()]),
        models=models,
    )

    # Two failed LLM attempts, then regex fallback lets the turn continue.
    assert result.state.path_taken.count("extract") == 2
    assert result.state.extract_retries == 2
    assert "extract_regex_fallback" in result.state.degradation
    # regex picked up the genre + the "after YYYY" bound.
    assert result.state.parsed is not None
    assert "Horror" in result.state.parsed.genres
    assert result.state.parsed.min_year == 2015
    # Turn still completes through generate/validate.
    assert result.state.path_taken[-2:] == ["generate", "validate"]


# -- empty-candidates -> fallback ---------------------------------------------


def test_empty_candidates_routes_to_fallback() -> None:
    result = run_turn(
        _initial_state(),
        retriever=_stub_retriever([]),  # nothing survives retrieval
        models=_good_models(),
    )

    assert result.state.path_taken == [
        "rewrite",
        "extract",
        "retrieve",
        "filter",
        "fallback",
    ]
    # fallback never runs generate/validate.
    assert "generate" not in result.state.path_taken
    assert "fallback" in result.state.degradation
    assert result.state.response is not None
    assert result.state.response.picks == []
    assert result.state.response.prose  # deterministic no-match message


# -- validate structural failure -> bounded regen -----------------------------


def test_validate_regen_cycle_then_success() -> None:
    calls = {"n": 0}

    def flaky_generate(query: str, candidates) -> RecommendationSet:
        calls["n"] += 1
        if calls["n"] == 1:
            return RecommendationSet(picks=[], prose="oops, no picks")  # structurally bad
        return _recommendation()

    models = _good_models()
    models.generate = flaky_generate

    result = run_turn(
        _initial_state(),
        retriever=_stub_retriever([_candidate()]),
        models=models,
    )

    # First generate fails validate (empty picks) -> regen edge -> second passes.
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
    assert result.state.gen_retries == 1
    assert result.state.validation_failed is False
    assert len(result.state.response.picks) == 1


def test_validate_failure_bounded_then_fallback() -> None:
    def always_empty(query: str, candidates) -> RecommendationSet:
        return RecommendationSet(picks=[], prose="still nothing")

    models = _good_models()
    models.generate = always_empty

    result = run_turn(
        _initial_state(),
        retriever=_stub_retriever([_candidate()]),
        models=models,
    )

    # Regen is bounded by 2, after which the turn drops to fallback.
    assert result.state.path_taken.count("generate") == 2
    assert result.state.path_taken[-1] == "fallback"
    assert result.state.gen_retries == 2
    assert "fallback" in result.state.degradation


# -- rewriter degradation -----------------------------------------------------


def test_rewrite_failure_degrades_to_raw_query() -> None:
    def broken_rewrite(raw_query: str, history) -> str:
        raise RuntimeError("rewriter slot unavailable")

    models = _good_models()
    models.rewrite = broken_rewrite

    result = run_turn(
        _initial_state("a good thriller"),
        retriever=_stub_retriever([_candidate()]),
        models=models,
    )

    # Rewriter is optional: turn continues using the raw query.
    assert result.state.rewritten_query == "a good thriller"
    assert "rewrite_skipped" in result.state.degradation
    assert result.state.path_taken[0] == "rewrite"
    assert result.state.response is not None
    assert len(result.state.response.picks) == 1


# -- regex fallback unit ------------------------------------------------------


def test_regex_extract_keywords_and_years() -> None:
    parsed = regex_extract("a scary horror thriller from before 1999")
    assert "Horror" in parsed.genres
    assert "Thriller" in parsed.genres
    assert parsed.max_year == 1999
    assert parsed.min_year is None


# -- build_graph is directly usable / injectable ------------------------------


def test_build_graph_invoke_directly() -> None:
    graph = build_graph(retriever=_stub_retriever([_candidate()]), models=_good_models())
    result = graph.invoke(_initial_state())
    final = TurnState.model_validate(result)
    assert final.path_taken[-1] == "validate"
    assert final.response is not None


def test_no_network_stub_retriever_only() -> None:
    """Guard: the whole graph runs with plain Python fakes - no live client."""
    seen = {"retriever": 0}

    def counting_retriever(query: str, parsed: ParsedQuery, shown_movies=()) -> list[ScoredMovie]:
        seen["retriever"] += 1
        return [_candidate()]

    run_turn(
        _initial_state(),
        retriever=counting_retriever,
        models=_good_models(),
    )
    assert seen["retriever"] == 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
