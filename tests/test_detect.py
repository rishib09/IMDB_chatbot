"""Tests for the detection rules -> taxonomy flags (ticket #24).

Deterministic and network-free: each test crafts a ``TurnTrace`` that should fire
exactly one taxonomy code (or none) and asserts ``detect`` returns exactly that.
"""

from __future__ import annotations

from datetime import UTC, datetime

from imdb_chatbot.eval.detect import (
    COST_BUDGET_USD,
    apply_detection,
    count_codes,
    detect,
)
from imdb_chatbot.schemas import (
    MovieRecommendation,
    RecommendationSet,
    ScoredMovie,
    TurnTrace,
)

# -- helpers ------------------------------------------------------------------

_HAPPY_PATH = ["rewrite", "extract", "retrieve", "filter", "generate", "validate"]


def _candidate(tmdb_id: int = 1, title: str = "John Wick", year: int = 2014) -> ScoredMovie:
    return ScoredMovie(tmdb_id=tmdb_id, title=title, year=year, rrf_score=0.5, rank=1)


def _response(*picks: MovieRecommendation) -> RecommendationSet:
    return RecommendationSet(picks=list(picks), prose="here you go")


def _trace(**overrides) -> TurnTrace:
    """A clean, first-try trace; overrides tweak individual fields."""
    base = {
        "trace_id": "t1",
        "ts": datetime.now(UTC),
        "session_id": "s1",
        "raw_query": "an action movie",
        "path_taken": list(_HAPPY_PATH),
        "candidates": [_candidate()],
        "response": _response(MovieRecommendation(title="John Wick", year=2014, reason="lean")),
    }
    base.update(overrides)
    return TurnTrace(**base)


# -- clean trace fires nothing ------------------------------------------------


def test_clean_trace_fires_no_codes() -> None:
    assert detect(_trace()) == []


# -- X2: extractor retries exhausted ------------------------------------------


def test_x2_when_extract_retries_exhausted() -> None:
    trace = _trace(
        extract_retries=2,
        path_taken=["rewrite", "extract", "extract", "retrieve", "filter", "generate", "validate"],
        degradation=["extract_regex_fallback"],
    )
    assert detect(trace) == ["X2"]


def test_x2_does_not_fire_on_single_retry() -> None:
    trace = _trace(extract_retries=1)
    assert "X2" not in detect(trace)


# -- G2: instruction/format near-miss caught (regen cycle) --------------------


def test_g2_when_validate_ran_twice() -> None:
    trace = _trace(
        gen_retries=1,
        path_taken=[
            "rewrite", "extract", "retrieve", "filter",
            "generate", "validate", "generate", "validate",
        ],
    )
    assert detect(trace) == ["G2"]


def test_g2_does_not_fire_on_single_validate() -> None:
    assert "G2" not in detect(_trace())


# -- S2: repetition / stale ---------------------------------------------------


def test_s2_on_duplicate_pick_within_response() -> None:
    dup = MovieRecommendation(title="John Wick", year=2014, reason="a")
    trace = _trace(response=_response(dup, dup))
    assert detect(trace) == ["S2"]


def test_s2_on_already_shown_movie_with_context() -> None:
    trace = _trace(
        candidates=[_candidate(tmdb_id=7)],
        response=_response(MovieRecommendation(title="John Wick", year=2014, reason="a")),
    )
    # tmdb_id 7 was already shown earlier in the session.
    assert detect(trace, shown_movies={7}) == ["S2"]
    # Without the session context, no repeat is visible (picks are unique).
    assert detect(trace) == []


# -- O1: over budget ----------------------------------------------------------


def test_o1_on_cost_over_threshold() -> None:
    trace = _trace(cost_usd=COST_BUDGET_USD + 0.01)
    assert detect(trace) == ["O1"]


def test_o1_on_budget_degradation_flag() -> None:
    trace = _trace(degradation=["O1_input_cap"])
    assert detect(trace) == ["O1"]


def test_o1_not_fired_under_budget() -> None:
    assert "O1" not in detect(_trace(cost_usd=COST_BUDGET_USD - 0.01))


# -- R1: retrieval-miss fallback ----------------------------------------------


def test_r1_on_empty_candidate_fallback() -> None:
    trace = _trace(
        path_taken=["rewrite", "extract", "retrieve", "filter", "fallback"],
        candidates=[],
        response=RecommendationSet(picks=[], prose="no match"),
    )
    assert detect(trace) == ["R1"]


def test_r1_not_fired_when_fallback_has_candidates() -> None:
    # A fallback reached AFTER a failed generation still has candidates -> not R1.
    trace = _trace(
        path_taken=[
            "rewrite", "extract", "retrieve", "filter",
            "generate", "validate", "generate", "validate", "fallback",
        ],
        candidates=[_candidate()],
        response=RecommendationSet(picks=[], prose="gave up"),
        gen_retries=2,
    )
    codes = detect(trace)
    assert "R1" not in codes
    assert "G2" in codes  # the regen cycle is still a caught near-miss


# -- combined + batch helpers -------------------------------------------------


def test_multiple_codes_sorted() -> None:
    trace = _trace(
        extract_retries=2,
        cost_usd=COST_BUDGET_USD + 1.0,
        path_taken=[
            "rewrite", "extract", "extract", "retrieve", "filter",
            "generate", "validate", "generate", "validate",
        ],
    )
    assert detect(trace) == ["G2", "O1", "X2"]


def test_apply_detection_writes_flags_in_place() -> None:
    trace = _trace(extract_retries=2, flags=["manual"])
    returned = apply_detection(trace)
    assert returned == ["X2", "manual"]
    assert trace.flags == ["X2", "manual"]


def test_count_codes_over_batch() -> None:
    traces = [
        _trace(),  # clean
        _trace(extract_retries=2),  # X2
        _trace(cost_usd=COST_BUDGET_USD + 1.0),  # O1
        _trace(extract_retries=2, cost_usd=COST_BUDGET_USD + 1.0),  # X2 + O1
    ]
    counts = count_codes(traces)
    assert counts["X2"] == 2
    assert counts["O1"] == 2
    assert counts["G2"] == 0
