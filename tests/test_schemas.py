"""Validation tests for the core Pydantic schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from imdb_chatbot.schemas import (
    ChangeRecord,
    MovieRecommendation,
    MovieRecord,
    ParsedQuery,
    RecommendationSet,
    ScoredMovie,
    TurnState,
    TurnTrace,
)


def test_movie_record_valid() -> None:
    m = MovieRecord(
        tmdb_id=27205,
        title="Inception",
        year=2010,
        genres=["Sci-Fi", "Thriller"],
        director="Christopher Nolan",
        cast=["Leonardo DiCaprio"],
        rating_raw=8.8,
        rating_z={"US": 1.2, "IN": 0.9},
        regions=["US", "GB"],
        poster_url="https://example.com/poster.jpg",
    )
    assert m.tmdb_id == 27205
    assert m.rating_z["US"] == 1.2


def test_movie_record_year_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        MovieRecord(tmdb_id=1, title="Too Old", year=1800)
    with pytest.raises(ValidationError):
        MovieRecord(tmdb_id=2, title="Too New", year=2099)


def test_movie_record_rating_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        MovieRecord(tmdb_id=3, title="Bad Rating", year=2000, rating_raw=99.0)


def test_movie_record_rating_z_accepts_per_region_dict() -> None:
    m = MovieRecord(tmdb_id=4, title="Z", year=2001, rating_z={"US": -0.5, "JP": 2.3})
    assert m.rating_z == {"US": -0.5, "JP": 2.3}


def test_parsed_query_defaults() -> None:
    p = ParsedQuery()
    assert p.genres == []
    assert p.similar_to is None
    p2 = ParsedQuery(genres=["Horror"], min_year=1990, exclude_actors=["X"])
    assert p2.min_year == 1990
    assert p2.exclude_actors == ["X"]


def test_scored_movie_defaults() -> None:
    s = ScoredMovie(tmdb_id=5, title="S", year=2005)
    assert s.dense_score == 0.0
    assert s.rrf_score == 0.0
    assert s.rank == 0


def test_recommendation_set() -> None:
    rec = MovieRecommendation(title="Arrival", year=2016, reason="cerebral sci-fi")
    rs = RecommendationSet(picks=[rec], prose="Here you go.")
    assert rs.picks[0].title == "Arrival"
    assert rs.prose == "Here you go."


def test_turn_state_valid() -> None:
    st = TurnState(
        trace_id="t1",
        session_id="s1",
        raw_query="something like Inception",
    )
    assert st.extract_retries == 0
    assert st.extract_failed is False
    assert st.validation_failed is False
    assert st.degradation == []
    assert st.retrieved == []


def test_turn_trace_has_model_config_version_field() -> None:
    tr = TurnTrace(
        trace_id="t1",
        ts=datetime.now(UTC),
        session_id="s1",
        raw_query="q",
        model_config_version="m-v3",
    )
    assert "model_config_version" in TurnTrace.model_fields
    assert tr.model_config_version == "m-v3"
    # Confirm we did not accidentally shadow Pydantic's reserved attribute.
    assert isinstance(TurnTrace.model_config, dict)


def test_turn_trace_defaults() -> None:
    tr = TurnTrace(
        trace_id="t2",
        ts=datetime.now(UTC),
        session_id="s2",
        raw_query="q",
    )
    assert tr.timings_ms == {}
    assert tr.token_usage == {}
    assert tr.cost_usd == 0.0
    assert tr.judge_scores is None


def test_change_record_valid() -> None:
    c = ChangeRecord(
        change_id="c1",
        ts=datetime.now(UTC),
        artifact_type="prompt",
        version_after="v2",
        motivating_trace_ids=["t1", "t2"],
        metric_before={"pass_rate": 0.7},
        metric_after={"pass_rate": 0.85},
        suite_size_before=100,
        suite_size_after=120,
    )
    assert c.artifact_type == "prompt"
    assert c.version_before is None
    assert c.metric_after["pass_rate"] == 0.85
