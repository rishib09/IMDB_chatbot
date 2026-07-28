"""Tests for the pure Chat-page render helpers.

Only the Streamlit-free functions in ``dashboard.render`` are exercised - no
Streamlit server is started here.
"""

from __future__ import annotations

from imdb_chatbot.dashboard.render import (
    PLACEHOLDER_POSTER,
    is_fallback,
    poster_src,
    recommendation_cards,
    relax_options,
)
from imdb_chatbot.schemas import MovieRecommendation, RecommendationSet


def test_recommendation_cards_maps_structured_fields() -> None:
    rec = RecommendationSet(
        picks=[
            MovieRecommendation(
                title="Parasite",
                year=2019,
                reason="A tense class satire.",
                poster_url="https://example.com/parasite.jpg",
            ),
            MovieRecommendation(
                title="Whiplash",
                year=2014,
                reason="Relentless drumming drama.",
            ),
        ],
        prose="Here are two picks.",
    )

    cards = recommendation_cards(rec)

    assert cards == [
        {
            "title": "Parasite",
            "year": 2019,
            "reason": "A tense class satire.",
            "poster": "https://example.com/parasite.jpg",
        },
        {
            "title": "Whiplash",
            "year": 2014,
            "reason": "Relentless drumming drama.",
            "poster": PLACEHOLDER_POSTER,
        },
    ]


def test_poster_src_placeholder_for_none_and_empty() -> None:
    assert poster_src(None) == PLACEHOLDER_POSTER
    assert poster_src("") == PLACEHOLDER_POSTER
    assert poster_src("   ") == PLACEHOLDER_POSTER


def test_poster_src_returns_url_otherwise() -> None:
    url = "https://example.com/poster.png"
    assert poster_src(url) == url


def test_fallback_set_yields_no_cards_and_signals_relax_path() -> None:
    rec = RecommendationSet(picks=[], prose="I could not find a match.")

    assert recommendation_cards(rec) == []
    assert is_fallback(rec) is True

    options = relax_options()
    assert len(options) >= 1
    for option in options:
        assert option["label"]
        assert option["query"]


def test_non_empty_set_is_not_fallback() -> None:
    rec = RecommendationSet(
        picks=[MovieRecommendation(title="Dune", year=2021, reason="Epic sci-fi.")]
    )
    assert is_fallback(rec) is False


def test_relax_options_returns_independent_copies() -> None:
    first = relax_options()
    first[0]["label"] = "mutated"
    assert relax_options()[0]["label"] != "mutated"
