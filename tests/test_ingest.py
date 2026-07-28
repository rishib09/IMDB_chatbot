"""Unit tests for TMDB -> MovieRecord ingestion.

These tests exercise ONLY the pure transform logic against embedded fixture
JSON. No test here makes a network call. The single live-API test is gated on
an explicit opt-in env var (``TMDB_LIVE_TEST``) so CI (and normal local runs)
never touch TMDB.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from imdb_chatbot.ingest import (
    is_complete,
    map_tmdb_movie,
    run_ingest,
    transform_movie,
)
from imdb_chatbot.ingest.certificates import load_certificate_map, registered_regions
from imdb_chatbot.store import TraceStore

# --------------------------------------------------------------------------
# Fixtures - representative TMDB /movie/{id}?append_to_response=credits,
# release_dates payloads. Deliberately trimmed but structurally faithful.
# --------------------------------------------------------------------------

US_MOVIE = {
    "id": 27205,
    "title": "Inception",
    "original_title": "Inception",
    "overview": "A thief who steals corporate secrets through dream-sharing tech.",
    "release_date": "2010-07-16",
    "poster_path": "/edv5CZvWj09upOsy2Y6IwDhK8bt.jpg",
    "vote_average": 8.4,
    "runtime": 148,
    "origin_country": ["US"],
    "genres": [{"id": 28, "name": "Action"}, {"id": 878, "name": "Science Fiction"}],
    "credits": {
        "cast": [
            {"name": "Leonardo DiCaprio", "order": 0},
            {"name": "Joseph Gordon-Levitt", "order": 1},
        ],
        "crew": [
            {"name": "Christopher Nolan", "job": "Director"},
            {"name": "Hans Zimmer", "job": "Original Music Composer"},
        ],
    },
    "release_dates": {
        "results": [
            {
                "iso_3166_1": "US",
                "release_dates": [{"certification": "PG-13", "type": 3}],
            }
        ]
    },
}

IN_MOVIE = {
    "id": 20453,
    "title": "3 Idiots",
    "original_title": "3 Idiots",
    "overview": "Two friends search for their long-lost companion from college.",
    "release_date": "2009-12-25",
    "poster_path": "/66A9MqXOyVFCssoloscw79z8Tew.jpg",
    "vote_average": 8.0,
    "runtime": 170,
    "origin_country": ["IN"],
    "genres": [{"id": 35, "name": "Comedy"}, {"id": 18, "name": "Drama"}],
    "credits": {
        "cast": [
            {"name": "Aamir Khan", "order": 0},
            {"name": "Kareena Kapoor Khan", "order": 1},
        ],
        "crew": [{"name": "Rajkumar Hirani", "job": "Director"}],
    },
    "release_dates": {
        "results": [
            {
                "iso_3166_1": "IN",
                "release_dates": [{"certification": "UA", "type": 3}],
            }
        ]
    },
}

CO_PRODUCTION = {
    "id": 999001,
    "title": "The Crossing",
    "original_title": "The Crossing",
    "overview": "A cross-border tale spanning two continents.",
    "release_date": "2018-05-01",
    "poster_path": "/coprod.jpg",
    "vote_average": 7.1,
    "runtime": 121,
    "origin_country": ["US", "IN"],
    "genres": [{"id": 18, "name": "Drama"}],
    "credits": {
        "cast": [{"name": "Some Actor", "order": 0}],
        "crew": [{"name": "Some Director", "job": "Director"}],
    },
    "release_dates": {
        "results": [
            {"iso_3166_1": "IN", "release_dates": [{"certification": "A", "type": 3}]},
        ]
    },
}

# All completeness fields present, but vote_average is out of the schema's
# [0, 10] range -> MovieRecord validation fails -> must be quarantined.
MALFORMED_MOVIE = {
    "id": 999002,
    "title": "Bad Rating",
    "overview": "This row is complete but carries an impossible rating.",
    "release_date": "2015-01-01",
    "poster_path": "/bad.jpg",
    "vote_average": 42.0,
    "origin_country": ["US"],
    "genres": [{"id": 18, "name": "Drama"}],
    "credits": {"cast": [{"name": "Nobody"}], "crew": []},
    "release_dates": {"results": []},
}

# Valid TMDB row that simply lacks a plot -> fails the completeness bar -> skip.
INCOMPLETE_MOVIE = {
    "id": 999003,
    "title": "No Plot Here",
    "overview": "",
    "release_date": "2016-01-01",
    "poster_path": "/nop.jpg",
    "vote_average": 6.0,
    "origin_country": ["US"],
    "genres": [{"id": 18, "name": "Drama"}],
    "credits": {"cast": [{"name": "Someone"}], "crew": []},
    "release_dates": {"results": []},
}


# --------------------------------------------------------------------------
# Pure transform
# --------------------------------------------------------------------------


def test_us_movie_maps_to_record():
    rec = map_tmdb_movie(US_MOVIE, preferred_region="US")
    assert rec.tmdb_id == 27205
    assert rec.title == "Inception"
    assert rec.year == 2010
    assert rec.director == "Christopher Nolan"
    assert rec.cast == ["Leonardo DiCaprio", "Joseph Gordon-Levitt"]
    assert rec.genres == ["Action", "Science Fiction"]
    assert rec.rating_raw == 8.4
    assert rec.regions == ["US"]
    assert rec.duration_min == 148
    assert str(rec.poster_url) == (
        "https://image.tmdb.org/t/p/w500/edv5CZvWj09upOsy2Y6IwDhK8bt.jpg"
    )
    # US MPAA PG-13 -> TEEN
    assert rec.certificate_raw == "PG-13"
    assert rec.certificate_system == "MPAA"
    assert rec.certificate_norm == "TEEN"


def test_in_movie_cbfc_cert_normalizes():
    rec = map_tmdb_movie(IN_MOVIE, preferred_region="IN")
    assert rec.regions == ["IN"]
    assert rec.certificate_system == "CBFC"
    assert rec.certificate_raw == "UA"
    assert rec.certificate_norm == "TEEN"
    assert rec.director == "Rajkumar Hirani"


def test_co_production_carries_both_regions():
    rec = map_tmdb_movie(CO_PRODUCTION, preferred_region="US")
    assert rec.regions == ["US", "IN"]
    # US pull had no US cert in release_dates; falls through to IN's CBFC 'A' -> MATURE.
    assert rec.certificate_system == "CBFC"
    assert rec.certificate_raw == "A"
    assert rec.certificate_norm == "MATURE"


def test_malformed_payload_is_rejected():
    outcome = transform_movie(MALFORMED_MOVIE, preferred_region="US")
    assert outcome.verdict == "rejected"
    assert outcome.record is None
    assert outcome.reason  # carries the ValidationError text
    assert "rating_raw" in outcome.reason


def test_incomplete_payload_is_skipped():
    assert is_complete(INCOMPLETE_MOVIE) is False
    outcome = transform_movie(INCOMPLETE_MOVIE, preferred_region="US")
    assert outcome.verdict == "skipped"
    assert outcome.record is None


def test_completeness_bar_requires_all_fields():
    assert is_complete(US_MOVIE) is True
    # drop each corpus field in turn -> incomplete
    for key in ("overview", "poster_path", "genres", "release_date"):
        bad = copy.deepcopy(US_MOVIE)
        bad[key] = "" if key in ("overview", "poster_path", "release_date") else []
        assert is_complete(bad) is False, f"expected incomplete when {key} missing"
    no_cast = copy.deepcopy(US_MOVIE)
    no_cast["credits"]["cast"] = []
    assert is_complete(no_cast) is False


def test_missing_year_is_skipped_not_rejected():
    no_year = copy.deepcopy(US_MOVIE)
    no_year["release_date"] = ""
    assert is_complete(no_year) is False
    assert transform_movie(no_year).verdict == "skipped"


# --------------------------------------------------------------------------
# Certificate registry (region adapter pattern)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("G", "ALL"), ("PG", "ALL"), ("PG-13", "TEEN"), ("R", "MATURE"), ("NC-17", "ADULT")],
)
def test_us_certificate_normalization(raw, expected):
    cmap = load_certificate_map("US")
    assert cmap is not None
    assert cmap.system == "MPAA"
    assert cmap.normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("U", "ALL"), ("UA", "TEEN"), ("A", "MATURE"), ("S", "ADULT")],
)
def test_in_certificate_normalization(raw, expected):
    cmap = load_certificate_map("IN")
    assert cmap is not None
    assert cmap.system == "CBFC"
    assert cmap.normalize(raw) == expected


def test_unknown_certificate_returns_none_without_failing(caplog):
    cmap = load_certificate_map("US")
    assert cmap is not None
    import logging

    with caplog.at_level(logging.WARNING):
        assert cmap.normalize("X-BOGUS") is None
    assert any("Unknown" in r.message for r in caplog.records)


def test_unregistered_region_has_no_adapter():
    assert load_certificate_map("ZZ") is None
    assert "US" in registered_regions()
    assert "IN" in registered_regions()


# --------------------------------------------------------------------------
# Store integration (still no network) - dedup / upsert / rejects
# --------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path):
    s = TraceStore(tmp_path / "corpus.sqlite")
    try:
        yield s
    finally:
        s.close()


def test_records_and_rejects_persist(store: TraceStore):
    ing = transform_movie(US_MOVIE, preferred_region="US")
    assert ing.verdict == "ingested" and ing.record is not None
    store.write_movie(ing.record)

    rej = transform_movie(MALFORMED_MOVIE, preferred_region="US")
    assert rej.verdict == "rejected"
    store.write_reject(rej.reason or "err", "{}")

    assert store.count_movies() == 1
    assert store.count_rejects() == 1
    fetched = store.read_movie(27205)
    assert fetched is not None and fetched.title == "Inception"


def test_upsert_is_idempotent(store: TraceStore):
    rec = map_tmdb_movie(US_MOVIE, preferred_region="US")
    store.write_movie(rec)
    store.write_movie(rec)  # re-run: same tmdb_id -> still one row
    assert store.count_movies() == 1


# --------------------------------------------------------------------------
# Live pull - GATED. Never runs unless a human opts in explicitly.
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("TMDB_LIVE_TEST") != "1",
    reason="live TMDB pull disabled (set TMDB_LIVE_TEST=1 to enable)",
)
def test_live_pull_smoke():
    store_path = Path(os.environ.get("TMDB_LIVE_DB", "data/live_test.sqlite"))
    store = TraceStore(store_path)
    try:
        stats = run_ingest(store, regions=["US"], max_pages=1)
    finally:
        store.close()
    assert stats.ingested + stats.skipped + stats.rejected > 0
