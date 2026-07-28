"""Tests for the added MovieRecord columns and region-group expansion."""

from imdb_chatbot.ingest.__main__ import expand_regions
from imdb_chatbot.ingest.tmdb import map_tmdb_movie


def _payload() -> dict:
    return {
        "id": 42,
        "title": "Sample",
        "release_date": "2001-05-04",
        "genres": [{"id": 1, "name": "Drama"}],
        "credits": {"cast": [{"name": "Actor A"}], "crew": [{"job": "Director", "name": "Dir"}]},
        "overview": "a plot",
        "vote_average": 7.5,
        "vote_count": 1200,
        "original_language": "ko",
        "runtime": 120,
        "origin_country": ["KR"],
        "poster_path": "/p.jpg",
        "status": "Released",
        "budget": 1_000_000,
        "revenue": 5_000_000,
        "production_companies": [{"name": "Studio A"}, {"name": "Studio B"}],
        "release_dates": {"results": []},
    }


def test_new_columns_are_mapped():
    rec = map_tmdb_movie(_payload(), preferred_region="KR")
    assert rec.vote_count == 1200
    assert rec.original_language == "ko"
    assert rec.status == "Released"
    assert rec.budget == 1_000_000
    assert rec.revenue == 5_000_000
    assert rec.production_companies == ["Studio A", "Studio B"]
    # metascore has been removed from the schema
    assert not hasattr(rec, "metascore")


def test_expand_regions_groups_and_dedup():
    assert expand_regions(["KR", "EUROPE"]) == ["KR", "GB", "FR", "DE", "IT", "ES"]
    assert expand_regions(["US", "US"]) == ["US"]
    assert expand_regions(["LATAM"])[0] == "MX"
