"""Adversary tests for the raw TMDB archive (#78).

The invariant every test here attacks is one sentence:

    A stored raw payload re-derives its ``MovieRecord`` exactly.

That claim is what makes ``MovieRecord`` safe to treat as a *derived* projection
instead of the system of record, so each test below picks an input that could
plausibly break it - the corpus's largest record, namespaces we deliberately do
not map, hostile JSON scalars, and a payload archived under an older fetch spec.

``tests/fixtures/tmdb/402431_wicked.json`` is a real TMDB response (Wicked 2024,
396 cast / 227 crew - the largest record in the corpus), fetched with the full
``APPEND_TO_RESPONSE`` spec. Refresh it with:

    npx @dotenvx/dotenvx run -f .env -- .venv/Scripts/python.exe -c "..."
"""

from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path

import pytest

from imdb_chatbot.ingest import APPEND_TO_RESPONSE, map_tmdb_movie, rederive_record
from imdb_chatbot.schemas import MovieRecord
from imdb_chatbot.store import RawArchive

FIXTURES = Path(__file__).parent / "fixtures" / "tmdb"

OLD_APPEND_SPEC = "credits,release_dates"  # what the 46k corpus was ingested under


@pytest.fixture()
def archive(tmp_path: Path):
    a = RawArchive(tmp_path / "raw_tmdb.sqlite")
    try:
        yield a
    finally:
        a.close()


@pytest.fixture(scope="module")
def wicked() -> dict:
    """The corpus's largest record, as TMDB actually returned it."""
    with (FIXTURES / "402431_wicked.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Invariant: a stored raw payload re-derives its MovieRecord exactly.
# Attack 1 - scale. 396 cast, 227 crew, ~205 KB of JSON through gzip + a BLOB.
# --------------------------------------------------------------------------


def test_largest_record_in_the_corpus_rederives_exactly(archive: RawArchive, wicked: dict):
    assert len(wicked["credits"]["cast"]) == 396
    assert len(wicked["credits"]["crew"]) == 227

    live = map_tmdb_movie(wicked, preferred_region="US")
    archive.put(wicked["id"], wicked, append_spec=APPEND_TO_RESPONSE)
    replayed = rederive_record(archive, wicked["id"], preferred_region="US")

    assert replayed == live
    # Not just equal-ish: every one of the 396 names, in order, survived.
    assert replayed is not None
    assert replayed.cast == [c["name"] for c in wicked["credits"]["cast"]]
    assert len(replayed.cast) == 396
    assert archive.get(wicked["id"]) == wicked


def test_missing_id_rederives_to_none(archive: RawArchive):
    assert rederive_record(archive, 402431) is None


# --------------------------------------------------------------------------
# Attack 2 - fields we do not map. The archive is only worth its 450 MB if it
# keeps what MovieRecord throws away; and unmapped data must not be able to
# perturb the projection.
# --------------------------------------------------------------------------


def test_unmapped_namespaces_are_kept_whole_and_cannot_change_the_record(
    archive: RawArchive, wicked: dict
):
    assert wicked["keywords"]["keywords"], "fixture must carry the unmapped namespaces"
    assert wicked["alternative_titles"]["titles"]

    stripped = copy.deepcopy(wicked)
    for unmapped in ("keywords", "alternative_titles", "imdb_id", "tagline", "homepage"):
        stripped.pop(unmapped, None)

    archive.put(wicked["id"], wicked, append_spec=APPEND_TO_RESPONSE)
    replayed = rederive_record(archive, wicked["id"], preferred_region="US")

    # Dropping every unmapped field leaves the derived record bit-identical:
    # nothing outside the mapped set can leak into MovieRecord today...
    assert replayed == map_tmdb_movie(stripped, preferred_region="US")
    # ...and yet the archive still has all of it, which is the entire point.
    stored = archive.get(wicked["id"])
    assert stored is not None
    assert stored["keywords"] == wicked["keywords"]
    assert stored["alternative_titles"] == wicked["alternative_titles"]
    assert stored["credits"]["crew"] == wicked["credits"]["crew"]
    # MovieRecord is unchanged by this ticket - no new field absorbed the data.
    assert set(MovieRecord.model_fields) == set(replayed.model_dump())


HOSTILE = {
    "id": 909001,
    "title": "기생충 — Parasite: Bande Originale ★",
    "original_title": "기생충",
    "overview": "Non-ASCII, emoji 🎬, and a lone surrogate-free   line separator.",
    "release_date": "2019-05-30",
    "poster_path": "/hostile.jpg",
    "vote_average": 8.508999999999999,  # a float that only survives exact repr
    "vote_count": 19_384,
    "budget": 11_363_000_000_000,  # bigger than any float64-exact integer usage
    "runtime": 132.5,
    "origin_country": ["KR", "US"],
    "genres": [{"id": 53, "name": "Thriller"}],
    "credits": {
        "cast": [{"name": "송강호", "order": 0}, {"name": "Choi Woo-shik", "order": 1}],
        "crew": [{"name": "봉준호", "job": "Director"}],
    },
    "release_dates": {"results": [{"iso_3166_1": "US", "release_dates": []}]},
    # Namespaces and junk we never map, deliberately awkward.
    "keywords": {"keywords": [{"id": 1, "name": "class conflict"}]},
    "alternative_titles": {"titles": [{"iso_3166_1": "FR", "title": "Parasite"}]},
    "belongs_to_collection": None,
    "spoken_languages": [],
    "_deeply": {"nested": {"list": [1, [2, [3, {"empty": {}, "none": None}]]]}},
}


def test_hostile_scalars_and_unicode_round_trip_losslessly(archive: RawArchive):
    """Attacks the storage codec: gzip + UTF-8 + JSON, not the mapping."""
    archive.put(HOSTILE["id"], HOSTILE, append_spec=APPEND_TO_RESPONSE)
    stored = archive.get(HOSTILE["id"])

    assert stored == HOSTILE
    assert stored is not None
    assert stored["vote_average"] == 8.508999999999999  # not 8.509
    assert stored["budget"] == 11_363_000_000_000
    assert stored["credits"]["cast"][0]["name"] == "송강호"

    live = map_tmdb_movie(HOSTILE, preferred_region="KR")
    replayed = rederive_record(archive, HOSTILE["id"], preferred_region="KR")
    assert replayed == live
    assert replayed is not None
    assert replayed.rating_raw == 8.508999999999999
    assert replayed.title == HOSTILE["title"]


def test_payload_is_stored_gzipped_not_as_plain_json(archive: RawArchive, wicked: dict):
    """The 450 MB estimate assumes compression actually happens."""
    size = archive.put(wicked["id"], wicked, append_spec=APPEND_TO_RESPONSE)
    row = archive._conn.execute(
        "SELECT payload FROM raw_movies WHERE tmdb_id = ?", (wicked["id"],)
    ).fetchone()
    blob = row["payload"]

    assert blob[:2] == b"\x1f\x8b"  # gzip magic
    assert len(blob) == size == archive.stored_bytes()
    assert size < len(json.dumps(wicked).encode("utf-8")) // 4
    assert json.loads(gzip.decompress(blob).decode("utf-8")) == wicked


# --------------------------------------------------------------------------
# Attack 3 - resumability. "tmdb_id is the PRIMARY KEY, so completed rows are
# skipped naturally" is true only while the fetch spec never changes. Widen
# append_to_response and that same skip silently freezes the narrow payload in
# place forever - the exact failure append_spec exists to prevent.
# --------------------------------------------------------------------------


def test_a_row_fetched_under_an_older_spec_does_not_count_as_done(
    archive: RawArchive, wicked: dict
):
    narrow = copy.deepcopy(wicked)
    narrow.pop("keywords")
    narrow.pop("alternative_titles")
    archive.put(narrow["id"], narrow, append_spec=OLD_APPEND_SPEC)

    assert archive.stored_ids() == {narrow["id"]}  # present...
    assert archive.stored_ids(append_spec=APPEND_TO_RESPONSE) == set()  # ...but not done
    assert archive.spec_of(narrow["id"]) == OLD_APPEND_SPEC

    archive.put(wicked["id"], wicked, append_spec=APPEND_TO_RESPONSE)
    assert archive.stored_ids(append_spec=APPEND_TO_RESPONSE) == {wicked["id"]}
    assert archive.count() == 1  # re-fetch replaced the row, it did not duplicate it
    stored = archive.get(wicked["id"])
    assert stored is not None and "keywords" in stored


def test_storage_is_deterministic_so_a_rerun_is_a_no_op(archive: RawArchive, wicked: dict):
    first = archive.put(wicked["id"], wicked, append_spec=APPEND_TO_RESPONSE)
    shuffled = dict(reversed(list(wicked.items())))  # same payload, different key order
    second = archive.put(wicked["id"], shuffled, append_spec=APPEND_TO_RESPONSE)
    assert first == second == archive.stored_bytes()
    assert archive.count() == 1


def test_archive_survives_reopening(tmp_path: Path, wicked: dict):
    """Resumability is worthless if the file does not outlive the process."""
    path = tmp_path / "raw_tmdb.sqlite"
    with RawArchive(path) as first:
        first.put(wicked["id"], wicked, append_spec=APPEND_TO_RESPONSE)
    with RawArchive(path) as second:
        assert second.stored_ids(append_spec=APPEND_TO_RESPONSE) == {wicked["id"]}
        assert rederive_record(second, wicked["id"], preferred_region="US") == map_tmdb_movie(
            wicked, preferred_region="US"
        )


# --------------------------------------------------------------------------
# Live integration - GATED. Fetches real payloads and asserts that what the
# archive gives back derives the same record the live payload does.
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("TMDB_LIVE_TEST") != "1",
    reason="live TMDB pull disabled (set TMDB_LIVE_TEST=1 to enable)",
)
def test_live_slice_archives_and_rederives_exactly(tmp_path: Path):
    from imdb_chatbot.ingest import TMDBClient

    ids = [27205, 402431, 496243]  # Inception, Wicked, Parasite
    with TMDBClient.from_env() as client, RawArchive(tmp_path / "raw.sqlite") as archive:
        for tmdb_id in ids:
            payload = client.fetch_details(tmdb_id)
            assert "keywords" in payload and "alternative_titles" in payload
            archive.put(tmdb_id, payload, append_spec=APPEND_TO_RESPONSE)
            assert rederive_record(archive, tmdb_id, preferred_region="US") == map_tmdb_movie(
                payload, preferred_region="US"
            )
        assert archive.stored_ids(append_spec=APPEND_TO_RESPONSE) == set(ids)
