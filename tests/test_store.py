"""Tests for the SQLite trace store (WAL mode, lock-serialized writes)."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from imdb_chatbot.schemas import (
    ChangeRecord,
    MovieRecommendation,
    MovieRecord,
    ParsedQuery,
    RecommendationSet,
    ScoredMovie,
    TurnTrace,
)
from imdb_chatbot.store import TraceStore


@pytest.fixture()
def store(tmp_path: Path) -> TraceStore:
    s = TraceStore(tmp_path / "trace.sqlite")
    try:
        yield s
    finally:
        s.close()


def _make_trace(trace_id: str) -> TurnTrace:
    return TurnTrace(
        trace_id=trace_id,
        ts=datetime.now(UTC),
        session_id="sess-1",
        user_id="user-1",
        raw_query="movies like Inception",
        rewritten_query="films similar to Inception",
        parsed=ParsedQuery(genres=["Sci-Fi"], min_year=2000, region="US"),
        retrieved=[ScoredMovie(tmdb_id=1, title="A", year=2001, rrf_score=0.5)],
        candidates=[ScoredMovie(tmdb_id=2, title="B", year=2002, rank=1)],
        filters_applied={"min_year": 2000},
        response=RecommendationSet(
            picks=[MovieRecommendation(title="A", year=2001, reason="great")],
            prose="Try these.",
        ),
        path_taken=["parse", "retrieve", "generate"],
        timings_ms={"retrieve": 12.5},
        token_usage={"prompt": 100, "completion": 50},
        cost_usd=0.0021,
        model_config_version="m-v3",
        prompt_version="p-v2",
        index_version="idx-2026-07",
        judge_scores={"relevance": 0.9},
    )


def test_store_opens_in_wal_mode(store: TraceStore) -> None:
    with store._read_lock:
        mode = store._read_conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert mode.lower() == "wal"


def test_trace_round_trips_losslessly(store: TraceStore) -> None:
    original = _make_trace("trace-round-trip")
    store.write_trace(original)
    fetched = store.read_trace("trace-round-trip")
    assert fetched is not None
    assert fetched == original


def test_read_missing_trace_returns_none(store: TraceStore) -> None:
    assert store.read_trace("does-not-exist") is None


def test_change_record_round_trips(store: TraceStore) -> None:
    rec = ChangeRecord(
        change_id="chg-1",
        ts=datetime.now(UTC),
        artifact_type="model",
        version_before="m-v2",
        version_after="m-v3",
        motivating_trace_ids=["t1", "t2"],
        metric_before={"pass_rate": 0.7},
        metric_after={"pass_rate": 0.85},
        suite_size_before=100,
        suite_size_after=120,
    )
    store.write_change(rec)
    fetched = store.read_change("chg-1")
    assert fetched is not None
    assert fetched == rec


def test_cache_round_trips(store: TraceStore) -> None:
    store.cache_put("k", b"some-bytes")
    assert store.cache_get("k") == b"some-bytes"
    assert store.cache_get("missing") is None


def test_concurrent_writes_are_serialized(store: TraceStore) -> None:
    """~20 real threads write distinct traces simultaneously; all must land intact.

    Attacks: "the write lock serializes concurrent writers" - broken if the
    shared write connection ever sees interleaved statements ("database is
    locked", lost rows, or a torn commit).
    """
    n = 20
    errors: list[BaseException] = []
    start = threading.Barrier(n)

    def worker(i: int) -> None:
        try:
            start.wait()  # release all threads at once for maximal contention
            store.write_trace(_make_trace(f"trace-{i:03d}"))
        except BaseException as exc:  # noqa: BLE001 - collected for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent writes raised: {errors!r}"
    for i in range(n):
        tid = f"trace-{i:03d}"
        fetched = store.read_trace(tid)
        assert fetched is not None, f"missing {tid}"
        assert fetched.trace_id == tid

    with store._read_lock:
        count = store._read_conn.execute("SELECT COUNT(*) FROM traces;").fetchone()[0]
    assert count == n


def test_failed_write_raises_and_next_write_still_lands(store: TraceStore) -> None:
    """A write either commits or raises; it never wedges the store.

    Attacks: "a raising write cannot leave the store hung" - the pre-#73 writer
    thread had an unbounded ``done.wait()``, so a dead writer blocked every later
    caller forever. Inject a write that fails mid-transaction, then prove (a) the
    caller got the exception, (b) the partial row was rolled back, (c) a good
    write completes within a bounded wait, (d) the lock is not held.
    """

    class Boom(RuntimeError):
        pass

    def _partial_then_fail(conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO cache (key, value) VALUES ('half', x'00')")
        raise Boom

    with pytest.raises(Boom):
        store._submit(_partial_then_fail)
    with pytest.raises(sqlite3.OperationalError):
        store._submit(lambda conn: conn.execute("INSERT INTO no_such_table VALUES (1)"))
    assert store.cache_get("half") is None, "partial write was not rolled back"

    landed = threading.Event()

    def _good() -> None:
        store.cache_put("after", b"ok")
        landed.set()

    t = threading.Thread(target=_good, daemon=True)
    t.start()
    assert landed.wait(timeout=5), "write after a failed write hung"
    assert store.cache_get("after") == b"ok"
    assert not store._write_lock.locked()


def test_read_connection_has_busy_timeout(store: TraceStore) -> None:
    """Attacks: "both connections are configured alike" - pre-#73 the read
    connection skipped ``_configure`` and so had ``busy_timeout=0``."""
    with store._read_lock:
        timeout = store._read_conn.execute("PRAGMA busy_timeout;").fetchone()[0]
    assert timeout == store._busy_timeout_ms


def test_ingest_writes_while_build_reads_under_wal(tmp_path: Path) -> None:
    """A second handle on the same file (as an index build would open) iterates
    the catalog while a real thread ingests movies through ``write_movie``.

    Attacks: "WAL lets one process read the corpus while another writes it" -
    broken by "database is locked" on either side or by a reader seeing a torn
    row. Two ``TraceStore`` instances stand in for the two processes.
    """
    path = tmp_path / "corpus.sqlite"
    n = 200
    errors: list[BaseException] = []

    def _movie(i: int) -> MovieRecord:
        return MovieRecord(
            tmdb_id=i,
            title=f"Movie {i}",
            year=2000,
            genres=["Drama"],
            director="D",
            cast=["A"],
            plot="p",
            regions=["US"],
            vote_count=1,
        )

    def ingest() -> None:
        try:
            with TraceStore(path) as writer:
                for i in range(1, n + 1):
                    writer.write_movie(_movie(i))
        except BaseException as exc:  # noqa: BLE001 - collected for assertion
            errors.append(exc)

    with TraceStore(path) as reader:
        t = threading.Thread(target=ingest)
        t.start()
        seen: list[int] = []
        while t.is_alive():
            seen.append(len(reader.read_all_movies()))
        t.join(timeout=30)
        assert not t.is_alive(), "ingest hung"
        assert not errors, f"ingest raised: {errors!r}"
        assert reader.count_movies() == n
        assert seen == sorted(seen), "reader saw the catalog shrink mid-ingest"
