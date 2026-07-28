"""SQLite trace store - the system-of-record for turn traces and the change ledger.

Design constraints (from ticket #12):

- The database runs in WAL journal mode with a sane ``busy_timeout`` so readers
  never block the single writer and vice versa.
- ALL writes are funneled through ONE dedicated writer thread that owns its own
  connection. Public write methods enqueue a job and block until the writer has
  committed it, so callers get deterministic completion. This serializes every
  write and sidesteps SQLite's "database is locked" failure mode under
  concurrency without sprinkling retries everywhere.
- Reads use a separate, read-only connection (WAL allows concurrent reads).

Rich objects (``TurnTrace``, ``ChangeRecord``) are stored as JSON via
``model_dump_json()`` in a ``data`` column, alongside a few extracted columns
(trace_id, ts, session_id, ...) for cheap querying/indexing.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from .schemas import ChangeRecord, MovieRecord, TurnTrace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    tmdb_id INTEGER PRIMARY KEY,
    title   TEXT,
    year    INTEGER,
    data    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rejects (
    reject_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT,
    reason     TEXT,
    data       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id   TEXT PRIMARY KEY,
    ts         TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id    TEXT,
    data       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS change_ledger (
    change_id     TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    data          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cache (
    key   TEXT PRIMARY KEY,
    value BLOB
);

CREATE INDEX IF NOT EXISTS idx_traces_session ON traces (session_id);
CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces (ts);
"""


@dataclass
class _WriteJob:
    """A unit of work handed to the writer thread.

    ``fn`` receives the writer's owned connection and performs the mutation.
    ``done`` is set once the work has committed (or raised); ``error`` carries
    any exception back to the caller so it can be re-raised there.
    """

    fn: Callable[[sqlite3.Connection], None]
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


_STOP = object()  # sentinel enqueued by close() to stop the writer loop


class TraceStore:
    """A single-writer, WAL-mode SQLite store for traces and the change ledger."""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self.path = str(path)
        self._busy_timeout_ms = busy_timeout_ms
        self._queue: queue.Queue[Any] = queue.Queue()
        self._closed = False
        self._read_lock = threading.Lock()

        # Dedicated writer thread owns its own connection for its whole life.
        self._writer = threading.Thread(
            target=self._writer_loop, name="tracestore-writer", daemon=True
        )
        self._writer.start()

        # Separate read connection. check_same_thread=False + a lock lets reads
        # come from any caller thread while WAL keeps them off the writer's back.
        self._read_conn = sqlite3.connect(self.path, check_same_thread=False)
        self._read_conn.row_factory = sqlite3.Row

        self.init_schema()

    # -- connection setup ------------------------------------------------------

    def _configure(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms};")
        conn.execute("PRAGMA foreign_keys=ON;")

    # -- writer thread ---------------------------------------------------------

    def _writer_loop(self) -> None:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        self._configure(conn)
        try:
            while True:
                job = self._queue.get()
                if job is _STOP:
                    break
                try:
                    job.fn(conn)
                    conn.commit()
                except BaseException as exc:  # noqa: BLE001 - relayed to caller
                    try:
                        conn.rollback()
                    except sqlite3.Error:
                        pass
                    job.error = exc
                finally:
                    job.done.set()
        finally:
            conn.close()

    def _submit(self, fn: Callable[[sqlite3.Connection], None]) -> None:
        """Enqueue a write and block until the writer thread has committed it."""
        if self._closed:
            raise RuntimeError("TraceStore is closed")
        job = _WriteJob(fn=fn)
        self._queue.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error

    # -- schema ----------------------------------------------------------------

    def init_schema(self) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.executescript(_SCHEMA)

        self._submit(_do)

    # -- traces ----------------------------------------------------------------

    def write_trace(self, trace: TurnTrace) -> None:
        payload = trace.model_dump_json()
        ts = trace.ts.isoformat()

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO traces "
                "(trace_id, ts, session_id, user_id, data) VALUES (?, ?, ?, ?, ?)",
                (trace.trace_id, ts, trace.session_id, trace.user_id, payload),
            )

        self._submit(_do)

    def read_trace(self, trace_id: str) -> TurnTrace | None:
        with self._read_lock:
            row = self._read_conn.execute(
                "SELECT data FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        if row is None:
            return None
        return TurnTrace.model_validate_json(row["data"])

    # -- movies (catalog) ------------------------------------------------------

    def write_movie(self, movie: MovieRecord) -> None:
        """Upsert a validated catalog row, keyed by ``tmdb_id`` (idempotent)."""
        payload = movie.model_dump_json()

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO movies "
                "(tmdb_id, title, year, data) VALUES (?, ?, ?, ?)",
                (movie.tmdb_id, movie.title, movie.year, payload),
            )

        self._submit(_do)

    def read_movie(self, tmdb_id: int) -> MovieRecord | None:
        with self._read_lock:
            row = self._read_conn.execute(
                "SELECT data FROM movies WHERE tmdb_id = ?", (tmdb_id,)
            ).fetchone()
        if row is None:
            return None
        return MovieRecord.model_validate_json(row["data"])

    def count_movies(self) -> int:
        with self._read_lock:
            return self._read_conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]

    # -- rejects (quarantine) --------------------------------------------------

    def write_reject(self, reason: str, data: str) -> None:
        """Quarantine a payload that failed validation, with the reason text.

        ``data`` is the raw source payload serialized as a JSON string; it is
        never silently dropped so failures stay auditable.
        """
        from datetime import UTC, datetime

        ts = datetime.now(UTC).isoformat()

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO rejects (ts, reason, data) VALUES (?, ?, ?)",
                (ts, reason, data),
            )

        self._submit(_do)

    def count_rejects(self) -> int:
        with self._read_lock:
            return self._read_conn.execute("SELECT COUNT(*) FROM rejects").fetchone()[0]

    # -- change ledger ---------------------------------------------------------

    def write_change(self, rec: ChangeRecord) -> None:
        payload = rec.model_dump_json()
        ts = rec.ts.isoformat()

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO change_ledger "
                "(change_id, ts, artifact_type, data) VALUES (?, ?, ?, ?)",
                (rec.change_id, ts, rec.artifact_type, payload),
            )

        self._submit(_do)

    def read_change(self, change_id: str) -> ChangeRecord | None:
        with self._read_lock:
            row = self._read_conn.execute(
                "SELECT data FROM change_ledger WHERE change_id = ?", (change_id,)
            ).fetchone()
        if row is None:
            return None
        return ChangeRecord.model_validate_json(row["data"])

    # -- generic cache ---------------------------------------------------------

    def cache_put(self, key: str, value: bytes) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, value),
            )

        self._submit(_do)

    def cache_get(self, key: str) -> bytes | None:
        with self._read_lock:
            row = self._read_conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return row["value"]

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Stop the writer thread and close all connections."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._writer.join(timeout=10)
        with self._read_lock:
            self._read_conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
