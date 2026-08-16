"""SQLite stores - turn traces / the change ledger, and the raw TMDB archive.

Two stores live here, on purpose kept in SEPARATE files:

- ``TraceStore`` -> ``data/corpus.sqlite``. Opened at runtime by the dashboard,
  so it ships with the deploy and must stay small.
- ``RawArchive`` -> ``data/raw_tmdb.sqlite``. Build-time only, hundreds of MB at
  full corpus size, no runtime consumer (ticket #78 / decision D0 #62).

Design constraints (from ticket #12):

- The database runs in WAL journal mode with a sane ``busy_timeout`` so readers
  never block the single writer and vice versa.
- ALL writes go through ONE write connection guarded by a ``threading.Lock``
  (ticket #73). Each public write method runs its statement and commits inside
  the lock, so writes are serialized and a write either commits or raises -
  there is no queue, no writer thread, and no unbounded wait.
- Reads use a separate, read-only connection (WAL allows concurrent reads).

Rich objects (``TurnTrace``, ``ChangeRecord``, ``MovieRecord``) are stored as
JSON via ``model_dump_json()`` in a ``data`` column, alongside a few extracted
columns (trace_id, ts, session_id, ...) for cheap querying/indexing. The catalog
additionally exposes every ``MovieRecord`` field as a generated column so a
human can read it without decoding JSON - see ``_ensure_movie_columns``.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self, get_args, get_origin

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

# -- catalog read surface (ticket #60) -----------------------------------------
#
# ``movies.data`` stays the single source of truth; every other ``MovieRecord``
# field is exposed as a VIRTUAL generated column so the catalog can be browsed,
# sorted and filtered in any DB viewer without decoding JSON. The columns are
# derived, never dual-written, so they cannot drift from the blob. The set is
# built from ``MovieRecord.model_fields``: a field added to the model becomes a
# column with no schema edit, on new *and* existing corpora alike.
#
# Requires SQLite >= 3.31 (generated columns; only VIRTUAL ones can be added by
# ALTER TABLE, which is what lets an existing corpus upgrade in place with no
# table rewrite). ``tests/test_store.py`` asserts the running version.

_MOVIE_STORED_COLUMNS = frozenset({"tmdb_id", "title", "year", "data"})


def _movie_column_ddl(field: str, ann: Any) -> str:
    """Column definition for one derived catalog field, chosen by annotation."""
    expr = f"json_extract(data, '$.{field}')"
    if get_origin(ann) is list:
        # A generated column may not use a subquery, so json_each() is out; join
        # the raw array text instead. '","' only ever separates two elements -
        # a quote inside one is escaped as \" - so the split cannot misfire.
        expr = (
            f"""CASE {expr} WHEN '[]' THEN '' ELSE replace(replace(replace("""
            f"""{expr}, '","', ' | '), '["', ''), '"]', '') END"""
        )
        sql_type = "TEXT"
    else:
        # Numeric affinity matters: under TEXT affinity `WHERE vote_count > 1000`
        # would compare strings and match every row.
        inner = [a for a in get_args(ann) if a is not type(None)] or [ann]
        sql_type = "INTEGER" if inner == [int] else "REAL" if inner == [float] else "TEXT"
    return f'"{field}" {sql_type} GENERATED ALWAYS AS ({expr}) VIRTUAL'


_MOVIE_DERIVED_COLUMNS = {
    name: _movie_column_ddl(name, f.annotation)
    for name, f in MovieRecord.model_fields.items()
    if name not in _MOVIE_STORED_COLUMNS
}


def _ensure_movie_columns(conn: sqlite3.Connection) -> None:
    """Add every missing derived catalog column; idempotent across opens.

    ``PRAGMA table_info`` does NOT list generated columns - only ``table_xinfo``
    does - so the existence check must use xinfo, or each open would re-ADD them
    and fail with "duplicate column name".
    """
    have = {row["name"] for row in conn.execute("PRAGMA table_xinfo(movies)")}
    for name, ddl in _MOVIE_DERIVED_COLUMNS.items():
        if name not in have:
            conn.execute(f"ALTER TABLE movies ADD COLUMN {ddl}")


class TraceStore:
    """A single-writer, WAL-mode SQLite store for traces and the change ledger."""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self.path = str(path)
        self._busy_timeout_ms = busy_timeout_ms
        self._closed = False
        # check_same_thread=False + a lock per connection lets any caller thread
        # read or write; WAL keeps readers off the writer's back.
        self._write_lock = threading.Lock()
        self._write_conn = self._connect()
        self._read_lock = threading.Lock()
        self._read_conn = self._connect()
        self.init_schema()

    # -- connection setup ------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms};")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _submit(self, fn: Callable[[sqlite3.Connection], None]) -> None:
        """Run ``fn`` on the write connection under the write lock and commit.

        Either commits or raises to the caller (after rollback); the lock is
        always released, so a failed write never blocks the next one.
        """
        if self._closed:
            raise RuntimeError("TraceStore is closed")
        with self._write_lock:
            try:
                fn(self._write_conn)
                self._write_conn.commit()
            except BaseException:
                self._write_conn.rollback()
                raise

    # -- schema ----------------------------------------------------------------

    def init_schema(self) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.executescript(_SCHEMA)
            _ensure_movie_columns(conn)

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

    def movie_ids(self) -> list[int]:
        """Every catalog ``tmdb_id``, ascending - without parsing 46k records."""
        with self._read_lock:
            rows = self._read_conn.execute(
                "SELECT tmdb_id FROM movies ORDER BY tmdb_id"
            ).fetchall()
        return [int(row["tmdb_id"]) for row in rows]

    def read_all_movies(self) -> list[MovieRecord]:
        """Return every catalog row, ordered by ``tmdb_id`` (read-only).

        Reads go through the read connection only, so this never needs the write
        lock: an ingest process may keep writing to the same corpus DB under WAL
        while an index build iterates the catalog here.
        """
        with self._read_lock:
            rows = self._read_conn.execute(
                "SELECT data FROM movies ORDER BY tmdb_id"
            ).fetchall()
        return [MovieRecord.model_validate_json(row["data"]) for row in rows]

    def iter_movies(self) -> Iterator[MovieRecord]:
        """Iterate every catalog row ordered by ``tmdb_id`` (read-only).

        Convenience wrapper over :meth:`read_all_movies`. The snapshot is taken
        eagerly under the read lock, then yielded, so iterating never holds the
        lock across caller work.
        """
        yield from self.read_all_movies()

    # -- rejects (quarantine) --------------------------------------------------

    def write_reject(self, reason: str, data: str) -> None:
        """Quarantine a payload that failed validation, with the reason text.

        ``data`` is the raw source payload serialized as a JSON string; it is
        never silently dropped so failures stay auditable.
        """
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
        """Close both connections."""
        if self._closed:
            return
        self._closed = True
        with self._write_lock:
            self._write_conn.close()
        with self._read_lock:
            self._read_conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


_RAW_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_movies (
    tmdb_id     INTEGER PRIMARY KEY,
    fetched_at  TEXT NOT NULL,
    append_spec TEXT NOT NULL,
    payload     BLOB NOT NULL
);
"""


class RawArchive:
    """Archive of raw TMDB payloads, gzipped JSON, one row per ``tmdb_id``.

    ``MovieRecord`` is a *derived* projection; this is what it is derived from.
    Any future schema question ("do we have producers? keywords?") becomes a
    re-derivation instead of a 46k re-fetch.

    Three properties do the work:

    - **Separate file.** ``data/corpus.sqlite`` is opened at runtime and ships;
      this archive is hundreds of MB at full corpus size and has no runtime
      consumer, so it never goes near the deploy artifact.
    - **``tmdb_id`` is the PRIMARY KEY**, so an interrupted backfill resumes by
      simply skipping ids already present - see :meth:`stored_ids`.
    - **``append_spec`` is stored per row**: the namespaces that were actually
      requested. Widening the fetch later re-fetches only the rows written under
      the older spec instead of paying another full sweep.

    Single-threaded by design (one build-time writer), so unlike ``TraceStore``
    it needs no write lock. Every :meth:`put` commits, which is exactly what
    makes a killed backfill resumable.
    """

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self.path = str(path)
        parent = Path(self.path).parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms};")
        self._conn.executescript(_RAW_SCHEMA)
        self._conn.commit()

    def put(
        self,
        tmdb_id: int,
        payload: Mapping[str, Any],
        *,
        append_spec: str,
        fetched_at: str | None = None,
    ) -> int:
        """Store one payload gzipped; returns the stored (compressed) byte count.

        Keys are sorted so the same payload always yields the same blob. JSON
        object order carries no meaning, and determinism makes the archive
        diffable.
        """
        blob = gzip.compress(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            compresslevel=6,
        )
        ts = fetched_at or datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO raw_movies "
            "(tmdb_id, fetched_at, append_spec, payload) VALUES (?, ?, ?, ?)",
            (int(tmdb_id), ts, append_spec, blob),
        )
        self._conn.commit()
        return len(blob)

    def get(self, tmdb_id: int) -> dict[str, Any] | None:
        """Return the stored payload, decompressed and parsed (None if absent)."""
        row = self._conn.execute(
            "SELECT payload FROM raw_movies WHERE tmdb_id = ?", (int(tmdb_id),)
        ).fetchone()
        if row is None:
            return None
        return json.loads(gzip.decompress(row["payload"]).decode("utf-8"))

    def spec_of(self, tmdb_id: int) -> str | None:
        """The ``append_to_response`` spec this row was fetched under."""
        row = self._conn.execute(
            "SELECT append_spec FROM raw_movies WHERE tmdb_id = ?", (int(tmdb_id),)
        ).fetchone()
        return None if row is None else str(row["append_spec"])

    def stored_ids(self, *, append_spec: str | None = None) -> set[int]:
        """Ids already archived - the backfill's resume set.

        Pass ``append_spec`` to count only rows fetched under that exact spec:
        a row written under an older, narrower spec is NOT done, and skipping it
        would freeze the stale payload in place forever.
        """
        if append_spec is None:
            rows = self._conn.execute("SELECT tmdb_id FROM raw_movies").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT tmdb_id FROM raw_movies WHERE append_spec = ?", (append_spec,)
            ).fetchall()
        return {int(row["tmdb_id"]) for row in rows}

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM raw_movies").fetchone()[0])

    def stored_bytes(self) -> int:
        """Total compressed payload bytes (excludes SQLite page overhead)."""
        row = self._conn.execute("SELECT SUM(LENGTH(payload)) FROM raw_movies").fetchone()
        return int(row[0] or 0)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
