"""Backfill the raw TMDB archive for every movie already in the corpus (#78).

The corpus was built before we archived anything, so its ~46k ``MovieRecord``
rows have no source payload behind them. This walks every ``tmdb_id`` in
``data/corpus.sqlite`` and fetches + stores the raw payload for any id not
already archived under the current ``APPEND_TO_RESPONSE`` spec.

Resumability is the archive's primary key, not bookkeeping here: a row lands
committed or not at all, and a re-run recomputes the to-do list as
``corpus ids - archived ids``. Kill it at movie 30,000 and the next run starts
at 30,000. A movie whose fetch fails is simply never stored, so the same
mechanism retries it next run - failures are logged individually and counted,
never silently dropped.

Usage (needs a TMDB credential, so run it under dotenvx):

    npx @dotenvx/dotenvx run -f .env -- .venv/Scripts/python.exe \
        scripts/backfill_raw_tmdb.py --limit 20      # validation slice
    npx @dotenvx/dotenvx run -f .env -- .venv/Scripts/python.exe \
        scripts/backfill_raw_tmdb.py                 # the full 46k run

Exit code is 1 if any movie failed after its retries, so an unattended run
cannot report success while leaving holes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import httpx

from imdb_chatbot.ingest.tmdb import APPEND_TO_RESPONSE, TMDBClient
from imdb_chatbot.store import RawArchive, TraceStore

logger = logging.getLogger("backfill_raw_tmdb")

# 404 means TMDB deleted or merged the movie since our ingest. Retrying cannot
# help, so it is reported as attrition rather than as a failure.
_GONE = object()


def _fetch_with_retry(
    client: TMDBClient,
    tmdb_id: int,
    *,
    retries: int,
    backoff: float,
) -> dict | object:
    """Fetch one payload, retrying transient failures with exponential backoff.

    Returns the payload, or ``_GONE`` for a 404. Raises on an unrecoverable
    error (bad credentials, or retries exhausted) so the caller decides whether
    to abort the run or record a per-movie failure.
    """
    delay = backoff
    for attempt in range(1, retries + 1):
        try:
            return client.fetch_details(tmdb_id)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 404:
                return _GONE
            if status in (401, 403):
                raise  # credentials: abort the whole run, do not hammer 46k times
            if status != 429 and status < 500:
                raise RuntimeError(
                    f"tmdb_id={tmdb_id}: HTTP {status} is not retryable"
                ) from exc
            wait = delay
            if status == 429:
                retry_after = exc.response.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    wait = float(retry_after)
            logger.warning(
                "tmdb_id=%d HTTP %d (attempt %d/%d) - retrying in %.1fs",
                tmdb_id,
                status,
                attempt,
                retries,
                wait,
            )
        except httpx.RequestError as exc:
            wait = delay
            logger.warning(
                "tmdb_id=%d %s (attempt %d/%d) - retrying in %.1fs",
                tmdb_id,
                type(exc).__name__,
                attempt,
                retries,
                wait,
            )
        if attempt == retries:
            break
        time.sleep(wait)
        delay *= 2
    raise RuntimeError(f"tmdb_id={tmdb_id}: giving up after {retries} attempts")


def _human_eta(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backfill_raw_tmdb")
    parser.add_argument("--db", default="data/corpus.sqlite", help="Corpus to read ids from.")
    parser.add_argument(
        "--raw-db", default="data/raw_tmdb.sqlite", help="Raw archive to write."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N movies (0 = all). Use a small slice to validate first.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.1, help="Seconds between requests (be polite)."
    )
    parser.add_argument("--retries", type=int, default=4, help="Attempts per movie.")
    parser.add_argument(
        "--backoff", type=float, default=2.0, help="Initial retry backoff, seconds."
    )
    parser.add_argument("--progress-every", type=int, default=250, help="Log every N movies.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    store = TraceStore(args.db)
    try:
        corpus_ids = store.movie_ids()
    finally:
        store.close()

    archive = RawArchive(args.raw_db)
    already = archive.stored_ids(append_spec=APPEND_TO_RESPONSE)
    todo = [tmdb_id for tmdb_id in corpus_ids if tmdb_id not in already]
    if args.limit > 0:
        todo = todo[: args.limit]

    logger.info(
        "corpus=%d archived=%d todo=%d spec=%r",
        len(corpus_ids),
        len(already),
        len(todo),
        APPEND_TO_RESPONSE,
    )

    stored = gone = 0
    failures: list[int] = []
    stored_bytes = 0
    started = time.monotonic()

    client = TMDBClient.from_env()
    try:
        for done, tmdb_id in enumerate(todo, start=1):
            try:
                payload = _fetch_with_retry(
                    client, tmdb_id, retries=args.retries, backoff=args.backoff
                )
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "aborting: TMDB rejected the credential (HTTP %d)",
                    exc.response.status_code,
                )
                break
            except RuntimeError as exc:
                # Not stored -> the next run picks this id up again.
                logger.error("FAILED %s", exc)
                failures.append(tmdb_id)
                continue

            if payload is _GONE:
                logger.warning("tmdb_id=%d gone from TMDB (404) - left unarchived", tmdb_id)
                gone += 1
            else:
                assert isinstance(payload, dict)
                stored_bytes += archive.put(
                    tmdb_id, payload, append_spec=APPEND_TO_RESPONSE
                )
                stored += 1

            if done % args.progress_every == 0 or done == len(todo):
                elapsed = time.monotonic() - started
                rate = done / elapsed if elapsed else 0.0
                remaining = (len(todo) - done) / rate if rate else 0.0
                logger.info(
                    "%d/%d (%.1f%%) | %.1f movies/s | avg %.1f KB gz | eta %s",
                    done,
                    len(todo),
                    100.0 * done / len(todo),
                    rate,
                    (stored_bytes / stored / 1024) if stored else 0.0,
                    _human_eta(remaining),
                )
            if args.sleep:
                time.sleep(args.sleep)
    finally:
        client.close()
        total_rows = archive.count()
        total_bytes = archive.stored_bytes()
        archive.close()

    on_disk = Path(args.raw_db).stat().st_size
    print(
        f"stored={stored} gone={gone} failed={len(failures)} "
        f"archive_rows={total_rows} "
        f"avg_gz_bytes={(total_bytes // total_rows) if total_rows else 0} "
        f"payload_bytes={total_bytes} on_disk_bytes={on_disk}"
    )
    if failures:
        head = ", ".join(str(i) for i in failures[:50])
        suffix = ", ..." if len(failures) > 50 else ""
        print(f"FAILED ids ({len(failures)}) - re-run to retry them: {head}{suffix}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
