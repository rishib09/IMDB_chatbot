"""CLI entry: ``python -m imdb_chatbot.ingest`` runs the live TMDB pull.

This is build-time only and requires a TMDB credential in the environment
(TMDB_READ_ACCESS_TOKEN or TMDB_API_KEY). It writes the validated corpus into a
SQLite ``TraceStore``; adding a region is ``--regions US IN JP`` plus that
region's YAML - no code change.
"""

from __future__ import annotations

import argparse
import logging
import sys

from ..store import TraceStore
from .tmdb import run_ingest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imdb_chatbot.ingest")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["US", "IN"],
        help="ISO 3166-1 alpha-2 origin-country codes to pull (default: US IN).",
    )
    parser.add_argument(
        "--db",
        default="data/corpus.sqlite",
        help="Path to the SQLite store to write (default: data/corpus.sqlite).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="TMDB discover pages to pull per region (default: 1).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    store = TraceStore(args.db)
    try:
        stats = run_ingest(store, regions=args.regions, max_pages=args.max_pages)
    finally:
        store.close()

    print(
        f"ingested={stats.ingested} skipped={stats.skipped} "
        f"rejected={stats.rejected} total_seen={len(stats.seen)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
