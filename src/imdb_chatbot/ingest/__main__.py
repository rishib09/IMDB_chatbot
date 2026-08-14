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

from ..store import RawArchive, TraceStore
from .tmdb import run_ingest

# Named region groups expand to ISO 3166-1 country codes so you can pull a whole
# region with one token, e.g. --regions KR EUROPE LATAM. Add a group here or pass
# raw country codes directly. TMDB /discover is capped at 500 pages (10k results)
# per query, so each country code yields at most ~10k movies (decision: default
# to that ceiling rather than building a year-partitioning crawler).
REGION_GROUPS: dict[str, list[str]] = {
    "EUROPE": ["GB", "FR", "DE", "IT", "ES"],
    "LATAM": ["MX", "BR", "AR", "CL", "CO"],
}


def expand_regions(regions: list[str]) -> list[str]:
    """Expand any group names into country codes, de-duplicated, order preserved."""
    out: list[str] = []
    for region in regions:
        key = region.upper()
        for code in REGION_GROUPS.get(key, [key]):
            if code not in out:
                out.append(code)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imdb_chatbot.ingest")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["US", "IN"],
        help="Country codes or group names (EUROPE, LATAM) to pull. Default: US IN.",
    )
    parser.add_argument(
        "--db",
        default="data/corpus.sqlite",
        help="Path to the SQLite store to write (default: data/corpus.sqlite).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help="TMDB discover pages per country (20 results/page; default 500 = ~10k, "
        "the TMDB discover ceiling).",
    )
    parser.add_argument(
        "--raw-db",
        default="data/raw_tmdb.sqlite",
        help="Raw TMDB payload archive, a SEPARATE build-time file that never "
        "ships (default: data/raw_tmdb.sqlite).",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip archiving raw payloads (the corpus record then has no source).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    regions = expand_regions(args.regions)
    store = TraceStore(args.db)
    archive = None if args.no_raw else RawArchive(args.raw_db)
    try:
        stats = run_ingest(
            store, regions=regions, max_pages=args.max_pages, archive=archive
        )
    finally:
        store.close()
        if archive is not None:
            archive.close()

    print(
        f"ingested={stats.ingested} skipped={stats.skipped} "
        f"rejected={stats.rejected} total_seen={len(stats.seen)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
