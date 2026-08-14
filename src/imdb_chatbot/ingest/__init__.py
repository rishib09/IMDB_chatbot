"""Build-time ingestion package: TMDB -> validated, deduped ``MovieRecord`` corpus.

Public surface:

- Pure transform (no network, unit-tested): ``is_complete``, ``map_tmdb_movie``,
  ``transform_movie``, ``Outcome``.
- Live pull (network, build-time only): ``TMDBClient``, ``run_ingest``.
- Raw archive replay (no network): ``rederive_record``, ``APPEND_TO_RESPONSE``.
- Region certificate adapters: ``certificates`` submodule.

Run the live pull with ``python -m imdb_chatbot.ingest`` (see ``__main__``).
"""

from __future__ import annotations

from .tmdb import (
    APPEND_TO_RESPONSE,
    IngestStats,
    Outcome,
    TMDBClient,
    is_complete,
    map_tmdb_movie,
    rederive_record,
    run_ingest,
    transform_movie,
)

__all__ = [
    "APPEND_TO_RESPONSE",
    "IngestStats",
    "Outcome",
    "TMDBClient",
    "is_complete",
    "map_tmdb_movie",
    "rederive_record",
    "run_ingest",
    "transform_movie",
]
