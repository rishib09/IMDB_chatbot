"""TMDB -> MovieRecord ingestion (build-time).

This module has two strictly separated halves:

1. PURE TRANSFORM (no I/O, no network) - ``is_complete``, ``map_tmdb_movie``,
   ``transform_movie``. These turn a TMDB movie-details JSON dict (with
   ``credits`` and ``release_dates`` appended) into a validated ``MovieRecord``,
   or classify it as skipped / rejected. They are the only thing the unit tests
   exercise, against embedded fixtures - never the live API.

2. LIVE PULL (network) - ``TMDBClient`` and ``run_ingest``. These are the ONLY
   things that touch ``httpx`` / the TMDB API. Tests never call them; they are
   invoked from ``python -m imdb_chatbot.ingest`` at build time.

Region adapter pattern: certificate normalization is delegated per-region to
YAML maps under ``config/certificates/`` via ``certificates.py``. Adding a
region (say Japan) is a new ``jp.yaml`` plus one registry entry - NOT a change
to the mapping code below. The core transform never learns a rating system.

Corpus rules:
- Validation failure (cannot build a ``MovieRecord``) -> quarantined to the
  ``rejects`` table with the ValidationError text. Never silently dropped.
- Field-completeness bar (plot AND poster AND >=1 genre AND >=1 cast AND a
  year) - a valid TMDB row that merely lacks corpus fields is SKIPPED (counted,
  not quarantined).
- Dedup + upsert by ``tmdb_id`` so re-runs are idempotent and a movie returned
  by two region pulls lands once (a co-production carrying e.g. ["US","IN"]).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from pydantic import ValidationError

from ..config import get_secret
from ..schemas import MovieRecord
from .certificates import CertificateMap, load_certificate_map

if TYPE_CHECKING:
    import httpx

    from ..store import TraceStore

logger = logging.getLogger(__name__)

IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
API_BASE = "https://api.themoviedb.org/3"


# ---------------------------------------------------------------------------
# PURE TRANSFORM (no I/O) - unit-tested against fixtures
# ---------------------------------------------------------------------------


def _parse_year(details: dict[str, Any]) -> int | None:
    """Extract a 4-digit release year from ``release_date`` ("YYYY-MM-DD")."""
    raw = details.get("release_date")
    if not raw or not isinstance(raw, str) or len(raw) < 4:
        return None
    head = raw[:4]
    if not head.isdigit():
        return None
    return int(head)


def _regions(details: dict[str, Any]) -> list[str]:
    """Origin countries, from ``origin_country`` (preferred) or production_countries."""
    origin = details.get("origin_country")
    if isinstance(origin, list) and origin:
        return [str(c).upper() for c in origin if c]
    prod = details.get("production_countries")
    if isinstance(prod, list) and prod:
        return [str(c.get("iso_3166_1", "")).upper() for c in prod if c.get("iso_3166_1")]
    return []


def _cast_names(details: dict[str, Any]) -> list[str]:
    cast = details.get("credits", {}).get("cast", []) or []
    return [c["name"] for c in cast if c.get("name")]


def _director(details: dict[str, Any]) -> str | None:
    crew = details.get("credits", {}).get("crew", []) or []
    for member in crew:
        if member.get("job") == "Director" and member.get("name"):
            return member["name"]
    return None


def _genre_names(details: dict[str, Any]) -> list[str]:
    return [g["name"] for g in details.get("genres", []) or [] if g.get("name")]


def _poster_url(details: dict[str, Any]) -> str | None:
    path = details.get("poster_path")
    if not path:
        return None
    return f"{IMAGE_BASE}{path}"


def _raw_certificate_for_region(details: dict[str, Any], region: str) -> str | None:
    """Pull a raw certification string for ``region`` from ``release_dates``."""
    results = details.get("release_dates", {}).get("results", []) or []
    for entry in results:
        if str(entry.get("iso_3166_1", "")).upper() != region.upper():
            continue
        for rd in entry.get("release_dates", []) or []:
            cert = (rd.get("certification") or "").strip()
            if cert:
                return cert
    return None


def _select_certificate(
    details: dict[str, Any],
    regions: list[str],
    preferred_region: str | None,
) -> tuple[CertificateMap | None, str | None]:
    """Choose which registered region's certificate to normalize against.

    Preference order: the pull region first (if given), then the movie's origin
    countries. The first registered region that actually carries a certificate
    wins. If none carry one, we still report the system of the first registered
    region (so ``certificate_system`` is populated, ``certificate_raw`` = None).
    """
    order: list[str] = []
    if preferred_region:
        order.append(preferred_region.upper())
    for r in regions:
        if r.upper() not in order:
            order.append(r.upper())

    fallback_map: CertificateMap | None = None
    for region in order:
        cmap = load_certificate_map(region)
        if cmap is None:
            continue
        if fallback_map is None:
            fallback_map = cmap
        raw = _raw_certificate_for_region(details, region)
        if raw:
            return cmap, raw
    return fallback_map, None


def is_complete(details: dict[str, Any]) -> bool:
    """Field-completeness bar: plot AND poster AND >=1 genre AND >=1 cast AND year.

    Evaluated on the raw payload so that a valid-but-thin TMDB row is SKIPPED
    rather than quarantined. ``year`` is checked here because it is the one
    required schema field; a row lacking it is a skip, not a validation reject.
    """
    has_plot = bool((details.get("overview") or "").strip())
    has_poster = bool(details.get("poster_path"))
    has_genre = len(_genre_names(details)) >= 1
    has_cast = len(_cast_names(details)) >= 1
    has_year = _parse_year(details) is not None
    return has_plot and has_poster and has_genre and has_cast and has_year


def map_tmdb_movie(
    details: dict[str, Any],
    *,
    preferred_region: str | None = None,
) -> MovieRecord:
    """Map a TMDB movie-details JSON dict -> ``MovieRecord`` (may raise).

    Raises ``pydantic.ValidationError`` if the payload cannot form a valid
    record; the caller quarantines such rows to ``rejects``.
    """
    regions = _regions(details)
    cmap, cert_raw = _select_certificate(details, regions, preferred_region)
    cert_norm = cmap.normalize(cert_raw) if cmap is not None else None
    cert_system = cmap.system if cmap is not None else None

    return MovieRecord(
        tmdb_id=details["id"],
        title=details.get("title") or details.get("original_title"),
        year=_parse_year(details),
        genres=_genre_names(details),
        director=_director(details),
        cast=_cast_names(details),
        plot=details.get("overview") or None,
        rating_raw=details.get("vote_average"),
        metascore=None,
        certificate_raw=cert_raw,
        certificate_norm=cert_norm,
        certificate_system=cert_system,
        regions=regions,
        duration_min=details.get("runtime"),
        poster_url=_poster_url(details),
    )


@dataclass
class Outcome:
    """Verdict of running one TMDB payload through the pure pipeline."""

    verdict: str  # "ingested" | "skipped" | "rejected"
    record: MovieRecord | None = None
    reason: str | None = None  # ValidationError text when rejected


def transform_movie(
    details: dict[str, Any],
    *,
    preferred_region: str | None = None,
) -> Outcome:
    """Pure classify-and-map: skip (thin), reject (invalid), or ingest (valid).

    No network, no store - the unit tests drive this directly with fixtures.
    """
    if not is_complete(details):
        return Outcome("skipped")
    try:
        record = map_tmdb_movie(details, preferred_region=preferred_region)
    except ValidationError as exc:
        return Outcome("rejected", reason=str(exc))
    return Outcome("ingested", record=record)


# ---------------------------------------------------------------------------
# LIVE PULL (network) - never exercised by the unit tests
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    ingested: int = 0
    skipped: int = 0
    rejected: int = 0
    seen: set[int] = field(default_factory=set)


class TMDBClient:
    """Thin httpx wrapper over the TMDB REST API (the only networked piece).

    Auth prefers the v4 read access token (Bearer) and falls back to the v3
    ``api_key`` query param. Secrets are resolved by name via ``config`` and
    never logged.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        import httpx

        self._token = token
        self._api_key = api_key
        if token is None and api_key is None:
            raise RuntimeError(
                "TMDBClient needs a TMDB_READ_ACCESS_TOKEN or TMDB_API_KEY."
            )
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = client or httpx.Client(
            base_url=API_BASE, headers=headers, timeout=timeout
        )

    @classmethod
    def from_env(cls) -> TMDBClient:
        token = get_secret("TMDB_READ_ACCESS_TOKEN", required=False)
        api_key = get_secret("TMDB_API_KEY", required=False)
        if not token and not api_key:
            raise RuntimeError(
                "Set TMDB_READ_ACCESS_TOKEN or TMDB_API_KEY to run the live pull."
            )
        return cls(token=token or None, api_key=api_key or None)

    def _params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = dict(extra or {})
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    def discover_ids(self, region: str, *, max_pages: int = 1) -> list[int]:
        """Discover movie ids with ``with_origin_country=<region>``."""
        ids: list[int] = []
        for page in range(1, max_pages + 1):
            resp = self._client.get(
                "/discover/movie",
                params=self._params(
                    {
                        "with_origin_country": region.upper(),
                        "page": page,
                        "sort_by": "popularity.desc",
                    }
                ),
            )
            resp.raise_for_status()
            body = resp.json()
            for row in body.get("results", []):
                if "id" in row:
                    ids.append(int(row["id"]))
            if page >= body.get("total_pages", page):
                break
        return ids

    def fetch_details(self, tmdb_id: int) -> dict[str, Any]:
        """Fetch full details with credits + release_dates appended."""
        resp = self._client.get(
            f"/movie/{tmdb_id}",
            params=self._params({"append_to_response": "credits,release_dates"}),
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def run_ingest(
    store: TraceStore,
    *,
    regions: Iterable[str],
    max_pages: int = 1,
    client: TMDBClient | None = None,
) -> IngestStats:
    """Pull each region, transform, and persist - idempotent (upsert on tmdb_id).

    This is the networked orchestration. It reuses the pure ``transform_movie``
    for every row so the classification logic stays identically testable.
    """
    owns_client = client is None
    client = client or TMDBClient.from_env()
    stats = IngestStats()
    try:
        for region in regions:
            region = region.upper()
            ids = client.discover_ids(region, max_pages=max_pages)
            logger.info("region %s: discovered %d ids", region, len(ids))
            for tmdb_id in ids:
                if tmdb_id in stats.seen:
                    continue  # dedup across region pulls
                stats.seen.add(tmdb_id)
                details = client.fetch_details(tmdb_id)
                outcome = transform_movie(details, preferred_region=region)
                if outcome.verdict == "skipped":
                    stats.skipped += 1
                elif outcome.verdict == "rejected":
                    store.write_reject(outcome.reason or "validation error", json.dumps(details))
                    stats.rejected += 1
                else:
                    assert outcome.record is not None
                    store.write_movie(outcome.record)
                    stats.ingested += 1
    finally:
        if owns_client:
            client.close()
    logger.info(
        "ingest done: ingested=%d skipped=%d rejected=%d",
        stats.ingested,
        stats.skipped,
        stats.rejected,
    )
    return stats
