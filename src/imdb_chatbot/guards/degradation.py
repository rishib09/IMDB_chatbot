"""The degradation ladder L1-L3 (section 5.5): degrade toward determinism.

- L1 = component fallbacks. These already exist as graph edges (generator ->
  fallback model, extractor -> regex, rewriter -> skip); here we only expose a
  tiny helper to record the ``degradation`` flag consistently.
- L2 = LLM-free mode. ``llm_free_answer`` runs ONLY the deterministic
  ``HybridRetriever`` over the live index and renders metadata cards
  (title/year/genre/rating/poster) under a fixed banner, with NO generated
  prose. Triggered when OpenRouter is down, the daily budget is exhausted, or
  every model slot is failing. Emits a ``degradation:L2`` flag.
- L3 = honest exit. ``health_check`` is a startup check (pointer resolves ->
  index loads -> artifacts exist -> DB opens) that REFUSES to serve rather than
  serve broken. It returns a clear state instead of raising an unhandled error.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..schemas import MovieRecommendation, RecommendationSet, ScoredMovie

# Canonical ladder flags (appended to ``TurnState.degradation`` / ``TurnTrace``).
L1_FLAG = "degradation:L1"
L2_FLAG = "degradation:L2"
L3_FLAG = "degradation:L3"

L2_BANNER = "Chat is temporarily limited - here is what matched your search"

HONEST_EXIT_MESSAGE = (
    "The movie index is unavailable right now, so I can't search reliably. "
    "Rather than show you something broken, I'm stepping aside - please try "
    "again shortly."
)


def honest_exit_message() -> str:
    """The plain, honest L3 message shown when the app refuses to serve."""
    return HONEST_EXIT_MESSAGE


# -- L1: record a component-fallback flag --------------------------------------


def record_degradation(flags: Iterable[str], flag: str) -> list[str]:
    """Append ``flag`` to a degradation list without duplicating it (pure)."""
    out = list(flags)
    if flag not in out:
        out.append(flag)
    return out


# -- L2: LLM-free deterministic retrieval mode ---------------------------------


class SupportsRetrieve(Protocol):
    """The retriever seam L2 depends on: deterministic, LLM-free ``retrieve``."""

    def retrieve(self, rewritten_query: str, parsed: Any = ..., shown_movies: Any = ...) -> Sequence[ScoredMovie]:
        ...


@dataclass
class MetadataCard:
    """A single deterministic result card - metadata only, no generated prose."""

    tmdb_id: int
    title: str
    year: int
    genres: list[str] = field(default_factory=list)
    rating: float | None = None
    poster_url: str | None = None


@dataclass
class L2Result:
    """The LLM-free answer: a banner, metadata cards, and a prose-free set."""

    banner: str
    cards: list[MetadataCard]
    recommendations: RecommendationSet
    flags: list[str] = field(default_factory=list)


def should_enter_l2(
    *,
    openrouter_down: bool = False,
    budget_exhausted: bool = False,
    all_slots_failing: bool = False,
) -> bool:
    """L2 trigger: OpenRouter down OR daily budget exhausted OR all slots failing."""
    return bool(openrouter_down or budget_exhausted or all_slots_failing)


def _enrich_card(candidate: ScoredMovie, movies_by_id: dict[int, Any]) -> MetadataCard:
    """Build a metadata card from a candidate, enriched from the movie map if present."""
    movie = movies_by_id.get(candidate.tmdb_id)
    genres: list[str] = []
    rating: float | None = None
    poster: str | None = None
    if movie is not None:
        genres = list(getattr(movie, "genres", []) or [])
        rating = getattr(movie, "rating_raw", None)
        poster_url = getattr(movie, "poster_url", None)
        poster = str(poster_url) if poster_url is not None else None
    return MetadataCard(
        tmdb_id=candidate.tmdb_id,
        title=candidate.title,
        year=candidate.year,
        genres=genres,
        rating=rating,
        poster_url=poster,
    )


def llm_free_answer(
    query: str,
    retriever: SupportsRetrieve,
    *,
    parsed: Any = None,
    shown_movies: Iterable[int] | None = None,
) -> L2Result:
    """Answer WITHOUT any LLM: deterministic retrieval rendered as metadata cards.

    Runs the injected ``HybridRetriever`` (or any object with a compatible
    ``retrieve``) over the live index, then renders each candidate as a
    metadata-only card under the fixed ``L2_BANNER``. No prose is generated. The
    returned ``RecommendationSet`` carries the same picks with EMPTY reasons so
    downstream code that expects that shape still works, and a ``degradation:L2``
    flag is attached.
    """
    candidates = list(retriever.retrieve(query, parsed, shown_movies))
    movies_by_id = getattr(retriever, "movies_by_id", {}) or {}

    cards = [_enrich_card(c, movies_by_id) for c in candidates]
    picks = [
        MovieRecommendation(
            title=card.title,
            year=card.year,
            reason="",  # LLM-free: metadata only, no generated prose.
            poster_url=card.poster_url,
        )
        for card in cards
    ]
    recommendations = RecommendationSet(picks=picks, prose="")
    return L2Result(banner=L2_BANNER, cards=cards, recommendations=recommendations, flags=[L2_FLAG])


# -- L3: startup health check + honest exit ------------------------------------


@dataclass
class HealthResult:
    """Startup health verdict. ``healthy`` False -> refuse to serve (L3)."""

    healthy: bool
    message: str
    checks: dict[str, bool] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def _index_dir_ok(index_dir: Path) -> bool:
    """The three index artifacts must exist and be non-empty on disk."""
    for name in ("index.faiss", "bm25.pkl", "sidecar.json"):
        artifact = index_dir / name
        try:
            if not artifact.is_file() or artifact.stat().st_size == 0:
                return False
        except OSError:
            return False
    return True


def health_check(
    *,
    live_index_path: Path | str,
    db_opener: Any = None,
) -> HealthResult:
    """Startup gate: pointer resolves -> index artifacts exist -> DB opens.

    Any failure returns an unhealthy ``HealthResult`` with a clear message - it
    NEVER raises an unhandled error. ``db_opener`` is an optional zero-arg
    callable that opens (and should close) the trace DB; when omitted the DB
    check is skipped (index availability is the load-bearing L3 gate).
    """
    checks: dict[str, bool] = {"pointer": False, "index": False, "db": False}
    live_index_path = Path(live_index_path)

    # 1. Pointer file resolves and names an active version + path.
    try:
        pointer = json.loads(live_index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return HealthResult(
            healthy=False,
            message=HONEST_EXIT_MESSAGE,
            checks=checks,
            flags=[L3_FLAG],
        )

    active = pointer.get("active")
    index_path = pointer.get("path")
    if not active or not index_path:
        return HealthResult(
            healthy=False,
            message=HONEST_EXIT_MESSAGE,
            checks=checks,
            flags=[L3_FLAG],
        )
    checks["pointer"] = True

    # 2. Index artifacts exist and are non-empty (the checksum/exists check).
    if not _index_dir_ok(Path(index_path)):
        return HealthResult(
            healthy=False,
            message=HONEST_EXIT_MESSAGE,
            checks=checks,
            flags=[L3_FLAG],
        )
    checks["index"] = True

    # 3. DB opens (optional). A raised opener means broken storage -> refuse.
    if db_opener is not None:
        try:
            handle = db_opener()
            close = getattr(handle, "close", None)
            if callable(close):
                close()
        except Exception:  # noqa: BLE001 - broken DB must degrade, not crash startup
            return HealthResult(
                healthy=False,
                message=HONEST_EXIT_MESSAGE,
                checks=checks,
                flags=[L3_FLAG],
            )
    checks["db"] = True

    return HealthResult(healthy=True, message="ok", checks=checks)
