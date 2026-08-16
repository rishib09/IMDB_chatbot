"""Typed data contracts for the IMDb RAG harness.

These Pydantic v2 models are the shared vocabulary across every ticket:

- ``MovieRecord``          - a normalized catalog row (system of record for a film).
- ``ParsedQuery``          - structured intent extracted from a raw user query.
- ``ScoredMovie``          - a retrieval candidate with dense/sparse/RRF scores.
- ``MovieRecommendation``  / ``RecommendationSet`` - the generated answer.
- ``TurnState``            - mutable state that flows through the LangGraph nodes.
- ``TurnTrace``            - the immutable, serialized record of one turn (system of record).
- ``ChangeRecord``         - one row in the change ledger (prompt/model/index/... edits).

Field names are load-bearing: downstream tickets depend on them exactly as written.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class MovieRecord(BaseModel):
    tmdb_id: int  # identity key
    title: str
    year: int = Field(ge=1888, le=2030)
    genres: list[str] = []
    director: str | None = None
    cast: list[str] = []
    plot: str | None = None
    original_language: str | None = None  # ISO 639-1, e.g. "en", "ko", "hi"
    rating_raw: float | None = Field(default=None, ge=0, le=10)
    rating_z: dict[str, float] = {}  # per-region z-score {region: z}
    vote_count: int | None = None  # TMDB vote count (popularity/quality gate, decision #4)
    certificate_raw: str | None = None
    certificate_norm: str | None = None  # {ALL, TEEN, MATURE, ADULT} or None
    certificate_system: str | None = None  # e.g. "MPAA", "CBFC"
    regions: list[str] = []  # origin countries, e.g. ["US","IN"]
    duration_min: float | None = None
    poster_url: HttpUrl | None = None
    status: str | None = None  # TMDB release status, e.g. "Released"
    budget: int | None = None
    revenue: int | None = None
    production_companies: list[str] = []  # company names


class ParsedQuery(BaseModel):
    genres: list[str] = []
    similar_to: str | None = None
    director: str | None = None  # "movies by/directed by X" -> that director's films
    actor: str | None = None  # "movies with/starring X" -> films with X in the cast
    exclude_actors: list[str] = []
    exclude_genres: list[str] = []
    min_year: int | None = None
    max_year: int | None = None
    min_rating: float | None = None
    region: str | None = None

    def merge_over(self, standing: ParsedQuery) -> ParsedQuery:
        """``standing`` with every positive field THIS turn specified overriding it.

        Derived from the model dump, so a new ``ParsedQuery`` field joins the
        standing set with no code change (ticket #72). "Specified" means
        non-default (``[]`` / ``None`` inherit; ``0`` does not). Exclusions
        (``EXCLUSION_FIELDS``) are never merged: they belong to the
        ``session_exclude_*`` + minimal-precedence mechanism (#21/#22), which
        must be able to suspend a remembered exclusion the user re-requests.
        """
        return standing.model_copy(
            update=self.model_dump(exclude_defaults=True, exclude=EXCLUSION_FIELDS),
        )


# ``ParsedQuery`` fields that are exclusions (never standing); classified by name.
EXCLUSION_FIELDS = frozenset(f for f in ParsedQuery.model_fields if f.startswith("exclude_"))


class ScoredMovie(BaseModel):
    tmdb_id: int
    title: str
    year: int
    regions: list[str] = []
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rank: int = 0


class MovieRecommendation(BaseModel):
    title: str
    year: int
    reason: str
    poster_url: HttpUrl | None = None


def index_by_title_year(records: Iterable[Any]) -> dict[tuple[str, int], int]:
    """``{(title, year): tmdb_id}`` - the join a pick uses to find its record.

    A ``MovieRecommendation`` carries no id, so every consumer of a generated
    answer (poster backfill, shown-movie bookkeeping, stale-repeat detection)
    has to map picks back onto the turn's candidates by ``(title, year)``. This
    is that map, built once instead of re-derived at each site. Duck-typed on
    purpose: ``ScoredMovie`` and ``MovieRecord`` both fit. Later records win on
    a duplicate key, as the inline comprehensions it replaces did.
    """
    return {(r.title, r.year): r.tmdb_id for r in records}


class RecommendationSet(BaseModel):
    picks: list[MovieRecommendation] = []
    prose: str = ""


class TurnState(BaseModel):
    """Mutable graph state that flows through the LangGraph nodes.

    Superset of the fields that get serialized into ``TurnTrace`` at END.
    """

    trace_id: str
    session_id: str
    user_id: str | None = None
    raw_query: str
    rewritten_query: str | None = None
    # Session memory (ticket #21), seeded per turn from ``ConversationState``:
    # ``history`` is the short-term window the rewriter reads to resolve
    # references; ``shown_movies`` and ``session_exclude_*`` are the standing
    # constraints carried into retrieval. All default empty -> a single-turn run
    # behaves exactly as before.
    history: list[str] = []
    shown_movies: list[int] = []
    session_exclude_actors: list[str] = []
    session_exclude_genres: list[str] = []
    # Standing constraints carried from earlier turns (ticket #54): a serialized
    # ParsedQuery whose set fields are merged UNDER this turn's parse (the current
    # turn overrides where it specifies). Empty on a fresh / replaced search.
    session_standing: dict = {}
    parsed: ParsedQuery | None = None
    retrieved: list[ScoredMovie] = []
    candidates: list[ScoredMovie] = []
    filters_applied: dict = {}
    response: RecommendationSet | None = None
    path_taken: list[str] = []
    extract_retries: int = 0
    gen_retries: int = 0
    extract_failed: bool = False
    validation_failed: bool = False
    # Machine-readable Gate-4 violation from the last validate; appended to the
    # regeneration prompt so the model is told exactly what to fix (ticket #19).
    validation_reason: str | None = None
    # EVERY violation this turn, in order, kept for the trace: the field above is
    # overwritten (and cleared on a clean regen), so it cannot answer "what did
    # Gate-4 reject?" after the fact (ticket #89).
    gate4_rejects: list[str] = []
    degradation: list[str] = []


class TurnTrace(BaseModel):
    trace_id: str
    ts: datetime
    session_id: str
    user_id: str | None = None
    raw_query: str
    rewritten_query: str | None = None
    parsed: ParsedQuery | None = None
    retrieved: list[ScoredMovie] = []
    candidates: list[ScoredMovie] = []
    filters_applied: dict = {}
    response: RecommendationSet | None = None
    path_taken: list[str] = []
    extract_retries: int = 0
    gen_retries: int = 0
    gate4_rejects: list[str] = []  # every validate violation, in order (ticket #89)
    timings_ms: dict[str, float] = {}
    token_usage: dict[str, int] = {}
    cost_usd: float = 0.0
    raw_completion: str | None = None
    degradation: list[str] = []
    prompt_version: str = ""
    # Named ``model_config_version`` (not ``model_config``) to avoid clashing
    # with Pydantic's reserved ``model_config`` class attribute.
    model_config_version: str = ""
    index_version: str = ""
    flags: list[str] = []
    judge_scores: dict[str, float] | None = None


class ChangeRecord(BaseModel):
    change_id: str
    ts: datetime
    artifact_type: str  # prompt|model|index|chunk_policy|threshold|filter
    version_before: str | None = None
    version_after: str
    motivating_trace_ids: list[str] = []
    metric_before: dict[str, float] = {}
    metric_after: dict[str, float] = {}
    suite_size_before: int = 0
    suite_size_after: int = 0
