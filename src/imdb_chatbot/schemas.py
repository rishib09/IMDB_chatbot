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

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class MovieRecord(BaseModel):
    tmdb_id: int  # identity key
    title: str
    year: int = Field(ge=1888, le=2030)
    genres: list[str] = []
    director: str | None = None
    cast: list[str] = []
    plot: str | None = None
    rating_raw: float | None = Field(default=None, ge=0, le=10)
    rating_z: dict[str, float] = {}  # per-region z-score {region: z}
    metascore: float | None = None
    certificate_raw: str | None = None
    certificate_norm: str | None = None  # {ALL, TEEN, MATURE, ADULT} or None
    certificate_system: str | None = None  # e.g. "MPAA", "CBFC"
    regions: list[str] = []  # origin countries, e.g. ["US","IN"]
    duration_min: float | None = None
    poster_url: HttpUrl | None = None


class ParsedQuery(BaseModel):
    genres: list[str] = []
    similar_to: str | None = None
    exclude_actors: list[str] = []
    exclude_genres: list[str] = []
    min_year: int | None = None
    max_year: int | None = None
    min_rating: float | None = None
    region: str | None = None


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
