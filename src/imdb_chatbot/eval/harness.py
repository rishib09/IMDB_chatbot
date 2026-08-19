"""Retrieval eval harness (ticket #17), anchored / known-item convention.

Given a built ``HybridRetriever`` and a labeled set, run retrieval for every
query and compute the retrieval metrics under the ANCHORED convention
(decision #6): a query's ``relevant_tmdb_ids`` are the known-item answer(s), and
Recall = did those id(s) land in the top-k. Because the labels pin only the
*known* relevant items (not every conceivably-relevant film), the reported recall
is a comparable LOWER BOUND across index versions - it is measured and reported,
never asserted.

Per query we compute:

- ``Hit@k``   - 1 if ANY relevant id appears in the top-k, else 0.
- ``Recall@k`` - fraction of the labeled relevant ids that appear in the top-k.
- reciprocal rank - ``1 / rank`` of the FIRST relevant id (0 if none); MRR is its
  mean over the answerable queries.
- exclusion clean - for queries whose ``parsed`` carries exclusions, whether the
  returned candidates contain ZERO excluded actor/genre (target precision 1.00).
- fell back - whether retrieval returned zero candidates. For
  ``ood_unanswerable`` a fallback is the CORRECT outcome.

Queries with no relevant ids (the ``ood_unanswerable`` rows) are excluded from
recall/hit/MRR - there is no right answer to recall - but still counted for the
fallback rate. Metrics are reported overall and per category.

TWO TIERS (ticket #68). By default the labeled ``parsed`` constraints are HANDED
to the retriever, so the tier measures retrieval alone - deterministic, free, and
blind to the extractor. Pass ``parse=`` to run the constraints the LLM extractor
actually produces instead: same rows, same metrics, and the delta between the two
tiers IS the extraction regression signal (the #90 failure mode, where a
vibe-phrased query flakes into an empty parse stamped ``region="US"``). Exclusions
are always scored against the LABEL, never against the parse the system produced:
the user asked for no Tom Cruise, so a Tom Cruise film in the results is a miss
whether the extractor dropped the exclusion or the filter leaked it.

To measure Recall@10 the retriever must be able to return >=10 candidates; build
it with ``final_k=10`` (the CLI does). If it returns fewer, the @10 figures
degrade gracefully to whatever depth is available.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from statistics import fmean

from ..retrieval.retrieve import HybridRetriever
from ..schemas import ParsedQuery
from .labels import CATEGORIES, LabeledQuery

# The depths at which Hit/Recall are reported.
K_VALUES: tuple[int, ...] = (1, 3, 5, 10)

# Tier 2: derive the retrieval constraints from the query instead of reading them
# off the label. In production this is the LLM extractor plus its deterministic
# guards; it costs one model call per row.
ParseFn = Callable[[LabeledQuery], ParsedQuery]


@dataclass
class QueryResult:
    """The per-query measurement, before aggregation."""

    query_id: str
    category: str
    num_relevant: int
    num_returned: int
    # 1-based ranks at which the labeled relevant ids landed (best-first order).
    relevant_ranks: list[int]
    first_relevant_rank: int | None
    hit_at: dict[int, bool]
    # None at every k when the query has no relevant ids (ood_unanswerable).
    recall_at: dict[int, float | None]
    reciprocal_rank: float | None
    has_exclusions: bool
    exclusion_clean: bool | None  # None when the query has no exclusions
    fell_back: bool  # retrieval returned zero candidates
    # Tier 2 only: the constraints the extractor actually produced (non-default
    # fields). None in the anchored tier, where the label's parse was used as-is.
    extracted_parse: dict | None = None


@dataclass
class CategorySummary:
    """Aggregated metrics for one group of queries (a category, or overall)."""

    label: str
    n: int
    # Number of queries that had >=1 relevant id (the answerable ones).
    n_answerable: int
    hit_at: dict[int, float]
    recall_at: dict[int, float]
    mrr: float
    # Mean over queries carrying exclusions; None when the group has none.
    exclusion_precision: float | None
    n_exclusion_queries: int
    fallback_rate: float
    # True when a fallback is the correct outcome for this group (ood only).
    fallback_is_correct: bool


@dataclass
class EvalReport:
    """The full report: per-query rows, per-category summaries, and overall."""

    overall: CategorySummary
    per_category: dict[str, CategorySummary]
    queries: list[QueryResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """A plain-dict view (handy for JSON dumps / assertions)."""
        return {
            "overall": asdict(self.overall),
            "per_category": {cat: asdict(summ) for cat, summ in self.per_category.items()},
        }


def _eval_query(
    retriever: HybridRetriever,
    label: LabeledQuery,
    k_values: Sequence[int],
    parse: ParseFn | None = None,
) -> QueryResult:
    """Run one query and measure it against its anchored relevant ids.

    ``parse`` (tier 2) derives the constraints from the query text; without it the
    label's own ``parsed`` is handed to the retriever (tier 1).
    """
    used = label.parsed if parse is None else parse(label)
    scored = retriever.retrieve(label.query, parsed=used)
    ranked_ids = [s.tmdb_id for s in scored]
    relevant = set(label.relevant_tmdb_ids)

    relevant_ranks = [rank for rank, tid in enumerate(ranked_ids, start=1) if tid in relevant]
    first_rank = relevant_ranks[0] if relevant_ranks else None

    hit_at = {k: any(r <= k for r in relevant_ranks) for k in k_values}

    if relevant:
        recall_at: dict[int, float | None] = {
            k: sum(1 for r in relevant_ranks if r <= k) / len(relevant) for k in k_values
        }
        reciprocal_rank: float | None = (1.0 / first_rank) if first_rank is not None else 0.0
    else:
        # No known-item answer: recall / MRR are undefined for this query.
        recall_at = {k: None for k in k_values}
        reciprocal_rank = None

    # Scored against the LABEL's exclusions in both tiers: what the user asked to
    # exclude is the standard, not what the extractor happened to notice.
    has_exclusions = bool(label.parsed.exclude_actors or label.parsed.exclude_genres)
    exclusion_clean: bool | None = None
    if has_exclusions:
        excluded_actors = set(label.parsed.exclude_actors)
        excluded_genres = set(label.parsed.exclude_genres)
        exclusion_clean = True
        for tid in ranked_ids:
            movie = retriever.movies_by_id.get(tid)
            if movie is None:
                continue
            if excluded_genres & set(movie.genres):
                exclusion_clean = False
                break
            if excluded_actors & set(movie.cast):
                exclusion_clean = False
                break

    return QueryResult(
        query_id=label.query_id,
        category=label.category,
        num_relevant=len(relevant),
        num_returned=len(ranked_ids),
        relevant_ranks=relevant_ranks,
        first_relevant_rank=first_rank,
        hit_at=hit_at,
        recall_at=recall_at,
        reciprocal_rank=reciprocal_rank,
        has_exclusions=has_exclusions,
        exclusion_clean=exclusion_clean,
        fell_back=len(ranked_ids) == 0,
        extracted_parse=None if parse is None else used.model_dump(exclude_defaults=True),
    )


def _summarize(
    label: str,
    results: Sequence[QueryResult],
    k_values: Sequence[int],
) -> CategorySummary:
    """Aggregate per-query results into a ``CategorySummary``.

    Hit/Recall/MRR are averaged over the ANSWERABLE queries (those with >=1
    relevant id); the fallback rate is over every query in the group.
    """
    n = len(results)
    answerable = [r for r in results if r.num_relevant > 0]
    n_answerable = len(answerable)

    # ``fmean`` raises on an empty sequence, so every mean keeps its guard.
    if n_answerable:
        hit_at = {k: fmean(r.hit_at[k] for r in answerable) for k in k_values}
        recall_at = {k: fmean(r.recall_at[k] or 0.0 for r in answerable) for k in k_values}
        mrr = fmean(r.reciprocal_rank or 0.0 for r in answerable)
    else:
        hit_at = {k: 0.0 for k in k_values}
        recall_at = {k: 0.0 for k in k_values}
        mrr = 0.0

    exclusion_queries = [r for r in results if r.has_exclusions]
    exclusion_precision: float | None = (
        fmean(bool(r.exclusion_clean) for r in exclusion_queries) if exclusion_queries else None
    )

    fallback_rate = fmean(r.fell_back for r in results) if n else 0.0
    fallback_is_correct = label == "ood_unanswerable"

    return CategorySummary(
        label=label,
        n=n,
        n_answerable=n_answerable,
        hit_at=hit_at,
        recall_at=recall_at,
        mrr=mrr,
        exclusion_precision=exclusion_precision,
        n_exclusion_queries=len(exclusion_queries),
        fallback_rate=fallback_rate,
        fallback_is_correct=fallback_is_correct,
    )


def evaluate(
    retriever: HybridRetriever,
    labels: Sequence[LabeledQuery],
    *,
    k_values: Sequence[int] = K_VALUES,
    parse: ParseFn | None = None,
) -> EvalReport:
    """Run ``retriever`` over ``labels`` and return the anchored eval report.

    Deterministic for a fixed index + labeled set (the retriever itself is
    deterministic) when ``parse`` is None. With ``parse`` this is the tier-2 run:
    one extractor call per row, so it costs money and is only as reproducible as
    the model. Recall is measured and reported, never asserted.
    """
    k_values = tuple(k_values)
    results = [_eval_query(retriever, label, k_values, parse) for label in labels]

    overall = _summarize("overall", results, k_values)
    per_category: dict[str, CategorySummary] = {}
    for cat in CATEGORIES:
        group = [r for r in results if r.category == cat]
        if group:
            per_category[cat] = _summarize(cat, group, k_values)

    return EvalReport(overall=overall, per_category=per_category, queries=results)


# -- reporting ----------------------------------------------------------------


def format_report(report: EvalReport, *, k_values: Sequence[int] = K_VALUES) -> str:
    """Render the report as a plain-ASCII metric table for the CLI."""
    k_values = tuple(k_values)
    lines: list[str] = []

    def _fmt_summary(summ: CategorySummary) -> None:
        hits = " ".join(f"Hit@{k}={summ.hit_at[k]:.2f}" for k in k_values)
        recalls = " ".join(f"R@{k}={summ.recall_at[k]:.2f}" for k in k_values)
        lines.append(f"  n={summ.n} answerable={summ.n_answerable}")
        lines.append(f"  {hits}")
        lines.append(f"  {recalls}")
        lines.append(f"  MRR={summ.mrr:.3f}")
        if summ.exclusion_precision is not None:
            lines.append(
                f"  exclusion_precision={summ.exclusion_precision:.2f} "
                f"({summ.n_exclusion_queries} queries)"
            )
        note = " (fallback is the correct outcome here)" if summ.fallback_is_correct else ""
        lines.append(f"  fallback_rate={summ.fallback_rate:.2f}{note}")

    lines.append("=== overall ===")
    _fmt_summary(report.overall)
    lines.append("")
    lines.append("=== per category ===")
    for cat, summ in report.per_category.items():
        lines.append(f"[{cat}]")
        _fmt_summary(summ)
    return "\n".join(lines)
