"""Anchored eval set + retrieval eval harness (ticket #17).

Public surface:

- ``labels``: the labeled-set JSONL format plus its typed loader. A labeled set
  is a list of ``LabeledQuery`` rows, one per line, each pinning a query to the
  known-item ``relevant_tmdb_ids`` a correct retrieval should surface.
- ``harness``: run a ``HybridRetriever`` over a labeled set and compute the
  retrieval metrics under the anchored / known-item convention (decision #6):
  Recall@k / Hit@k for k in {1,3,5,10}, MRR, exclusion precision, and the
  fallback rate. Recall is a measured LOWER BOUND, never asserted.

Run it against the live corpus with::

    python -m imdb_chatbot.eval --db data/corpus.sqlite --labels labels.jsonl

The categories are the 8 from section 8.7: standard_semantic, exact_title,
negation_exclusion, short, region_conditioned, numeric_filter, ambiguous,
ood_unanswerable.
"""

from __future__ import annotations

from .harness import (
    K_VALUES,
    CategorySummary,
    EvalReport,
    QueryResult,
    evaluate,
    format_report,
)
from .labels import CATEGORIES, LabeledQuery, load_labels

__all__ = [
    "CATEGORIES",
    "K_VALUES",
    "CategorySummary",
    "EvalReport",
    "LabeledQuery",
    "QueryResult",
    "evaluate",
    "format_report",
    "load_labels",
]
