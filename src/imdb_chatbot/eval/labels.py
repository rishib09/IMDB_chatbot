"""Labeled-set format + loader for the anchored retrieval eval (ticket #17).

A labeled set is a JSONL file - one JSON object per line - where each line pins a
query to the known-item ``relevant_tmdb_ids`` a correct retrieval should surface:

    {"query_id": "q1", "query": "rainy city detective at midnight",
     "category": "standard_semantic", "relevant_tmdb_ids": [1],
     "parsed": {"exclude_actors": ["Repeat Star"]}}

Fields:

- ``query_id``            - stable id for the row (used in the metric table).
- ``query``              - the raw user query text handed to the retriever.
- ``category``           - one of the 8 categories from section 8.7 (below).
- ``relevant_tmdb_ids``  - the anchored known-item answer(s). May be empty for
                           ``ood_unanswerable`` rows (there is no right movie).
- ``parsed``             - OPTIONAL ``ParsedQuery`` fields (filters/exclusions);
                           absent means an empty ``ParsedQuery``.

The loader validates each row into a typed ``LabeledQuery`` (pydantic) and rejects
unknown categories loudly so a typo never silently mislabels a row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..schemas import ParsedQuery

# The 8 eval categories from section 8.7. Order is stable so the metric table
# always reports categories in the same sequence.
CATEGORIES: tuple[str, ...] = (
    "standard_semantic",
    "exact_title",
    "negation_exclusion",
    "short",
    "region_conditioned",
    "numeric_filter",
    "ambiguous",
    "ood_unanswerable",
)

Category = Literal[
    "standard_semantic",
    "exact_title",
    "negation_exclusion",
    "short",
    "region_conditioned",
    "numeric_filter",
    "ambiguous",
    "ood_unanswerable",
]


class LabeledQuery(BaseModel):
    """One anchored eval row: a query + its known-item relevant tmdb_id(s)."""

    query_id: str
    query: str
    category: Category
    relevant_tmdb_ids: list[int] = Field(default_factory=list)
    parsed: ParsedQuery = Field(default_factory=ParsedQuery)


def load_labels(path: str | Path) -> list[LabeledQuery]:
    """Parse a labeled-set JSONL file into a list of typed ``LabeledQuery`` rows.

    Blank lines are skipped. Every non-blank line must be a JSON object that
    validates against ``LabeledQuery`` (including a known ``category``); an
    invalid row raises rather than being dropped, so a malformed eval set fails
    loudly instead of silently shrinking the suite.
    """
    path = Path(path)
    rows: list[LabeledQuery] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            try:
                rows.append(LabeledQuery.model_validate(obj))
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: invalid labeled query: {exc}") from exc
    return rows
