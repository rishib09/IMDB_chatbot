"""ChangeRecord emission - the auditable trail written on promotion.

A ``ChangeRecord`` is emitted AUTOMATICALLY by the promotion flow, never
hand-authored: passing the gate is what earns a ledger row. ``emit_change_record``
builds the record and writes it to the store's ``change_ledger`` table;
``promote_and_record`` runs the gate first and only emits on a promote.

The wall clock is injected (``clock``) so tests are deterministic - production
callers pass ``lambda: datetime.now(UTC)``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

from ..schemas import ChangeRecord
from ..store import TraceStore
from .gate import CandidateResult, PromotionDecision, evaluate_promotion

Clock = Callable[[], datetime]

# The artifact kinds the gate governs (PRD section 8.4). Any behavioral change is
# one of these; anything else is a programming error, so we reject it loudly.
ALLOWED_ARTIFACT_TYPES: frozenset[str] = frozenset(
    {"prompt", "model", "index", "chunk_policy", "threshold", "filter"}
)


def emit_change_record(
    store: TraceStore,
    *,
    artifact_type: str,
    version_before: str | None,
    version_after: str,
    motivating_trace_ids: Sequence[str],
    metric_before: Mapping[str, float],
    metric_after: Mapping[str, float],
    suite_size_before: int,
    suite_size_after: int,
    clock: Clock,
) -> ChangeRecord:
    """Build a ``ChangeRecord`` and write it to the store's change ledger.

    ``clock`` supplies the timestamp (injected for test determinism). The
    ``change_id`` is a fresh uuid4 - records are append-only and machine-emitted,
    so identity is generated, never supplied by a human. Raises ``ValueError`` for
    an ``artifact_type`` outside :data:`ALLOWED_ARTIFACT_TYPES`.
    """
    if artifact_type not in ALLOWED_ARTIFACT_TYPES:
        raise ValueError(
            f"artifact_type must be one of {sorted(ALLOWED_ARTIFACT_TYPES)}, "
            f"got {artifact_type!r}"
        )

    record = ChangeRecord(
        change_id=uuid.uuid4().hex,
        ts=clock(),
        artifact_type=artifact_type,
        version_before=version_before,
        version_after=version_after,
        motivating_trace_ids=list(motivating_trace_ids),
        metric_before=dict(metric_before),
        metric_after=dict(metric_after),
        suite_size_before=suite_size_before,
        suite_size_after=suite_size_after,
    )
    store.write_change(record)
    return record


def promote_and_record(
    store: TraceStore,
    *,
    candidate_result: CandidateResult,
    baseline_result: CandidateResult,
    motivating_case_ids: Sequence[str],
    artifact_type: str,
    version_before: str | None,
    version_after: str,
    motivating_trace_ids: Sequence[str],
    clock: Clock,
    noise_tol: float | None = None,
) -> tuple[PromotionDecision, ChangeRecord | None]:
    """Run the promotion gate and, on a promote, emit a ``ChangeRecord``.

    Returns ``(decision, record)``. On a REJECT the ledger is untouched and
    ``record`` is ``None``. The emitted record's metric snapshots and suite sizes
    are read straight off the baseline/candidate eval results, so the ledger
    always reflects the exact numbers the gate weighed.
    """
    kwargs = {} if noise_tol is None else {"noise_tol": noise_tol}
    decision = evaluate_promotion(
        candidate_result, baseline_result, motivating_case_ids, **kwargs
    )
    if not decision.promote:
        return decision, None

    record = emit_change_record(
        store,
        artifact_type=artifact_type,
        version_before=version_before,
        version_after=version_after,
        motivating_trace_ids=motivating_trace_ids,
        metric_before=baseline_result.metrics,
        metric_after=candidate_result.metrics,
        suite_size_before=baseline_result.n,
        suite_size_after=candidate_result.n,
        clock=clock,
    )
    return decision, record
