"""One full harness cycle, end to end and deterministic (ticket #26).

This module stitches the reliability loop together so it can be *watched* in one
place and *asserted* in a test, with no network and no LLM. It walks the exact
path the PRD describes (sections 8.1 - 8.4):

    seed a failure  ->  detect its taxonomy code  ->  represent a candidate fix
        ->  run the promotion gate  ->  record a ChangeRecord in the ledger

Each stage reuses the real production code, not a stand-in:

- SEED  - a hand-built ``TurnTrace`` exhibiting an S2 repetition (the same pick
  recommended twice in one response). The trace is written to a ``TraceStore``,
  which is the system of record.
- DETECT - ``eval.detect.detect`` reads that trace and proves the ``S2`` code.
- FIX + GATE - a baseline-vs-candidate eval snapshot where the motivating case
  (the seeded trace) FAILED on the baseline and now PASSES on the candidate, and
  no previously-passing case and no topline metric regresses. ``promote.ledger``'s
  ``promote_and_record`` runs the gate and, on a PROMOTE, emits the record.
- LEDGER - the emitted ``ChangeRecord`` lands in the store's ``change_ledger``,
  carrying the motivating trace id and the metric deltas the gate weighed.

``run_loop`` returns a machine-readable :class:`LoopResult` (each stage's
evidence) so a test can assert every step; ``narrate`` renders the same result
as the readable story the CLI prints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..eval.detect import detect
from ..promote.gate import CandidateResult, PromotionDecision
from ..promote.ledger import promote_and_record
from ..schemas import (
    ChangeRecord,
    MovieRecommendation,
    RecommendationSet,
    ScoredMovie,
    TurnTrace,
)
from ..store import TraceStore

Clock = Callable[[], datetime]

# A fixed wall clock keeps the demo (and its test) fully deterministic. Production
# callers of ``promote_and_record`` pass ``lambda: datetime.now(UTC)`` instead.
_DEMO_TS = datetime(2026, 7, 25, 12, 0, 0, tzinfo=UTC)


def _demo_clock() -> datetime:
    return _DEMO_TS


# The taxonomy code the seeded trace is built to trigger (a state-repetition
# near-miss). Named so the narrative and the test agree on the expected code.
SEEDED_CODE = "S2"

# The one non-motivating case that passes on BOTH baseline and candidate; it is
# the regression guard the gate checks (a previously-passing case must not fail).
REGRESSION_GUARD_CASE = "recall_regression_guard"


def seed_failure_trace(trace_id: str = "demo-trace-s2") -> TurnTrace:
    """Build a ``TurnTrace`` that exhibits a detectable S2 repetition failure.

    The response recommends the SAME film (title + year) twice, which is the
    intra-response duplicate signal ``eval.detect`` proves as ``S2``. Everything
    else on the trace is a plausible, clean single turn.
    """
    duplicate = MovieRecommendation(
        title="Inception",
        year=2010,
        reason="A heist inside dreams - twisty and rewatchable.",
    )
    response = RecommendationSet(
        picks=[duplicate, duplicate],  # same pick twice -> S2 repetition
        prose="Here are two picks for you.",
    )
    candidates = [
        ScoredMovie(tmdb_id=27205, title="Inception", year=2010, rank=1),
    ]
    return TurnTrace(
        trace_id=trace_id,
        ts=_DEMO_TS,
        session_id="demo-session",
        user_id=None,
        raw_query="recommend me a mind-bending sci-fi",
        rewritten_query="recommend me a mind-bending sci-fi",
        candidates=candidates,
        response=response,
        path_taken=["rewrite", "extract", "retrieve", "filter", "generate", "validate", "END"],
    )


def _baseline_candidate(
    trace_id: str, *, regress: bool
) -> tuple[CandidateResult, CandidateResult]:
    """Build the baseline vs candidate eval snapshot the gate weighs.

    The motivating case (keyed by ``trace_id``) FAILS on the baseline and PASSES
    on the candidate. ``REGRESSION_GUARD_CASE`` passes on both, so the gate can
    prove nothing that used to pass now fails. When ``regress`` is True the
    candidate's topline ``recall_at_5`` drops hard, so the gate must REJECT even
    though the motivating case now passes - the "candidate regressed" variant.
    """
    baseline = CandidateResult(
        case_pass={trace_id: False, REGRESSION_GUARD_CASE: True},
        metrics={"recall_at_5": 0.80, "exclusion_precision": 0.95},
        suite_size=25,
    )
    candidate_recall = 0.50 if regress else 0.84
    candidate = CandidateResult(
        case_pass={trace_id: True, REGRESSION_GUARD_CASE: True},
        metrics={"recall_at_5": candidate_recall, "exclusion_precision": 0.96},
        suite_size=25,
    )
    return baseline, candidate


@dataclass
class LoopResult:
    """The evidence from one walked cycle - each stage's machine-readable proof."""

    trace_id: str
    detected_codes: list[str]
    decision: PromotionDecision
    record: ChangeRecord | None
    metric_before: dict[str, float] = field(default_factory=dict)
    metric_after: dict[str, float] = field(default_factory=dict)

    @property
    def promoted(self) -> bool:
        return self.decision.promote

    @property
    def metric_deltas(self) -> dict[str, float]:
        """Per-metric ``after - before`` deltas (empty when nothing was recorded)."""
        return {
            key: self.metric_after[key] - self.metric_before[key]
            for key in self.metric_before
            if key in self.metric_after
        }


def run_loop(
    store: TraceStore,
    *,
    regress: bool = False,
    trace_id: str = "demo-trace-s2",
    clock: Clock = _demo_clock,
) -> LoopResult:
    """Walk the full seed -> detect -> gate -> ledger cycle over ``store``.

    Deterministic and network-free. Writes the seeded failure trace, runs
    detection, runs the promotion gate against a baseline/candidate snapshot, and
    (on a PROMOTE) emits the ``ChangeRecord``. With ``regress=True`` the candidate
    regresses so the gate REJECTS and no record is written. Returns a
    :class:`LoopResult` carrying every stage's evidence.
    """
    # 1. SEED - write a failing trace into the system of record.
    trace = seed_failure_trace(trace_id)
    store.write_trace(trace)

    # 2. DETECT - prove the taxonomy code from the trace alone.
    detected_codes = detect(trace)

    # 3 + 4. FIX + GATE - weigh a candidate fix and, on a promote, record it.
    baseline, candidate = _baseline_candidate(trace_id, regress=regress)
    decision, record = promote_and_record(
        store,
        candidate_result=candidate,
        baseline_result=baseline,
        motivating_case_ids=[trace_id],
        artifact_type="prompt",
        version_before="rec-prompt-v1",
        version_after="rec-prompt-v2-dedupe",
        motivating_trace_ids=[trace_id],
        clock=clock,
    )

    return LoopResult(
        trace_id=trace_id,
        detected_codes=detected_codes,
        decision=decision,
        record=record,
        metric_before=dict(baseline.metrics),
        metric_after=dict(candidate.metrics),
    )


def narrate(result: LoopResult) -> str:
    """Render a :class:`LoopResult` as a readable end-to-end narrative."""
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append("IMDb harness - end-to-end reliability loop (ticket #26)")
    lines.append("=" * 68)
    lines.append("")

    lines.append("1. SEED a failure")
    lines.append(
        f"   Wrote trace {result.trace_id!r}: a response that recommends the same"
    )
    lines.append("   film twice - a stale/repeated recommendation (state failure).")
    lines.append("")

    lines.append("2. DETECT the taxonomy code")
    codes = ", ".join(result.detected_codes) if result.detected_codes else "(none)"
    lines.append(f"   detect(trace) fired: {codes}")
    lines.append(f"   -> S2 = repetition / stale recommendation (expected {SEEDED_CODE}).")
    lines.append("")

    lines.append("3. FIX + GATE the candidate")
    lines.append("   Candidate: a de-duplicating recommendation prompt.")
    lines.append(
        f"   Motivating case {result.trace_id!r}: FAIL on baseline -> PASS on candidate."
    )
    for key, before in result.metric_before.items():
        after = result.metric_after.get(key, before)
        delta = after - before
        lines.append(f"   metric {key}: {before:.4g} -> {after:.4g} ({delta:+.3g})")
    verdict = "PROMOTE" if result.promoted else "REJECT"
    lines.append(f"   Gate decision: {verdict}")
    if not result.promoted:
        for reason in result.decision.reasons:
            lines.append(f"     - {reason}")
    lines.append("")

    lines.append("4. LEDGER row")
    if result.record is not None:
        rec = result.record
        lines.append(f"   Wrote ChangeRecord {rec.change_id}")
        lines.append(f"   artifact_type={rec.artifact_type} "
                     f"version {rec.version_before} -> {rec.version_after}")
        lines.append(f"   motivating_trace_ids={rec.motivating_trace_ids}")
        lines.append(f"   metric_before={rec.metric_before} metric_after={rec.metric_after}")
    else:
        lines.append("   No ChangeRecord written - the gate REJECTED the candidate,")
        lines.append("   so the ledger stays clean (no unearned promotion).")
    lines.append("")
    lines.append("=" * 68)
    return "\n".join(lines)
