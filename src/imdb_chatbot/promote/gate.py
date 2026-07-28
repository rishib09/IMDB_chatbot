"""The promotion gate (PRD section 8.4) - one gate, pure and deterministic.

A behavioral change is PROMOTED if and only if ALL three hold:

1. Every motivating case now PASSES on the candidate (the change fixes what it
   set out to fix).
2. ZERO previously-passing cases regress - a case that passed on the baseline
   must still pass on the candidate (no collateral damage to the suite).
3. NO topline metric regresses beyond the noise tolerance (Recall@5, exclusion
   precision, adherence, and friends do not slip, allowing for measurement
   noise).

Otherwise the change is REJECTED, with a machine-readable list of reasons.

Statistical-power caveat: with a POC-scale suite (n around 25) the gate CANNOT
reliably detect small (about 1%) topline regressions - the noise band swallows
them. The gate is a guardrail against obvious breakage, not a proof of
non-regression; treat marginal metric moves as "within noise", and grow the
suite before trusting sub-noise deltas.

Inputs are deliberately simple structures (a per-case pass/fail map plus a
topline metric dict), so this module has no dependency on the eval harness, the
retriever, or any I/O - it can be exercised with plain dicts in a unit test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# Default noise tolerance: a topline metric may move against us by up to this
# fraction (relative to the baseline value) and still count as "within noise".
DEFAULT_NOISE_TOL = 0.01  # 1% relative

# Topline metrics where a LOWER value is better (latency, spend, fallback rate).
# Every other metric key is treated as higher-is-better.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "p95_latency_ms",
        "latency_ms",
        "cost_per_conv_usd",
        "cost_usd",
        "fallback_rate",
    }
)


@dataclass(frozen=True)
class CandidateResult:
    """A minimal eval snapshot: per-case pass/fail and topline metrics.

    - ``case_pass`` maps a case id to whether it passed on this artifact.
    - ``metrics`` is the topline metric dict (e.g. ``{"recall_at_5": 0.82}``).
    - ``suite_size`` is the number of cases behind ``metrics`` (defaults to the
      number of cases in ``case_pass``).
    """

    case_pass: Mapping[str, bool]
    metrics: Mapping[str, float] = field(default_factory=dict)
    suite_size: int | None = None

    @property
    def n(self) -> int:
        return self.suite_size if self.suite_size is not None else len(self.case_pass)


@dataclass(frozen=True)
class PromotionDecision:
    """The gate's verdict: promote or not, plus human/machine-readable reasons."""

    promote: bool
    reasons: list[str]
    # Diagnostics for the ledger / dashboard.
    failing_motivating: list[str] = field(default_factory=list)
    regressed_cases: list[str] = field(default_factory=list)
    regressed_metrics: list[str] = field(default_factory=list)


def _is_regression(key: str, before: float, after: float, noise_tol: float) -> bool:
    """True when ``after`` is worse than ``before`` by more than the noise band.

    Direction is metric-aware (``LOWER_IS_BETTER``). The tolerance is relative to
    the magnitude of the baseline value, so it works across units (a fraction in
    [0, 1] and a latency in milliseconds alike); when the baseline is zero we fall
    back to an absolute comparison against ``noise_tol``.
    """
    lower_better = key in LOWER_IS_BETTER
    # "worsening" is always a non-negative amount of movement in the bad direction.
    worsening = (after - before) if lower_better else (before - after)
    if worsening <= 0:
        return False  # improved or unchanged
    denom = abs(before)
    if denom == 0:
        return worsening > noise_tol
    return (worsening / denom) > noise_tol


def evaluate_promotion(
    candidate_result: CandidateResult,
    baseline_result: CandidateResult,
    motivating_case_ids: Sequence[str],
    *,
    noise_tol: float = DEFAULT_NOISE_TOL,
) -> PromotionDecision:
    """Decide whether ``candidate_result`` may be promoted over ``baseline_result``.

    Pure and deterministic (see module docstring for the section 8.4 rule and the
    statistical-power caveat). Returns a :class:`PromotionDecision`; on rejection
    ``reasons`` lists every failed condition.
    """
    reasons: list[str] = []

    # (1) Motivating cases must now pass on the candidate.
    failing_motivating = [
        cid for cid in motivating_case_ids if not candidate_result.case_pass.get(cid, False)
    ]
    if failing_motivating:
        reasons.append(
            "motivating cases still failing: " + ", ".join(sorted(failing_motivating))
        )

    # (2) No previously-passing case may regress.
    regressed_cases = [
        cid
        for cid, passed in baseline_result.case_pass.items()
        if passed and not candidate_result.case_pass.get(cid, False)
    ]
    if regressed_cases:
        reasons.append(
            "previously-passing cases now failing: " + ", ".join(sorted(regressed_cases))
        )

    # (3) No topline metric may regress beyond the noise tolerance.
    regressed_metrics: list[str] = []
    for key, before in baseline_result.metrics.items():
        if key not in candidate_result.metrics:
            continue
        after = candidate_result.metrics[key]
        if _is_regression(key, before, after, noise_tol):
            regressed_metrics.append(key)
            reasons.append(
                f"topline metric regressed beyond noise: {key} "
                f"{before:.4g} -> {after:.4g} (tol={noise_tol:g})"
            )

    promote = not reasons
    return PromotionDecision(
        promote=promote,
        reasons=reasons,
        failing_motivating=sorted(failing_motivating),
        regressed_cases=sorted(regressed_cases),
        regressed_metrics=regressed_metrics,
    )
