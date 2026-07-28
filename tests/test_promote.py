"""Tests for the promotion gate and ChangeRecord emission (ticket #25).

Deterministic and network-free: the gate runs on plain pass/fail + metric dicts,
and emission writes to an in-memory-style temp SQLite store with an injected
clock.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from imdb_chatbot.promote.gate import CandidateResult, evaluate_promotion
from imdb_chatbot.promote.ledger import (
    ALLOWED_ARTIFACT_TYPES,
    emit_change_record,
    promote_and_record,
)
from imdb_chatbot.store import TraceStore

_TS = datetime(2026, 7, 25, 12, 0, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path: Path) -> TraceStore:
    s = TraceStore(tmp_path / "trace.sqlite")
    try:
        yield s
    finally:
        s.close()


def _fixed_clock():
    return _TS


# -- gate ---------------------------------------------------------------------


def test_gate_promotes_when_motivating_pass_and_no_regression() -> None:
    baseline = CandidateResult(
        case_pass={"c1": True, "c2": True, "c3": False},
        metrics={"recall_at_5": 0.80, "p95_latency_ms": 3000},
    )
    candidate = CandidateResult(
        case_pass={"c1": True, "c2": True, "c3": True},
        metrics={"recall_at_5": 0.85, "p95_latency_ms": 2900},
    )

    decision = evaluate_promotion(candidate, baseline, motivating_case_ids=["c3"])

    assert decision.promote is True
    assert decision.reasons == []


def test_gate_rejects_on_topline_regression_beyond_noise() -> None:
    baseline = CandidateResult(
        case_pass={"c1": True},
        metrics={"recall_at_5": 0.80},
    )
    # 0.80 -> 0.60 is a 25% relative drop, far beyond the 1% noise band.
    candidate = CandidateResult(
        case_pass={"c1": True},
        metrics={"recall_at_5": 0.60},
    )

    decision = evaluate_promotion(candidate, baseline, motivating_case_ids=[])

    assert decision.promote is False
    assert decision.regressed_metrics == ["recall_at_5"]
    assert any("regressed beyond noise" in r for r in decision.reasons)


def test_gate_allows_sub_noise_metric_move() -> None:
    baseline = CandidateResult(case_pass={"c1": True}, metrics={"recall_at_5": 1.00})
    # A 0.5% dip is within the 1% noise tolerance -> not a regression.
    candidate = CandidateResult(case_pass={"c1": True}, metrics={"recall_at_5": 0.995})

    decision = evaluate_promotion(
        candidate, baseline, motivating_case_ids=["c1"], noise_tol=0.01
    )

    assert decision.promote is True


def test_gate_rejects_when_previously_passing_case_fails() -> None:
    baseline = CandidateResult(case_pass={"c1": True, "c2": True}, metrics={"recall_at_5": 0.80})
    candidate = CandidateResult(case_pass={"c1": True, "c2": False}, metrics={"recall_at_5": 0.90})

    decision = evaluate_promotion(candidate, baseline, motivating_case_ids=["c1"])

    assert decision.promote is False
    assert decision.regressed_cases == ["c2"]
    assert any("previously-passing" in r for r in decision.reasons)


def test_gate_rejects_when_motivating_case_still_fails() -> None:
    baseline = CandidateResult(case_pass={"c1": False}, metrics={"recall_at_5": 0.80})
    candidate = CandidateResult(case_pass={"c1": False}, metrics={"recall_at_5": 0.80})

    decision = evaluate_promotion(candidate, baseline, motivating_case_ids=["c1"])

    assert decision.promote is False
    assert decision.failing_motivating == ["c1"]


def test_gate_lower_is_better_regression_detected() -> None:
    baseline = CandidateResult(case_pass={"c1": True}, metrics={"p95_latency_ms": 3000})
    # Latency climbing is a regression for a lower-is-better metric.
    candidate = CandidateResult(case_pass={"c1": True}, metrics={"p95_latency_ms": 4000})

    decision = evaluate_promotion(candidate, baseline, motivating_case_ids=["c1"])

    assert decision.promote is False
    assert decision.regressed_metrics == ["p95_latency_ms"]


# -- emission -----------------------------------------------------------------


def test_emit_change_record_writes_one_row(store: TraceStore) -> None:
    rec = emit_change_record(
        store,
        artifact_type="prompt",
        version_before="p-v1",
        version_after="p-v2",
        motivating_trace_ids=["t1", "t2"],
        metric_before={"recall_at_5": 0.80},
        metric_after={"recall_at_5": 0.85},
        suite_size_before=25,
        suite_size_after=25,
        clock=_fixed_clock,
    )

    assert rec.ts == _TS
    stored = store.read_change(rec.change_id)
    assert stored is not None
    assert stored.artifact_type == "prompt"
    assert stored.version_before == "p-v1"
    assert stored.version_after == "p-v2"
    assert stored.motivating_trace_ids == ["t1", "t2"]
    assert stored.metric_after == {"recall_at_5": 0.85}


def test_emit_change_record_rejects_bad_artifact_type(store: TraceStore) -> None:
    with pytest.raises(ValueError):
        emit_change_record(
            store,
            artifact_type="not_a_real_type",
            version_before=None,
            version_after="v2",
            motivating_trace_ids=[],
            metric_before={},
            metric_after={},
            suite_size_before=0,
            suite_size_after=0,
            clock=_fixed_clock,
        )


def test_allowed_artifact_types_match_schema_contract() -> None:
    assert ALLOWED_ARTIFACT_TYPES == frozenset(
        {"prompt", "model", "index", "chunk_policy", "threshold", "filter"}
    )


def _rows(store: TraceStore) -> list:
    with store._read_lock:
        return store._read_conn.execute("SELECT change_id FROM change_ledger").fetchall()


def test_promote_and_record_writes_exactly_one_on_promote(store: TraceStore) -> None:
    baseline = CandidateResult(
        case_pass={"c1": True, "c2": False},
        metrics={"recall_at_5": 0.80},
        suite_size=25,
    )
    candidate = CandidateResult(
        case_pass={"c1": True, "c2": True},
        metrics={"recall_at_5": 0.86},
        suite_size=25,
    )

    decision, record = promote_and_record(
        store,
        candidate_result=candidate,
        baseline_result=baseline,
        motivating_case_ids=["c2"],
        artifact_type="index",
        version_before="idx-1",
        version_after="idx-2",
        motivating_trace_ids=["trace-c2"],
        clock=_fixed_clock,
    )

    assert decision.promote is True
    assert record is not None
    assert len(_rows(store)) == 1
    assert record.metric_before == {"recall_at_5": 0.80}
    assert record.metric_after == {"recall_at_5": 0.86}
    assert record.suite_size_before == 25
    assert record.suite_size_after == 25


def test_promote_and_record_writes_none_on_reject(store: TraceStore) -> None:
    baseline = CandidateResult(case_pass={"c1": True}, metrics={"recall_at_5": 0.80})
    # A hard topline regression -> reject; ledger must stay empty.
    candidate = CandidateResult(case_pass={"c1": True}, metrics={"recall_at_5": 0.50})

    decision, record = promote_and_record(
        store,
        candidate_result=candidate,
        baseline_result=baseline,
        motivating_case_ids=["c1"],
        artifact_type="prompt",
        version_before="p-v1",
        version_after="p-v2",
        motivating_trace_ids=["trace-c1"],
        clock=_fixed_clock,
    )

    assert decision.promote is False
    assert record is None
    assert _rows(store) == []
