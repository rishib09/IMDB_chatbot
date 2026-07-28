"""Tests for the Change Ledger dashboard render-data helpers (ticket #25).

Deterministic and Streamlit-free: only the pure ``data.py`` shaping functions are
exercised (no server is started). After writing a few ChangeRecords the timeline
markers, table ordering, and topline colors are asserted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from imdb_chatbot.dashboard.data import (
    ledger_table,
    ledger_table_display,
    metric_timeline,
    topline_strip,
)
from imdb_chatbot.promote.ledger import emit_change_record
from imdb_chatbot.schemas import (
    MovieRecommendation,
    RecommendationSet,
    ScoredMovie,
    TurnTrace,
)
from imdb_chatbot.store import TraceStore


@pytest.fixture()
def store(tmp_path: Path) -> TraceStore:
    s = TraceStore(tmp_path / "trace.sqlite")
    try:
        yield s
    finally:
        s.close()


def _clock_at(ts: datetime):
    return lambda: ts


def _seed_three_changes(store: TraceStore) -> list:
    """Write three ChangeRecords at increasing timestamps; return them oldest-first."""
    recs = []
    for i, (day, before, after) in enumerate(
        [(1, 0.70, 0.75), (2, 0.75, 0.80), (3, 0.80, 0.90)], start=1
    ):
        rec = emit_change_record(
            store,
            artifact_type="prompt",
            version_before=f"p-v{i}",
            version_after=f"p-v{i + 1}",
            motivating_trace_ids=[],
            metric_before={"recall_at_5": before},
            metric_after={"recall_at_5": after},
            suite_size_before=25,
            suite_size_after=25,
            clock=_clock_at(datetime(2026, 7, day, 12, 0, 0, tzinfo=UTC)),
        )
        recs.append(rec)
    return recs


def test_metric_timeline_points_and_markers(store: TraceStore) -> None:
    recs = _seed_three_changes(store)

    timeline = metric_timeline(store, "recall_at_5")

    # One point + one marker per change, chronological (oldest first).
    assert [p["value"] for p in timeline["points"]] == [0.75, 0.80, 0.90]
    assert [m["change_id"] for m in timeline["markers"]] == [r.change_id for r in recs]
    # Each marker carries the delta for the selected metric.
    assert timeline["markers"][2]["delta"] == pytest.approx(0.10)


def test_metric_timeline_skips_missing_metric(store: TraceStore) -> None:
    _seed_three_changes(store)

    timeline = metric_timeline(store, "adherence")

    # No change carries 'adherence' -> no points, but markers still exist.
    assert timeline["points"] == []
    assert len(timeline["markers"]) == 3
    assert all(m["value"] is None for m in timeline["markers"])


def test_ledger_table_is_newest_first(store: TraceStore) -> None:
    recs = _seed_three_changes(store)

    rows = ledger_table(store)

    assert [r["change_id"] for r in rows] == [rec.change_id for rec in reversed(recs)]
    # Newest row first, with the version-move summary and a positive delta.
    assert rows[0]["what_changed"] == "prompt: p-v3 -> p-v4"
    assert rows[0]["metric_deltas"]["recall_at_5"] == pytest.approx(0.10)


def test_ledger_table_why_counts_taxonomy_codes(store: TraceStore) -> None:
    # A trace that fires exactly one taxonomy code (O1: over cost budget).
    trace = TurnTrace(
        trace_id="trace-o1",
        ts=datetime(2026, 7, 1, 9, 0, 0, tzinfo=UTC),
        session_id="s1",
        raw_query="an action movie",
        path_taken=["rewrite", "extract", "retrieve", "filter", "generate", "validate"],
        candidates=[ScoredMovie(tmdb_id=1, title="John Wick", year=2014, rank=1)],
        response=RecommendationSet(
            picks=[MovieRecommendation(title="John Wick", year=2014, reason="lean")],
            prose="here",
        ),
        cost_usd=0.99,  # over COST_BUDGET_USD -> O1
    )
    store.write_trace(trace)
    emit_change_record(
        store,
        artifact_type="threshold",
        version_before="t-1",
        version_after="t-2",
        motivating_trace_ids=["trace-o1"],
        metric_before={"recall_at_5": 0.80},
        metric_after={"recall_at_5": 0.82},
        suite_size_before=25,
        suite_size_after=25,
        clock=_clock_at(datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)),
    )

    rows = ledger_table(store)

    assert rows[0]["why"] == {"O1": 1}


def test_ledger_table_display_flattens_cells(store: TraceStore) -> None:
    _seed_three_changes(store)

    rows = ledger_table_display(store)

    assert isinstance(rows[0]["why"], str)
    assert isinstance(rows[0]["metric_deltas"], str)
    assert rows[0]["metric_deltas"].startswith("recall_at_5 +0.1")


def test_topline_strip_colors_respect_thresholds(store: TraceStore) -> None:
    metrics = {
        "recall_at_5": 0.85,  # >= 0.80 green
        "exclusion_precision": 0.96,  # < 1.00 but >= 0.95 amber
        "adherence": 0.50,  # < 0.90 red
        "p95_latency_ms": 2500,  # <= 3000 green (lower is better)
        "cost_per_conv_usd": 0.20,  # > 0.10 red
    }

    strip = topline_strip(metrics)
    by_key = {c["key"]: c for c in strip}

    assert by_key["recall_at_5"]["status"] == "green"
    assert by_key["exclusion_precision"]["status"] == "amber"
    assert by_key["adherence"]["status"] == "red"
    assert by_key["p95_latency_ms"]["status"] == "green"
    assert by_key["cost_per_conv_usd"]["status"] == "red"


def test_topline_strip_from_store_uses_latest_snapshot(store: TraceStore) -> None:
    _seed_three_changes(store)  # newest metric_after recall_at_5 = 0.90

    strip = topline_strip(store)
    by_key = {c["key"]: c for c in strip}

    assert by_key["recall_at_5"]["value"] == pytest.approx(0.90)
    assert by_key["recall_at_5"]["status"] == "green"
    # A metric never recorded shows as n/a + gray.
    assert by_key["adherence"]["value"] is None
    assert by_key["adherence"]["status"] == "gray"
