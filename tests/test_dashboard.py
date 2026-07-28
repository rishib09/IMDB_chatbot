"""Tests for the dashboard data-access layer.

Only the plain, non-Streamlit ``load_change_ledger`` function is exercised
here - the Streamlit server is never started in tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from imdb_chatbot.dashboard.data import load_change_ledger
from imdb_chatbot.schemas import ChangeRecord
from imdb_chatbot.store import TraceStore


@pytest.fixture()
def store(tmp_path: Path) -> TraceStore:
    s = TraceStore(tmp_path / "trace.sqlite")
    try:
        yield s
    finally:
        s.close()


def test_load_change_ledger_empty(store: TraceStore) -> None:
    assert load_change_ledger(store) == []


def test_load_change_ledger_one_row(store: TraceStore) -> None:
    rec = ChangeRecord(
        change_id="chg-1",
        ts=datetime(2026, 7, 25, 12, 0, 0, tzinfo=UTC),
        artifact_type="prompt",
        version_after="p-v2",
        motivating_trace_ids=["trace-1"],
    )
    store.write_change(rec)

    rows = load_change_ledger(store)

    assert len(rows) == 1
    row = rows[0]
    assert row["change_id"] == "chg-1"
    assert row["artifact_type"] == "prompt"
    assert row["ts"] == rec.ts.isoformat()
