"""Tests for the end-to-end loop demo and the HF Spaces app entry (ticket #26).

Deterministic and network-free:

- The full seed -> detect -> gate -> ledger cycle asserts each stage (the S2 code
  fired, a PROMOTE decision, exactly one ChangeRecord with the right fields).
- The regressing-candidate variant asserts the gate REJECTS and writes no record.
- ``app.py`` is exercised at the function level (health check returns unhealthy
  for the repo's blank index pointer WITHOUT raising, and never starts a server).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from imdb_chatbot.demo import narrate, run_loop, seed_failure_trace
from imdb_chatbot.demo.__main__ import main as demo_main
from imdb_chatbot.demo.loop import SEEDED_CODE
from imdb_chatbot.store import TraceStore

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def store(tmp_path: Path) -> TraceStore:
    s = TraceStore(tmp_path / "loop.sqlite")
    try:
        yield s
    finally:
        s.close()


def _ledger_rows(store: TraceStore) -> list:
    with store._read_lock:
        return store._read_conn.execute("SELECT change_id FROM change_ledger").fetchall()


# -- the full cycle -----------------------------------------------------------


def test_full_loop_seed_detect_gate_ledger(store: TraceStore) -> None:
    result = run_loop(store, trace_id="demo-trace-s2")

    # SEED: the failing trace is written to the system of record.
    assert store.read_trace("demo-trace-s2") is not None

    # DETECT: the seeded repetition fires exactly the expected taxonomy code.
    assert SEEDED_CODE == "S2"
    assert result.detected_codes == ["S2"]

    # GATE: motivating case now passes, nothing regresses -> PROMOTE.
    assert result.promoted is True
    assert result.decision.reasons == []

    # LEDGER: exactly one ChangeRecord, carrying the motivating trace id + deltas.
    assert result.record is not None
    rows = _ledger_rows(store)
    assert len(rows) == 1
    stored = store.read_change(result.record.change_id)
    assert stored is not None
    assert stored.motivating_trace_ids == ["demo-trace-s2"]
    assert stored.artifact_type == "prompt"
    assert stored.metric_before == {"recall_at_5": 0.80, "exclusion_precision": 0.95}
    assert stored.metric_after == {"recall_at_5": 0.84, "exclusion_precision": 0.96}
    # The recorded delta matches what the gate weighed.
    assert result.metric_deltas["recall_at_5"] == pytest.approx(0.04)


def test_full_loop_narrative_mentions_each_stage(store: TraceStore) -> None:
    text = narrate(run_loop(store))
    for token in ("SEED", "DETECT", "S2", "PROMOTE", "LEDGER", "ChangeRecord"):
        assert token in text


def test_seed_trace_exhibits_repetition() -> None:
    trace = seed_failure_trace("t-abc")
    assert trace.response is not None
    picks = trace.response.picks
    # Same film recommended twice -> the intra-response duplicate S2 signal.
    assert len(picks) == 2
    assert (picks[0].title, picks[0].year) == (picks[1].title, picks[1].year)


# -- the regressing variant ---------------------------------------------------


def test_regressing_candidate_is_rejected_and_writes_no_record(store: TraceStore) -> None:
    result = run_loop(store, regress=True)

    # The failure is still detected...
    assert result.detected_codes == ["S2"]
    # ...but the candidate regresses recall, so the gate REJECTS.
    assert result.promoted is False
    assert result.record is None
    assert any("regressed beyond noise" in r for r in result.decision.reasons)
    # No unearned ledger row.
    assert _ledger_rows(store) == []


# -- CLI ----------------------------------------------------------------------


def test_demo_cli_runs_green(capsys: pytest.CaptureFixture[str]) -> None:
    assert demo_main([]) == 0
    out = capsys.readouterr().out
    assert "PROMOTE" in out


def test_demo_cli_regress_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert demo_main(["--regress"]) == 0
    out = capsys.readouterr().out
    assert "REJECT" in out


# -- app.py entry (function level, no Streamlit server) -----------------------


def _load_app_module():
    spec = importlib.util.spec_from_file_location("hf_app", REPO_ROOT / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_health_check_unhealthy_for_blank_index_without_raising() -> None:
    app = _load_app_module()
    # The repo's config/live_index.json has active=null (no index shipped), so
    # the L3 health check must report unhealthy - and must NOT raise.
    health = app.get_health()
    assert health.healthy is False
    assert health.checks["pointer"] is False
    assert "unavailable" in health.message.lower()


def test_app_budget_not_exhausted_by_default() -> None:
    app = _load_app_module()
    # A fresh budget tracker has recorded no spend -> normal serving (not L2).
    assert app.budget_exhausted() is False
