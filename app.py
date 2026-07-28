"""Hugging Face Spaces entry point (Streamlit SDK) - ticket #26.

HF Spaces runs this file with ``streamlit run app.py``. Before serving a single
turn it runs the L3 startup health check (``guards.degradation.health_check``):
the live-index pointer must resolve, the index artifacts must exist, and the
trace DB must open. If any of that fails the app renders the honest "index
unavailable" message (L3) instead of crashing - a degrade, never a stack trace
in the user's face.

When healthy, budget exhaustion is wired to L2 (LLM-free mode): if the daily
production budget is spent the app shows the L2 banner and still serves the
deterministic dashboard, rather than erroring out.

The health check and budget logic live in importable functions so they can be
unit-tested at the function level WITHOUT starting a Streamlit server (the
``st`` import is deferred into ``main``).
"""

from __future__ import annotations

from pathlib import Path

from imdb_chatbot.guards.budget import BudgetTracker
from imdb_chatbot.guards.degradation import (
    L2_BANNER,
    HealthResult,
    health_check,
    honest_exit_message,
)
from imdb_chatbot.store import TraceStore

REPO_ROOT = Path(__file__).resolve().parent
LIVE_INDEX_PATH = REPO_ROOT / "config" / "live_index.json"
DB_PATH = REPO_ROOT / "data" / "traces.db"


def _db_opener() -> TraceStore:
    """Open the trace DB for the L3 health check (creates the data dir if needed).

    The health check calls this and immediately closes the handle; a raised
    opener is treated as broken storage and degrades to the honest exit.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return TraceStore(DB_PATH)


def get_health() -> HealthResult:
    """Run the L3 startup health check against the repo's live index + trace DB.

    Never raises: a missing/blank index pointer, missing index artifacts, or a
    broken DB all come back as an unhealthy ``HealthResult`` so the caller can
    render the honest-exit message instead of crashing.
    """
    return health_check(live_index_path=LIVE_INDEX_PATH, db_opener=_db_opener)


def budget_exhausted(tracker: BudgetTracker | None = None) -> bool:
    """True when today's production spend has tripped L2 (LLM-free mode).

    A fresh tracker (no recorded spend) reports ``False`` -> normal serving. The
    tracker is injectable so this stays deterministic in tests.
    """
    tracker = tracker or BudgetTracker()
    return tracker.exhausted()


def main() -> None:
    """Streamlit entry: L3 health gate -> (L2 budget notice) -> dashboard."""
    import streamlit as st

    from imdb_chatbot.dashboard.app import main as dashboard_main

    st.set_page_config(page_title="IMDb Chatbot", layout="wide")

    health = get_health()
    if not health.healthy:
        # L3: refuse to serve broken - show the honest exit, do not crash.
        st.title("IMDb Chatbot")
        st.error("Index unavailable")
        st.write(honest_exit_message())
        return

    if budget_exhausted():
        # L2: degrade to a clear banner rather than erroring on spend exhaustion.
        st.warning(L2_BANNER)

    dashboard_main()


if __name__ == "__main__":
    main()
