"""Streamlit dashboard shell for the IMDb RAG harness.

Two pages proving the UI/DB seam before any real data exists:

1. Chat - a placeholder only. Real chat logic arrives in a later ticket.
2. Change Ledger - reads the change_ledger table via TraceStore. With an
   empty DB it renders an empty ledger without error.

Run with:  streamlit run src/imdb_chatbot/dashboard/app.py

All data access is delegated to ``data.load_change_ledger`` so the read logic
stays unit-testable independently of this Streamlit rendering code.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from ..store import TraceStore
from .data import LEDGER_COLUMNS, load_change_ledger

# Default trace-store location; override with TRACE_STORE_PATH.
DEFAULT_STORE_PATH = Path(__file__).resolve().parents[3] / "data" / "traces.db"


def _store_path() -> str:
    return os.environ.get("TRACE_STORE_PATH", str(DEFAULT_STORE_PATH))


def render_chat_page() -> None:
    st.title("Chat")
    st.info("Chat arrives in a later ticket. This is a placeholder for now.")


def render_change_ledger_page() -> None:
    st.title("Change Ledger")
    path = _store_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    store = TraceStore(path)
    try:
        rows = load_change_ledger(store)
    finally:
        store.close()

    if not rows:
        st.info("No changes recorded yet.")
        return

    st.dataframe(rows, use_container_width=True, column_order=LEDGER_COLUMNS)


PAGES = {
    "Chat": render_chat_page,
    "Change Ledger": render_change_ledger_page,
}


def main() -> None:
    st.set_page_config(page_title="IMDb Chatbot Dashboard", layout="wide")
    choice = st.sidebar.radio("Page", list(PAGES.keys()))
    PAGES[choice]()


if __name__ == "__main__":
    main()
