"""Streamlit dashboard shell for the IMDb RAG harness.

Two pages:

1. Chat - a conversation (st.chat_input / st.chat_message) that runs a turn
   handler and renders the RecommendationSet as poster cards, with a
   relax-a-constraint fallback for empty/dead-end turns.
2. Change Ledger - reads the change_ledger table via TraceStore. With an
   empty DB it renders an empty ledger without error.

Run with:  streamlit run src/imdb_chatbot/dashboard/app.py

The graph call is kept behind a thin injected handler boundary and all
render-data prep lives in ``render`` (pure, Streamlit-free) so the card mapping
and the fallback branch stay unit-testable without starting a server.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import streamlit as st

from ..schemas import RecommendationSet
from ..store import TraceStore
from .data import (
    TOPLINE_SPECS,
    ledger_table_display,
    load_change_ledger,
    metric_timeline,
    topline_strip,
)
from .render import is_fallback, recommendation_cards, relax_options

# Traffic-light color -> a hex swatch for the topline strip.
_STATUS_HEX = {
    "green": "#2e7d32",
    "amber": "#f9a825",
    "red": "#c62828",
    "gray": "#9e9e9e",
}

# Default trace-store location; override with TRACE_STORE_PATH.
DEFAULT_STORE_PATH = Path(__file__).resolve().parents[3] / "data" / "traces.db"

# A turn handler is the thin boundary over the graph: query -> RecommendationSet.
ChatHandler = Callable[[str], RecommendationSet]

_CHAT_HISTORY_KEY = "chat_history"


def _store_path() -> str:
    return os.environ.get("TRACE_STORE_PATH", str(DEFAULT_STORE_PATH))


def _default_handler(query: str) -> RecommendationSet:
    """Fallback handler used when no graph-backed handler is injected.

    Wiring the real graph needs a retriever + models bundle (network + index),
    so the default returns an empty set that exercises the relax-a-constraint
    path. Callers inject a graph-backed handler to get real picks.
    """
    return RecommendationSet(
        picks=[],
        prose=(
            "I could not find a match for that. Try relaxing one constraint "
            "using a quick reply below."
        ),
    )


def _render_cards(rec: RecommendationSet) -> None:
    """Render each pick as a poster card from structured fields only."""
    cards = recommendation_cards(rec)
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards, strict=True):
        with column:
            # Broken/missing poster is already a placeholder via poster_src.
            st.image(card["poster"], use_container_width=True)
            st.markdown(f"**{card['title']}** ({card['year']})")
            st.caption(card["reason"])


def _render_fallback(rec: RecommendationSet) -> None:
    """Render fallback prose plus deterministic relax-a-constraint buttons."""
    if rec.prose:
        st.write(rec.prose)
    columns = st.columns(len(relax_options()))
    for column, option in zip(columns, relax_options(), strict=True):
        with column:
            if st.button(option["label"], key=f"relax-{option['label']}"):
                _handle_turn(option["query"])
                st.rerun()


def _render_response(rec: RecommendationSet) -> None:
    if is_fallback(rec):
        _render_fallback(rec)
    else:
        _render_cards(rec)


def _handle_turn(query: str) -> None:
    """Run one turn: append the user message, call the handler, store the reply."""
    handler: ChatHandler = st.session_state.get("chat_handler", _default_handler)
    rec = handler(query)
    history = st.session_state.setdefault(_CHAT_HISTORY_KEY, [])
    history.append({"role": "user", "text": query})
    history.append({"role": "assistant", "rec": rec})


def render_chat_page(handler: ChatHandler | None = None) -> None:
    st.title("Chat")
    if handler is not None:
        st.session_state["chat_handler"] = handler

    history = st.session_state.setdefault(_CHAT_HISTORY_KEY, [])
    for message in history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["text"])
            else:
                _render_response(message["rec"])

    prompt = st.chat_input("Ask for a movie recommendation")
    if prompt:
        _handle_turn(prompt)
        st.rerun()


def _render_topline_strip(store: TraceStore) -> None:
    """Draw the current topline metrics as colored status tiles."""
    strip = topline_strip(store)
    columns = st.columns(len(strip))
    for column, cell in zip(columns, strip, strict=True):
        with column:
            hexcolor = _STATUS_HEX.get(cell["status"], _STATUS_HEX["gray"])
            st.markdown(
                f"<div style='border-left:6px solid {hexcolor};padding-left:8px'>"
                f"<small>{cell['label']}</small><br>"
                f"<span style='font-size:1.4rem'>{cell['display']}</span></div>",
                unsafe_allow_html=True,
            )


def _render_metric_timeline(store: TraceStore) -> None:
    """Draw a metric-selectable timeline with a vertical marker per promoted change."""
    metric_key = st.selectbox(
        "Metric",
        [spec.key for spec in TOPLINE_SPECS],
        format_func=lambda k: next(s.label for s in TOPLINE_SPECS if s.key == k),
    )
    timeline = metric_timeline(store, metric_key)
    points = timeline["points"]
    if points:
        series = {p["ts"]: p["value"] for p in points}
        st.line_chart(series)
    else:
        st.caption("No promoted changes carry this metric yet.")
    for marker in timeline["markers"]:
        delta = marker["delta"]
        delta_txt = f" (delta {delta:+.3g})" if delta is not None else ""
        # Each marker is clickable -> reveals its change_id / detail.
        with st.expander(f"{marker['ts']} - {marker['label']}{delta_txt}"):
            st.write(f"change_id: {marker['change_id']}")
            st.write(f"artifact_type: {marker['artifact_type']}")
            st.write(f"value: {marker['value']}")


def render_change_ledger_page() -> None:
    st.title("Change Ledger")
    path = _store_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    store = TraceStore(path)
    try:
        rows = load_change_ledger(store)
        if not rows:
            st.info("No changes recorded yet.")
            return

        st.subheader("Topline health")
        _render_topline_strip(store)

        st.subheader("Metric timeline")
        _render_metric_timeline(store)

        st.subheader("Ledger")
        st.dataframe(ledger_table_display(store), use_container_width=True)
    finally:
        store.close()


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
