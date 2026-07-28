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
from .live import ChatReply, TurnTelemetry
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

# A turn handler is the thin boundary over the graph: query -> ChatReply
# (the answer plus optional per-turn telemetry).
ChatHandler = Callable[[str], ChatReply]

_CHAT_HISTORY_KEY = "chat_history"
# A submitted-but-not-yet-answered query. Set on submit, consumed on the next
# rerun so the assistant bubble can show a loading spinner WHILE the turn runs
# (ticket #45). Cleared once the reply is stored.
_PENDING_KEY = "pending_query"
# Running per-session usage totals for the compact line below the input (#46).
_SESSION_USAGE_KEY = "session_usage"

# CSS that lifts Streamlit's pinned chat input a little and fixes a thin session
# usage bar at the very bottom, so the running total sits BELOW the input box.
_SESSION_FOOTER_CSS = """
<style>
[data-testid="stBottom"] { bottom: 1.9rem !important; }
.session-usage-footer {
    position: fixed;
    left: 0; right: 0; bottom: 0;
    z-index: 1000;
    height: 1.9rem;
    line-height: 1.9rem;
    padding: 0 14px;
    text-align: center;
    font-size: 0.78rem;
    color: #9aa0a6;
    background: rgba(14, 17, 23, 0.94);
    border-top: 1px solid rgba(255, 255, 255, 0.07);
}
</style>
"""


def _store_path() -> str:
    return os.environ.get("TRACE_STORE_PATH", str(DEFAULT_STORE_PATH))


def _default_handler(query: str) -> ChatReply:
    """Fallback handler used when no graph-backed handler is injected.

    Wiring the real graph needs a retriever + models bundle (network + index),
    so the default returns an empty set that exercises the relax-a-constraint
    path, with no telemetry (no LLM ran). Callers inject a graph-backed handler
    to get real picks and token/cost telemetry.
    """
    rec = RecommendationSet(
        picks=[],
        prose=(
            "I could not find a match for that. Try relaxing one constraint "
            "using a quick reply below."
        ),
    )
    return ChatReply(rec=rec, telemetry=None)


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


def _render_fallback(rec: RecommendationSet, key_prefix: str = "") -> None:
    """Render fallback prose plus deterministic relax-a-constraint buttons.

    ``key_prefix`` disambiguates the button keys: the same relax options are
    rendered once per fallback turn in the history, so keying on the label alone
    collides (StreamlitDuplicateElementKey) the moment two turns dead-end. The
    caller passes the message's position in history to make each key unique.
    """
    if rec.prose:
        st.write(rec.prose)
    columns = st.columns(len(relax_options()))
    for column, option in zip(columns, relax_options(), strict=True):
        with column:
            if st.button(option["label"], key=f"relax-{key_prefix}-{option['label']}"):
                _submit_query(option["query"])
                st.rerun()


def _render_telemetry(telemetry: TurnTelemetry) -> None:
    """One compact line per turn: answer model, tokens up/down, cost (#46).

    Replaces the old three large metric tiles - the numbers were dominating the
    conversation. The generator is the answer model; other slots are omitted here
    (the full per-slot breakdown still lives in the persisted TurnTrace).
    """
    model = telemetry.models.get("generator") or next(iter(telemetry.models.values()), "n/a")
    line = (
        f"{model}  |  up {telemetry.input_tokens:,} / down "
        f"{telemetry.output_tokens:,} tok  |  est ${telemetry.cost_usd:.4f}"
    )
    if telemetry.degradation:
        line += "  |  degraded: " + ", ".join(telemetry.degradation)
    st.caption(line)


def _accumulate_usage(telemetry: TurnTelemetry) -> None:
    """Fold one turn's usage into the running session totals."""
    totals = st.session_state.setdefault(
        _SESSION_USAGE_KEY, {"input": 0, "output": 0, "cost": 0.0, "turns": 0}
    )
    totals["input"] += telemetry.input_tokens
    totals["output"] += telemetry.output_tokens
    totals["cost"] += telemetry.cost_usd
    totals["turns"] += 1


def _render_session_usage() -> None:
    """A single compact session-total line, fixed below the input box (#46, #3)."""
    totals = st.session_state.get(_SESSION_USAGE_KEY)
    if not totals or not totals["turns"]:
        return
    line = (
        f"Session: {totals['turns']} turn(s)  |  up {totals['input']:,} / down "
        f"{totals['output']:,} tok  |  est ${totals['cost']:.4f}"
    )
    st.markdown(_SESSION_FOOTER_CSS, unsafe_allow_html=True)
    st.markdown(f"<div class='session-usage-footer'>{line}</div>", unsafe_allow_html=True)


def _render_response(reply: ChatReply, key_prefix: str = "") -> None:
    rec = reply.rec
    if reply.conversational:
        # Persona reply (Maya's greeting / purpose): prose only, no relax buttons.
        if rec.prose:
            st.write(rec.prose)
    elif is_fallback(rec):
        _render_fallback(rec, key_prefix)
    else:
        _render_cards(rec)
    if reply.telemetry is not None:
        _render_telemetry(reply.telemetry)


def _submit_query(query: str) -> None:
    """Record a user message and mark it pending so the next rerun answers it.

    Splitting submit from answer (ticket #45) lets the user's message and a
    loading spinner render immediately, before the (multi-second) turn runs.
    """
    history = st.session_state.setdefault(_CHAT_HISTORY_KEY, [])
    history.append({"role": "user", "text": query})
    st.session_state[_PENDING_KEY] = query


def _process_pending() -> None:
    """Answer the pending query inside an assistant bubble, with a live spinner."""
    pending = st.session_state.get(_PENDING_KEY)
    if pending is None:
        return
    handler: ChatHandler = st.session_state.get("chat_handler", _default_handler)
    history = st.session_state.setdefault(_CHAT_HISTORY_KEY, [])
    with st.chat_message("assistant"):
        with st.spinner("Maya is thinking..."):
            reply = handler(pending)
        # key_prefix = the index this message will occupy once appended, so any
        # relax buttons keep unique keys across the following reruns.
        _render_response(reply, key_prefix=str(len(history)))
    history.append({"role": "assistant", "reply": reply})
    if reply.telemetry is not None:
        _accumulate_usage(reply.telemetry)
    st.session_state[_PENDING_KEY] = None


def render_chat_page(handler: ChatHandler | None = None) -> None:
    st.title("Chat")
    if handler is not None:
        st.session_state["chat_handler"] = handler

    # When the live graph handler could not be built, the entry point records a
    # reason; show it once so the user knows recommendations are stubbed.
    offline_reason = st.session_state.get("chat_offline_reason")
    if offline_reason and st.session_state.get("chat_handler") is None:
        st.info(
            "Offline demo mode - real recommendations are disabled: "
            f"{offline_reason}"
        )

    history = st.session_state.setdefault(_CHAT_HISTORY_KEY, [])
    for idx, message in enumerate(history):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["text"])
            else:
                _render_response(message["reply"], key_prefix=str(idx))

    # Answer a just-submitted query (renders the spinner in-line, below history).
    _process_pending()

    # Compact running usage line, just above the input (#46).
    _render_session_usage()

    prompt = st.chat_input("Ask for a movie recommendation")
    if prompt:
        _submit_query(prompt)
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
