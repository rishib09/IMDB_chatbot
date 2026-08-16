"""Tests for the Chat-page render helpers.

The bulk exercise the Streamlit-free functions in ``dashboard.render`` (no
server started). The final test drives the whole page through Streamlit's
``AppTest`` harness to cover the relax-button key-uniqueness invariant, which
only breaks once the history holds more than one dead-end turn - something the
pure-function tests cannot see.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from imdb_chatbot.dashboard.app import render_chat_page
from imdb_chatbot.dashboard.live import ChatReply, TurnTelemetry
from imdb_chatbot.dashboard.render import (
    PLACEHOLDER_POSTER,
    is_fallback,
    poster_src,
    recommendation_cards,
    relax_options,
)
from imdb_chatbot.schemas import MovieRecommendation, RecommendationSet


def test_recommendation_cards_maps_structured_fields() -> None:
    rec = RecommendationSet(
        picks=[
            MovieRecommendation(
                title="Parasite",
                year=2019,
                reason="A tense class satire.",
                poster_url="https://example.com/parasite.jpg",
            ),
            MovieRecommendation(
                title="Whiplash",
                year=2014,
                reason="Relentless drumming drama.",
            ),
        ],
        prose="Here are two picks.",
    )

    cards = recommendation_cards(rec)

    assert cards == [
        {
            "title": "Parasite",
            "year": 2019,
            "reason": "A tense class satire.",
            "poster": "https://example.com/parasite.jpg",
        },
        {
            "title": "Whiplash",
            "year": 2014,
            "reason": "Relentless drumming drama.",
            "poster": PLACEHOLDER_POSTER,
        },
    ]


def test_poster_src_placeholder_for_none_and_empty() -> None:
    assert poster_src(None) == PLACEHOLDER_POSTER
    assert poster_src("") == PLACEHOLDER_POSTER
    assert poster_src("   ") == PLACEHOLDER_POSTER


def test_poster_src_returns_url_otherwise() -> None:
    url = "https://example.com/poster.png"
    assert poster_src(url) == url


def test_fallback_set_yields_no_cards_and_signals_relax_path() -> None:
    rec = RecommendationSet(picks=[], prose="I could not find a match.")

    assert recommendation_cards(rec) == []
    assert is_fallback(rec) is True

    options = relax_options()
    assert len(options) >= 1
    for option in options:
        assert option["label"]
        assert option["query"]


def test_non_empty_set_is_not_fallback() -> None:
    rec = RecommendationSet(
        picks=[MovieRecommendation(title="Dune", year=2021, reason="Epic sci-fi.")]
    )
    assert is_fallback(rec) is False


def test_relax_options_returns_independent_copies() -> None:
    first = relax_options()
    first[0]["label"] = "mutated"
    assert relax_options()[0]["label"] != "mutated"


def _chat_page_script(render) -> None:
    """Top-level script body for AppTest: render the Chat page only.

    AppTest re-executes this body in a fresh namespace, so the page function
    arrives as a kwarg rather than an ``import`` statement. Importing it here
    instead put the whole ``dashboard.app`` chain (store, eval.detect, graph)
    inside the timed script run - see ``_chat_app``.
    """
    render()


def _chat_app() -> AppTest:
    """An AppTest for the Chat page, with a script-run deadline that fits (#91).

    ``default_timeout`` bounds ONE script run, and Streamlit's 3s default was
    being spent on cold start rather than rendering: the first run in a process
    also pays the ScriptRunner bootstrap, measured at ~0.9-1.1s idle here, on
    top of ~1s for the cold ``dashboard.app`` import the script body used to do.
    Under full-suite load that overshot 3s and the test failed on the timeout,
    never reaching its assertions. Passing ``render`` in removes the import from
    the timed window; the raised budget covers the bootstrap that is left.
    Rendering itself is ~0.3s, so this is slack for machine load - a genuinely
    hung page still fails, just later.
    """
    return AppTest.from_function(
        _chat_page_script, default_timeout=15, kwargs={"render": render_chat_page}
    )


def test_two_dead_end_turns_do_not_collide_relax_button_keys() -> None:
    """Two fallback turns in history must render without a duplicate-key crash.

    Regression: relax-button keys were derived from the option label alone, so a
    second dead-end turn re-rendered the identical keys and Streamlit raised
    StreamlitDuplicateElementKey. Keying by history position fixes it.
    """
    fallback = ChatReply(
        rec=RecommendationSet(picks=[], prose="I could not find a match."),
        telemetry=None,
    )
    at = _chat_app()
    at.session_state["chat_history"] = [
        {"role": "user", "text": "first query"},
        {"role": "assistant", "reply": fallback},
        {"role": "user", "text": "second query"},
        {"role": "assistant", "reply": fallback},
    ]

    at.run()

    assert not at.exception
    # Both dead-end turns rendered their full relax button row.
    assert len(at.button) == 2 * len(relax_options())


def test_telemetry_strip_renders_model_tokens_and_cost() -> None:
    """A reply carrying telemetry shows the model, token, and cost metrics."""
    reply = ChatReply(
        rec=RecommendationSet(
            picks=[MovieRecommendation(title="Parasite", year=2019, reason="Tense.")],
            prose="One pick.",
        ),
        telemetry=TurnTelemetry(
            models={"generator": "deepseek/deepseek-chat"},
            input_tokens=1234,
            output_tokens=56,
            cost_usd=0.0021,
        ),
    )
    at = _chat_app()
    at.session_state["chat_history"] = [
        {"role": "user", "text": "a pick please"},
        {"role": "assistant", "reply": reply},
    ]

    at.run()

    assert not at.exception
    # Compact single-line telemetry (#46): a caption, NOT three big metric tiles.
    assert len(at.metric) == 0
    captions = " ".join(c.value for c in at.caption)
    assert "deepseek/deepseek-chat" in captions
    assert "1,234" in captions  # tokens uploaded, formatted
    assert "$0.0021" in captions  # cost


def _telemetry_reply():
    return ChatReply(
        rec=RecommendationSet(
            picks=[MovieRecommendation(title="Dune", year=2021, reason="Epic sci-fi.")],
            prose="One pick.",
        ),
        telemetry=TurnTelemetry(
            models={"generator": "deepseek/deepseek-chat"},
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        ),
    )


def _fake_reply_handler(query: str) -> ChatReply:
    return _telemetry_reply()


def test_pending_query_is_answered_and_usage_accumulates() -> None:
    """A submitted (pending) query is answered on rerun and folds into session usage."""
    at = _chat_app()
    at.session_state["chat_handler"] = _fake_reply_handler
    at.session_state["chat_history"] = [{"role": "user", "text": "a sci-fi movie"}]
    at.session_state["pending_query"] = "a sci-fi movie"

    at.run()

    assert not at.exception
    # The pending query was consumed and the assistant reply rendered.
    assert at.session_state["pending_query"] is None
    markdown = " ".join(m.value for m in at.markdown)
    assert "Dune" in markdown
    # Session usage line accumulated exactly one turn.
    assert at.session_state["session_usage"]["turns"] == 1
    assert at.session_state["session_usage"]["input"] == 100
    # The session total is now a fixed footer rendered via st.markdown (#3).
    markdown_all = " ".join(m.value for m in at.markdown)
    assert "Session: 1 turn(s)" in markdown_all
