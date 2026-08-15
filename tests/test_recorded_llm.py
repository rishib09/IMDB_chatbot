"""Recorded-cassette tests at the LLM boundary (ticket #69).

These exercise the same OpenRouter seams the live handler wires up - the cheap
chat model, the title extractor, the follow-up classifier, and the
structured-output extractor slot - through vcrpy cassettes committed under
``tests/cassettes/``. The cassettes are captures of REAL responses (recorded
once under dotenvx), not fakes, so CI replays them deterministically with no
network and no key. None of these needs the corpus or the index.

Re-record after a prompt or model-slot change:

    npx @dotenvx/dotenvx run -f .env -- pytest tests/test_recorded_llm.py --record-mode=rewrite

``vcr_config`` (tests/conftest.py) strips the Authorization header, so no key
material ever lands in a cassette.
"""

from __future__ import annotations

import os

import pytest

from imdb_chatbot.config import load_models_config
from imdb_chatbot.dashboard.live import _FOLLOWUP_PROMPT, _TITLE_EXTRACT_PROMPT
from imdb_chatbot.graph.models import _init_slot_model, build_models
from imdb_chatbot.persona import chat_system_prompt
from imdb_chatbot.router import FollowupKind, route_followup


@pytest.fixture(autouse=True)
def _placeholder_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay needs no real key, but the client requires one to be present."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-cassette-replay-placeholder")


def _ask_cheap_model(system_prompt: str, user_msg: str) -> str:
    """One round trip on the cheap (rewriter-slot) model, as the live handler does."""
    model = _init_slot_model("rewriter", load_models_config())
    resp = model.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
    )
    return str(resp.content).strip()


@pytest.mark.vcr
def test_chitchat_reply_stays_in_character() -> None:
    text = _ask_cheap_model(chat_system_prompt(), "how are you today?")

    assert text
    # Maya's chat prompt always pivots back to offering a movie.
    assert "movie" in text.lower()


@pytest.mark.vcr
def test_title_extractor_pulls_the_title() -> None:
    title = _ask_cheap_model(_TITLE_EXTRACT_PROMPT, "who directed Parasite?").strip('"')

    assert title.lower() == "parasite"


def _followup_classifier(query: str, last_titles: list[str]) -> str:
    """The follow-up one-shot exactly as ``load_live_resources`` phrases it."""
    titles = ", ".join(last_titles) or "(none)"
    return _ask_cheap_model(_FOLLOWUP_PROMPT, f"Movies just shown: {titles}\nUser's message: {query}")


@pytest.mark.vcr
def test_followup_classifier_routes_a_narrowing_message() -> None:
    kind = route_followup(
        "only the ones from after 2010",
        ["Inception (2010)", "Interstellar (2014)", "Memento (2000)"],
        _followup_classifier,
    )

    assert kind is FollowupKind.REFINE


@pytest.mark.vcr
def test_followup_classifier_routes_a_vague_rejection() -> None:
    # Phrased past the bare-negation regex on purpose, so the LLM decides.
    raw = _followup_classifier(
        "no, I don't like these at all", ["Inception (2010)", "Interstellar (2014)"]
    )

    assert "clarify" in raw.lower()


@pytest.mark.vcr
def test_extractor_slot_parses_a_structured_query() -> None:
    parsed = build_models(load_models_config()).extract(
        "movies directed by Christopher Nolan after 2010"
    )

    assert parsed.director is not None and "nolan" in parsed.director.lower()
    assert parsed.min_year in (2010, 2011)
