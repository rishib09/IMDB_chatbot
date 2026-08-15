"""Live integration tier (ticket #69): real model, real index, real corpus.

One test per intent path through the live chat handler - greeting, chit-chat,
meta, movie question, search, refine, replace, clarify, and the degraded /
offline fallbacks. Everything here is ``@pytest.mark.live``: deselected by
default and run explicitly (real network, real spend) with

    npx @dotenvx/dotenvx run -f .env -- pytest -m live -q

Assertions are shape/constraint checks (never exact title lists), and every
one carries ``_describe(reply)`` so a red run is diagnosable from the log
alone: path taken, degradation flags, models, tokens, cost, and picks.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from imdb_chatbot.dashboard.live import ChatReply, build_live_chat_handler
from imdb_chatbot.persona import chitchat_fallback
from imdb_chatbot.router import clarify_question

pytestmark = pytest.mark.live

# Robust topic probes: "did the answer stay on topic" is judged against a
# filmography, not an exact expected list, so reranking never breaks the test.
NOLAN_FILMS = {
    "Following",
    "Memento",
    "Insomnia",
    "Batman Begins",
    "The Prestige",
    "The Dark Knight",
    "The Dark Knight Rises",
    "Inception",
    "Interstellar",
    "Dunkirk",
    "Tenet",
    "Oppenheimer",
}

HANKS_FILMS = {
    "Big",
    "Philadelphia",
    "Forrest Gump",
    "Apollo 13",
    "Toy Story",
    "Saving Private Ryan",
    "You've Got Mail",
    "The Green Mile",
    "Cast Away",
    "Catch Me If You Can",
    "The Terminal",
    "The Polar Express",
    "Captain Phillips",
    "Sully",
    "Finch",
    "A Man Called Otto",
}


def _describe(reply: ChatReply) -> str:
    """Everything needed to diagnose a failed live turn from the log alone."""
    parts = [
        f"picks={[f'{p.title} ({p.year})' for p in reply.rec.picks]}",
        f"prose={reply.rec.prose!r}",
        f"conversational={reply.conversational}",
    ]
    meter = reply.telemetry
    if meter is None:
        parts.append("telemetry=None")
    else:
        parts.append(f"path={meter.path_taken}")
        parts.append(f"degradation={meter.degradation}")
        parts.append(f"models={meter.models}")
        parts.append(f"tokens={meter.input_tokens}/{meter.output_tokens}")
        parts.append(f"cost=${meter.cost_usd:.6f}")
    return " | ".join(parts)


@pytest.fixture()
def handler(live_resources):
    """A fresh session per test: no state bleeds between intent probes."""
    return build_live_chat_handler(live_resources)


@pytest.fixture()
def retry_scenario(live_resources):
    """Run a whole search scenario on a fresh session, retrying a model wobble.

    Even at temperature 0, OpenRouter occasionally serves a generation that
    Gate-4 rejects (ungrounded prose), turning a good search turn into the
    fallback. That is provider nondeterminism, not a code regression, so a
    scenario gets up to three fresh-session attempts; a genuine regression
    still fails every attempt and surfaces the last diagnostic.
    """

    def run(scenario, attempts: int = 3) -> None:
        for attempt in range(attempts):
            try:
                scenario(build_live_chat_handler(live_resources))
                return
            except AssertionError:
                if attempt == attempts - 1:
                    raise

    return run


def test_greeting_is_deterministic_persona_reply(handler) -> None:
    reply = handler("hi")

    assert reply.conversational, _describe(reply)
    assert reply.telemetry is None, _describe(reply)  # no LLM ran
    assert reply.rec.picks == [], _describe(reply)
    assert "maya" in reply.rec.prose.lower(), _describe(reply)


def test_chitchat_gets_one_cheap_llm_reply(handler) -> None:
    reply = handler("how are you today?")

    assert reply.conversational, _describe(reply)
    assert reply.rec.picks == [], _describe(reply)
    assert reply.rec.prose.strip(), _describe(reply)
    meter = reply.telemetry
    assert meter is not None, _describe(reply)
    assert list(meter.models) == ["chat"], _describe(reply)  # exactly one cheap call
    assert meter.input_tokens > 0 and meter.output_tokens > 0, _describe(reply)


def test_meta_purpose_reply(handler) -> None:
    reply = handler("what can you do?")

    assert reply.conversational, _describe(reply)
    assert reply.telemetry is None, _describe(reply)
    assert reply.rec.picks == [], _describe(reply)
    assert "recommend" in reply.rec.prose.lower(), _describe(reply)


def test_movie_question_answers_from_corpus(handler) -> None:
    reply = handler("who directed Parasite?")

    assert reply.conversational, _describe(reply)
    assert reply.rec.picks == [], _describe(reply)
    prose = reply.rec.prose.lower().replace("-", " ")
    assert "parasite" in prose, _describe(reply)
    assert "bong joon" in prose, _describe(reply)  # grounded in the corpus record


def test_search_returns_on_topic_cards(retry_scenario) -> None:
    def scenario(handler) -> None:
        reply = handler("recommend some christopher nolan movies")

        meter = reply.telemetry
        assert meter is not None, _describe(reply)
        assert reply.rec.picks, _describe(reply)
        assert "generate" in meter.path_taken, _describe(reply)
        assert "validate" in meter.path_taken, _describe(reply)
        assert "fallback" not in meter.path_taken, _describe(reply)
        titles = {p.title for p in reply.rec.picks}
        assert len(titles & NOLAN_FILMS) * 2 >= len(titles), _describe(reply)  # mostly Nolan

    retry_scenario(scenario)


def test_refine_narrows_the_previous_results(retry_scenario) -> None:
    # Narrowing to a decade moves the eligible pool almost entirely off the
    # titles already shown, so the refined turn has fresh candidates. (A
    # narrowing that overlaps the shown top-5 - e.g. Nolan "after 2010" -
    # strands today because the vote-count cap is applied before the
    # shown-movies filter; see the live-tier notes in ticket #69.)
    def scenario(handler) -> None:
        first = handler("recommend some good sci-fi movies")
        assert first.rec.picks, _describe(first)

        second = handler("only the ones from the 1980s")

        assert second.telemetry is not None, _describe(second)
        assert second.rec.picks, _describe(second)
        assert all(1980 <= p.year <= 1989 for p in second.rec.picks), _describe(second)
        titles = {p.title for p in second.rec.picks}
        assert not titles & {p.title for p in first.rec.picks}, _describe(second)  # no repeats

    retry_scenario(scenario)


def test_replace_switches_to_a_new_topic(retry_scenario) -> None:
    def scenario(handler) -> None:
        first = handler("recommend some christopher nolan movies")
        assert first.rec.picks, _describe(first)

        second = handler("actually, recommend some movies starring tom hanks instead")

        assert second.rec.picks, _describe(second)
        titles = {p.title for p in second.rec.picks}
        assert titles & HANKS_FILMS, _describe(second)  # the new topic landed
        assert not titles & NOLAN_FILMS, _describe(second)  # the old topic was dropped
        assert not titles & {p.title for p in first.rec.picks}, _describe(second)

    retry_scenario(scenario)


def test_vague_rejection_gets_a_clarifying_question(retry_scenario) -> None:
    # Seeded with the most stable search probe: vibe-phrased first turns
    # ("feel-good animated movies") intermittently lose the extractor's
    # structured output and then the generator honestly returns empty picks.
    def scenario(handler) -> None:
        first = handler("recommend some christopher nolan movies")
        assert first.rec.picks, _describe(first)

        second = handler("no, I don't like these at all")

        assert second.conversational, _describe(second)
        assert second.rec.picks == [], _describe(second)
        assert second.rec.prose == clarify_question(), _describe(second)

    retry_scenario(scenario)


def test_offline_chitchat_uses_deterministic_fallback(live_resources) -> None:
    offline = replace(live_resources, chat_fn=None)
    handler = build_live_chat_handler(offline)

    reply = handler("how are you today?")

    assert reply.conversational, _describe(reply)
    assert reply.telemetry is None, _describe(reply)
    assert reply.rec.prose == chitchat_fallback(), _describe(reply)


def test_degraded_models_fall_back_without_crashing(live_resources, monkeypatch) -> None:
    """Every LLM call failing (broken key) must degrade, never raise: the
    rewrite is skipped, extraction falls back to regex, and the failed
    generation is caught by validate and routed to the fallback prose."""
    from imdb_chatbot.config import load_models_config
    from imdb_chatbot.graph.models import build_models

    monkeypatch.setenv("IMDB_TEST_BROKEN_KEY", "sk-or-v1-deliberately-broken")
    broken_cfg = {**load_models_config(), "secret": "IMDB_TEST_BROKEN_KEY"}
    degraded = replace(
        live_resources,
        models_factory=lambda meter: build_models(broken_cfg, meter=meter),
        chat_fn=None,
        followup_fn=None,
        title_extractor=None,
    )
    handler = build_live_chat_handler(degraded)

    reply = handler("recommend a good heist movie")

    meter = reply.telemetry
    assert meter is not None, _describe(reply)
    assert reply.rec.prose.strip(), _describe(reply)  # a graceful reply, not a crash
    assert "rewrite_skipped" in meter.degradation, _describe(reply)
    assert "extract_regex_fallback" in meter.degradation, _describe(reply)
    assert "fallback" in meter.path_taken, _describe(reply)
