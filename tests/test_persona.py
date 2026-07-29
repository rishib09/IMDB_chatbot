"""Maya persona + intent gate (ticket #44).

Covers the deterministic classifier, the persona replies, and the live handler
routing small talk / meta questions away from the movie pipeline (no picks, no
telemetry) while genuine searches still run the graph.
"""

from __future__ import annotations

import pytest

from imdb_chatbot.dashboard.live import LiveResources, build_live_chat_handler
from imdb_chatbot.graph.models import GraphModels
from imdb_chatbot.graph.usage import UsageMeter
from imdb_chatbot.movie_info import build_title_lookup
from imdb_chatbot.persona import (
    Intent,
    chat_system_prompt,
    chitchat_fallback,
    classify_intent,
    generator_system_prompt,
    persona_reply,
)
from imdb_chatbot.schemas import (
    MovieRecommendation,
    MovieRecord,
    ParsedQuery,
    RecommendationSet,
    ScoredMovie,
)


@pytest.mark.parametrize(
    "message",
    ["hi", "Hello", "hey there", "yo", "  hi!!  ", "thanks", "thank you", "bye", ""],
)
def test_greetings_classify_as_greeting(message: str) -> None:
    assert classify_intent(message) is Intent.GREETING


@pytest.mark.parametrize(
    "message",
    [
        "what is the purpose of this chatbot?",
        "who are you",
        "what can you do?",
        "tell me about yourself",
        "how does this work",
        "help",
    ],
)
def test_meta_questions_classify_as_meta(message: str) -> None:
    assert classify_intent(message) is Intent.META


@pytest.mark.parametrize(
    "message",
    [
        "how are you doing?",
        "how's it going",
        "what's up",
        "tell me a joke",
        "do you like movies?",
        "how have you been?",
    ],
)
def test_conversational_messages_classify_as_chitchat(message: str) -> None:
    assert classify_intent(message) is Intent.CHITCHAT


@pytest.mark.parametrize(
    "message",
    [
        "who is the cast of Parasite?",
        "what is Inception about",
        "who directed The Dark Knight",
        "have you heard of movie obsession?",
        "tell me about the movie Oldboy",
    ],
)
def test_movie_questions_classify_as_movie_question(message: str) -> None:
    assert classify_intent(message) is Intent.MOVIE_QUESTION


@pytest.mark.parametrize(
    "message",
    [
        "recommend a good thriller",
        "what's a good horror movie",
        "a gritty korean revenge thriller",
        "movies with tom hanks",
    ],
)
def test_recommendation_requests_are_not_movie_questions(message: str) -> None:
    assert classify_intent(message) is not Intent.MOVIE_QUESTION


@pytest.mark.parametrize(
    "message",
    [
        "high school movies",  # contains 'hi' but is a search
        "a gritty korean revenge thriller",
        "recommend me a horror movie from 2000",
        "help me find a feel-good animated film",  # 'help' + real request
        "something like Parasite",
        "recommend some christopher nolan movies",
    ],
)
def test_real_requests_classify_as_search(message: str) -> None:
    assert classify_intent(message) is Intent.SEARCH


def test_chat_and_chitchat_fallback_prompts_present() -> None:
    assert "Maya" in chat_system_prompt()
    assert "recommend" in chitchat_fallback().lower()


def test_persona_reply_greeting_and_meta_mention_maya() -> None:
    greeting = persona_reply(Intent.GREETING)
    purpose = persona_reply(Intent.META)
    assert greeting.picks == [] and "Maya" in greeting.prose
    assert purpose.picks == [] and "Maya" in purpose.prose


def test_persona_reply_rejects_search() -> None:
    with pytest.raises(ValueError):
        persona_reply(Intent.SEARCH)


def test_generator_system_prompt_is_mayas_voice() -> None:
    prompt = generator_system_prompt()
    assert "Maya" in prompt
    assert prompt  # non-empty


# -- live handler routing -----------------------------------------------------


def _fake_retriever(query: str, parsed: ParsedQuery):
    return [ScoredMovie(tmdb_id=1, title="Parasite", year=2019, regions=["KR"])]


def _fake_models_factory(meter: UsageMeter | None) -> GraphModels:
    def rewrite(raw_query, history):
        if meter is not None:
            meter.record("rewriter", model="m", input_tokens=10, output_tokens=2)
        return raw_query

    def extract(query):
        if meter is not None:
            meter.record("extractor", model="m", input_tokens=10, output_tokens=2)
        return ParsedQuery()

    def generate(query, candidates):
        if meter is not None:
            meter.record("generator", model="deepseek/deepseek-chat", input_tokens=100, output_tokens=30)
        return RecommendationSet(
            picks=[MovieRecommendation(title="Parasite", year=2019, reason="Tense.")],
            prose="A gripping pick.",
        )

    return GraphModels(rewrite=rewrite, extract=extract, generate=generate)


def _fake_chat_fn(message, meter):
    if meter is not None:
        meter.record("chat", model="google/gemma-3-12b-it", input_tokens=20, output_tokens=15)
    return "I'm doing great! Would you like me to recommend a movie, or just chatting?"


class _FakeStore:
    """Minimal store: write_trace is a no-op; read_movie returns a poster holder."""

    def __init__(self, posters: dict[int, str]):
        self._posters = posters

    def write_trace(self, trace):
        pass

    def read_movie(self, tmdb_id: int):
        url = self._posters.get(tmdb_id)
        if url is None:
            return None
        return type("M", (), {"poster_url": url})()


def _resources(
    chat_fn=None, store=None, followup_fn=None, movie_lookup=None, title_extractor=None
) -> LiveResources:
    return LiveResources(
        retriever=_fake_retriever,
        models_factory=_fake_models_factory,
        pricing={
            "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
            "google/gemma-3-12b-it": {"input": 0.05, "output": 0.10},
        },
        versions={"index": "test", "model_config": "test", "prompt": "persona-v2"},
        store=store,
        chat_fn=chat_fn,
        followup_fn=followup_fn,
        movie_lookup=movie_lookup,
        title_extractor=title_extractor,
    )


def test_handler_routes_greeting_to_persona_no_pipeline() -> None:
    handler = build_live_chat_handler(_resources())

    reply = handler("hi")

    assert reply.conversational is True
    assert reply.rec.picks == []
    assert "Maya" in reply.rec.prose
    # No LLM / retrieval ran -> no telemetry strip.
    assert reply.telemetry is None


def test_handler_routes_meta_to_persona() -> None:
    handler = build_live_chat_handler(_resources())

    reply = handler("what is the purpose of this chatbot?")

    assert reply.conversational is True
    assert reply.rec.picks == []
    assert "Maya" in reply.rec.prose


def test_handler_runs_graph_for_real_search() -> None:
    handler = build_live_chat_handler(_resources())

    reply = handler("a gritty korean revenge thriller")

    assert reply.conversational is False
    assert [p.title for p in reply.rec.picks] == ["Parasite"]
    assert reply.telemetry is not None
    assert reply.telemetry.input_tokens > 0


def test_handler_routes_chitchat_to_llm_with_telemetry() -> None:
    handler = build_live_chat_handler(_resources(chat_fn=_fake_chat_fn))

    reply = handler("how are you doing?")

    assert reply.conversational is True
    assert reply.rec.picks == []
    assert "recommend a movie" in reply.rec.prose
    # The cheap chat call is metered -> small telemetry (not None).
    assert reply.telemetry is not None
    assert reply.telemetry.input_tokens == 20
    assert reply.telemetry.cost_usd > 0


def test_handler_chitchat_falls_back_without_chat_fn() -> None:
    handler = build_live_chat_handler(_resources(chat_fn=None))

    reply = handler("how are you doing?")

    assert reply.conversational is True
    assert reply.rec.prose  # deterministic fallback, non-empty
    assert reply.telemetry is None  # no LLM ran


def test_handler_backfills_posters_from_store() -> None:
    store = _FakeStore({1: "https://img.example/parasite.jpg"})
    handler = build_live_chat_handler(_resources(store=store))

    reply = handler("a gritty korean revenge thriller")

    assert [p.title for p in reply.rec.picks] == ["Parasite"]
    # poster_url is the authoritative corpus URL, not whatever the model emitted.
    assert str(reply.rec.picks[0].poster_url) == "https://img.example/parasite.jpg"


# -- follow-up routing (ticket #54) ------------------------------------------


def _followup(label):
    return lambda query, titles: label


def test_followup_clarify_asks_a_question_without_searching() -> None:
    handler = build_live_chat_handler(_resources(followup_fn=_followup("clarify")))
    handler("a korean thriller")  # turn 1 -> shows Parasite

    reply = handler("i'm not sure")  # vague -> clarify

    assert reply.conversational is True
    assert reply.rec.picks == []
    assert "region" in reply.rec.prose.lower()


def test_followup_replace_lets_previously_shown_reappear() -> None:
    handler = build_live_chat_handler(_resources(followup_fn=_followup("replace")))
    handler("a korean thriller")  # turn 1 -> Parasite (now marked shown)

    reply = handler("show me some comedies")  # replace -> shown reset

    assert [p.title for p in reply.rec.picks] == ["Parasite"]


def test_followup_refine_keeps_shown_so_they_do_not_repeat() -> None:
    handler = build_live_chat_handler(_resources(followup_fn=_followup("refine")))
    handler("a korean thriller")  # turn 1 -> Parasite (shown)

    reply = handler("make it darker")  # refine -> Parasite excluded (already shown)

    assert reply.rec.picks == []


def test_handler_answers_movie_question_from_corpus() -> None:
    lookup = build_title_lookup(
        [
            MovieRecord(
                tmdb_id=1,
                title="Parasite",
                year=2019,
                director="Bong Joon Ho",
                cast=["Song Kang-ho"],
                plot="A poor family schemes.",
            )
        ]
    )
    handler = build_live_chat_handler(
        _resources(movie_lookup=lookup, title_extractor=lambda q: "Parasite")
    )

    reply = handler("who is the cast of parasite?")

    assert reply.conversational is True
    assert reply.rec.picks == []
    assert "Song Kang-ho" in reply.rec.prose
