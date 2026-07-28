"""The live Chat handler, driven with fakes (no network, no index, no LLM).

``build_live_chat_handler`` takes a ``LiveResources`` bundle whose retriever and
models factory are injectable, so a full ChatReply - real picks plus token/cost
telemetry - is produced deterministically here.
"""

from __future__ import annotations

from imdb_chatbot.dashboard.live import (
    ChatReply,
    LiveResources,
    build_live_chat_handler,
)
from imdb_chatbot.graph.models import GraphModels
from imdb_chatbot.graph.usage import UsageMeter
from imdb_chatbot.schemas import (
    MovieRecommendation,
    ParsedQuery,
    RecommendationSet,
    ScoredMovie,
)

PRICING = {"deepseek/deepseek-chat": {"input": 0.14, "output": 0.28}}


def _fake_retriever(query: str, parsed: ParsedQuery):
    return [ScoredMovie(tmdb_id=1, title="Parasite", year=2019, regions=["KR"])]


def _fake_models_factory(meter: UsageMeter | None) -> GraphModels:
    """A GraphModels whose slots record token usage into ``meter`` like the real one."""

    def rewrite(raw_query, history):
        if meter is not None:
            meter.record("rewriter", model="google/gemma-3-12b-it", input_tokens=30, output_tokens=8)
        return raw_query

    def extract(query):
        if meter is not None:
            meter.record("extractor", model="google/gemma-3-12b-it", input_tokens=40, output_tokens=12)
        return ParsedQuery(region="KR")

    def generate(query, candidates):
        if meter is not None:
            meter.record("generator", model="deepseek/deepseek-chat", input_tokens=200, output_tokens=60)
        return RecommendationSet(
            picks=[MovieRecommendation(title="Parasite", year=2019, reason="Tense.")],
            prose="A gripping pick for you.",
        )

    return GraphModels(rewrite=rewrite, extract=extract, generate=generate)


def _resources() -> LiveResources:
    return LiveResources(
        retriever=_fake_retriever,
        models_factory=_fake_models_factory,
        pricing=PRICING,
        versions={"index": "test", "model_config": "test", "prompt": "test"},
        store=None,
    )


def test_live_handler_returns_picks_and_telemetry() -> None:
    handler = build_live_chat_handler(_resources())

    reply = handler("gritty korean revenge thriller")

    assert isinstance(reply, ChatReply)
    assert [p.title for p in reply.rec.picks] == ["Parasite"]

    telemetry = reply.telemetry
    assert telemetry is not None
    # All three slots ran: tokens summed across rewriter+extractor+generator.
    assert telemetry.input_tokens == 30 + 40 + 200
    assert telemetry.output_tokens == 8 + 12 + 60
    assert telemetry.models["generator"] == "deepseek/deepseek-chat"
    # Cost is estimated from the price table (>0 given real token counts).
    assert telemetry.cost_usd > 0
    assert "validate" in telemetry.path_taken


def test_live_handler_session_memory_suppresses_repeats() -> None:
    """A movie shown on turn 1 is filtered out of the candidates on turn 2."""
    handler = build_live_chat_handler(_resources())

    first = handler("korean thriller")
    assert [p.title for p in first.rec.picks] == ["Parasite"]

    # Turn 2: the only candidate was already shown -> retrieval drops it ->
    # empty candidates -> deterministic fallback (no picks).
    second = handler("another one like that")
    assert second.rec.picks == []
    # Telemetry still reports (the rewriter/extractor ran this turn).
    assert second.telemetry is not None
    assert "fallback" in second.telemetry.path_taken
