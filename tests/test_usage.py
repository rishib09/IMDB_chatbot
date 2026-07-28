"""Tests for per-turn token/cost accounting (graph/usage.py) and its flow into
the serialized TurnTrace. No network or LangChain: usage is read off plain
fake message objects.
"""

from __future__ import annotations

from imdb_chatbot.graph.tracing import TraceCollector, serialize_trace
from imdb_chatbot.graph.usage import (
    UsageMeter,
    estimate_cost,
    usage_from_message,
)
from imdb_chatbot.schemas import TurnState

PRICING = {
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
    "google/gemma-3-12b-it": {"input": 0.05, "output": 0.10},
}


class _FakeMessage:
    """Stands in for a LangChain AIMessage: just the attributes usage reads."""

    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


def test_usage_from_message_reads_usage_metadata() -> None:
    msg = _FakeMessage(
        usage_metadata={"input_tokens": 120, "output_tokens": 45, "total_tokens": 165},
        response_metadata={"model_name": "deepseek/deepseek-chat", "token_usage": {}},
    )
    assert usage_from_message(msg) == (120, 45, "deepseek/deepseek-chat", 0.0)


def test_usage_from_message_falls_back_to_token_usage_and_cost() -> None:
    # No usage_metadata: read prompt/completion tokens and the OpenRouter cost.
    msg = _FakeMessage(
        usage_metadata=None,
        response_metadata={
            "model": "google/gemma-3-12b-it",
            "token_usage": {"prompt_tokens": 30, "completion_tokens": 8, "cost": 0.00012},
        },
    )
    assert usage_from_message(msg) == (30, 8, "google/gemma-3-12b-it", 0.00012)


def test_usage_from_message_handles_missing_metadata() -> None:
    assert usage_from_message(None) == (0, 0, "", 0.0)
    assert usage_from_message(_FakeMessage()) == (0, 0, "", 0.0)


def test_meter_aggregates_across_slots() -> None:
    meter = UsageMeter()
    meter.record("rewriter", model="google/gemma-3-12b-it", input_tokens=30, output_tokens=8)
    meter.record("generator", model="deepseek/deepseek-chat", input_tokens=120, output_tokens=45)
    assert meter.input_tokens == 150
    assert meter.output_tokens == 53
    assert meter.total_tokens == 203
    assert meter.models() == {
        "rewriter": "google/gemma-3-12b-it",
        "generator": "deepseek/deepseek-chat",
    }


def test_estimate_cost_uses_price_table_when_no_reported_cost() -> None:
    meter = UsageMeter()
    meter.record("generator", model="deepseek/deepseek-chat", input_tokens=1_000_000, output_tokens=1_000_000)
    # 1M input * 0.14 + 1M output * 0.28 = 0.42
    assert estimate_cost(meter, PRICING) == 0.42


def test_estimate_cost_prefers_reported_provider_cost() -> None:
    meter = UsageMeter()
    meter.record(
        "generator",
        model="deepseek/deepseek-chat",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        reported_cost_usd=0.007,
    )
    # Reported cost wins over the (much larger) table estimate.
    assert estimate_cost(meter, PRICING) == 0.007


def test_estimate_cost_ignores_unknown_models() -> None:
    meter = UsageMeter()
    meter.record("generator", model="some/unlisted-model", input_tokens=1000, output_tokens=1000)
    assert estimate_cost(meter, PRICING) == 0.0


def test_serialize_trace_populates_token_usage_and_cost() -> None:
    meter = UsageMeter()
    meter.record("generator", model="deepseek/deepseek-chat", input_tokens=200, output_tokens=100)
    state = TurnState(trace_id="t1", session_id="s1", raw_query="hi")

    trace = serialize_trace(state, TraceCollector(), usage=meter, pricing=PRICING)

    assert trace.token_usage == {
        "input_tokens": 200,
        "output_tokens": 100,
        "total_tokens": 300,
    }
    # 200/1e6*0.14 + 100/1e6*0.28 = 0.000028 + 0.000028 = 0.000056
    assert trace.cost_usd == 0.000056


def test_serialize_trace_without_usage_leaves_defaults() -> None:
    state = TurnState(trace_id="t1", session_id="s1", raw_query="hi")
    trace = serialize_trace(state, TraceCollector())
    assert trace.token_usage == {}
    assert trace.cost_usd == 0.0
