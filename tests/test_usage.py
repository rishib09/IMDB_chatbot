"""Per-turn token/cost accounting (``graph/models.py``) and its flow into the
serialized ``TurnTrace``.

Ticket #67 removed the hand-maintained price table: a turn's cost is now nothing
but the sum of what OpenRouter itself reported per call. The offline tests here
attack that summation and the "what if the provider told us nothing" edge; the
live test reconciles a real call's reported cost against OpenRouter's own
``/api/v1/generation`` record of the same call.
"""

from __future__ import annotations

import time

import pytest

from imdb_chatbot.graph.models import UsageMeter, usage_from_message
from imdb_chatbot.graph.tracing import TraceCollector, serialize_trace
from imdb_chatbot.schemas import TurnState


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
    """Attacks: 'every response carries a usage block we can read'."""
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


def test_meter_does_not_round_away_real_per_call_costs() -> None:
    """Attacks: 'six decimal places is enough precision for a turn's cost'.

    A real gemma-3-12b call costs ~1.05e-06. A turn makes several, and the old
    ``round(total, 6)`` quantised the sum to the nearest micro-dollar - here it
    would report $0.000008 for $0.0000084, a silent 5% under-count that grows
    with every cheap model added.
    """
    meter = UsageMeter()
    for _ in range(8):
        meter.record("rewriter", model="google/gemma-3-12b-it", cost_usd=1.05e-06)
    assert meter.cost_usd == 8.4e-06
    assert round(meter.cost_usd, 6) != meter.cost_usd  # the old rounding lost it


def test_meter_reports_zero_when_the_provider_reported_no_cost() -> None:
    """Attacks: 'a turn that burned tokens always has a cost to show'.

    With the price table gone there is nothing to fall back on, and that is the
    point: an unreported cost surfaces as an honest 0.0, never as a locally
    invented estimate that drifts from the invoice.
    """
    meter = UsageMeter()
    meter.record("generator", model="some/unlisted-model", input_tokens=1000, output_tokens=1000)
    assert meter.total_tokens == 2000
    assert meter.cost_usd == 0.0


def test_models_yaml_carries_no_price_table() -> None:
    """Attacks: 'the hand-maintained pricing table is gone for good' (#67).

    Nothing reads a ``pricing:`` key any more, so one silently re-added would
    never be noticed - and would re-open the drift this ticket closed.
    """
    from imdb_chatbot.config import load_models_config

    assert "pricing" not in load_models_config()


def test_serialize_trace_populates_token_usage_and_cost() -> None:
    meter = UsageMeter()
    meter.record(
        "generator",
        model="deepseek/deepseek-chat",
        input_tokens=200,
        output_tokens=100,
        cost_usd=5.6e-05,
    )
    state = TurnState(trace_id="t1", session_id="s1", raw_query="hi")

    trace = serialize_trace(state, TraceCollector(), usage=meter)

    assert trace.token_usage == {
        "input_tokens": 200,
        "output_tokens": 100,
        "total_tokens": 300,
    }
    assert trace.cost_usd == 5.6e-05


def test_serialize_trace_without_usage_leaves_defaults() -> None:
    state = TurnState(trace_id="t1", session_id="s1", raw_query="hi")
    trace = serialize_trace(state, TraceCollector())
    assert trace.token_usage == {}
    assert trace.cost_usd == 0.0


# -- live reconciliation against OpenRouter ------------------------------------

GENERATION_URL = "https://openrouter.ai/api/v1/generation"


def _fetch_generation(gen_id: str, key: str) -> dict:
    """OpenRouter's own record of one call. Eventually consistent, so poll."""
    import httpx

    for _ in range(10):
        resp = httpx.get(
            GENERATION_URL,
            params={"id": gen_id},
            headers={"Authorization": f"Bearer {key}"},
            timeout=15.0,
        )
        if resp.status_code == 200:
            return resp.json()["data"]
        if resp.status_code != 404:
            resp.raise_for_status()
        time.sleep(2.0)
    pytest.fail(f"OpenRouter never returned a generation record for {gen_id}")


@pytest.mark.live
def test_displayed_cost_matches_openrouters_own_record(request) -> None:
    """The cost the UI shows is the cost OpenRouter billed for the same call.

    One cheap real call: meter what the response's usage block reported, then ask
    ``/api/v1/generation`` for that generation id and compare. This is the check
    the deleted price table could never pass.

    Tokens are compared against ``native_tokens_*``, not ``tokens_*``: the latter
    are OpenRouter's normalised GPT-tokenizer counts (23/1 where the model itself
    saw 16/2), so billing and the telemetry strip both track the native figures.

    The key is pulled via ``getfixturevalue`` rather than a named parameter so a
    failure's fixture dump never prints it.
    """
    from imdb_chatbot.config import load_models_config
    from imdb_chatbot.graph.models import _init_slot_model

    key = request.getfixturevalue("openrouter_key")
    cfg = load_models_config()
    response = _init_slot_model("rewriter", cfg).invoke("Reply with the single word OK.")

    input_tokens, output_tokens, model, cost = usage_from_message(response)
    meter = UsageMeter()
    meter.record(
        "rewriter",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )

    gen_id = response.response_metadata["id"]
    record = _fetch_generation(gen_id, key)
    detail = (
        f"{gen_id} model={model} meter=${meter.cost_usd} "
        f"tokens={meter.input_tokens}/{meter.output_tokens} "
        f"record=${record['total_cost']} "
        f"native={record['native_tokens_prompt']}/{record['native_tokens_completion']}"
    )

    assert meter.cost_usd > 0, detail
    assert meter.cost_usd == pytest.approx(record["total_cost"], rel=1e-9), detail
    assert meter.input_tokens == record["native_tokens_prompt"], detail
    assert meter.output_tokens == record["native_tokens_completion"], detail
