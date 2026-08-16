"""Langfuse callback wiring (ticket #66) - the no-op path and the traced path.

No Langfuse account, no network and no LLM is involved. The traced path drives
the REAL ``langfuse.langchain.CallbackHandler`` with throwaway keys and points
the SDK at an in-memory OpenTelemetry sink, so the assertions are on the spans
Langfuse would have shipped rather than on a fake that echoes its input back.

The adversary tests attack the two assumptions the wiring rests on: that an
unconfigured checkout attaches nothing at all, and that a tracing backend having
a bad day cannot take a turn down with it.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from imdb_chatbot.graph import GraphModels, run_turn, tracing
from imdb_chatbot.schemas import (
    MovieRecommendation,
    ParsedQuery,
    RecommendationSet,
    ScoredMovie,
    TurnState,
)

EXPECTED_PATH = ["rewrite", "extract", "retrieve", "filter", "generate", "validate"]


def _retriever(query: str, parsed: ParsedQuery, shown_movies=()) -> list[ScoredMovie]:
    return [ScoredMovie(tmdb_id=1, title="John Wick", year=2014, rrf_score=0.5, rank=1)]


def _models() -> GraphModels:
    return GraphModels(
        rewrite=lambda raw, history: f"standalone: {raw}",
        extract=lambda query: ParsedQuery(genres=["Action"]),
        generate=lambda query, candidates: RecommendationSet(
            picks=[MovieRecommendation(title="John Wick", year=2014, reason="lean action")],
            prose="A lean action pick.",
        ),
    )


def _state(**overrides) -> TurnState:
    fields = {
        "trace_id": "trace-lf-1",
        "session_id": "sess-lf-1",
        "user_id": "user-lf-1",
        "raw_query": "action movies",
    }
    return TurnState(**{**fields, **overrides})


@pytest.fixture
def no_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """An environment with no Langfuse credentials (the default checkout)."""
    for name in (*tracing.LANGFUSE_SECRETS, "LANGFUSE_HOST"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def _sink():
    """A configured-but-offline Langfuse: real SDK, throwaway keys, local sink.

    ``span_exporter`` replaces the OTLP exporter, so nothing leaves the process;
    the host points at a dead port as a second guarantee of that. Module-scoped
    because the SDK keeps one client per public key.
    """
    from langfuse import Langfuse

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-local-sink")
        mp.setenv("LANGFUSE_SECRET_KEY", "sk-lf-local-sink")
        mp.setenv("LANGFUSE_HOST", "http://127.0.0.1:1")
        exporter = InMemorySpanExporter()
        client = Langfuse(
            public_key="pk-lf-local-sink",
            secret_key="sk-lf-local-sink",
            host="http://127.0.0.1:1",
            span_exporter=exporter,
        )
        try:
            yield client, exporter
        finally:
            client.shutdown()


@pytest.fixture
def local_sink(_sink):
    """Call to flush the SDK and read back the spans Langfuse would have shipped."""
    client, exporter = _sink
    exporter.clear()

    def spans():
        client.flush()
        return exporter.get_finished_spans()

    return spans


# -- the no-op path ------------------------------------------------------------


def test_unconfigured_langfuse_attaches_nothing(no_langfuse: None) -> None:
    """Attacks: 'tracing is harmless when unconfigured' - it must attach NO handler.

    A disabled-but-constructed handler still logs auth errors on every turn and
    ships spans nowhere; the only clean no-op is an empty config.
    """
    assert tracing.langfuse_handler() is None
    assert tracing.langfuse_config(_state()) == {}


def test_still_encrypted_keys_count_as_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attacks: 'the env var is set, so the key is usable'.

    The repo's ``.env`` is dotenvx ciphertext. Run without dotenvx it resolves to
    a truthy ``encrypted:...`` string - a key-shaped non-key that would
    authenticate against Langfuse forever and log a failure every turn.
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "encrypted:BB1S2Bnotarealkey")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "encrypted:BB1S2Bnotarealkey")

    assert tracing.langfuse_handler() is None


def test_turn_succeeds_with_no_credentials_present(no_langfuse: None) -> None:
    """A checkout with no Langfuse account runs a full turn unchanged."""
    result = run_turn(_state(), retriever=_retriever, models=_models())

    assert result.state.path_taken == EXPECTED_PATH
    assert result.state.response is not None
    assert result.trace.trace_id == "trace-lf-1"


def test_anonymous_turn_does_not_forge_a_user_id(local_sink) -> None:
    """Attacks: 'every turn has a user' - an anonymous one must not invent one.

    ``TurnState.user_id`` is optional. Passing it through unguarded would attribute
    cost to a literal "None" user in the dashboard, which is worse than no
    attribution because it looks real.
    """
    config = tracing.langfuse_config(_state(user_id=None))

    assert "langfuse_user_id" not in config["metadata"]
    assert config["metadata"]["langfuse_session_id"] == "sess-lf-1"


# -- the traced path -----------------------------------------------------------


def test_one_callback_captures_every_node(local_sink) -> None:
    """A single attached callback yields one trace with a span per graph node."""
    result = run_turn(_state(), retriever=_retriever, models=_models())
    spans = local_sink()

    assert result.state.path_taken == EXPECTED_PATH
    # Every node the turn walked shows up as its own span...
    assert set(EXPECTED_PATH) <= {span.name for span in spans}
    # ...all of them under ONE trace (no per-node instrumentation, no orphans).
    assert len({span.context.trace_id for span in spans}) == 1


def test_cost_is_attributed_to_the_user_and_session(local_sink) -> None:
    """Every exported span carries the session and user the turn belongs to."""
    run_turn(_state(), retriever=_retriever, models=_models())
    spans = local_sink()

    assert spans
    assert {span.attributes.get("session.id") for span in spans} == {"sess-lf-1"}
    assert {span.attributes.get("user.id") for span in spans} == {"user-lf-1"}
    # The Langfuse trace joins back to the TurnTrace row that owns it.
    assert {span.attributes.get("langfuse.trace.metadata.imdb_trace_id") for span in spans} == {
        "trace-lf-1"
    }


@pytest.mark.live
def test_live_turn_lands_in_the_langfuse_project(live_resources) -> None:
    """The #66 acceptance check: a real turn is queryable in the real project.

    Needs Langfuse credentials, which no test tier can synthesise - it skips
    (naming what is missing) until they exist. Run it with:

        npx @dotenvx/dotenvx run -f .env -- pytest -m live -q -k langfuse
    """
    import time
    import uuid

    from langfuse import get_client

    from imdb_chatbot.memory.session import ConversationState, run_session_turn

    if tracing.langfuse_handler() is None:
        pytest.skip("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set; tracing is off")
    client = get_client()
    assert client.auth_check(), "Langfuse rejected the credentials (check keys and LANGFUSE_HOST)"

    conversation = ConversationState(session_id=f"lf-live-{uuid.uuid4()}")
    run_session_turn(
        conversation,
        "a lean action movie from the 2010s",
        retriever=live_resources.retriever,
        models=live_resources.models_factory(None),
        trace_id=str(uuid.uuid4()),
        user_id="langfuse-live-check",
        vocab=live_resources.vocab,
    )
    client.flush()

    # Cloud ingestion is asynchronous: poll rather than assume it is instant.
    traces = []
    for _ in range(12):
        traces = client.api.trace.list(session_id=conversation.session_id, limit=5).data
        if traces:
            break
        time.sleep(5)
    assert traces, f"no Langfuse trace for session {conversation.session_id} after 60s"

    detail = client.api.trace.get(traces[0].id)
    names = {observation.name for observation in detail.observations}
    assert set(EXPECTED_PATH) <= names, f"missing node spans: {sorted(set(EXPECTED_PATH) - names)}"
    assert detail.session_id == conversation.session_id
    assert detail.user_id == "langfuse-live-check"
    # Tokens and cost came from the SDK, not from a hand-maintained price table.
    assert detail.total_cost is not None
    assert any(observation.type == "GENERATION" for observation in detail.observations)


def test_a_failing_tracer_never_breaks_a_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attacks: 'the callback is trusted code' - a handler that raises on every event.

    This is the tracing-outage acceptance criterion made hostile: not a slow or
    unreachable backend, but a handler whose every hook raises. The turn must
    still produce its answer and its TurnTrace.
    """
    from langchain_core.callbacks.base import BaseCallbackHandler

    class ExplodingHandler(BaseCallbackHandler):
        def __getattribute__(self, name: str):
            if name.startswith("on_"):
                raise RuntimeError("langfuse is down")
            return super().__getattribute__(name)

    monkeypatch.setattr(tracing, "langfuse_handler", ExplodingHandler)

    result = run_turn(_state(), retriever=_retriever, models=_models())

    assert result.state.path_taken == EXPECTED_PATH
    assert result.state.response is not None
    assert result.state.response.picks[0].title == "John Wick"
