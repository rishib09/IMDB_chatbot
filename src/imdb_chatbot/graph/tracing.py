"""Per-node tracing: timings + path, then serialize TurnState -> TurnTrace.

The ``@traced`` decorator wraps every graph node so that observability falls out
of the orchestration for free (PRD section 7.5):

- it appends the node name to ``path_taken`` (so a retry cycle is visible as a
  repeated node);
- it records per-node wall-clock into a ``TraceCollector`` (``timings_ms``),
  accumulating across retries of the same node;
- it captures an unexpected exception into the collector and records a
  degradation flag on state rather than letting the turn crash.

At END, ``serialize_trace`` snapshots the final ``TurnState`` (plus the collected
timings and placeholder artifact versions) into an immutable ``TurnTrace`` that
``TraceStore`` persists.

``langfuse_config`` is the second, optional observability layer (ticket #66):
ONE LangChain callback on the graph invocation, which Langfuse expands into a
trace with a span per node and a generation per nested LLM call. The persisted
``TurnTrace`` stays the system of record (decision D1, #63) - Langfuse is the
dashboard, so it is always optional and never allowed to break a turn.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..schemas import TurnState, TurnTrace
from .usage import UsageMeter, estimate_cost

# A node returns a dict of TurnState field updates (LangGraph merges them).
NodeFn = Callable[[TurnState], dict[str, Any]]


@dataclass
class TraceCollector:
    """Side-channel accumulator owned by one turn (not part of TurnState).

    ``timings_ms`` and ``errors`` live here because ``TurnState`` deliberately
    carries no timing fields - a fresh collector per ``run_turn`` keeps concurrent
    turns isolated.
    """

    timings_ms: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def traced(node_name: str, collector: TraceCollector) -> Callable[[NodeFn], NodeFn]:
    """Wrap a node so it records path + timing and never lets an error escape."""

    def decorator(fn: NodeFn) -> NodeFn:
        def wrapper(state: TurnState) -> dict[str, Any]:
            start = time.perf_counter()
            updates: dict[str, Any] = {}
            try:
                result = fn(state)
                if result:
                    updates.update(result)
            except Exception as exc:  # noqa: BLE001 - captured into the trace
                collector.errors[node_name] = repr(exc)
                degradation = list(state.degradation)
                degradation.append(f"{node_name}_error")
                updates["degradation"] = degradation
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            collector.timings_ms[node_name] = (
                collector.timings_ms.get(node_name, 0.0) + elapsed_ms
            )
            # Append to path_taken (visible retry cycles); overwrite channel with
            # the extended list since LangGraph uses last-value semantics here. A
            # node may pre-extend the path itself (an in-node retry, e.g. the
            # ``relax_standing`` step of retrieve) - honour that before appending.
            updates["path_taken"] = [*updates.get("path_taken", state.path_taken), node_name]
            return updates

        wrapper.__name__ = f"traced_{node_name}"
        return wrapper

    return decorator


# -- Langfuse callback (ticket #66) --------------------------------------------

# Langfuse Cloud (decision D1, #63) reads these from the environment: the
# dotenvx-encrypted .env locally, platform secrets in CI / on HF Spaces.
# LANGFUSE_HOST is optional - the SDK defaults to the EU cloud region.
LANGFUSE_SECRETS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def _usable_key(name: str) -> bool:
    """True when ``name`` holds a real key, not a placeholder for one.

    A dotenvx-encrypted ``.env`` read WITHOUT dotenvx resolves to a truthy
    ``"encrypted:..."`` ciphertext; authenticating with that just floods the log
    with auth failures, so it counts as unconfigured (cf. the same check on
    ``OPENROUTER_API_KEY`` in tests/conftest.py).
    """
    value = os.environ.get(name, "")
    return bool(value) and not value.startswith("encrypted:")


def langfuse_handler() -> Any | None:
    """The Langfuse LangChain callback, or ``None`` when tracing is unconfigured.

    Returns ``None`` - a clean no-op, no handler attached at all - when either key
    is absent, so a checkout with no Langfuse account traces nothing and pays
    nothing. An unimportable SDK or a constructor failure degrades the same way:
    an observability outage must never break a turn.
    """
    if not all(_usable_key(name) for name in LANGFUSE_SECRETS):
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:  # noqa: BLE001 - tracing is optional; never fail a turn for it
        return None


def langfuse_config(state: TurnState) -> dict[str, Any]:
    """LangGraph ``invoke`` config that traces this turn, or ``{}`` if unconfigured.

    One callback on the invocation captures the whole chain - LangChain propagates
    it down into every node and every nested model call - so no node needs its own
    instrumentation. The ``langfuse_*`` metadata keys are the SDK's contract for
    attributing a trace's tokens and cost to a user and a session; ``imdb_trace_id``
    joins the Langfuse trace back to the ``TurnTrace`` row that owns it.
    """
    handler = langfuse_handler()
    if handler is None:
        return {}
    metadata: dict[str, Any] = {
        "langfuse_session_id": state.session_id,
        "imdb_trace_id": state.trace_id,
    }
    if state.user_id:
        metadata["langfuse_user_id"] = state.user_id
    return {"callbacks": [handler], "metadata": metadata}


def serialize_trace(
    state: TurnState,
    collector: TraceCollector,
    *,
    versions: dict[str, str] | None = None,
    usage: UsageMeter | None = None,
    pricing: dict[str, dict[str, float]] | None = None,
) -> TurnTrace:
    """Snapshot the final ``TurnState`` into an immutable ``TurnTrace``.

    ``versions`` supplies the prompt / model_config / index artifact versions;
    where a real version is not available yet a ``"dev"`` placeholder is used.

    ``usage`` (a per-turn ``UsageMeter``) populates the trace's ``token_usage``
    and ``cost_usd`` - the system-of-record for per-turn spend that the daily
    budget tracker sums. Absent a meter both stay at their empty defaults.
    """
    versions = versions or {}
    token_usage: dict[str, int] = {}
    cost_usd = 0.0
    if usage is not None:
        token_usage = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
        cost_usd = estimate_cost(usage, pricing)
    return TurnTrace(
        trace_id=state.trace_id,
        ts=datetime.now(UTC),
        session_id=state.session_id,
        user_id=state.user_id,
        raw_query=state.raw_query,
        rewritten_query=state.rewritten_query,
        parsed=state.parsed,
        retrieved=state.retrieved,
        candidates=state.candidates,
        filters_applied=state.filters_applied,
        response=state.response,
        path_taken=state.path_taken,
        extract_retries=state.extract_retries,
        gen_retries=state.gen_retries,
        gate4_rejects=state.gate4_rejects,
        timings_ms=dict(collector.timings_ms),
        token_usage=token_usage,
        cost_usd=cost_usd,
        degradation=state.degradation,
        prompt_version=versions.get("prompt", "dev"),
        model_config_version=versions.get("model_config", "dev"),
        index_version=versions.get("index", "dev"),
    )
