"""Per-turn token + cost accounting for the LLM slots.

The graph's model callables (rewriter / extractor / generator) record each LLM
call's token usage into a ``UsageMeter`` - a per-turn side-channel, exactly like
``TraceCollector``. Token counts come straight from LangChain's
``usage_metadata`` (reliable across providers). Cost is either the actual value
OpenRouter returns (when the request asks for ``usage.include``) or, failing
that, an estimate from the configurable per-model price table in
``config/models.yaml``.

Nothing here touches the network or LangChain: ``usage_from_message`` reads plain
attributes off whatever object the model returned, so a fake object with a
``usage_metadata`` dict works in tests just like a real ``AIMessage``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import load_models_config


@dataclass
class SlotUsage:
    """Accumulated token usage for one model slot within a single turn."""

    slot: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    reported_cost_usd: float = 0.0  # actual provider cost, when available


@dataclass
class UsageMeter:
    """One turn's LLM accounting, keyed by slot.

    Callables built by ``build_models`` call :meth:`record` after each invoke.
    The aggregate token totals and cost then flow into the ``TurnTrace`` (via
    ``serialize_trace``) and the Chat UI telemetry strip.
    """

    slots: dict[str, SlotUsage] = field(default_factory=dict)

    def record(
        self,
        slot: str,
        *,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        reported_cost_usd: float = 0.0,
    ) -> SlotUsage:
        usage = self.slots.get(slot)
        if usage is None:
            usage = SlotUsage(slot=slot)
            self.slots[slot] = usage
        if model:
            usage.model = model
        usage.input_tokens += int(input_tokens or 0)
        usage.output_tokens += int(output_tokens or 0)
        usage.reported_cost_usd += float(reported_cost_usd or 0.0)
        usage.calls += 1
        return usage

    @property
    def input_tokens(self) -> int:
        """Total prompt (uploaded) tokens across every slot this turn."""
        return sum(u.input_tokens for u in self.slots.values())

    @property
    def output_tokens(self) -> int:
        """Total completion (downloaded) tokens across every slot this turn."""
        return sum(u.output_tokens for u in self.slots.values())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def reported_cost_usd(self) -> float:
        return sum(u.reported_cost_usd for u in self.slots.values())

    def has_reported_cost(self) -> bool:
        """True if any slot carried a real provider-reported cost."""
        return any(u.reported_cost_usd for u in self.slots.values())

    def models(self) -> dict[str, str]:
        """slot -> model id actually used (only slots that were invoked)."""
        return {slot: u.model for slot, u in self.slots.items() if u.model}


def usage_from_message(message: Any) -> tuple[int, int, str, float]:
    """Extract ``(input_tokens, output_tokens, model, reported_cost_usd)``.

    Reads LangChain's structured ``usage_metadata`` first, then falls back to the
    provider-shaped ``response_metadata['token_usage']`` (prompt/completion tokens
    and, for OpenRouter with ``usage.include``, a ``cost`` field). All accesses
    are defensive so a partial or missing metadata block yields zeros, never an
    exception.
    """
    if message is None:
        return 0, 0, "", 0.0

    input_tokens = 0
    output_tokens = 0
    meta = getattr(message, "usage_metadata", None)
    if isinstance(meta, dict):
        input_tokens = int(meta.get("input_tokens", 0) or 0)
        output_tokens = int(meta.get("output_tokens", 0) or 0)

    response_meta = getattr(message, "response_metadata", None) or {}
    model = ""
    reported_cost = 0.0
    if isinstance(response_meta, dict):
        model = str(response_meta.get("model_name") or response_meta.get("model") or "")
        token_usage = response_meta.get("token_usage")
        if isinstance(token_usage, dict):
            reported_cost = float(token_usage.get("cost", 0.0) or 0.0)
            if not input_tokens:
                input_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
            if not output_tokens:
                output_tokens = int(token_usage.get("completion_tokens", 0) or 0)

    return input_tokens, output_tokens, model, reported_cost


def load_pricing(cfg: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    """The per-model price table from ``models.yaml`` (USD per 1,000,000 tokens)."""
    cfg = cfg or load_models_config()
    return cfg.get("pricing", {}) or {}


def estimate_cost(
    meter: UsageMeter,
    pricing: dict[str, dict[str, float]] | None = None,
) -> float:
    """USD cost for one turn.

    Prefers the provider-reported cost when any slot carried one; otherwise sums a
    price-table estimate per slot: ``input/1e6 * input_price + output/1e6 *
    output_price``. A model missing from the table contributes nothing (so the
    estimate is a floor, never a crash).
    """
    if meter.has_reported_cost():
        return round(meter.reported_cost_usd, 6)

    pricing = pricing if pricing is not None else load_pricing()
    total = 0.0
    for usage in meter.slots.values():
        price = pricing.get(usage.model)
        if not price:
            continue
        total += usage.input_tokens / 1_000_000 * float(price.get("input", 0.0))
        total += usage.output_tokens / 1_000_000 * float(price.get("output", 0.0))
    return round(total, 6)
