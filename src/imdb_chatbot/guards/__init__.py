"""Security / cost controls and the degradation ladder L0-L3 (ticket #23, section 5.5).

The app protects itself from oversized / off-topic / abusive input and from spend,
degrading toward determinism, never toward a stack trace. Four concerns, one per
module, all pure / injectable so tests drive them with zero network I/O:

- ``limits``      - input caps (chars/tokens), context cap, per-session turn cap,
                    and the ``max_tokens=400`` LLM-call backstop.
- ``topic_gate``  - a cheap deterministic (plus optional cheap-LLM) on-topic check
                    that runs BEFORE rewrite; off-topic -> a fixed movies-only refusal.
- ``degradation`` - the L1/L2/L3 ladder: L1 records component-fallback flags, L2 is
                    the LLM-free deterministic-retrieval mode, L3 is the startup
                    health check + honest-exit message.
- ``budget``      - a per-day production ``$`` counter that trips L2 (not an outage)
                    when the daily budget is exceeded.
"""

from __future__ import annotations

from .budget import BudgetTracker
from .degradation import (
    HealthResult,
    L2Result,
    MetadataCard,
    health_check,
    honest_exit_message,
    llm_free_answer,
    record_degradation,
    should_enter_l2,
)
from .limits import (
    CapResult,
    count_tokens,
    enforce_context_cap,
    enforce_input_caps,
    llm_call_params,
    session_turn_guard,
)
from .topic_gate import TopicResult, topic_gate

__all__ = [
    "BudgetTracker",
    "CapResult",
    "HealthResult",
    "L2Result",
    "MetadataCard",
    "TopicResult",
    "count_tokens",
    "enforce_context_cap",
    "enforce_input_caps",
    "health_check",
    "honest_exit_message",
    "llm_call_params",
    "llm_free_answer",
    "record_degradation",
    "session_turn_guard",
    "should_enter_l2",
    "topic_gate",
]
