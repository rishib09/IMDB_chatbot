"""LangGraph orchestration for one conversational turn (ticket #18).

The deterministic-spine / stochastic-leaves graph: LangGraph is the control
plane, Pydantic is the data plane, and the graph state (``TurnState``) is itself
a Pydantic model, so every hop is both an orchestration step and a validation
gate (PRD section 7.3 / 7.4).

Public surface:

- ``build_graph``  - compile the ``StateGraph`` given an injected retriever and
  a ``GraphModels`` bundle (both are fully monkeypatchable so tests never touch
  the network).
- ``run_turn`` / ``TurnResult`` - run a single turn end-to-end and serialize the
  final ``TurnState`` into a persisted ``TurnTrace``.
- ``GraphModels`` / ``build_models`` - the injectable model factory (extractor and
  generator use Pydantic structured output).
"""

from __future__ import annotations

from .build import RetrieverFn, TurnResult, build_graph, run_turn
from .gate4 import Gate4Result, run_gate4
from .models import GraphModels, build_models
from .tracing import TraceCollector, serialize_trace, traced

__all__ = [
    "Gate4Result",
    "GraphModels",
    "RetrieverFn",
    "TraceCollector",
    "TurnResult",
    "build_graph",
    "build_models",
    "run_gate4",
    "run_turn",
    "serialize_trace",
    "traced",
]
