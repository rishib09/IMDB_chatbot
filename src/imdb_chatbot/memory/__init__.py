"""Session-scoped conversational memory (ticket #21, PRD section 6).

This package implements the two *session* memory tiers from the PRD memory spec:

- Short-term window: the last N turns verbatim plus a running summary of older
  turns (``ConversationState.history_lines``), fed to the history-aware rewriter.
- Constraint state: exclusions, ``shown_movies``, preferences and feedback that
  persist for the life of a session (``ConversationState``).

The ``session`` submodule is in-memory and session-scoped. The durable,
cross-session tier (tier 3) lives in the ``durable`` submodule (ticket #22): an
append-only log of explicit-feedback ``Triple`` facts that is loaded at session
start (passive injection) and where a live request outranks a remembered
preference (minimal precedence).

Public surface:

- ``ConversationState`` / ``Turn`` - the per-session constraint + history state.
- ``run_session_turn`` - run one turn through the graph with the session's history
  and constraints applied, then fold the outcome back into the state.
- ``build_turn_state`` / ``update_state_from_result`` - the lower-level halves of
  ``run_session_turn`` for callers that drive the graph themselves.
- ``Triple`` / ``MemoryStore`` / ``LocalJsonlStore`` / ``HFDatasetStore`` - the
  durable memory model and its stores.
- ``widget_to_triple`` / ``known_preferences`` / ``resolve_collision`` - the
  widget mapping, passive injection and minimal-precedence functions.
"""

from __future__ import annotations

from .durable import (
    MAX_TRIPLES_PER_USER,
    HFDatasetStore,
    KnownPreferences,
    LocalJsonlStore,
    MemoryStore,
    Triple,
    known_preferences,
    normalize_user_id,
    resolve_collision,
    widget_to_triple,
)
from .session import (
    WINDOW_SIZE,
    ConversationState,
    Turn,
    build_turn_state,
    run_session_turn,
    update_state_from_result,
)

__all__ = [
    "WINDOW_SIZE",
    "ConversationState",
    "Turn",
    "build_turn_state",
    "run_session_turn",
    "update_state_from_result",
    "MAX_TRIPLES_PER_USER",
    "HFDatasetStore",
    "KnownPreferences",
    "LocalJsonlStore",
    "MemoryStore",
    "Triple",
    "known_preferences",
    "normalize_user_id",
    "resolve_collision",
    "widget_to_triple",
]
