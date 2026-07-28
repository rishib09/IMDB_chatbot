"""Session-scoped conversational memory (ticket #21, PRD section 6).

This package implements the two *session* memory tiers from the PRD memory spec:

- Short-term window: the last N turns verbatim plus a running summary of older
  turns (``ConversationState.history_lines``), fed to the history-aware rewriter.
- Constraint state: exclusions, ``shown_movies``, preferences and feedback that
  persist for the life of a session (``ConversationState``).

Everything here is in-memory and session-scoped. The durable, cross-session
knowledge graph (tier 3) is a separate later ticket (#22); there is deliberately
NO persistence in this module.

Public surface:

- ``ConversationState`` / ``Turn`` - the per-session constraint + history state.
- ``run_session_turn`` - run one turn through the graph with the session's history
  and constraints applied, then fold the outcome back into the state.
- ``build_turn_state`` / ``update_state_from_result`` - the lower-level halves of
  ``run_session_turn`` for callers that drive the graph themselves.
"""

from __future__ import annotations

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
]
