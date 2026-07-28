"""Session memory: constraint state + short-term window (ticket #21).

``ConversationState`` is the per-session container the PRD calls the "constraint
state" tier (section 6): exclusions, ``shown_movies``, preferences and feedback
that must persist for the whole session. It also owns the "short-term window"
tier - the last ``window_size`` turns kept verbatim plus a running summary of
everything older - which is what the history-aware rewriter reads to resolve
references (PRD section 7.5).

The graph itself stays single-turn and stateless. This module is the thin driver
that carries session state *into* each ``TurnState`` (history for the rewriter,
``shown_movies`` and standing exclusions for retrieval) and folds the turn's
outcome back *out* into the session. Nothing here is persisted - session scope
only. The durable cross-session KG is ticket #22.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from ..graph import GraphModels, RetrieverFn, TurnResult, run_turn
from ..schemas import TurnState
from ..store import TraceStore

# Last N turns kept verbatim in the short-term window (PRD section 6).
WINDOW_SIZE = 6


class Turn(BaseModel):
    """One recorded exchange in the session history.

    Stores the RAW user query (per PRD section 6: the raw query is what history
    keeps, while the rewritten query is used only for retrieval) alongside the
    rewritten query and the titles actually recommended, so the rewriter can
    resolve later references like "something like that".
    """

    raw_query: str
    rewritten_query: str | None = None
    recommended: list[str] = []  # "Title (Year)" strings actually shown

    def as_history_line(self) -> str:
        rec = ", ".join(self.recommended) if self.recommended else "(no picks)"
        return f"User: {self.raw_query} | Bot recommended: {rec}"


class ConversationState(BaseModel):
    """Per-session constraint state + short-term history window.

    Exclusions accumulate and persist for the session (a constraint set in turn 1
    still applies at turn N); ``shown_movies`` accumulates so nothing repeats.
    """

    session_id: str
    exclude_actors: list[str] = []
    exclude_genres: list[str] = []
    shown_movies: set[int] = Field(default_factory=set)
    preferences: dict = {}
    feedback: list = []
    # Full turn log; only the last ``window_size`` are exposed verbatim.
    turns: list[Turn] = []
    summary: str = ""  # running summary of turns that fell out of the window
    window_size: int = WINDOW_SIZE

    # -- short-term window ----------------------------------------------------

    def window(self) -> list[Turn]:
        """The last ``window_size`` turns, kept verbatim."""
        if self.window_size <= 0:
            return []
        return self.turns[-self.window_size :]

    def history_lines(self) -> list[str]:
        """History for the rewriter: the running summary then the verbatim window."""
        lines: list[str] = []
        if self.summary:
            lines.append(f"Summary of earlier turns: {self.summary}")
        lines.extend(turn.as_history_line() for turn in self.window())
        return lines

    # -- constraint state -----------------------------------------------------

    def add_exclusions(
        self,
        actors: Iterable[str] = (),
        genres: Iterable[str] = (),
    ) -> None:
        """Merge new exclusions in, de-duplicated and order-preserving."""
        for actor in actors:
            if actor and actor not in self.exclude_actors:
                self.exclude_actors.append(actor)
        for genre in genres:
            if genre and genre not in self.exclude_genres:
                self.exclude_genres.append(genre)

    def mark_shown(self, tmdb_ids: Iterable[int]) -> None:
        """Record movie ids as already shown so they are never recommended twice."""
        for tmdb_id in tmdb_ids:
            self.shown_movies.add(int(tmdb_id))

    def record_turn(self, turn: Turn) -> None:
        """Append a completed turn, summarizing anything pushed out of the window."""
        self.turns.append(turn)
        overflow = (
            self.turns[: -self.window_size]
            if self.window_size > 0 and len(self.turns) > self.window_size
            else []
        )
        if overflow:
            self.summary = _summarize(overflow)


def _summarize(turns: list[Turn]) -> str:
    """A deterministic (no-LLM) running summary of turns older than the window.

    Session memory must run with zero network I/O in tests, so this is a plain
    compaction of the raw queries rather than an LLM call. It is enough for the
    rewriter to retain older context without keeping every turn verbatim.
    """
    return "; ".join(turn.raw_query for turn in turns)


# -- graph driver -------------------------------------------------------------


def build_turn_state(
    conversation: ConversationState,
    raw_query: str,
    *,
    trace_id: str,
    user_id: str | None = None,
) -> TurnState:
    """Seed a ``TurnState`` for one turn from the live session state.

    Carries the history window (for the rewriter), the accumulated
    ``shown_movies`` and the standing exclusions (for retrieval) into the turn.
    """
    return TurnState(
        trace_id=trace_id,
        session_id=conversation.session_id,
        user_id=user_id,
        raw_query=raw_query,
        history=conversation.history_lines(),
        shown_movies=sorted(conversation.shown_movies),
        session_exclude_actors=list(conversation.exclude_actors),
        session_exclude_genres=list(conversation.exclude_genres),
    )


def update_state_from_result(conversation: ConversationState, result: TurnResult) -> None:
    """Fold a finished turn back into the session state.

    - New exclusions the extractor found this turn persist for the session.
    - Movies actually recommended are marked ``shown`` (no repeats later).
    - The raw query + rewritten query + recommended titles enter the history
      window (raw query stored per PRD section 6).
    """
    final = result.state

    if final.parsed is not None:
        conversation.add_exclusions(
            final.parsed.exclude_actors,
            final.parsed.exclude_genres,
        )

    recommended_titles: list[str] = []
    shown_ids: list[int] = []
    if final.response is not None and final.response.picks:
        by_title_year = {(c.title, c.year): c.tmdb_id for c in final.candidates}
        for pick in final.response.picks:
            recommended_titles.append(f"{pick.title} ({pick.year})")
            tmdb_id = by_title_year.get((pick.title, pick.year))
            if tmdb_id is not None:
                shown_ids.append(tmdb_id)
    conversation.mark_shown(shown_ids)

    conversation.record_turn(
        Turn(
            raw_query=final.raw_query,
            rewritten_query=final.rewritten_query,
            recommended=recommended_titles,
        )
    )


def run_session_turn(
    conversation: ConversationState,
    raw_query: str,
    *,
    retriever: RetrieverFn,
    models: GraphModels,
    trace_id: str,
    user_id: str | None = None,
    store: TraceStore | None = None,
    versions: dict[str, str] | None = None,
) -> TurnResult:
    """Run one turn with session memory applied, then update the session.

    This is the multi-turn entry point: history resolves references in the
    rewriter, standing exclusions and ``shown_movies`` constrain retrieval, and
    the outcome is folded back so the next turn sees the updated state.
    """
    state = build_turn_state(
        conversation,
        raw_query,
        trace_id=trace_id,
        user_id=user_id,
    )
    result = run_turn(
        state,
        retriever=retriever,
        models=models,
        store=store,
        versions=versions,
    )
    update_state_from_result(conversation, result)
    return result
