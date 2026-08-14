"""Detection rules: turn a ``TurnTrace`` into taxonomy flag codes (ticket #24).

This is the "detection pass" from PRD section 8.1 / 8.2: a deterministic,
post-write pass that reads a finished ``TurnTrace`` and appends the taxonomy
``flags`` it can prove fired. It runs AFTER the trace is written (the trace is the
system of record), never in the hot path, so production latency never waits on
evaluation. Async judge scoring fills ``judge_scores`` separately.

The failure taxonomy (section 8.2):

    R1 relevant docs not retrieved         (retrieval)
    R2 exclusion leaked into candidates     (retrieval)
    X1 filter extraction wrong/incomplete   (extraction)
    X2 malformed JSON, retries exhausted    (extraction)
    G1 hallucinated title/fact not in ctx   (generation)
    G2 instruction/format violation caught  (generation)
    S1 constraint dropped across turns       (state)
    S2 repetition / stale recommendation     (state)
    F1 deterministic filter bug              (filter)
    O1 latency/cost budget exceeded          (ops)

This module implements the codes that are cleanly derivable from a SINGLE
``TurnTrace`` (plus, for S2, an optional session ``shown_movies`` context, since a
trace does not itself carry the session's shown set):

- ``X2`` - the extractor's JSON retries were exhausted (``extract_retries`` hit
  the cap). See ``_extract_retries_exhausted``.
- ``G2`` - a caught instruction/format near-miss: ``validate`` ran more than once,
  i.e. a ``generate -> validate`` regen cycle is visible in ``path_taken``.
- ``S2`` - a repetition / stale recommendation: a recommended pick is duplicated
  within the response, or (when ``shown_movies`` context is supplied) a pick maps
  to a movie already shown earlier in the session.
- ``O1`` - the turn went over budget: ``cost_usd`` exceeded the threshold, or a
  budget / degradation flag is present.
- ``R1`` - a retrieval miss surfaced as the fallback path with zero candidates.

The remaining codes (R2, X1, G1, S1, F1) need a labeled gold parse, a semantic
judge, or the cross-turn session view, so they are detected elsewhere (the
retrieval eval harness, Gate 4, the async judge, or the multi-turn replay suite)
rather than from a lone trace. Each rule here is small and pure; ``detect``
returns the sorted, de-duplicated list of codes it fired.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from ..schemas import TurnTrace, index_by_title_year

# Mirror the graph's retry caps (see graph/build.py ``MAX_EXTRACT_RETRIES`` /
# ``MAX_GEN_RETRIES``). Kept as local constants so this pure detection module
# does not have to import the LangGraph assembly (which pulls langgraph).
EXTRACT_RETRY_CAP = 2

# O1 cost budget per turn in USD. A turn whose recorded ``cost_usd`` exceeds this
# is flagged over-budget. Deliberately conservative for a POC-scale turn.
COST_BUDGET_USD = 0.05

# Degradation / flag substrings that independently signal an ops-budget breach
# (e.g. the ``O1_input_cap`` oversize-input notice from PRD section 7). Matched
# case-insensitively as substrings so future budget flags are picked up too.
_BUDGET_FLAG_MARKERS = ("o1", "budget", "input_cap")


def _extract_retries_exhausted(trace: TurnTrace) -> bool:
    """X2: the extractor burned through its JSON retries (counter hit the cap).

    When the LLM extractor fails to emit valid JSON ``EXTRACT_RETRY_CAP`` times
    the graph falls back to regex extraction and records the exhausted counter on
    ``extract_retries``. That counter reaching the cap is the X2 evidence.
    """
    return trace.extract_retries >= EXTRACT_RETRY_CAP


def _validate_cycle_caught(trace: TurnTrace) -> bool:
    """G2: an instruction/format near-miss was caught by the output gate.

    ``validate`` running more than once means a ``generate -> validate`` regen
    cycle happened: the first answer was rejected by Gate 4 (excluded actor leak,
    bad format, ungrounded fact) and regenerated. A single clean validate does
    not fire G2.
    """
    return trace.path_taken.count("validate") > 1


def _stale_or_repeated(trace: TurnTrace, shown_movies: set[int]) -> bool:
    """S2: a repeated / stale recommendation.

    Two independent signals:

    - a pick duplicated within THIS response (same title + year twice), which is
      derivable from the trace alone; or
    - a pick that maps (via the trace's candidates) to a ``tmdb_id`` already in
      the session's ``shown_movies`` - a movie recommended again after it was
      already surfaced earlier in the session (needs the session context).
    """
    picks = trace.response.picks if trace.response else []
    seen: set[tuple[str, int]] = set()
    for pick in picks:
        key = (pick.title, pick.year)
        if key in seen:
            return True
        seen.add(key)

    if shown_movies:
        by_title_year = index_by_title_year(trace.candidates)
        for pick in picks:
            tmdb_id = by_title_year.get((pick.title, pick.year))
            if tmdb_id is not None and tmdb_id in shown_movies:
                return True
    return False


def _over_budget(trace: TurnTrace) -> bool:
    """O1: the turn exceeded the latency/cost budget.

    Fires when the recorded ``cost_usd`` is over ``COST_BUDGET_USD`` or a budget /
    degradation flag (e.g. ``O1_input_cap``) is present on the trace.
    """
    if trace.cost_usd > COST_BUDGET_USD:
        return True
    markers = _BUDGET_FLAG_MARKERS
    for flag in (*trace.degradation, *trace.flags):
        lowered = flag.lower()
        if any(marker in lowered for marker in markers):
            return True
    return False


def _retrieval_miss_fallback(trace: TurnTrace) -> bool:
    """R1: a retrieval miss - the fallback path was taken with zero candidates.

    The graph routes ``filter -> fallback`` precisely when nothing survived
    retrieval. That empty-candidate fallback is the user-rephrase / R1 signal
    (distinct from a fallback reached AFTER a failed generation, which keeps its
    candidates).
    """
    return "fallback" in trace.path_taken and len(trace.candidates) == 0


def detect(trace: TurnTrace, *, shown_movies: Iterable[int] | None = None) -> list[str]:
    """Return the sorted taxonomy codes that fire for ``trace``.

    A deterministic, pure post-write pass. ``shown_movies`` is the optional
    session context needed for the cross-turn half of S2 (the session's already
    shown ``tmdb_id`` set); omitted, S2 still fires on intra-response duplicates.
    A clean, first-try trace fires nothing.
    """
    shown = {int(m) for m in shown_movies} if shown_movies else set()
    codes: set[str] = set()
    if _extract_retries_exhausted(trace):
        codes.add("X2")
    if _validate_cycle_caught(trace):
        codes.add("G2")
    if _stale_or_repeated(trace, shown):
        codes.add("S2")
    if _over_budget(trace):
        codes.add("O1")
    if _retrieval_miss_fallback(trace):
        codes.add("R1")
    return sorted(codes)


def apply_detection(trace: TurnTrace, *, shown_movies: Iterable[int] | None = None) -> list[str]:
    """Fold ``detect`` output into ``trace.flags`` (de-duplicated, sorted) and return it.

    A convenience for callers that want the trace's ``flags`` field updated in
    place after detection. Pre-existing flags are preserved.
    """
    merged = sorted(set(trace.flags) | set(detect(trace, shown_movies=shown_movies)))
    trace.flags = merged
    return merged


def count_codes(traces: Iterable[TurnTrace]) -> Counter[str]:
    """Run detection over a batch of traces and count how often each code fired.

    A small aggregation helper for the dashboard / triage view: the returned
    ``Counter`` maps each taxonomy code to the number of traces that fired it.
    """
    counter: Counter[str] = Counter()
    for trace in traces:
        counter.update(detect(trace))
    return counter
