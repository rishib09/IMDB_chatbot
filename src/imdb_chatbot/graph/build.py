"""Assemble and run the single-turn LangGraph (PRD section 7.3).

Topology (deterministic spine, stochastic leaves):

    START -> rewrite -> extract
    extract  --(extract_failed and extract_retries < 2)--> extract   (JSON retry)
             --(else)------------------------------------> retrieve
    retrieve -> filter
    filter   --(len(candidates) == 0)--> fallback
             --(else)-----------------> generate
    generate -> validate
    validate --(validation_failed and gen_retries < 2)--> generate   (regen)
             --(validation_failed)-------------------------> fallback
             --(else)---------------------------------------> END
    fallback -> END

``validate`` is the full Gate-4 output gate (ticket #19): a STRUCTURAL check (does
the response parse as a ``RecommendationSet`` with non-empty picks?) followed by
the deterministic SEMANTIC checks in ``graph/gate4.py`` (no excluded actor leaks,
every title exists among the candidates, and prose facts are grounded in the
candidate records). A violation drives the regen edge (bounded), then fallback.

Both stochastic dependencies are injected: ``retriever`` is a callable
``(rewritten_query, parsed) -> list[ScoredMovie]`` and ``models`` is a
``GraphModels`` bundle. Tests supply stubs/fakes so a turn runs with zero
network I/O.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from ..schemas import ParsedQuery, RecommendationSet, ScoredMovie, TurnState
from ..store import TraceStore
from .gate4 import run_gate4
from .models import GraphModels
from .tracing import TraceCollector, serialize_trace, traced
from .usage import UsageMeter

# Injected retriever: (rewritten_query, parsed) -> ranked candidates.
RetrieverFn = Callable[[str, ParsedQuery], Sequence[ScoredMovie]]

MAX_EXTRACT_RETRIES = 2
MAX_GEN_RETRIES = 2

_FALLBACK_PROSE = (
    "I could not find a match for that. Could you relax one constraint - "
    "for example, widen the year range or drop an excluded genre?"
)

_GENRE_WORDS = {
    "action",
    "adventure",
    "animation",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "horror",
    "mystery",
    "romance",
    "sci-fi",
    "thriller",
    "war",
    "western",
}


# -- regex fallback extraction (used after 2 failed LLM extract attempts) -------


def regex_extract(query: str) -> ParsedQuery:
    """A dependency-free keyword/regex extractor - the safety net for the LLM.

    Deliberately simple: recognise genre words and a bounding year so the turn can
    always continue with a valid ``ParsedQuery`` instead of stalling.
    """
    lowered = query.lower()
    genres = [word.capitalize() for word in sorted(_GENRE_WORDS) if word in lowered]
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", query)]
    min_year: int | None = None
    max_year: int | None = None
    if years:
        if any(w in lowered for w in ("after", "since", "newer", "recent")):
            min_year = min(years)
        elif any(w in lowered for w in ("before", "older", "until")):
            max_year = max(years)
    return ParsedQuery(genres=genres, min_year=min_year, max_year=max_year)


# -- nodes ---------------------------------------------------------------------


def _effective_query(state: TurnState) -> str:
    return state.rewritten_query or state.raw_query


def _make_rewrite(models: GraphModels, collector: TraceCollector):
    @traced("rewrite", collector)
    def rewrite(state: TurnState) -> dict:
        # Rewriter is optional: on any failure, skip and use the raw query,
        # recording a degradation flag (PRD section 7.3).
        try:
            # History-aware: the rewriter reads the session's short-term window
            # (ticket #21) to resolve references like "something like that".
            rewritten = models.rewrite(state.raw_query, state.history)
            if not rewritten or not str(rewritten).strip():
                raise ValueError("empty rewrite")
            return {"rewritten_query": str(rewritten).strip()}
        except Exception:  # noqa: BLE001 - graceful degradation, not an error
            return {
                "rewritten_query": state.raw_query,
                "degradation": [*state.degradation, "rewrite_skipped"],
            }

    return rewrite


def _make_extract(models: GraphModels, collector: TraceCollector):
    @traced("extract", collector)
    def extract(state: TurnState) -> dict:
        query = _effective_query(state)
        try:
            parsed = models.extract(query)
            if not isinstance(parsed, ParsedQuery):
                parsed = ParsedQuery.model_validate(parsed)
            return {"parsed": parsed, "extract_failed": False}
        except Exception:  # noqa: BLE001 - drives the JSON-retry edge
            retries = state.extract_retries + 1
            if retries >= MAX_EXTRACT_RETRIES:
                # Exhausted LLM attempts: fall back to regex extraction and
                # continue (extract_failed cleared so the edge routes to retrieve).
                return {
                    "parsed": regex_extract(query),
                    "extract_failed": False,
                    "extract_retries": retries,
                    "degradation": [*state.degradation, "extract_regex_fallback"],
                }
            return {"extract_failed": True, "extract_retries": retries}

    return extract


def _session_merged_parsed(state: TurnState) -> ParsedQuery:
    """Fold the session's standing exclusions (ticket #21) into this turn's parse.

    Exclusions set in an earlier turn are carried on ``TurnState`` and unioned
    into ``parsed`` here so they still constrain retrieval on later turns.
    """
    parsed = state.parsed or ParsedQuery()
    if not (state.session_exclude_actors or state.session_exclude_genres):
        return parsed
    return parsed.model_copy(
        update={
            "exclude_actors": sorted(
                {*parsed.exclude_actors, *state.session_exclude_actors}
            ),
            "exclude_genres": sorted(
                {*parsed.exclude_genres, *state.session_exclude_genres}
            ),
        }
    )


def _make_retrieve(retriever: RetrieverFn, collector: TraceCollector):
    @traced("retrieve", collector)
    def retrieve(state: TurnState) -> dict:
        parsed = _session_merged_parsed(state)
        retrieved = list(retriever(_effective_query(state), parsed))
        # Session no-repeat guarantee (ticket #21): drop anything already shown.
        if state.shown_movies:
            shown = set(state.shown_movies)
            retrieved = [m for m in retrieved if m.tmdb_id not in shown]
        return {"retrieved": retrieved, "parsed": parsed}

    return retrieve


def _make_filter(collector: TraceCollector):
    @traced("filter", collector)
    def filter_node(state: TurnState) -> dict:
        # The heavy deterministic filtering already ran inside the retriever
        # (ticket #16); keep this node thin. ``shown_movies`` de-duplication is
        # likewise handled there, so here we simply promote survivors.
        candidates = list(state.retrieved)
        filters_applied = state.parsed.model_dump() if state.parsed else {}
        return {"candidates": candidates, "filters_applied": filters_applied}

    return filter_node


def _regen_query(state: TurnState) -> str:
    """Query fed to the generator, with the prior Gate-4 violation appended.

    On a regeneration pass (``validation_reason`` set) we tell the model exactly
    which rule it broke last time, so the retry has a chance of fixing it rather
    than resampling blind. On the first pass this is just the effective query.
    """
    query = _effective_query(state)
    if state.validation_reason:
        query = (
            f"{query}\n\nThe previous answer was REJECTED by output validation "
            f"for: {state.validation_reason}. Regenerate using ONLY facts present "
            "in the provided records and fix that violation."
        )
    return query


def _make_generate(models: GraphModels, collector: TraceCollector):
    @traced("generate", collector)
    def generate(state: TurnState) -> dict:
        query = _regen_query(state)
        try:
            response = models.generate(query, state.candidates)
            if not isinstance(response, RecommendationSet):
                response = RecommendationSet.model_validate(response)
            return {"response": response}
        except Exception:  # noqa: BLE001 - empty response is caught by validate
            return {
                "response": RecommendationSet(picks=[], prose=""),
                "degradation": [*state.degradation, "generate_error"],
            }

    return generate


def _make_validate(collector: TraceCollector):
    @traced("validate", collector)
    def validate(state: TurnState) -> dict:
        # Gate-4 (ticket #19): a deterministic output gate. First the STRUCTURAL
        # check (#18) - the response must parse as a RecommendationSet with at
        # least one pick - then the SEMANTIC Gate-4 checks (excluded actors,
        # title existence / anti-hallucination, prose fact-grounding). Any
        # violation drives the existing validate->generate regen edge (bounded),
        # then fallback. The machine-readable reason is stashed so the regen
        # prompt can tell the model exactly what to fix.
        response = state.response
        structurally_ok = isinstance(response, RecommendationSet) and len(response.picks) > 0
        if not structurally_ok:
            return {
                "validation_failed": True,
                "validation_reason": "structural:empty_picks",
                "gen_retries": state.gen_retries + 1,
            }

        exclude_actors = state.parsed.exclude_actors if state.parsed else []
        result = run_gate4(response, state.candidates, exclude_actors)
        if result.ok:
            return {"validation_failed": False, "validation_reason": None}
        return {
            "validation_failed": True,
            "validation_reason": result.reason,
            "gen_retries": state.gen_retries + 1,
        }

    return validate


def _make_fallback(collector: TraceCollector):
    @traced("fallback", collector)
    def fallback(state: TurnState) -> dict:
        response = RecommendationSet(picks=[], prose=_FALLBACK_PROSE)
        return {"response": response, "degradation": [*state.degradation, "fallback"]}

    return fallback


# -- conditional edges ---------------------------------------------------------


def _after_extract(state: TurnState) -> str:
    if state.extract_failed and state.extract_retries < MAX_EXTRACT_RETRIES:
        return "extract"
    return "retrieve"


def _after_filter(state: TurnState) -> str:
    return "fallback" if len(state.candidates) == 0 else "generate"


def _after_validate(state: TurnState) -> str:
    if state.validation_failed and state.gen_retries < MAX_GEN_RETRIES:
        return "generate"
    if state.validation_failed:
        return "fallback"
    return END


# -- assembly ------------------------------------------------------------------


def build_graph(
    *,
    retriever: RetrieverFn,
    models: GraphModels,
    collector: TraceCollector | None = None,
):
    """Compile the single-turn ``StateGraph``.

    ``collector`` is the per-turn tracing side-channel; ``run_turn`` supplies a
    fresh one per turn. Passing it here (rather than via graph config) keeps the
    nodes plain single-argument functions and keeps concurrent turns isolated.
    """
    collector = collector or TraceCollector()

    builder = StateGraph(TurnState)
    builder.add_node("rewrite", _make_rewrite(models, collector))
    builder.add_node("extract", _make_extract(models, collector))
    builder.add_node("retrieve", _make_retrieve(retriever, collector))
    builder.add_node("filter", _make_filter(collector))
    builder.add_node("generate", _make_generate(models, collector))
    builder.add_node("validate", _make_validate(collector))
    builder.add_node("fallback", _make_fallback(collector))

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "extract")
    builder.add_conditional_edges(
        "extract", _after_extract, {"extract": "extract", "retrieve": "retrieve"}
    )
    builder.add_edge("retrieve", "filter")
    builder.add_conditional_edges(
        "filter", _after_filter, {"fallback": "fallback", "generate": "generate"}
    )
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges(
        "validate",
        _after_validate,
        {"generate": "generate", "fallback": "fallback", END: END},
    )
    builder.add_edge("fallback", END)

    return builder.compile()


@dataclass
class TurnResult:
    """The outcome of one turn: final graph state plus its serialized trace."""

    state: TurnState
    trace: object  # TurnTrace (kept loose to avoid a schema import cycle in typing)


def run_turn(
    state: TurnState,
    *,
    retriever: RetrieverFn,
    models: GraphModels,
    store: TraceStore | None = None,
    versions: dict[str, str] | None = None,
    usage: UsageMeter | None = None,
    pricing: dict[str, dict[str, float]] | None = None,
) -> TurnResult:
    """Run one conversational turn end-to-end and serialize a ``TurnTrace``.

    A fresh ``TraceCollector`` is created per call so timings never bleed between
    turns. When ``store`` is provided the resulting trace is persisted (the trace
    is the system-of-record for the turn). When a ``usage`` meter is supplied its
    token totals and cost are folded into the trace. Returns state and trace.
    """
    collector = TraceCollector()
    graph = build_graph(retriever=retriever, models=models, collector=collector)
    result = graph.invoke(state)
    final = TurnState.model_validate(result)
    trace = serialize_trace(
        final, collector, versions=versions, usage=usage, pricing=pricing
    )
    if store is not None:
        store.write_trace(trace)
    return TurnResult(state=final, trace=trace)
