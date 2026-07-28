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

``validate`` here is a STRUCTURAL check only (does the response parse as a
``RecommendationSet`` with non-empty picks?). The Gate-4 semantic validation is a
separate later ticket (#19).

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
from .models import GraphModels
from .tracing import TraceCollector, serialize_trace, traced

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
            rewritten = models.rewrite(state.raw_query, [])
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


def _make_retrieve(retriever: RetrieverFn, collector: TraceCollector):
    @traced("retrieve", collector)
    def retrieve(state: TurnState) -> dict:
        parsed = state.parsed or ParsedQuery()
        retrieved = list(retriever(_effective_query(state), parsed))
        return {"retrieved": retrieved}

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


def _make_generate(models: GraphModels, collector: TraceCollector):
    @traced("generate", collector)
    def generate(state: TurnState) -> dict:
        query = _effective_query(state)
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
        # STRUCTURAL check only (ticket #18): response parses as a
        # RecommendationSet AND has at least one pick. Semantic Gate-4 is #19.
        response = state.response
        structurally_ok = isinstance(response, RecommendationSet) and len(response.picks) > 0
        if structurally_ok:
            return {"validation_failed": False}
        return {"validation_failed": True, "gen_retries": state.gen_retries + 1}

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
) -> TurnResult:
    """Run one conversational turn end-to-end and serialize a ``TurnTrace``.

    A fresh ``TraceCollector`` is created per call so timings never bleed between
    turns. When ``store`` is provided the resulting trace is persisted (the trace
    is the system-of-record for the turn). Returns both the final state and trace.
    """
    collector = TraceCollector()
    graph = build_graph(retriever=retriever, models=models, collector=collector)
    result = graph.invoke(state)
    final = TurnState.model_validate(result)
    trace = serialize_trace(final, collector, versions=versions)
    if store is not None:
        store.write_trace(trace)
    return TurnResult(state=final, trace=trace)
