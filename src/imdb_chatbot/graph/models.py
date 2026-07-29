"""Injectable model factory: one chat model per slot (PRD section 5.1).

``GraphModels`` bundles the three LLM-backed behaviours the graph needs, each as
a plain callable:

- ``rewrite(raw_query, history) -> str``            (rewriter slot, temp 0)
- ``extract(query) -> ParsedQuery``                 (extractor slot, temp 0, JSON)
- ``generate(query, candidates) -> RecommendationSet`` (generator slot, temp 0.7)

Modelling the seam as three callables keeps the graph nodes trivially testable:
tests construct ``GraphModels`` directly with fakes (see tests/test_graph.py), so
no live OpenRouter call is ever made. ``build_models`` wires the real LangChain
models from ``config/models.yaml`` - the extractor and generator use structured
output (``with_structured_output``) against the Pydantic contracts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import get_secret, load_models_config
from ..schemas import ParsedQuery, RecommendationSet, ScoredMovie
from .usage import UsageMeter, usage_from_message

RewriteFn = Callable[[str, Sequence[Any]], str]
ExtractFn = Callable[[str], ParsedQuery]
GenerateFn = Callable[[str, Sequence[ScoredMovie]], RecommendationSet]


@dataclass
class GraphModels:
    """The three LLM behaviours the graph depends on, as injectable callables.

    Any of the three may raise to exercise a degradation / retry edge; the graph
    nodes translate a raised/invalid result into the appropriate state flag
    (rewrite -> degrade to raw query; extract -> ``extract_failed``; generate ->
    empty response caught by ``validate``).
    """

    rewrite: RewriteFn
    extract: ExtractFn
    generate: GenerateFn


# -- prompts (v1 placeholders; the versioned prompt artifact lands in a later ticket)


def _rewrite_prompt(raw_query: str, history: Sequence[Any]) -> str:
    hist = "\n".join(str(h) for h in history) if history else "(none)"
    return (
        "Rewrite the user's latest message into a single standalone movie-search "
        "query, resolving pronouns from the conversation history. Return only the "
        "rewritten query.\n\n"
        f"History:\n{hist}\n\nLatest message: {raw_query}\n\nStandalone query:"
    )


def _extract_prompt(query: str) -> str:
    return (
        "Extract structured search filters from the query as JSON matching the "
        "ParsedQuery schema (genres, similar_to, director, actor, exclude_actors, "
        "exclude_genres, min_year, max_year, min_rating, region). Decide a named "
        "person's ROLE from the wording: 'directed by', 'by', 'from director', or "
        "a bare name ('Tarantino movies') -> set 'director'; 'with', 'starring', "
        "'featuring', 'stars' -> set 'actor'. Examples: 'movies by Christopher "
        "Nolan' -> director='Christopher Nolan'; 'movies with Tom Hanks' -> "
        "actor='Tom Hanks'; 'films starring Meryl Streep' -> actor='Meryl Streep'. "
        "Never put an actor in 'director' or a director in 'actor'. Omit unknown "
        "fields.\n\n"
        f"Query: {query}"
    )


def _generate_prompt(query: str, candidates: Sequence[ScoredMovie]) -> str:
    lines = [f"- {c.title} ({c.year})" for c in candidates]
    listing = "\n".join(lines) if lines else "(no candidates)"
    return (
        "You are Maya, an expert movie recommender. Recommend ONLY from the "
        "candidate movies listed below, and ONLY those that GENUINELY match the "
        "user's request. A candidate matches only if it truly is what was asked "
        "for: a documentary, featurette, or making-of ABOUT a film or director is "
        "NOT one of that director's movies, so do not offer it as one. If NONE of "
        "the candidates genuinely match the request, return an EMPTY picks list "
        "rather than stretching to a loose match. State only facts present in the "
        "provided records; never invent titles, years, directors, cast, or plot, "
        "and never mention an actor the user asked to exclude.\n\n"
        f"User request: {query}\n\nCandidates:\n{listing}"
    )


# -- real factory (lazy import so tests never require langchain at import time)


def _init_slot_model(slot: str, cfg: dict[str, Any]) -> Any:
    """Build one OpenRouter-backed chat model for ``slot`` from a models.yaml dict.

    Imported lazily so importing this module (and running the fake-model tests)
    never pulls LangChain or touches the network. Uses ``ChatOpenAI`` directly
    (OpenRouter is OpenAI-API compatible) and asks OpenRouter to include the real
    per-request cost in the usage block so the trace can report actual spend.
    """
    from langchain_openai import ChatOpenAI

    slot_cfg = cfg["slots"][slot]
    return ChatOpenAI(
        model=slot_cfg["default"],
        base_url=cfg["base_url"],
        api_key=get_secret(cfg["secret"]),
        temperature=slot_cfg.get("temperature", 0),
        # OpenRouter returns usage.cost (and detailed token usage) when asked.
        extra_body={"usage": {"include": True}},
    )


def build_models(
    cfg: dict[str, Any] | None = None,
    *,
    meter: UsageMeter | None = None,
) -> GraphModels:
    """Wire the real OpenRouter-backed ``GraphModels`` from ``config/models.yaml``.

    Tests do NOT call this - they construct ``GraphModels`` with fakes. It is only
    exercised at runtime when a live ``OPENROUTER_API_KEY`` is available.

    When a ``meter`` is supplied, each slot's token usage / model / cost is
    recorded into it after every invoke (the extractor and generator use
    ``include_raw=True`` so the underlying ``AIMessage`` - and its usage metadata -
    survives structured-output parsing).
    """
    cfg = cfg or load_models_config()

    # Maya's persona (ticket #44): prepended as the generator's system message so
    # recommendations speak in her voice. Imported lazily-cheap; falls back to an
    # empty string if the artifact is missing so the generator still runs.
    try:
        from ..persona import generator_system_prompt

        persona_system = generator_system_prompt()
    except Exception:  # noqa: BLE001 - persona is optional flavor, never fatal
        persona_system = ""

    rewriter = _init_slot_model("rewriter", cfg)
    extractor = _init_slot_model("extractor", cfg).with_structured_output(
        ParsedQuery, include_raw=True
    )
    generator = _init_slot_model("generator", cfg).with_structured_output(
        RecommendationSet, include_raw=True
    )

    def _record(slot: str, raw_message: Any) -> None:
        if meter is None:
            return
        input_tokens, output_tokens, model, cost = usage_from_message(raw_message)
        meter.record(
            slot,
            model=model or cfg["slots"][slot]["default"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reported_cost_usd=cost,
        )

    def rewrite(raw_query: str, history: Sequence[Any]) -> str:
        resp = rewriter.invoke(_rewrite_prompt(raw_query, history))
        _record("rewriter", resp)
        text = getattr(resp, "content", resp)
        return str(text).strip()

    def extract(query: str) -> ParsedQuery:
        out = extractor.invoke(_extract_prompt(query))
        _record("extractor", out.get("raw"))
        parsed = out.get("parsed")
        if parsed is None:
            raise ValueError("extractor returned no parsed output")
        return parsed

    def generate(query: str, candidates: Sequence[ScoredMovie]) -> RecommendationSet:
        user_prompt = _generate_prompt(query, candidates)
        messages = (
            [{"role": "system", "content": persona_system}] if persona_system else []
        )
        messages.append({"role": "user", "content": user_prompt})
        out = generator.invoke(messages)
        _record("generator", out.get("raw"))
        response = out.get("parsed")
        if response is None:
            raise ValueError("generator returned no parsed output")
        return response

    return GraphModels(rewrite=rewrite, extract=extract, generate=generate)
