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
        "ParsedQuery schema (genres, similar_to, exclude_actors, exclude_genres, "
        "min_year, max_year, min_rating, region). Omit unknown fields.\n\n"
        f"Query: {query}"
    )


def _generate_prompt(query: str, candidates: Sequence[ScoredMovie]) -> str:
    lines = [f"- {c.title} ({c.year})" for c in candidates]
    listing = "\n".join(lines) if lines else "(no candidates)"
    return (
        "You are a movie recommender. Using ONLY the candidate movies below, write "
        "a friendly recommendation as a RecommendationSet (picks + prose). State "
        "only facts present in the provided records; no external trivia. Do not "
        "invent titles, years, directors, or cast, and do not mention any actor "
        "the user asked to exclude.\n\n"
        f"User request: {query}\n\nCandidates:\n{listing}"
    )


# -- real factory (lazy import so tests never require langchain at import time)


def _init_slot_model(slot: str, cfg: dict[str, Any]) -> Any:
    """Build one LangChain chat model for ``slot`` from a models.yaml dict.

    Imported lazily so importing this module (and running the fake-model tests)
    never pulls LangChain or touches the network.
    """
    try:
        from langchain.chat_models import init_chat_model
    except ImportError:  # pragma: no cover - depends on installed extras
        from langchain_core.language_models import init_chat_model  # type: ignore[no-redef]

    slot_cfg = cfg["slots"][slot]
    return init_chat_model(
        slot_cfg["default"],
        model_provider="openai",  # OpenRouter is OpenAI-API compatible
        base_url=cfg["base_url"],
        api_key=get_secret(cfg["secret"]),
        temperature=slot_cfg.get("temperature", 0),
    )


def build_models(cfg: dict[str, Any] | None = None) -> GraphModels:
    """Wire the real OpenRouter-backed ``GraphModels`` from ``config/models.yaml``.

    Tests do NOT call this - they construct ``GraphModels`` with fakes. It is only
    exercised at runtime when a live ``OPENROUTER_API_KEY`` is available.
    """
    cfg = cfg or load_models_config()

    rewriter = _init_slot_model("rewriter", cfg)
    extractor = _init_slot_model("extractor", cfg).with_structured_output(ParsedQuery)
    generator = _init_slot_model("generator", cfg).with_structured_output(RecommendationSet)

    def rewrite(raw_query: str, history: Sequence[Any]) -> str:
        resp = rewriter.invoke(_rewrite_prompt(raw_query, history))
        text = getattr(resp, "content", resp)
        return str(text).strip()

    def extract(query: str) -> ParsedQuery:
        return extractor.invoke(_extract_prompt(query))

    def generate(query: str, candidates: Sequence[ScoredMovie]) -> RecommendationSet:
        return generator.invoke(_generate_prompt(query, candidates))

    return GraphModels(rewrite=rewrite, extract=extract, generate=generate)
