"""Assemble the live, graph-backed Chat handler for the Streamlit page.

The dashboard ships with a deterministic stub handler (``dashboard.app`` /
``_default_handler``) that always returns the relax-a-constraint fallback. This
module builds the REAL handler: it loads the live index, the embedder, the corpus
store and the hybrid retriever, wires the OpenRouter-backed ``GraphModels``, and
runs each user message through the full LangGraph turn with session memory.

Everything heavy is imported and constructed lazily inside
``load_live_resources`` so importing this module never pulls FAISS/torch/langchain
and never needs a network key. Two failure modes are surfaced as clear
``RuntimeError`` messages (never a crash) so the app can fall back to the stub and
tell the user exactly why:

- no ``OPENROUTER_API_KEY`` -> "set the key";
- the key is still dotenvx-encrypted -> "launch under dotenvx".

The turn logic is split from resource loading so it is unit-testable with fakes:
``build_live_chat_handler`` takes a ``LiveResources`` bundle whose ``retriever``
and ``models_factory`` can be fakes, so a full ChatReply (with telemetry) can be
produced with zero network or index I/O.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..graph import GraphModels, RetrieverFn, UsageMeter, estimate_cost
from ..movie_info import TitleExtractor, TitleLookup, answer_movie_question
from ..persona import (
    Intent,
    chitchat_fallback,
    classify_intent,
    persona_reply,
)
from ..router import FollowupClassifier, FollowupKind, clarify_question, route_followup
from ..schemas import ParsedQuery, RecommendationSet, index_by_title_year

# A factory that builds the turn's GraphModels bound to a usage meter.
ModelsFactory = Callable[[UsageMeter | None], GraphModels]
# A conversational (chit-chat) reply generator: (message, meter) -> reply text.
ChatFn = Callable[[str, "UsageMeter | None"], str]

DEFAULT_CORPUS_PATH = Path("data") / "corpus.sqlite"

# System prompts for the two one-shot classifications the cheap model performs.
_TITLE_EXTRACT_PROMPT = (
    "Extract the movie title the user is asking about. Reply with ONLY the "
    "title, nothing else. If there is no title, reply with an empty line."
)
_FOLLOWUP_PROMPT = (
    "You route a follow-up message in a movie chat. Reply with EXACTLY one "
    "word: 'refine' (the user wants to adjust or narrow the movies just "
    "shown), 'replace' (a brand-new, unrelated movie search), or 'clarify' "
    "(they rejected the results but gave no new criteria)."
)


@dataclass
class TurnTelemetry:
    """The per-turn numbers the Chat UI shows: model(s), tokens up/down, cost."""

    models: dict[str, str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    path_taken: list[str] = field(default_factory=list)
    degradation: list[str] = field(default_factory=list)


@dataclass
class ChatReply:
    """A chat handler's return value: the answer plus its telemetry.

    ``telemetry`` is ``None`` when no LLM ran (the deterministic stub, or a
    persona/small-talk reply), so the UI omits the telemetry strip in that case.

    ``conversational`` marks a persona reply (Maya's greeting / purpose): the UI
    renders its prose as a plain message, WITHOUT the relax-a-constraint buttons
    that a genuine empty-result search fallback shows.
    """

    rec: RecommendationSet
    telemetry: TurnTelemetry | None = None
    conversational: bool = False


@dataclass
class LiveResources:
    """Everything a live turn needs, minus the per-session conversation state.

    Split out from ``build_live_chat_handler`` so the expensive bits (index,
    embedder, store, retriever) can be built once and cached, while a fresh
    handler (with its own ``ConversationState``) is cheap to make per session.
    """

    retriever: RetrieverFn
    models_factory: ModelsFactory
    pricing: dict[str, dict[str, float]]
    versions: dict[str, str]
    store: object | None = None  # TraceStore | None (kept loose to avoid import)
    # Cheap LLM used for conversational (chit-chat) replies; None -> deterministic
    # fallback (offline). Kept separate from the graph so a "how are you" costs one
    # small call, not a full retrieval turn.
    chat_fn: ChatFn | None = None
    # Classifies a follow-up turn (refine / replace / clarify) when prior results
    # exist (ticket #54). None -> no routing (every search is treated as fresh).
    followup_fn: FollowupClassifier | None = None
    # Answering factual movie questions (ticket #58): a corpus title lookup and an
    # LLM that pulls the title out of the question. None -> a generic redirect.
    movie_lookup: TitleLookup | None = None
    title_extractor: TitleExtractor | None = None


def load_live_resources(
    *,
    corpus_path: Path | str | None = None,
    embedder_profile: str | None = None,
) -> LiveResources:
    """Build the live resource bundle from the on-disk index + corpus + config.

    Raises ``RuntimeError`` with an actionable message when the app is not ready
    to serve live (no active index, missing key, encrypted key). The caller
    catches it and falls back to the stub handler.
    """
    from ..config import get_secret, load_live_index, load_models_config
    from ..graph.models import _init_slot_model, build_models
    from ..graph.usage import load_pricing, usage_from_message
    from ..index.build import load_index
    from ..index.embedder import build_embedder
    from ..persona import chat_system_prompt, persona_version
    from ..retrieval.retrieve import HybridRetriever
    from ..store import TraceStore

    cfg = load_models_config()

    # Fail fast with a clear reason BEFORE loading the (large) index.
    key = get_secret(cfg["secret"], required=False)
    if not key:
        raise RuntimeError(
            f"{cfg['secret']} is not set. Add it to your local .env or the "
            "platform secrets, then reload."
        )
    if key.startswith("encrypted:"):
        raise RuntimeError(
            f"{cfg['secret']} is still dotenvx-encrypted. Launch the app under "
            "dotenvx so the real key is injected (see the README launch command)."
        )

    ptr = load_live_index()
    if not ptr.get("active"):
        raise RuntimeError(
            "No live index is active (config/live_index.json). Build an index "
            "first, then reload."
        )

    loaded = load_index(ptr["path"])
    # Which embedder runs is config (models.yaml -> embedder.default), and it must
    # match the one the live index was built with. A hosted profile that cannot be
    # constructed degrades to the local model inside build_embedder.
    embedder = build_embedder(embedder_profile, cfg)
    store = TraceStore(Path(corpus_path or DEFAULT_CORPUS_PATH))
    hybrid = HybridRetriever.from_store(loaded, embedder, store)

    def retriever(rewritten_query: str, parsed: ParsedQuery):
        return hybrid.retrieve(rewritten_query, parsed)

    def models_factory(meter: UsageMeter | None) -> GraphModels:
        return build_models(cfg, meter=meter)

    # Cheap conversational model (rewriter slot) for chit-chat replies.
    chat_model = _init_slot_model("rewriter", cfg)
    chat_system = chat_system_prompt()

    def ask(system_prompt: str, user_msg: str, meter: UsageMeter | None = None) -> str:
        """One cheap-model round trip, optionally metered into the turn's usage."""
        resp = chat_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
        )
        if meter is not None:
            input_tokens, output_tokens, model, cost = usage_from_message(resp)
            meter.record(
                "chat",
                model=model or cfg["slots"]["rewriter"]["default"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reported_cost_usd=cost,
            )
        return str(getattr(resp, "content", resp))

    def chat_fn(message: str, meter: UsageMeter | None) -> str:
        return ask(chat_system, message, meter).strip()

    def title_extractor(query: str) -> str:
        return ask(_TITLE_EXTRACT_PROMPT, query).strip().strip('"')

    def followup_fn(query: str, last_titles: list[str]) -> str:
        titles = ", ".join(last_titles) or "(none)"
        return ask(_FOLLOWUP_PROMPT, f"Movies just shown: {titles}\nUser's message: {query}")

    from ..movie_info import build_title_lookup

    movie_lookup = build_title_lookup(hybrid.movies_by_id.values())

    return LiveResources(
        retriever=retriever,
        models_factory=models_factory,
        pricing=load_pricing(cfg),
        versions={
            "index": str(ptr.get("active", "dev")),
            "model_config": "dev",
            "prompt": persona_version(),
        },
        store=store,
        chat_fn=chat_fn,
        followup_fn=followup_fn,
        movie_lookup=movie_lookup,
        title_extractor=title_extractor,
    )


def _enrich_posters(
    rec: RecommendationSet,
    candidates: list,
    store: object | None,
) -> RecommendationSet:
    """Backfill each pick's poster_url from the corpus (never trust the LLM for it).

    The generator's structured output includes a ``poster_url`` field, which the
    model *guesses* - usually blank or a broken URL. Here we overwrite it with the
    authoritative TMDB poster from the matching ``MovieRecord`` (looked up by the
    candidate's ``tmdb_id``). A missing store or unmatched pick is left as-is.
    """
    if store is None or not rec.picks:
        return rec
    by_title_year = index_by_title_year(candidates)
    for pick in rec.picks:
        tmdb_id = by_title_year.get((pick.title, pick.year))
        if tmdb_id is None:
            continue
        try:
            movie = store.read_movie(tmdb_id)
        except Exception:  # noqa: BLE001 - a lookup miss must never break rendering
            movie = None
        if movie is not None and getattr(movie, "poster_url", None):
            pick.poster_url = movie.poster_url
    return rec


def _last_recommended(conversation: object) -> list[str]:
    """The titles shown on the most recent turn that produced picks (or [])."""
    for turn in reversed(getattr(conversation, "turns", [])):
        if turn.recommended:
            return list(turn.recommended)
    return []


def build_live_chat_handler(
    resources: LiveResources,
) -> Callable[[str], ChatReply]:
    """Return a ``(query) -> ChatReply`` handler with its own session memory.

    Each call runs one full turn through the graph, measuring token usage into a
    fresh ``UsageMeter``, folding the outcome back into the session (so the next
    turn's rewriter sees history and retrieval skips already-shown movies), and
    packaging the answer plus telemetry.
    """
    from ..memory.session import ConversationState, run_session_turn

    conversation = ConversationState(session_id=str(uuid.uuid4()))

    def handler(query: str) -> ChatReply:
        # Intent gate: greetings / meta questions get Maya's deterministic reply
        # (no retrieval, no LLM); conversational chit-chat gets a cheap LLM reply
        # that offers to help; only genuine searches run the movie pipeline.
        intent = classify_intent(query)
        if intent in (Intent.GREETING, Intent.META):
            return ChatReply(
                rec=persona_reply(intent),
                telemetry=None,
                conversational=True,
            )

        if intent is Intent.MOVIE_QUESTION:
            # A factual question about a specific film: answer from the corpus.
            if resources.movie_lookup is not None and resources.title_extractor is not None:
                text = answer_movie_question(
                    query, resources.movie_lookup, resources.title_extractor
                )
            else:
                text = (
                    "Tell me a movie title and I'll share what I know about it - "
                    "or I can recommend something."
                )
            return ChatReply(
                rec=RecommendationSet(picks=[], prose=text),
                telemetry=None,
                conversational=True,
            )

        if intent is Intent.CHITCHAT:
            meter = UsageMeter()
            if resources.chat_fn is not None:
                text = resources.chat_fn(query, meter)
            else:
                text = chitchat_fallback()
            telemetry = None
            if meter.slots:
                telemetry = TurnTelemetry(
                    models=meter.models(),
                    input_tokens=meter.input_tokens,
                    output_tokens=meter.output_tokens,
                    cost_usd=estimate_cost(meter, resources.pricing),
                )
            return ChatReply(
                rec=RecommendationSet(picks=[], prose=text),
                telemetry=telemetry,
                conversational=True,
            )

        # Follow-up routing (ticket #54): if results are already on screen, decide
        # whether this turn refines them, replaces them, or is a vague rejection.
        last_titles = _last_recommended(conversation)
        if last_titles and resources.followup_fn is not None:
            kind = route_followup(query, last_titles, resources.followup_fn)
            if kind is FollowupKind.CLARIFY:
                return ChatReply(
                    rec=RecommendationSet(picks=[], prose=clarify_question()),
                    telemetry=None,
                    conversational=True,
                )
            if kind is FollowupKind.REPLACE:
                # A brand-new search: forget accumulated constraints and let
                # previously-shown movies be eligible again.
                conversation.reset_standing()
                conversation.shown_movies.clear()
            # REFINE keeps the standing constraints (they accumulate) and the
            # shown-movies set (so a narrowed search surfaces fresh titles).

        meter = UsageMeter()
        models = resources.models_factory(meter)
        result = run_session_turn(
            conversation,
            query,
            retriever=resources.retriever,
            models=models,
            trace_id=str(uuid.uuid4()),
            store=resources.store,
            versions=resources.versions,
            usage=meter,
            pricing=resources.pricing,
        )
        rec = result.state.response or RecommendationSet(picks=[], prose="")
        # Backfill authoritative posters from the corpus (ticket #2 fix).
        rec = _enrich_posters(rec, result.state.candidates, resources.store)
        telemetry = TurnTelemetry(
            models=meter.models(),
            input_tokens=meter.input_tokens,
            output_tokens=meter.output_tokens,
            cost_usd=result.trace.cost_usd,
            path_taken=list(result.state.path_taken),
            degradation=list(result.state.degradation),
        )
        return ChatReply(rec=rec, telemetry=telemetry)

    return handler
