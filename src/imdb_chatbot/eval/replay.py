"""Multi-turn replay suite: scripted conversations + invariant checks (ticket #24).

The retrieval eval (ticket #17) protects a SINGLE turn's retrieval quality. This
module protects the MULTI-TURN memory behaviour - the guarantees from PRD section
6 (constraint state) and section 7.5 (reference resolution / minimal precedence)
that only appear across turns. It is the "multi-turn replay invariants" row of the
failure taxonomy (section 8.2: S1 constraint dropped across turns, S2 repetition).

A replay SCRIPT is a scripted conversation: a list of turns, each carrying the raw
user message plus a set of code-verifiable INVARIANTS to assert AFTER that turn:

    constraint_holds     - an exclusion set earlier still constrains this turn and
                           no recommended film violates it (S1).
    no_repeat            - nothing recommended this turn was already shown (S2).
    resolves_reference   - the rewriter resolved a reference ("that") into a
                           standalone query containing the expected anchor tokens
                           (section 7.5 turn-2 resolution).
    precedence_complies  - a durable EXCLUDED that this turn's live request
                           contradicts is suspended for the turn so the bot
                           complies (section 6.2 / ticket #22 minimal precedence).
    recommends_genre / min_picks / expect_fallback - positive helpers.

The RUNNER replays a script through the real single-turn graph (``run_turn``) with
injected FAKE models and a stub retriever over a small fixed catalog - fully
deterministic, ZERO network I/O. Session memory is carried across turns by
``ConversationState`` (ticket #21); durable EXCLUDED facts are seeded through the
real ``LocalJsonlStore`` / ``known_preferences`` path (ticket #22); minimal
precedence is applied per turn with ``resolve_collision`` so a live re-request of
a remembered exclusion suspends it for that turn only (the durable fact survives).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..graph import GraphModels, run_turn
from ..memory import (
    ConversationState,
    LocalJsonlStore,
    Triple,
    build_turn_state,
    known_preferences,
    normalize_user_id,
    resolve_collision,
    update_state_from_result,
)
from ..schemas import (
    MovieRecommendation,
    ParsedQuery,
    RecommendationSet,
    ScoredMovie,
    TurnState,
)

# How many picks the fake generator returns per turn. One keeps the no-repeat
# demonstration crisp: successive same-genre turns surface a NEW film each time.
PICKS_PER_TURN = 1


# -- the deterministic movie world -------------------------------------------


@dataclass(frozen=True)
class _Film:
    """A tiny catalog row for the replay world: identity + genres + cast."""

    tmdb_id: int
    title: str
    year: int
    genres: tuple[str, ...]
    cast: tuple[str, ...] = ()


# A small, stable catalog spanning the genres the scripts exercise. Order is
# load-bearing: the stub retriever preserves it, so same-genre turns surface
# films in this sequence (which is what makes no-repeat observable).
CATALOG: tuple[_Film, ...] = (
    _Film(1, "John Wick", 2014, ("Action",), ("Keanu Reeves",)),
    _Film(2, "Mad Max: Fury Road", 2015, ("Action",), ("Tom Hardy",)),
    _Film(3, "The Raid", 2011, ("Action",), ("Iko Uwais",)),
    _Film(4, "The Conjuring", 2013, ("Horror",), ("Vera Farmiga",)),
    _Film(5, "Hereditary", 2018, ("Horror",), ("Toni Collette",)),
    _Film(6, "Superbad", 2007, ("Comedy",), ("Jonah Hill",)),
    _Film(7, "Se7en", 1995, ("Thriller",), ("Brad Pitt",)),
    _Film(8, "Parasite", 2019, ("Thriller", "Drama"), ("Song Kang-ho",)),
    _Film(9, "Blade Runner 2049", 2017, ("Sci-Fi",), ("Ryan Gosling",)),
    _Film(10, "The Notebook", 2004, ("Romance",), ("Ryan Gosling",)),
)

_BY_TITLE_YEAR: dict[tuple[str, int], _Film] = {(f.title, f.year): f for f in CATALOG}
_KNOWN_ACTORS: tuple[str, ...] = tuple(sorted({a for f in CATALOG for a in f.cast}))

# genre synonyms -> canonical label. The fake extractor recognises these.
_GENRE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Action": ("action",),
    "Horror": ("horror", "scary", "scares", "frightening"),
    "Comedy": ("comedy", "comedies", "funny"),
    "Thriller": ("thriller", "thrillers"),
    "Drama": ("drama", "dramas"),
    "Sci-Fi": ("sci-fi", "scifi", "sci fi", "science fiction"),
    "Romance": ("romance", "romantic"),
}

# Negation triggers that flip a following genre/actor mention into an exclusion.
_NEG = r"(?:without|no|not|nothing|except|excluding|avoid)"


def _resolve_film(title: str, year: int) -> _Film | None:
    return _BY_TITLE_YEAR.get((title, year))


# -- fake models + stub retriever --------------------------------------------


def _last_user_query(history: Sequence[str]) -> str | None:
    """Pull the previous turn's raw user query out of the history window lines."""
    for line in reversed(list(history)):
        if line.startswith("User: "):
            return line[len("User: ") :].split(" | Bot")[0].strip()
    return None


def fake_rewrite(raw_query: str, history: Sequence[str]) -> str:
    """History-aware rewrite: resolve a reference ("that") to the prior query.

    If the message contains a reference phrase and there is a previous turn in the
    window, the reference is replaced with the previous raw query - so
    "something like that but without Keanu Reeves" after an "action movies" turn
    becomes "action movies but without Keanu Reeves". Otherwise the raw query is
    returned unchanged (a standalone message needs no resolution).
    """
    prev = _last_user_query(history)
    if not prev:
        return raw_query
    # Longest phrases first so "something like that" wins over bare "that".
    for phrase in (
        "something like that",
        "more like that",
        "more of those",
        "one of those",
        "like that",
        "those",
        "that one",
        "that",
    ):
        if phrase in raw_query.lower():
            return re.sub(re.escape(phrase), prev, raw_query, count=1, flags=re.IGNORECASE)
    return raw_query


def fake_extract(query: str) -> ParsedQuery:
    """Deterministic keyword/negation extractor - no LLM, no network.

    Recognises requested genres, negated genre exclusions ("without comedy",
    "nothing scary") and negated actor exclusions ("without Keanu Reeves"). A
    genre in a negation window becomes an exclusion, never a request.
    """
    text = query.lower()
    requested: list[str] = []
    exclude_genres: list[str] = []
    for canonical, syns in _GENRE_SYNONYMS.items():
        for syn in syns:
            negated = re.search(rf"{_NEG}\s+(?:\w+\s+){{0,2}}{re.escape(syn)}", text)
            if negated:
                exclude_genres.append(canonical)
                break
            if re.search(rf"\b{re.escape(syn)}\b", text):
                requested.append(canonical)
                break
    requested = [g for g in requested if g not in exclude_genres]

    exclude_actors: list[str] = []
    for actor in _KNOWN_ACTORS:
        if re.search(rf"{_NEG}\s+(?:\w+\s+){{0,2}}{re.escape(actor.lower())}", text):
            exclude_actors.append(actor)

    return ParsedQuery(
        genres=sorted(set(requested)),
        exclude_genres=sorted(set(exclude_genres)),
        exclude_actors=sorted(set(exclude_actors)),
    )


def fake_generate(query: str, candidates: Sequence[ScoredMovie]) -> RecommendationSet:
    """Recommend the top ``PICKS_PER_TURN`` surviving candidates, in order."""
    picks = [
        MovieRecommendation(title=c.title, year=c.year, reason=f"matches your request: {c.title}")
        for c in list(candidates)[:PICKS_PER_TURN]
    ]
    prose = "Here is what I would watch." if picks else ""
    return RecommendationSet(picks=picks, prose=prose)


def build_models() -> GraphModels:
    """The fake ``GraphModels`` bundle used by the whole replay suite."""
    return GraphModels(rewrite=fake_rewrite, extract=fake_extract, generate=fake_generate)


def stub_retriever(query: str, parsed: ParsedQuery) -> list[ScoredMovie]:
    """Deterministic retriever over ``CATALOG`` honouring the parsed constraints.

    Keeps a film iff it matches a requested genre (or none were requested) and is
    not excluded by genre or by cast. The graph's ``retrieve`` node has already
    folded the session's standing exclusions into ``parsed`` and drops
    ``shown_movies`` afterwards, so this stub only enforces the parse it is given.
    """
    requested = {g.lower() for g in parsed.genres}
    exclude_genres = {g.lower() for g in parsed.exclude_genres}
    exclude_actors = {a.lower() for a in parsed.exclude_actors}
    out: list[ScoredMovie] = []
    for rank, film in enumerate(CATALOG, start=1):
        film_genres = {g.lower() for g in film.genres}
        film_cast = {a.lower() for a in film.cast}
        if requested and not (film_genres & requested):
            continue
        if film_genres & exclude_genres:
            continue
        if film_cast & exclude_actors:
            continue
        out.append(
            ScoredMovie(
                tmdb_id=film.tmdb_id,
                title=film.title,
                year=film.year,
                rrf_score=1.0 / rank,
                rank=rank,
            )
        )
    return out


# -- script format ------------------------------------------------------------

InvariantKind = Literal[
    "constraint_holds",
    "no_repeat",
    "resolves_reference",
    "precedence_complies",
    "recommends_genre",
    "min_picks",
    "expect_fallback",
]


class Invariant(BaseModel):
    """One code-verifiable assertion checked after a turn."""

    kind: InvariantKind
    exclude_genres: list[str] = []
    exclude_actors: list[str] = []
    contains: list[str] = []
    genre: str | None = None
    n: int | None = None


class ReplayTurn(BaseModel):
    """One scripted turn: the raw user message + invariants to check after it."""

    user: str
    invariants: list[Invariant] = []


class ReplayScript(BaseModel):
    """A scripted multi-turn conversation with per-turn invariants.

    ``durable_excluded`` seeds durable EXCLUDED(genre) facts (ticket #22) before
    the first turn, so a later turn can exercise minimal precedence.
    """

    name: str
    description: str = ""
    user_id: str = "rishi"
    durable_excluded: list[str] = []
    turns: list[ReplayTurn]


def load_script(path: str | Path) -> ReplayScript:
    """Parse one replay-script JSON file into a typed ``ReplayScript``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReplayScript.model_validate(data)


def load_scripts(directory: str | Path) -> list[ReplayScript]:
    """Load every ``*.json`` replay script in ``directory``, sorted by filename."""
    return [load_script(p) for p in sorted(Path(directory).glob("*.json"))]


# -- runner -------------------------------------------------------------------


@dataclass
class _TurnContext:
    """Everything an invariant checker needs about one completed turn."""

    state: TurnState
    recommended: list[_Film]
    shown_before: set[int]
    fell_to_fallback: bool


@dataclass
class InvariantOutcome:
    kind: str
    ok: bool
    detail: str


@dataclass
class TurnOutcome:
    index: int
    user: str
    rewritten_query: str | None
    picks: list[tuple[str, int]]
    invariants: list[InvariantOutcome]

    @property
    def ok(self) -> bool:
        return all(inv.ok for inv in self.invariants)


@dataclass
class ScriptOutcome:
    name: str
    turns: list[TurnOutcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(turn.ok for turn in self.turns)

    def failures(self) -> list[str]:
        """Human-readable descriptions of every failed invariant."""
        out: list[str] = []
        for turn in self.turns:
            for inv in turn.invariants:
                if not inv.ok:
                    out.append(f"{self.name} turn {turn.index} [{inv.kind}]: {inv.detail}")
        return out

    def assert_ok(self) -> None:
        """Raise ``AssertionError`` listing every failed invariant (for tests)."""
        if not self.ok:
            raise AssertionError("; ".join(self.failures()))


# -- invariant checkers (pure) ------------------------------------------------


def _check_constraint_holds(inv: Invariant, ctx: _TurnContext) -> InvariantOutcome:
    parsed = ctx.state.parsed or ParsedQuery()
    for genre in inv.exclude_genres:
        if genre not in parsed.exclude_genres:
            return InvariantOutcome("constraint_holds", False, f"exclusion {genre!r} not applied")
        leaked = [f.title for f in ctx.recommended if genre in f.genres]
        if leaked:
            return InvariantOutcome(
                "constraint_holds", False, f"{genre!r} leaked via {leaked}"
            )
    for actor in inv.exclude_actors:
        if actor not in parsed.exclude_actors:
            return InvariantOutcome("constraint_holds", False, f"exclusion {actor!r} not applied")
        leaked = [f.title for f in ctx.recommended if actor in f.cast]
        if leaked:
            return InvariantOutcome(
                "constraint_holds", False, f"{actor!r} leaked via {leaked}"
            )
    return InvariantOutcome("constraint_holds", True, "ok")


def _check_no_repeat(inv: Invariant, ctx: _TurnContext) -> InvariantOutcome:
    repeated = sorted(f.title for f in ctx.recommended if f.tmdb_id in ctx.shown_before)
    if repeated:
        return InvariantOutcome("no_repeat", False, f"already-shown film(s) recommended: {repeated}")
    return InvariantOutcome("no_repeat", True, "ok")


def _check_resolves_reference(inv: Invariant, ctx: _TurnContext) -> InvariantOutcome:
    rewritten = (ctx.state.rewritten_query or "").lower()
    missing = [s for s in inv.contains if s.lower() not in rewritten]
    if missing:
        return InvariantOutcome(
            "resolves_reference",
            False,
            f"rewritten query {ctx.state.rewritten_query!r} missing {missing}",
        )
    return InvariantOutcome("resolves_reference", True, "ok")


def _check_precedence_complies(inv: Invariant, ctx: _TurnContext) -> InvariantOutcome:
    genre = inv.genre or ""
    parsed = ctx.state.parsed or ParsedQuery()
    if genre in parsed.exclude_genres:
        return InvariantOutcome(
            "precedence_complies", False, f"exclusion {genre!r} was NOT suspended for the live request"
        )
    if not any(genre in f.genres for f in ctx.recommended):
        return InvariantOutcome(
            "precedence_complies", False, f"no {genre!r} film recommended despite the live request"
        )
    return InvariantOutcome("precedence_complies", True, "ok")


def _check_recommends_genre(inv: Invariant, ctx: _TurnContext) -> InvariantOutcome:
    genre = inv.genre or ""
    if not any(genre in f.genres for f in ctx.recommended):
        return InvariantOutcome("recommends_genre", False, f"no {genre!r} film recommended")
    return InvariantOutcome("recommends_genre", True, "ok")


def _check_min_picks(inv: Invariant, ctx: _TurnContext) -> InvariantOutcome:
    need = inv.n or 1
    have = len(ctx.recommended)
    if have < need:
        return InvariantOutcome("min_picks", False, f"expected >= {need} picks, got {have}")
    return InvariantOutcome("min_picks", True, "ok")


def _check_expect_fallback(inv: Invariant, ctx: _TurnContext) -> InvariantOutcome:
    if not ctx.fell_to_fallback:
        return InvariantOutcome("expect_fallback", False, "turn did not take the fallback path")
    return InvariantOutcome("expect_fallback", True, "ok")


_CHECKERS = {
    "constraint_holds": _check_constraint_holds,
    "no_repeat": _check_no_repeat,
    "resolves_reference": _check_resolves_reference,
    "precedence_complies": _check_precedence_complies,
    "recommends_genre": _check_recommends_genre,
    "min_picks": _check_min_picks,
    "expect_fallback": _check_expect_fallback,
}


def _check(inv: Invariant, ctx: _TurnContext) -> InvariantOutcome:
    return _CHECKERS[inv.kind](inv, ctx)


# -- session seeding + turn driver --------------------------------------------


def _seed_durable(
    conversation: ConversationState,
    user_id: str,
    durable_excluded: Sequence[str],
    workdir: Path,
) -> None:
    """Seed durable EXCLUDED facts through the real #22 store + passive injection.

    Writes ``EXCLUDED(genre)`` triples to a ``LocalJsonlStore`` and folds them
    into the session's constraint state via ``known_preferences`` - the same
    passive-injection path production uses at session start.
    """
    store = LocalJsonlStore(workdir)
    uid = normalize_user_id(user_id)
    for genre in durable_excluded:
        store.append(uid, Triple(user_id=uid, relation="EXCLUDED", object=genre, provenance="replay_seed"))
    known_preferences(uid, store).apply_to(conversation)


def _drive_turn(
    conversation: ConversationState,
    raw_query: str,
    *,
    trace_id: str,
    models: GraphModels,
) -> tuple[object, set[int]]:
    """Run one turn with minimal precedence applied, then fold the result back.

    Mirrors ``run_session_turn`` but inserts the section-6.2 precedence step: the
    live request's explicitly-requested genres suspend any colliding standing
    exclusion FOR THIS TURN ONLY (via ``resolve_collision``), leaving the durable
    / session exclusion set itself untouched for later turns.
    """
    requested_genres = fake_extract(raw_query).genres
    state = build_turn_state(conversation, raw_query, trace_id=trace_id)
    effective = resolve_collision(state.session_exclude_genres, requested_genres)
    state = state.model_copy(update={"session_exclude_genres": effective})

    shown_before = set(conversation.shown_movies)
    result = run_turn(state, retriever=stub_retriever, models=models)
    update_state_from_result(conversation, result)
    return result, shown_before


def run_script(script: ReplayScript, *, workdir: str | Path | None = None) -> ScriptOutcome:
    """Replay ``script`` through the graph and evaluate each turn's invariants.

    Deterministic and network-free. ``workdir`` is a scratch directory for the
    durable-memory store when the script seeds ``durable_excluded`` (tests pass
    ``tmp_path``); scripts without durable facts need none.
    """
    models = build_models()
    conversation = ConversationState(session_id=f"replay-{script.name}")
    if script.durable_excluded:
        if workdir is None:
            raise ValueError(
                f"script {script.name!r} seeds durable_excluded but no workdir was given"
            )
        _seed_durable(conversation, script.user_id, script.durable_excluded, Path(workdir))

    outcome = ScriptOutcome(name=script.name)
    for index, turn in enumerate(script.turns, start=1):
        result, shown_before = _drive_turn(
            conversation, turn.user, trace_id=f"{script.name}-t{index}", models=models
        )
        final = result.state
        recommended: list[_Film] = []
        if final.response is not None:
            for pick in final.response.picks:
                film = _resolve_film(pick.title, pick.year)
                if film is not None:
                    recommended.append(film)
        ctx = _TurnContext(
            state=final,
            recommended=recommended,
            shown_before=shown_before,
            fell_to_fallback="fallback" in final.path_taken,
        )
        outcome.turns.append(
            TurnOutcome(
                index=index,
                user=turn.user,
                rewritten_query=final.rewritten_query,
                picks=[(f.title, f.year) for f in recommended],
                invariants=[_check(inv, ctx) for inv in turn.invariants],
            )
        )
    return outcome
