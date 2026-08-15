"""Session memory + multi-turn tests (ticket #21).

Every scenario is a scripted multi-turn conversation driven by FAKE models and a
stub retriever - there is NO network I/O and no live LLM call. The invariants are
the PRD's multi-turn replay guarantees (section 6 / section 7.5):

- a constraint set in an early turn still holds turns later,
- no movie is recommended twice in a session (``shown_movies``),
- turn-2 reference resolution: "that" resolves to the turn-1 film + an exclusion,
- the short-term window keeps <= 6 turns verbatim and summarizes older ones.
"""

from __future__ import annotations

import pytest

from imdb_chatbot.graph import GraphModels
from imdb_chatbot.memory import (
    WINDOW_SIZE,
    ConversationState,
    Turn,
    run_session_turn,
)
from imdb_chatbot.schemas import (
    MovieRecommendation,
    ParsedQuery,
    RecommendationSet,
    ScoredMovie,
)

# -- fakes --------------------------------------------------------------------


def _movie(tmdb_id: int, title: str, year: int) -> ScoredMovie:
    return ScoredMovie(tmdb_id=tmdb_id, title=title, year=year, rrf_score=0.5, rank=1)


def _recording_retriever(catalog: list[ScoredMovie], seen: list[ParsedQuery]):
    """Retriever that returns a fixed catalog and records each parse it receives.

    Recording the parse lets a test assert that standing exclusions are still
    present on a later turn (the constraint held across turns).
    """

    def retrieve(query: str, parsed: ParsedQuery, shown_movies=()) -> list[ScoredMovie]:
        seen.append(parsed)
        return list(catalog)

    return retrieve


def _pick_first_candidate_models(
    rewrite_fn=None,
    extract_fn=None,
) -> GraphModels:
    """Fake models whose generator recommends the FIRST surviving candidate.

    Because the pick mirrors a real candidate (title/year), the session driver
    can map it back to a ``tmdb_id`` and mark it shown - which is exactly what
    makes the no-repeat invariant observable in a test.
    """

    def default_rewrite(raw_query: str, history) -> str:
        return raw_query

    def default_extract(query: str) -> ParsedQuery:
        return ParsedQuery(genres=["Action"])

    def generate(query: str, candidates) -> RecommendationSet:
        if not candidates:
            return RecommendationSet(picks=[], prose="no candidates")
        top = candidates[0]
        return RecommendationSet(
            picks=[
                MovieRecommendation(title=top.title, year=top.year, reason="great pick")
            ],
            prose=f"Try {top.title}.",
        )

    return GraphModels(
        rewrite=rewrite_fn or default_rewrite,
        extract=extract_fn or default_extract,
        generate=generate,
    )


# -- constraint persists across turns -----------------------------------------


def test_exclusion_set_in_turn_1_still_holds_at_turn_3() -> None:
    seen: list[ParsedQuery] = []
    catalog = [_movie(1, "A", 2020), _movie(2, "B", 2021), _movie(3, "C", 2022)]
    retriever = _recording_retriever(catalog, seen)

    # Turn 1 introduces the exclusion; later turns must not re-state it.
    def extract(query: str) -> ParsedQuery:
        if "tom cruise" in query.lower():
            return ParsedQuery(genres=["Action"], exclude_actors=["Tom Cruise"])
        return ParsedQuery(genres=["Action"])

    models = _pick_first_candidate_models(extract_fn=extract)
    conv = ConversationState(session_id="sess-1")

    run_session_turn(
        conv,
        "an action movie without Tom Cruise",
        retriever=retriever,
        models=models,
        trace_id="t1",
    )
    run_session_turn(
        conv, "something fun", retriever=retriever, models=models, trace_id="t2"
    )
    run_session_turn(
        conv, "another one", retriever=retriever, models=models, trace_id="t3"
    )

    # The exclusion persisted in session state...
    assert conv.exclude_actors == ["Tom Cruise"]
    # ...and was still applied to retrieval on turn 3, even though turns 2 and 3
    # never mentioned the actor.
    assert "Tom Cruise" in seen[-1].exclude_actors
    assert all("Tom Cruise" in parse.exclude_actors for parse in seen)


# -- no movie recommended twice -----------------------------------------------


def test_no_movie_recommended_twice_in_a_session() -> None:
    seen: list[ParsedQuery] = []
    catalog = [_movie(1, "A", 2020), _movie(2, "B", 2021), _movie(3, "C", 2022)]
    retriever = _recording_retriever(catalog, seen)
    models = _pick_first_candidate_models()
    conv = ConversationState(session_id="sess-2")

    recommended: list[str] = []
    for i in range(3):
        result = run_session_turn(
            conv,
            "recommend something",
            retriever=retriever,
            models=models,
            trace_id=f"t{i}",
        )
        for pick in result.state.response.picks:
            recommended.append(pick.title)

    # Each turn surfaced a DIFFERENT movie (the previous pick was de-duped out).
    assert recommended == ["A", "B", "C"]
    assert len(set(recommended)) == len(recommended)
    assert conv.shown_movies == {1, 2, 3}


# -- section 7.5 turn-2 reference resolution -----------------------------------


def test_turn_2_reference_resolution_resolves_that_and_adds_exclusion() -> None:
    seen: list[ParsedQuery] = []
    catalog = [_movie(10, "Top Gun: Maverick", 2022), _movie(11, "John Wick", 2014)]
    retriever = _recording_retriever(catalog, seen)

    # Canned, history-aware rewrite: turn 2 resolves "that" to the film the bot
    # recommended in turn 1 (found in the history window) and keeps the "without
    # <actor>" clause. Proves the rewriter USES history rather than echoing.
    def rewrite(raw_query: str, history) -> str:
        joined = " ".join(history)
        if "that" in raw_query.lower() and "Top Gun: Maverick" in joined:
            return "action movie similar to Top Gun: Maverick without Tom Cruise"
        return raw_query

    def extract(query: str) -> ParsedQuery:
        parsed = ParsedQuery(genres=["Action"])
        if "similar to top gun: maverick" in query.lower():
            parsed = parsed.model_copy(update={"similar_to": "Top Gun: Maverick"})
        if "without tom cruise" in query.lower():
            parsed = parsed.model_copy(update={"exclude_actors": ["Tom Cruise"]})
        return parsed

    models = _pick_first_candidate_models(rewrite_fn=rewrite, extract_fn=extract)
    conv = ConversationState(session_id="sess-3")

    # Turn 1: bot recommends Top Gun: Maverick (the first catalog candidate).
    turn1 = run_session_turn(
        conv,
        "recommend an action movie",
        retriever=retriever,
        models=models,
        trace_id="t1",
    )
    assert turn1.state.response.picks[0].title == "Top Gun: Maverick"

    # Turn 2: "something like that but without Tom Cruise".
    turn2 = run_session_turn(
        conv,
        "something like that but without Tom Cruise",
        retriever=retriever,
        models=models,
        trace_id="t2",
    )

    # The pronoun "that" was resolved to the turn-1 film...
    assert turn2.state.rewritten_query is not None
    assert "Top Gun: Maverick" in turn2.state.rewritten_query
    # ...and the exclusion was parsed and applied to retrieval.
    assert turn2.state.parsed is not None
    assert turn2.state.parsed.exclude_actors == ["Tom Cruise"]
    assert "Tom Cruise" in seen[-1].exclude_actors
    # The raw query (not the rewrite) is what history keeps (PRD section 6).
    assert conv.turns[-1].raw_query == "something like that but without Tom Cruise"


# -- short-term window --------------------------------------------------------


def test_short_term_window_keeps_six_verbatim_and_summarizes_older() -> None:
    seen: list[ParsedQuery] = []
    retriever = _recording_retriever([_movie(1, "A", 2020)], seen)
    models = _pick_first_candidate_models()
    conv = ConversationState(session_id="sess-4")

    for i in range(8):
        run_session_turn(
            conv,
            f"query number {i}",
            retriever=retriever,
            models=models,
            trace_id=f"t{i}",
        )

    window = conv.window()
    assert len(window) <= WINDOW_SIZE == 6
    # The verbatim window is the six MOST RECENT turns.
    assert [turn.raw_query for turn in window] == [f"query number {i}" for i in range(2, 8)]
    # Older turns (0 and 1) fell out of the window and were summarized instead.
    assert conv.summary != ""
    assert "query number 0" in conv.summary
    assert "query number 1" in conv.summary

    # The rewriter's history reflects both tiers: the summary line + the window.
    lines = conv.history_lines()
    assert lines[0].startswith("Summary of earlier turns:")
    assert len(lines) == 1 + len(window)


def test_history_line_records_raw_query_and_recommendation() -> None:
    turn = Turn(raw_query="an action movie", recommended=["John Wick (2014)"])
    line = turn.as_history_line()
    assert "an action movie" in line
    assert "John Wick (2014)" in line


# -- standing constraints are earned (ticket #85) -----------------------------


def _region_aware_retriever(catalog: list[ScoredMovie], seen: list[ParsedQuery]):
    """Stub retriever where region='korean' is unsatisfiable (empties the pool)."""

    def retrieve(query: str, parsed: ParsedQuery, shown_movies=()) -> list[ScoredMovie]:
        seen.append(parsed)
        return [] if parsed.region == "korean" else list(catalog)

    return retrieve


def _nolan_after_korean_extract(query: str) -> ParsedQuery:
    if "nolan" in query.lower():
        return ParsedQuery(director="Christopher Nolan")
    return ParsedQuery(region="korean", genres=["Thriller"])


def test_turn_that_returned_nothing_cannot_constrain_later_turns() -> None:
    """Attacks: "every turn's parse is worth remembering as a standing constraint".

    Turn 1 carries an unsatisfiable constraint and yields nothing; turn 2 is an
    unrelated query. Nothing from turn 1 may reach turn 2's merged parse.
    """
    seen: list[ParsedQuery] = []
    catalog = [_movie(1, "Inception", 2010), _movie(2, "Interstellar", 2014)]
    retriever = _region_aware_retriever(catalog, seen)
    models = _pick_first_candidate_models(extract_fn=_nolan_after_korean_extract)
    conv = ConversationState(session_id="sess-85")

    first = run_session_turn(
        conv, "a gritty korean revenge thriller", retriever=retriever, models=models, trace_id="t1"
    )
    assert first.state.response.picks == []
    assert conv.standing == ParsedQuery()  # a turn that returned nothing earned nothing

    second = run_session_turn(
        conv, "recommend some christopher nolan movies", retriever=retriever, models=models, trace_id="t2"
    )
    assert seen[-1].region is None and seen[-1].genres == []
    assert seen[-1].director == "Christopher Nolan"
    assert second.state.response.picks
    assert "relax_standing" not in second.state.path_taken  # nothing to relax


def test_standing_that_empties_the_pool_is_relaxed_once_and_recorded() -> None:
    """Attacks: "an inherited standing constraint is applied unconditionally".

    Even if a bad constraint got into the standing set, a turn it would starve
    retries once without it, records ``relax_standing`` in the path, and the
    session drops the offending standing set afterwards.
    """
    seen: list[ParsedQuery] = []
    catalog = [_movie(1, "Inception", 2010)]
    retriever = _region_aware_retriever(catalog, seen)
    models = _pick_first_candidate_models(extract_fn=_nolan_after_korean_extract)
    conv = ConversationState(session_id="sess-85b")
    conv.standing = ParsedQuery(region="korean")  # poisoned by hand

    result = run_session_turn(
        conv, "recommend some christopher nolan movies", retriever=retriever, models=models, trace_id="t1"
    )

    assert result.state.response.picks
    path = result.state.path_taken
    assert path[path.index("retrieve") : path.index("retrieve") + 3] == [
        "retrieve",
        "relax_standing",
        "retrieve",
    ]
    assert [p.region for p in seen] == ["korean", None]  # merged first, then own parse only
    assert conv.standing.region is None and conv.standing.director == "Christopher Nolan"


def test_deliberate_refinement_across_successful_turns_still_accumulates() -> None:
    """Attacks: "earning standing constraints breaks the #54 refinement loop".

    Two successful turns: the second (years only) must still inherit the first
    turn's genre, with no relaxation.
    """
    seen: list[ParsedQuery] = []
    catalog = [_movie(1, "Alien", 1979), _movie(2, "Aliens", 1986)]
    retriever = _region_aware_retriever(catalog, seen)

    def extract(query: str) -> ParsedQuery:
        if "1980s" in query:
            return ParsedQuery(min_year=1980, max_year=1989)
        return ParsedQuery(genres=["Sci-Fi"])

    models = _pick_first_candidate_models(extract_fn=extract)
    conv = ConversationState(session_id="sess-85c")
    run_session_turn(conv, "good sci-fi movies", retriever=retriever, models=models, trace_id="t1")
    second = run_session_turn(
        conv, "only the ones from the 1980s", retriever=retriever, models=models, trace_id="t2"
    )

    assert seen[-1].genres == ["Sci-Fi"] and (seen[-1].min_year, seen[-1].max_year) == (1980, 1989)
    assert "relax_standing" not in second.state.path_taken
    assert conv.standing.genres == ["Sci-Fi"] and conv.standing.min_year == 1980


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
