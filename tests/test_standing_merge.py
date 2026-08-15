"""Adversary tests for the standing-constraint merge (ticket #72).

Invariant under attack: *standing constraints survive a merge for every field*.
The old merges hand-listed field names, so a new ``ParsedQuery`` field was
silently dropped. These tests never name a field themselves - they introspect
``ParsedQuery.model_fields`` and add a field the merge code has never seen.
"""

from __future__ import annotations

import types
import typing

from imdb_chatbot.graph.build import _session_merged_parsed
from imdb_chatbot.memory.durable import resolve_collision
from imdb_chatbot.memory.session import ConversationState
from imdb_chatbot.schemas import EXCLUSION_FIELDS, ParsedQuery, TurnState


def _sample(annotation: object) -> object:
    """A non-default, non-falsy value for a ``ParsedQuery`` field annotation."""
    if isinstance(annotation, types.UnionType):
        annotation = next(a for a in typing.get_args(annotation) if a is not type(None))
    origin = typing.get_origin(annotation) or annotation
    return {list: ["x"], int: 1999, float: 7.5, str: "x"}[origin]


def _fully_specified() -> ParsedQuery:
    return ParsedQuery(**{n: _sample(f.annotation) for n, f in ParsedQuery.model_fields.items()})


def _state(parsed: ParsedQuery | None, standing: ParsedQuery) -> TurnState:
    return TurnState(
        trace_id="t",
        session_id="s",
        raw_query="q",
        parsed=parsed,
        session_standing=standing.model_dump(),
    )


def test_every_positive_field_survives_a_merge_at_both_sites() -> None:
    """Attacks: "the merge names every field" - iterates model_fields instead of trusting it."""
    turn1 = _fully_specified()
    conv = ConversationState(session_id="s")
    conv.merge_standing(turn1)
    conv.merge_standing(ParsedQuery())  # turn 2 specifies nothing
    merged = _session_merged_parsed(_state(ParsedQuery(), conv.standing))
    for name in ParsedQuery.model_fields:
        expected = [] if name in EXCLUSION_FIELDS else getattr(turn1, name)
        assert getattr(conv.standing, name) == expected, name
        assert getattr(merged, name) == expected, name


def test_a_field_added_after_the_merge_was_written_still_survives() -> None:
    """Attacks: field drift - a subclass adds a field the merge code has never seen."""

    class Extended(ParsedQuery):
        mood: str | None = None
        runtime_max: int | None = None

    turn1 = Extended(mood="tense", runtime_max=0)  # 0 is specified, not "unset"
    merged = Extended().merge_over(turn1)  # turn 2 says nothing -> everything inherits
    assert (merged.mood, merged.runtime_max) == ("tense", 0)
    assert Extended(mood="calm").merge_over(turn1).mood == "calm"  # current wins


def test_zero_is_specified_and_empty_is_not() -> None:
    """Attacks: falsy-vs-unset conflation (``min_year=0`` must override, ``genres=[]`` must not)."""
    standing = ParsedQuery(min_year=1990, genres=["Drama"])
    merged = ParsedQuery(min_year=0, genres=[]).merge_over(standing)
    assert (merged.min_year, merged.genres) == (0, ["Drama"])


def test_exclusions_never_enter_standing_but_union_with_session_set() -> None:
    """Attacks: the exclusion classification - every ``exclude_*`` field is one, and only those."""
    assert EXCLUSION_FIELDS == {n for n in ParsedQuery.model_fields if n.startswith("exclude_")}
    for f in EXCLUSION_FIELDS:  # every exclusion has its session mirror on TurnState
        assert f"session_{f}" in TurnState.model_fields, f
    standing = _fully_specified()
    state = _state(ParsedQuery(exclude_genres=["Horror"]), standing).model_copy(
        update={"session_exclude_actors": ["Cage"]}
    )
    merged = _session_merged_parsed(state)
    assert merged.exclude_genres == ["Horror"]  # standing's ["x"] must NOT leak in
    assert merged.exclude_actors == ["Cage"]


def test_resolve_collision_drops_blank_exclusions_on_both_sides() -> None:
    """Attacks: asymmetric blank handling (blanks were filtered from ``requested`` only)."""
    assert resolve_collision(["", "  ", "Horror"], ["", "horror"]) == []
    assert resolve_collision(["Horror", " "], []) == ["Horror"]
