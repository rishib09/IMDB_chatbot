"""Follow-up router (ticket #54) + standing-constraint accumulation."""

from __future__ import annotations

from imdb_chatbot.memory.session import ConversationState
from imdb_chatbot.router import FollowupKind, clarify_question, route_followup
from imdb_chatbot.schemas import ParsedQuery


def _fake(label: str):
    return lambda query, titles: label


def test_bare_negation_short_circuits_to_clarify() -> None:
    # Even if the classifier says otherwise, a whole-message rejection -> clarify.
    assert route_followup("no", ["A"], _fake("refine")) is FollowupKind.CLARIFY
    assert route_followup("not these", ["A"], _fake("replace")) is FollowupKind.CLARIFY
    assert route_followup("something else", ["A"], _fake("refine")) is FollowupKind.CLARIFY


def test_classifier_labels_map_to_kinds() -> None:
    assert route_followup("make it korean", ["A"], _fake("refine")) is FollowupKind.REFINE
    assert route_followup("now something else entirely", ["A"], _fake("replace")) is FollowupKind.REPLACE
    assert route_followup("hmm", ["A"], _fake("clarify")) is FollowupKind.CLARIFY


def test_unknown_classification_defaults_to_refine() -> None:
    assert route_followup("adjust that", ["A"], _fake("???")) is FollowupKind.REFINE


def test_clarify_question_mentions_the_filters() -> None:
    q = clarify_question().lower()
    assert "region" in q and "genre" in q


def test_merge_standing_accumulates_positive_constraints() -> None:
    c = ConversationState(session_id="s")
    c.merge_standing(ParsedQuery(region="KR", genres=["Thriller"]))
    c.merge_standing(ParsedQuery(genres=["Comedy"]))  # override genre, keep region
    assert c.standing.region == "KR"
    assert c.standing.genres == ["Comedy"]


def test_merge_standing_does_not_accumulate_exclusions() -> None:
    # Exclusions are owned by the session_exclude_* + precedence mechanism.
    c = ConversationState(session_id="s")
    c.merge_standing(ParsedQuery(exclude_genres=["Comedy"], exclude_actors=["X"]))
    assert c.standing.exclude_genres == [] and c.standing.exclude_actors == []


def test_reset_standing_clears_everything() -> None:
    c = ConversationState(session_id="s")
    c.merge_standing(ParsedQuery(region="KR", director="Nolan"))
    c.reset_standing()
    assert c.standing == ParsedQuery()
