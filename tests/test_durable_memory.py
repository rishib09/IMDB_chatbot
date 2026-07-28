"""Durable cross-session memory tests (ticket #22, PRD section 6, rungs 2-3).

Every scenario uses ``LocalJsonlStore`` on a tmp dir - fully deterministic, NO
network and no HF calls. The invariants are the durable-memory guarantees:

- a widget click writes the right ``Triple`` and it survives a "new session"
  (reload from the store),
- ``known_preferences`` injects durable EXCLUDED into the turn's effective
  filters and puts WATCHED movies in ``shown_movies`` (no-repeat),
- minimal precedence: a live "good horror movies" request suspends a durable
  EXCLUDED(horror) FOR THIS TURN while leaving the durable triple unchanged,
- retention caps a user's log at ~200 triples (oldest dropped),
- ``HFDatasetStore`` raises a clear error without a token (asserted WITHOUT any
  network access).
"""

from __future__ import annotations

import pytest

from imdb_chatbot.memory import (
    MAX_TRIPLES_PER_USER,
    ConversationState,
    HFDatasetStore,
    LocalJsonlStore,
    Triple,
    known_preferences,
    resolve_collision,
    widget_to_triple,
)
from imdb_chatbot.schemas import MovieRecord


def _movie(tmdb_id: int, title: str, genres: list[str] | None = None) -> MovieRecord:
    return MovieRecord(tmdb_id=tmdb_id, title=title, year=2020, genres=genres or [])


# -- widget click writes a durable triple that survives a new session ----------


def test_widget_click_writes_durable_triple_and_survives_new_session(tmp_path) -> None:
    store = LocalJsonlStore(tmp_path)
    movie = _movie(603, "The Matrix", ["Action"])

    triple = widget_to_triple("Rishi", "thumbs_up", movie)
    store.append(triple.user_id, triple)

    assert triple.relation == "LIKED"
    assert triple.object == "The Matrix"
    assert triple.provenance == "widget:thumbs_up"
    # Name-field identity: user_id is lowercased.
    assert triple.user_id == "rishi"

    # A brand-new store instance (simulating a fresh session / process) still
    # sees the fact - it was persisted, not just held in memory.
    reloaded = LocalJsonlStore(tmp_path)
    facts = reloaded.load("rishi")
    assert len(facts) == 1
    assert facts[0].relation == "LIKED"
    assert facts[0].object == "The Matrix"
    assert "rishi" in reloaded.all_users()


def test_each_widget_kind_maps_to_its_relation(tmp_path) -> None:
    movie = _movie(11, "Star Wars")
    assert widget_to_triple("rishi", "thumbs_up", movie).relation == "LIKED"
    assert widget_to_triple("rishi", "thumbs_down", movie).relation == "DISLIKED"
    assert widget_to_triple("rishi", "seen_it", movie).relation == "WATCHED"
    assert widget_to_triple("rishi", "not_interested", movie).relation == "REJECTED"
    # WATCHED / REJECTED record the movie id (a no-repeat fact).
    assert widget_to_triple("rishi", "seen_it", movie).object == "11"
    assert widget_to_triple("rishi", "not_interested", movie).object == "11"


def test_unknown_widget_kind_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        widget_to_triple("rishi", "shrug", _movie(1, "X"))


# -- known_preferences injects durable EXCLUDED + WATCHED (rung 2) --------------


def test_known_preferences_injects_excluded_filter_and_watched_shown(tmp_path) -> None:
    store = LocalJsonlStore(tmp_path)
    store.append("rishi", Triple(user_id="rishi", relation="EXCLUDED", object="Horror"))
    # A "Seen it" click on movie 42 -> WATCHED, must land in shown_movies.
    store.append("rishi", widget_to_triple("rishi", "seen_it", _movie(42, "Jaws")))
    store.append("rishi", widget_to_triple("rishi", "thumbs_up", _movie(7, "Heat")))

    prefs = known_preferences("Rishi", store)

    assert "Horror" in prefs.exclude_genres
    assert 42 in prefs.shown_movies
    assert "Heat" in prefs.liked

    # Folded into a fresh session's constraint state (start of a new session).
    conv = ConversationState(session_id="new-sess")
    prefs.apply_to(conv)
    assert "Horror" in conv.exclude_genres
    assert 42 in conv.shown_movies
    assert conv.preferences.get("liked") == ["Heat"]


# -- minimal precedence: live request outranks durable EXCLUDED (rung 3) --------


def test_minimal_precedence_live_request_suspends_durable_exclusion(tmp_path) -> None:
    store = LocalJsonlStore(tmp_path)
    store.append("rishi", Triple(user_id="rishi", relation="EXCLUDED", object="Horror"))

    prefs = known_preferences("rishi", store)
    assert prefs.exclude_genres == ["Horror"]

    # The live parsed query asks FOR horror ("good horror movies").
    requested_genres = ["Horror"]
    effective = resolve_collision(prefs.exclude_genres, requested_genres)

    # The exclusion is DROPPED for this turn - the bot complies with the live ask.
    assert "Horror" not in effective
    assert effective == []

    # ...but the durable triple is UNCHANGED afterwards (we did not un-remember
    # the standing dislike). A reload still shows EXCLUDED(Horror).
    reloaded = LocalJsonlStore(tmp_path).load("rishi")
    assert len(reloaded) == 1
    assert reloaded[0].relation == "EXCLUDED"
    assert reloaded[0].object == "Horror"


def test_resolve_collision_is_case_insensitive_and_keeps_others(tmp_path) -> None:
    effective = resolve_collision(["Horror", "Romance"], ["horror"])
    # horror suspended (case-insensitive match), romance still excluded.
    assert effective == ["Romance"]


# -- retention cap ~200 (oldest dropped) ---------------------------------------


def test_retention_cap_drops_oldest(tmp_path) -> None:
    store = LocalJsonlStore(tmp_path)
    total = MAX_TRIPLES_PER_USER + 25
    for i in range(total):
        store.append(
            "rishi",
            Triple(user_id="rishi", relation="WATCHED", object=str(i), ts=float(i)),
        )

    facts = store.load("rishi")
    assert len(facts) == MAX_TRIPLES_PER_USER
    # The OLDEST (0..24) were dropped; the newest survive, newest last.
    objects = [t.object for t in facts]
    assert objects[0] == str(total - MAX_TRIPLES_PER_USER)
    assert objects[-1] == str(total - 1)


# -- HFDatasetStore refuses to run without a token (no network) -----------------


def test_hf_store_raises_clear_error_without_token() -> None:
    store = HFDatasetStore(repo_id="someone/durable-memory", token=None)
    with pytest.raises(RuntimeError, match="requires a Hugging Face token"):
        store.load("rishi")
    with pytest.raises(RuntimeError, match="requires a Hugging Face token"):
        store.append("rishi", Triple(user_id="rishi", relation="LIKED", object="Heat"))
    with pytest.raises(RuntimeError, match="requires a Hugging Face token"):
        store.all_users()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
