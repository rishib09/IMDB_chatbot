"""Tests for the multi-turn replay suite (ticket #24).

Deterministic and network-free: every script replays through the real single-turn
graph with FAKE models and a stub retriever over a fixed catalog. The scripts
assert the PRD's cross-turn memory invariants (section 6 / section 7.5); a
deliberately-broken script proves the suite actually catches a violation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from imdb_chatbot.eval.replay import (
    CATALOG,
    Invariant,
    _check,
    _TurnContext,
    build_models,
    load_script,
    load_scripts,
    run_script,
    stub_retriever,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "replay"
_BROKEN = Path(__file__).parent / "fixtures" / "replay_broken"


def _film(title: str, year: int):
    return next(f for f in CATALOG if f.title == title and f.year == year)


# -- the authored suite all passes --------------------------------------------


def test_fixture_directory_has_enough_scripts() -> None:
    scripts = load_scripts(_FIXTURES)
    assert len(scripts) >= 8, "expected ~8-10 authored replay scripts"


@pytest.mark.parametrize("path", sorted(_FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_replay_script_passes(path: Path, tmp_path: Path) -> None:
    script = load_script(path)
    outcome = run_script(script, workdir=tmp_path)
    # Every turn ran at least one invariant, and all invariants held.
    assert sum(len(t.invariants) for t in outcome.turns) > 0
    outcome.assert_ok()


def test_every_invariant_kind_is_exercised_somewhere() -> None:
    kinds: set[str] = set()
    for script in load_scripts(_FIXTURES):
        for turn in script.turns:
            for inv in turn.invariants:
                kinds.add(inv.kind)
    # The four headline invariants from the ticket are all covered.
    assert {
        "constraint_holds",
        "no_repeat",
        "resolves_reference",
        "precedence_complies",
    } <= kinds


# -- the suite has teeth: a broken script is DETECTED as failing ---------------


def test_deliberately_broken_script_is_detected(tmp_path: Path) -> None:
    script = load_script(_BROKEN / "broken_dropped_constraint.json")
    outcome = run_script(script, workdir=tmp_path)

    assert outcome.ok is False
    failures = outcome.failures()
    assert failures, "the broken script should report at least one failed invariant"
    # It fails on turn 2's constraint_holds: the re-requested exclusion was suspended.
    assert any("turn 2" in f and "constraint_holds" in f for f in failures)

    with pytest.raises(AssertionError):
        outcome.assert_ok()


# -- direct invariant checks (teeth at the checker level) ----------------------


def test_no_repeat_checker_flags_a_repeat() -> None:
    john_wick = _film("John Wick", 2014)
    ctx = _TurnContext(
        state=None,  # type: ignore[arg-type]  # no_repeat does not read state
        recommended=[john_wick],
        shown_before={john_wick.tmdb_id},
        fell_to_fallback=False,
    )
    outcome = _check(Invariant(kind="no_repeat"), ctx)
    assert outcome.ok is False
    assert "John Wick" in outcome.detail


def test_no_repeat_checker_passes_on_fresh_pick() -> None:
    fresh = _film("Mad Max: Fury Road", 2015)
    ctx = _TurnContext(
        state=None,  # type: ignore[arg-type]
        recommended=[fresh],
        shown_before={1},
        fell_to_fallback=False,
    )
    assert _check(Invariant(kind="no_repeat"), ctx).ok is True


# -- specific behavioural assertions on a couple of scripts --------------------


def test_reference_resolution_rewrites_that_to_prior_query(tmp_path: Path) -> None:
    outcome = run_script(load_script(_FIXTURES / "04_reference_resolution_turn2.json"), workdir=tmp_path)
    outcome.assert_ok()
    # Turn 2's "that" was resolved into a standalone query mentioning "action".
    assert "action" in (outcome.turns[1].rewritten_query or "").lower()
    # Keanu Reeves' film (John Wick) was excluded, so a different action film shows.
    assert ("John Wick", 2014) not in outcome.turns[1].picks


def test_precedence_suspends_then_reapplies(tmp_path: Path) -> None:
    outcome = run_script(load_script(_FIXTURES / "06_precedence_horror.json"), workdir=tmp_path)
    outcome.assert_ok()
    # Turn 1 complied with the live horror request (a horror film was recommended)...
    assert any(_film(t, y).genres == ("Horror",) for (t, y) in outcome.turns[0].picks)
    # ...turn 2, which did not re-request horror, got a comedy and no horror.
    assert all("Horror" not in _film(t, y).genres for (t, y) in outcome.turns[1].picks)


# -- world sanity + no-network guard ------------------------------------------


def test_stub_retriever_honours_exclusions() -> None:
    from imdb_chatbot.schemas import ParsedQuery

    got = stub_retriever("", ParsedQuery(genres=["Action"], exclude_actors=["Keanu Reeves"]))
    titles = {m.title for m in got}
    assert "John Wick" not in titles  # Keanu Reeves excluded
    assert "Mad Max: Fury Road" in titles


def test_models_are_plain_callables_no_network() -> None:
    models = build_models()
    # Pure Python fakes: calling them never touches a client.
    assert models.rewrite("hello", []) == "hello"
    parsed = models.extract("a horror movie without Keanu Reeves")
    assert parsed.genres == ["Horror"]
    assert parsed.exclude_actors == ["Keanu Reeves"]
