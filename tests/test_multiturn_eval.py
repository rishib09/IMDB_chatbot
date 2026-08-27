"""The multi-turn eval runner (ticket #106): live baseline + teeth.

Two tiers, both aimed at the same worry - a runner that reports green without
measuring anything is worse than no runner.

- The LIVE test is the deliverable: the nine golden scripts in
  ``eval/multiturn/`` replayed through the production handler (intent gate,
  follow-up router, session memory, real extractor, real index, Gate-4, real
  generator). It prints the pass/fail table and the run's cost, and asserts that
  the runner OBSERVED each turn - not that the system under measurement passed
  (which is the baseline's job to record, not the test's job to demand).

      npx @dotenvx/dotenvx run -f .env -- pytest -m live tests/test_multiturn_eval.py -q -s

- The adversary tests run in the deterministic, network-free replay world and
  attack the checkers themselves: a script whose expectations are all wrong must
  come back with all of them flagged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imdb_chatbot.eval.multiturn import (
    format_multiturn_report,
    format_script_row,
    run_live_scripts,
    to_dict,
)
from imdb_chatbot.eval.replay import (
    Invariant,
    _check,
    _TurnContext,
    load_script,
    load_scripts,
    run_script,
)

MULTITURN = Path(__file__).resolve().parents[1] / "eval" / "multiturn"
_BROKEN = Path(__file__).parent / "fixtures" / "replay_broken"

# The five kinds ticket #106 implements; the broken fixture exercises every one.
_B3B_KINDS = {"routes_to", "no_retrieval", "parsed", "omits_genre", "answers_about_pick"}


# -- teeth: a script whose expectations are all wrong fails on all of them ------


def test_a_script_that_should_fail_does_fail() -> None:
    """Attacks: 'the runner would notice a violated expectation'.

    Mirror of the #24 broken-script test, aimed at the five kinds #106 adds. Each
    invariant in the fixture contradicts what the deterministic world actually
    does, so a green result here means the checkers are decorative.
    """
    outcome = run_script(load_script(_BROKEN / "broken_live_kinds.json"))

    assert outcome.ok is False
    flagged = {inv.kind for turn in outcome.turns for inv in turn.invariants if not inv.ok}
    assert flagged == _B3B_KINDS, f"kinds that failed to notice: {_B3B_KINDS - flagged}"
    for failure in outcome.failures():
        assert "broken_live_kinds turn 1" in failure  # names script + turn (AC 3)


def test_answering_about_the_wrong_thing_is_not_an_answer() -> None:
    """Attacks: 'naming the film is enough to have answered the question'.

    The tempting shortcut for ``answers_about_pick`` is to check that the prose
    mentions the film. "Parasite is a great thriller!" mentions it and answers
    nothing; the requested FIELD has to be in there too.
    """

    class _Movie:
        tmdb_id, title, year = 496243, "Parasite", 2019
        genres, cast = ["Thriller"], []
        director = "Bong Joon-ho"

    def _ctx(prose: str) -> _TurnContext:
        return _TurnContext(
            parsed=None,
            rewritten_query=None,
            recommended=[],
            shown_before=set(),
            fell_to_fallback=False,
            prose=prose,
            previous_recommended=[_Movie()],
        )

    inv = Invariant(kind="answers_about_pick", n=1, field="director")
    assert _check(inv, _ctx("Parasite is a great thriller!")).ok is False
    assert _check(inv, _ctx("Bong Joon-ho is a wonderful filmmaker.")).ok is False
    # Punctuation-folded, so the corpus's "Bong Joon-ho" matches prose spelling it
    # "Bong Joon Ho" - the check must not turn into a string-formatting lottery.
    assert _check(inv, _ctx("Parasite (2019) was directed by Bong Joon Ho.")).ok is True


def test_a_satisfied_contains_does_not_excuse_a_wrong_equals() -> None:
    """Attacks: 'the parsed checker reads the constraint it was given'.

    ``parsed`` takes either ``equals`` (scalars) or ``contains`` (lists). An
    invariant carrying neither asserts nothing, and must be reported as broken
    rather than quietly counted as a pass.
    """
    from imdb_chatbot.schemas import ParsedQuery

    ctx = _TurnContext(
        parsed=ParsedQuery(genres=["Thriller"], region="KR"),
        rewritten_query=None,
        recommended=[],
        shown_before=set(),
        fell_to_fallback=False,
    )
    assert _check(Invariant(kind="parsed", field="region", equals="FR"), ctx).ok is False
    assert _check(Invariant(kind="parsed", field="genres", contains=["Comedy"]), ctx).ok is False
    assert _check(Invariant(kind="parsed", field="regino", equals="KR"), ctx).ok is False
    assert _check(Invariant(kind="parsed", field="region"), ctx).ok is False
    assert _check(Invariant(kind="parsed", field="region", equals="KR"), ctx).ok is True


# -- the live baseline: the deliverable ---------------------------------------


@pytest.mark.live
def test_golden_multiturn_baseline(live_resources) -> None:
    """Replay all nine golden scripts through the live handler and report.

    Asserts that the runner MEASURED: every declared expectation got a verdict
    with a reason, at least one search turn came back with real picks (so the
    instrumentation really saw inside the graph), and real money was spent (so
    this was the live tier and not a silently degraded one). Which scripts pass
    is the baseline the report records, not a condition of the test.
    """
    scripts = load_scripts(MULTITURN)
    # ~70 real model calls: stream each verdict so a long run is watchable and a
    # killed run still leaves the scripts it did finish on the record.
    outcomes = run_live_scripts(
        scripts, live_resources, on_script=lambda o: print(format_script_row(o), flush=True)
    )
    report = format_multiturn_report(outcomes)
    print("\n" + report)

    assert len(outcomes) == len(scripts)
    for outcome, script in zip(outcomes, scripts, strict=True):
        declared = [len(turn.invariants) for turn in script.turns]
        evaluated = [len(turn.invariants) for turn in outcome.turns]
        assert evaluated == declared, f"{script.name}: {evaluated} verdicts for {declared}\n{report}"
        assert all(inv.detail for t in outcome.turns for inv in t.invariants), report

    assert any(t.route == "SEARCH" and t.picks for o in outcomes for t in o.turns), report
    assert sum(o.cost_usd for o in outcomes) > 0, report
    # The baseline is an artifact a human reads later, so it has to serialize.
    assert json.loads(json.dumps(to_dict(outcomes)))["scripts"], report
