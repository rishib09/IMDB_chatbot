"""Live multi-turn eval runner: the golden scripts through the real system (#106).

``eval/harness.py`` measures ONE turn: it calls ``retriever.retrieve`` once, with
no session, no router and no memory. Every promise the product makes ACROSS turns
- no-repeat, reference resolution, standing-constraint replacement vs
accumulation, exclusion carry-over and suspension, dead-end relaxation - is
therefore unmeasured by it. The nine golden scripts in ``eval/multiturn/*.json``
(B3 / #68) declare those promises as data; this module runs them.

The runner drives the PRODUCTION handler - ``dashboard.live.build_live_chat_handler``
- not a re-implementation of it, so what the scripts measure is what a user gets:
the deterministic intent gate, the follow-up router, ``run_session_turn`` with a
real ``ConversationState``, the real extractor, the real hybrid retriever over the
real index, Gate-4, and the real generator.

That handler returns only a ``ChatReply``, so the runner sees inside the turn by
INSTRUMENTING the one seam production already exposes: the injected retriever. Its
call log yields, per turn, the effective (rewritten) query, the session-merged
``ParsedQuery`` retrieval actually ran with (after any relaxation), the candidate
pool the picks join against, and - crucially for ``no_retrieval`` - whether it was
called at all. Nothing about the handler changes; the wrapper only watches.

The one thing the wrapper cannot see is which of the three production paths served
the turn, because a persona / movie-question reply never reaches the retriever. So
the route is re-derived by calling the same deterministic ``classify_intent`` the
handler calls. It is a pure function of the message, so this cannot disagree with
the handler for the same input - but it does mean a NEW route added to the handler
must be mirrored in ``_route_of`` below.

Run it (real model, real index, real spend)::

    npx @dotenvx/dotenvx run -f .env -- python -m imdb_chatbot.eval.multiturn \\
        --db data/corpus.sqlite --index-dir data/index/<version>

The table it prints is the deliverable: which scripts pass today, which turn and
which expectation each failure violated, and what the run cost.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT
from ..persona import Intent, classify_intent
from ..schemas import ParsedQuery, ScoredMovie, index_by_title_year
from .replay import (
    ReplayScript,
    ScriptOutcome,
    TurnOutcome,
    _check,
    _Film,
    _TurnContext,
    load_scripts,
)

DEFAULT_SCRIPTS = PROJECT_ROOT / "eval" / "multiturn"

# persona ``Intent`` -> the route name the scripts use. GREETING / META / CHITCHAT
# are one route from the script's point of view: a conversational reply that costs
# no retrieval.
_ROUTE_BY_INTENT = {
    Intent.GREETING: "CHITCHAT",
    Intent.META: "CHITCHAT",
    Intent.CHITCHAT: "CHITCHAT",
    Intent.MOVIE_QUESTION: "MOVIE_INFO",
    Intent.SEARCH: "SEARCH",
}


def _route_of(query: str) -> str:
    """Which production path serves ``query`` (mirrors the handler's intent gate)."""
    return _ROUTE_BY_INTENT[classify_intent(query)]


@dataclass
class _Probe:
    """The retriever call log for ONE turn - the runner's window into the graph.

    ``parsed`` and ``query`` hold the LAST call's arguments because that is the
    parse the turn ended up retrieving with: ``graph.build._make_retrieve`` walks
    a relaxation ladder, re-calling the retriever with a weaker parse each rung,
    and the final rung is what actually constrained the results.
    """

    calls: int = 0
    query: str | None = None
    parsed: ParsedQuery | None = None
    candidates: list[ScoredMovie] = field(default_factory=list)

    def reset(self) -> None:
        self.calls, self.query, self.parsed = 0, None, None
        self.candidates = []


def _instrument(retriever, probe: _Probe):
    """Wrap ``retriever`` so every call is recorded, changing nothing about it."""

    def watched(query: str, parsed: ParsedQuery, shown_movies: Sequence[int] = ()):
        probe.calls += 1
        probe.query, probe.parsed = query, parsed
        out = list(retriever(query, parsed, shown_movies))
        probe.candidates = out
        return out

    return watched


def _recommended(picks: Sequence[Any], probe: _Probe, store: Any) -> list[Any]:
    """Resolve this turn's picks to corpus ``MovieRecord``s, in pick order.

    A pick carries no id, so it is joined onto the turn's candidate pool by
    ``(title, year)`` - the same join production uses for poster backfill and
    shown-movie bookkeeping. A pick that will not join (Gate-4 should make this
    impossible) still occupies its slot as a bare ``_Film``, so ``min_picks``
    counts what the user saw rather than what the runner could look up.
    """
    by_title_year = index_by_title_year(probe.candidates)
    out: list[Any] = []
    for pick in picks:
        tmdb_id = by_title_year.get((pick.title, pick.year))
        record = store.read_movie(tmdb_id) if tmdb_id is not None and store is not None else None
        out.append(record if record is not None else _Film(-1, pick.title, pick.year, (), ()))
    return out


def run_live_script(script: ReplayScript, resources) -> ScriptOutcome:
    """Replay one golden script through the live handler and check its invariants.

    A fresh handler (so a fresh ``ConversationState``) per script: no state bleeds
    between scripts. ``shown_before`` accumulates every film recommended earlier in
    THIS script, which is what ``no_repeat`` is asserted against.
    """
    from ..dashboard.live import build_live_chat_handler

    probe = _Probe()
    handler = build_live_chat_handler(
        replace(resources, retriever=_instrument(resources.retriever, probe))
    )
    store = resources.store

    outcome = ScriptOutcome(name=script.name)
    shown: set[int] = set()
    previous: list[Any] = []
    for index, turn in enumerate(script.turns, start=1):
        probe.reset()
        reply = handler(turn.user)
        recommended = _recommended(reply.rec.picks, probe, store)
        telemetry = reply.telemetry
        ctx = _TurnContext(
            parsed=probe.parsed,
            rewritten_query=probe.query,
            recommended=recommended,
            shown_before=set(shown),
            fell_to_fallback="fallback" in (telemetry.path_taken if telemetry else []),
            route=_route_of(turn.user),
            retrieval_calls=probe.calls,
            prose=reply.rec.prose,
            previous_recommended=previous,
        )
        outcome.turns.append(
            TurnOutcome(
                index=index,
                user=turn.user,
                rewritten_query=probe.query,
                picks=[(m.title, m.year) for m in recommended],
                invariants=[_check(inv, ctx) for inv in turn.invariants],
                route=ctx.route,
                cost_usd=telemetry.cost_usd if telemetry else 0.0,
            )
        )
        shown.update(m.tmdb_id for m in recommended if m.tmdb_id > 0)
        previous = recommended
    return outcome


def run_live_scripts(
    scripts: Sequence[ReplayScript],
    resources,
    *,
    on_script: Callable[[ScriptOutcome], None] | None = None,
) -> list[ScriptOutcome]:
    """Run every script on its own session, in order.

    A full run is ~70 real model calls and takes minutes, so ``on_script`` is
    called as each one lands - the CLI uses it to stream the table rather than
    leaving a long run looking hung.
    """
    outcomes: list[ScriptOutcome] = []
    for script in scripts:
        outcome = run_live_script(script, resources)
        outcomes.append(outcome)
        if on_script is not None:
            on_script(outcome)
    return outcomes


# -- reporting ----------------------------------------------------------------


def format_script_row(outcome: ScriptOutcome) -> str:
    """One script's verdict: pass/fail, how much held, cost, and every violation."""
    checks = [inv for turn in outcome.turns for inv in turn.invariants]
    held = sum(1 for inv in checks if inv.ok)
    verdict = "PASS" if outcome.ok else "FAIL"
    lines = [
        (
            f"[{verdict}] {outcome.name:<28} turns={len(outcome.turns)} "
            f"invariants={held}/{len(checks)} cost=${outcome.cost_usd:.4f}"
        )
    ]
    lines.extend(f"        {failure}" for failure in outcome.failures())
    return "\n".join(lines)


def format_multiturn_report(outcomes: Sequence[ScriptOutcome]) -> str:
    """The pass/fail table: one row per script, then every violated expectation."""
    return "\n".join(
        [
            "=== multi-turn scripts ===",
            *(format_script_row(o) for o in outcomes),
            "",
            format_totals(outcomes),
        ]
    )


def format_totals(outcomes: Sequence[ScriptOutcome]) -> str:
    """The one-line bottom of the table: scripts passed, turns run, dollars spent."""
    passed = sum(1 for o in outcomes if o.ok)
    return (
        f"{len(outcomes)} scripts: {passed} passed, {len(outcomes) - passed} failed | "
        f"{sum(len(o.turns) for o in outcomes)} turns | "
        f"cost ${sum(o.cost_usd for o in outcomes):.4f}"
    )


def to_dict(outcomes: Sequence[ScriptOutcome]) -> dict:
    """A JSON-dumpable baseline record of the run (the #106 'baseline' artifact)."""
    return {
        "cost_usd": sum(o.cost_usd for o in outcomes),
        "scripts": [
            {
                "name": o.name,
                "ok": o.ok,
                "cost_usd": o.cost_usd,
                "turns": [
                    {
                        "index": t.index,
                        "user": t.user,
                        "route": t.route,
                        "rewritten_query": t.rewritten_query,
                        "picks": [f"{title} ({year})" for title, year in t.picks],
                        "invariants": [
                            {"kind": i.kind, "ok": i.ok, "detail": i.detail} for i in t.invariants
                        ],
                    }
                    for t in o.turns
                ],
            }
            for o in outcomes
        ],
    }


# -- CLI -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imdb_chatbot.eval.multiturn")
    parser.add_argument("--db", default="data/corpus.sqlite", help="Corpus SQLite store.")
    parser.add_argument(
        "--scripts", default=str(DEFAULT_SCRIPTS), help=f"Script dir (default: {DEFAULT_SCRIPTS})."
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Index version dir (default: whatever config/live_index.json points at).",
    )
    parser.add_argument("--json-out", default=None, help="Write the baseline record here.")
    args = parser.parse_args(argv)

    from .. import config as config_module
    from ..dashboard.live import load_live_resources

    if args.index_dir:
        index_dir = Path(args.index_dir)
        pointer = {"active": index_dir.name, "path": str(index_dir)}
        config_module.load_live_index = lambda: pointer  # type: ignore[assignment]

    scripts = load_scripts(args.scripts)
    resources = load_live_resources(corpus_path=args.db)
    print(f"scripts={args.scripts} n={len(scripts)} db={args.db}", flush=True)
    print("=== multi-turn scripts ===", flush=True)

    def show(outcome: ScriptOutcome) -> None:
        print(format_script_row(outcome), flush=True)

    try:
        outcomes = run_live_scripts(scripts, resources, on_script=show)
    finally:
        if resources.store is not None:
            resources.store.close()

    print("\n" + format_totals(outcomes))
    if args.json_out:
        import json

        Path(args.json_out).write_text(json.dumps(to_dict(outcomes), indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0 if all(o.ok for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
