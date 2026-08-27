"""Adversary tests against the GOLDEN EVAL SET ITSELF (ticket #68).

The label set is the measuring instrument: every retrieval decision is settled by
the numbers it produces. A silent defect in the instrument is therefore worse than
a bug in the code it measures - a typo'd category, a duplicated query_id, an
anchor whose tmdb_id no longer exists after a re-ingest, or a labeled constraint
that excludes its own anchor all shrink or bias the suite while it still reports a
number. Each test below names the assumption it attacks.

The corpus-dependent tests resolve the real corpus via ``live_corpus_path`` and
skip (naming the missing piece) when it is absent, so they run locally and on any
checkout that has the data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from imdb_chatbot.eval.labels import CATEGORIES, load_labels
from imdb_chatbot.eval.replay import _CHECKERS, Invariant, _check, _TurnContext, load_scripts
from imdb_chatbot.store import TraceStore

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "labels.jsonl"
MULTITURN = Path(__file__).resolve().parents[1] / "eval" / "multiturn"

# The ticket's spec: ~50 anchored rows across all 8 cells, ~8-10 multi-turn scripts.
MIN_ROWS = 50
MIN_SCRIPTS = 8


@pytest.fixture(scope="module")
def golden():
    return load_labels(GOLDEN)


# -- the file itself ----------------------------------------------------------


def test_golden_set_covers_every_category(golden) -> None:
    """Attacks: 'the golden set still covers all 8 cells'.

    A dropped or renamed row leaves a coverage hole that the metric table hides -
    it simply stops printing that category instead of failing.
    """
    assert len(golden) >= MIN_ROWS, f"golden set shrank to {len(golden)} rows"
    covered = {row.category for row in golden}
    assert covered == set(CATEGORIES), f"missing cells: {set(CATEGORIES) - covered}"


def test_query_ids_and_queries_are_unique(golden) -> None:
    """Attacks: 'every row is a distinct case'.

    ``load_labels`` validates rows one at a time, so a copy-pasted row keeps its
    neighbour's id: the per-query table then reports one id twice and the suite is
    quietly one case smaller than it looks.
    """
    ids = [row.query_id for row in golden]
    queries = [row.query.strip().lower() for row in golden]
    assert len(set(ids)) == len(ids), f"duplicate query_id(s): {sorted({i for i in ids if ids.count(i) > 1})}"
    assert len(set(queries)) == len(queries), "the same query text is labeled twice"


def test_anchors_match_the_answerability_of_their_category(golden) -> None:
    """Attacks: 'an ood row is unanswerable and every other row is anchored'.

    An ood row that gains an anchor starts contributing to recall (it has no right
    answer); an answerable row that loses its anchors silently drops out of the
    recall denominator instead of scoring zero.
    """
    for row in golden:
        if row.category == "ood_unanswerable":
            assert not row.relevant_tmdb_ids, f"{row.query_id}: ood row carries an anchor"
        else:
            assert row.relevant_tmdb_ids, f"{row.query_id}: answerable row has no anchor"


def test_every_negation_row_actually_carries_an_exclusion(golden) -> None:
    """Attacks: 'the exclusion cell measures exclusions'.

    Exclusion precision is averaged over the rows whose ``parsed`` carries an
    exclusion. A negation row that lost its ``exclude_*`` field is dropped from
    that average silently - the metric stays 1.00 on fewer and fewer queries.
    """
    negation = [r for r in golden if r.category == "negation_exclusion"]
    assert negation, "no negation rows at all"
    for row in negation:
        assert row.parsed.exclude_actors or row.parsed.exclude_genres, (
            f"{row.query_id}: negation row with no exclusion in parsed"
        )


def test_a_typo_in_a_category_is_rejected_not_dropped(tmp_path: Path, golden) -> None:
    """Attacks: 'a mistyped category fails loudly'.

    Mutation test on the REAL file: retype one row's category and the loader must
    refuse the whole set rather than skip the row (which would shrink the suite by
    one case and lose a cell without a word).
    """
    lines = GOLDEN.read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(lines[0])
    row["category"] = "standard_semantics"  # one plausible keystroke away
    lines[0] = json.dumps(row)
    mutated = tmp_path / "mutated.jsonl"
    mutated.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_labels(mutated)


# -- the anchors against the real corpus --------------------------------------


@pytest.fixture(scope="module")
def corpus(live_corpus_path):
    store = TraceStore(live_corpus_path)
    try:
        yield store
    finally:
        store.close()


def test_every_anchor_resolves_to_a_real_corpus_row(golden, corpus) -> None:
    """Attacks: 'every anchor is a real tmdb_id in THIS corpus'.

    An anchor that does not exist can never be retrieved, so its row is pinned at
    Recall 0 forever - and reads as a retrieval failure rather than a bad label.
    Re-ingest is when this breaks.
    """
    missing = [
        (row.query_id, tid)
        for row in golden
        for tid in row.relevant_tmdb_ids
        if corpus.read_movie(tid) is None
    ]
    assert not missing, f"anchors absent from the corpus: {missing}"


def test_no_anchor_is_excluded_by_its_own_labeled_constraints(golden, corpus) -> None:
    """Attacks: 'the labeled constraints and the anchor agree'.

    The retriever applies ``parsed`` as a hard filter. An anchor its own row
    filters out (wrong region, an excluded genre, outside the year window) is
    unreachable by construction: the cell scores 0 and the number blames
    retrieval for a label defect.
    """
    problems: list[str] = []
    for row in golden:
        p = row.parsed
        for tid in row.relevant_tmdb_ids:
            movie = corpus.read_movie(tid)
            if movie is None:
                continue  # covered by the test above
            genres = set(movie.genres)
            cast = set(movie.cast)
            for reason, bad in (
                (f"region {p.region} not in {movie.regions}", p.region and p.region not in movie.regions),
                (f"missing labeled genre(s) {set(p.genres) - genres}", not set(p.genres) <= genres),
                (f"carries excluded genre {genres & set(p.exclude_genres)}", genres & set(p.exclude_genres)),
                (f"carries excluded actor {cast & set(p.exclude_actors)}", cast & set(p.exclude_actors)),
                (f"year {movie.year} < min_year {p.min_year}", p.min_year and movie.year < p.min_year),
                (f"year {movie.year} > max_year {p.max_year}", p.max_year and movie.year > p.max_year),
                (
                    f"rating {movie.rating_raw} < min_rating {p.min_rating}",
                    p.min_rating and (movie.rating_raw or 0.0) < p.min_rating,
                ),
            ):
                if bad:
                    problems.append(f"{row.query_id}/{tid} {movie.title!r}: {reason}")
    assert not problems, "anchors filtered out by their own labels: " + "; ".join(problems)


# -- the multi-turn scripts ---------------------------------------------------


def test_multiturn_scripts_load_and_every_turn_asserts_something() -> None:
    """Attacks: 'a multi-turn script is a test, not a transcript'.

    A turn with no invariants replays and passes without checking anything, so a
    dropped assertion looks exactly like a green script.
    """
    scripts = load_scripts(MULTITURN)
    assert len(scripts) >= MIN_SCRIPTS, f"only {len(scripts)} multi-turn scripts"
    for script in scripts:
        assert script.turns, f"{script.name}: no turns"
        for index, turn in enumerate(script.turns, start=1):
            assert turn.invariants, f"{script.name} turn {index}: no invariant"


def test_a_typo_in_an_invariant_kind_is_rejected(tmp_path: Path) -> None:
    """Attacks: 'a mistyped invariant kind fails loudly'.

    Mutation test on a real script: an unknown kind must be refused at load time,
    not carried into the runner where it would be looked up and ignored.
    """
    path = min(MULTITURN.glob("*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["turns"][0]["invariants"][0]["kind"] = "min_pick"  # one keystroke away
    (tmp_path / "mutated.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_scripts(tmp_path)


def test_every_parsed_invariant_names_a_real_parsedquery_field() -> None:
    """Attacks: 'the scripts' field names still match the ParsedQuery contract'.

    Six invariants across the scripts assert on ``ParsedQuery`` fields BY NAME.
    Renaming one (``region`` -> ``origin``) or typing it wrong turns those into
    permanent failures blamed on the system rather than on the label - the
    multi-turn twin of an anchor that no longer exists in the corpus.
    """
    from imdb_chatbot.schemas import ParsedQuery

    named = {
        (script.name, inv.field)
        for script in load_scripts(MULTITURN)
        for turn in script.turns
        for inv in turn.invariants
        if inv.kind == "parsed"
    }
    assert named, "no script asserts on the parse at all"
    unknown = {pair for pair in named if pair[1] not in ParsedQuery.model_fields}
    assert not unknown, f"invariants naming a field ParsedQuery does not have: {sorted(unknown)}"


def test_every_kind_the_scripts_use_is_actually_checked() -> None:
    """Attacks: 'every declared invariant kind is actually evaluated'.

    A kind added to ``InvariantKind`` (so scripts may use it) but never wired into
    ``_CHECKERS`` would make ``_check`` raise mid-run - or, if the lookup were ever
    made lenient, turn the scripts into green no-ops. Both start here.
    """
    used = {inv.kind for s in load_scripts(MULTITURN) for t in s.turns for inv in t.invariants}
    assert used, "the golden scripts assert nothing"
    assert used <= set(_CHECKERS), f"kinds with no checker: {sorted(used - set(_CHECKERS))}"


def test_an_unobserved_invariant_fails_rather_than_passing_silently() -> None:
    """Attacks: 'a checker with no evidence reports the truth'.

    The live-tier kinds read things a given runner may not have captured - the
    route, the retrieval count, the parse, the previous turn's picks. A checker
    that treated "I saw nothing" as "nothing was wrong" would report nine green
    scripts while measuring nothing, which is worse than no runner at all.
    """
    blind = _TurnContext(
        parsed=None,
        rewritten_query=None,
        recommended=[],
        shown_before=set(),
        fell_to_fallback=False,
    )
    for inv in (
        Invariant(kind="parsed", field="region", equals="KR"),
        Invariant(kind="routes_to", route="CHITCHAT"),
        Invariant(kind="no_retrieval"),
        Invariant(kind="answers_about_pick", n=2, field="director"),
    ):
        outcome = _check(inv, blind)
        assert outcome.ok is False, f"{inv.kind} passed on a context that observed nothing"
        assert outcome.detail, f"{inv.kind} failed without saying why"


def test_an_unwired_invariant_kind_raises_instead_of_being_ignored() -> None:
    """Attacks: 'an unknown kind is refused, not skipped'.

    ``_check`` is the last gate: a kind that reaches it with no checker must stop
    the run loudly rather than be silently dropped from the script's assertions.
    """
    with pytest.raises(NotImplementedError):
        _check(Invariant.model_construct(kind="not_a_kind"), None)  # type: ignore[arg-type]
