"""Tests for the anchored eval set + retrieval eval harness (ticket #17).

Everything runs on a small SYNTHETIC corpus and the dependency-free
``StubEmbedder``: a tiny FAISS + BM25 index is built with #15's ``build_index``,
then a ``HybridRetriever`` runs over it and the harness scores a handful of
example labeled queries whose ``relevant_tmdb_ids`` we control.

The corpus + stub embedder are fully deterministic, so the per-query relevant
ranks are fixed constants. From those fixed ranks we HAND-COMPUTE the expected
Recall@k / Hit@k / MRR and assert the harness reports exactly those values.
Two properties are structural (independent of the exact ranking): an exclusion
query has exclusion precision 1.00, and an ood query with no relevant ids is
handled (recall/MRR undefined, no crash).

Tests that need ``faiss`` are skipped when it is not installed; they run on CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from imdb_chatbot.eval import evaluate, format_report, load_labels
from imdb_chatbot.eval.labels import CATEGORIES, LabeledQuery
from imdb_chatbot.index.build import build_index, load_index
from imdb_chatbot.index.embedder import StubEmbedder
from imdb_chatbot.retrieval.retrieve import HybridRetriever
from imdb_chatbot.schemas import MovieRecord, ParsedQuery
from imdb_chatbot.store import TraceStore

HAS_FAISS = importlib.util.find_spec("faiss") is not None
requires_faiss = pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")

FIXTURE = Path(__file__).parent / "fixtures" / "labels_example.jsonl"

# Distinctive rare plot tokens (zephyrium, quixotl, monsoon/backwaters, comedy)
# make each known-item query's single relevant movie a clear BM25 match; RRF then
# fuses that with the stub's dense ranking to a fixed, deterministic position.
REPEAT_STAR = "Repeat Star"


def _corpus() -> list[MovieRecord]:
    return [
        MovieRecord(tmdb_id=1, title="Midnight Detective", year=2001,
                    genres=["Thriller", "Mystery"], director="Dir A",
                    cast=[REPEAT_STAR, "Alice Kim"],
                    plot="A detective hunts a killer through a rainy zephyrium city at midnight.",
                    regions=["US"], rating_raw=7.5),
        MovieRecord(tmdb_id=2, title="Seoul Mystery", year=2019,
                    genres=["Mystery", "Drama"], director="Dir B",
                    cast=["Song Lee", "Alice Kim"],
                    plot="A quiet detective story about a missing sister in Seoul.",
                    regions=["KR"], rating_raw=8.2),
        MovieRecord(tmdb_id=3, title="Haunted Manor", year=2015,
                    genres=["Horror", "Thriller"], director="Dir C",
                    cast=[REPEAT_STAR, "Bob Stone"],
                    plot="A family is terrorized by ghosts in a haunted quixotl manor.",
                    regions=["US"], rating_raw=6.1),
        MovieRecord(tmdb_id=4, title="Delhi Nights", year=2012,
                    genres=["Drama", "Romance"], director="Dir D",
                    cast=["Ravi Menon", "Priya Nair"],
                    plot="A romance blooms across the night markets of Delhi.",
                    regions=["IN"], rating_raw=7.0),
        MovieRecord(tmdb_id=5, title="Cold Case Files", year=2008,
                    genres=["Thriller", "Crime"], director="Dir E",
                    cast=[REPEAT_STAR, "Nina West"],
                    plot="Detectives reopen a cold murder case from decades past.",
                    regions=["US", "IN"], rating_raw=6.8),
        MovieRecord(tmdb_id=6, title="Mumbai Horror", year=2020,
                    genres=["Horror"], director="Dir F",
                    cast=["Ravi Menon", "Bob Stone"],
                    plot="A cursed film reel unleashes terror in a Mumbai cinema.",
                    regions=["IN"], rating_raw=5.4),
        MovieRecord(tmdb_id=7, title="The Silent Witness", year=1998,
                    genres=["Mystery", "Thriller"], director="Dir G",
                    cast=["Alice Kim", "Nina West"],
                    plot="A mute witness holds the key to a baffling murder mystery.",
                    regions=["US"], rating_raw=7.9),
        MovieRecord(tmdb_id=8, title="Busan Ghost", year=2016,
                    genres=["Horror", "Drama"], director="Dir H",
                    cast=["Song Lee", REPEAT_STAR],
                    plot="A grieving mother is haunted by a ghost on a train to Busan.",
                    regions=["KR"], rating_raw=8.0),
        MovieRecord(tmdb_id=9, title="Sunny Comedy", year=2011,
                    genres=["Comedy"], director="Dir I",
                    cast=["Priya Nair", "Bob Stone"],
                    plot="A sunny feel-good comedy about a dysfunctional wedding comedy.",
                    regions=["IN"], rating_raw=6.5),
        MovieRecord(tmdb_id=10, title="Detective Dawn", year=2022,
                    genres=["Crime", "Mystery"], director="Dir J",
                    cast=["Nina West", "Song Lee"],
                    plot="At dawn a rookie detective cracks a citywide smuggling ring.",
                    regions=["US"], rating_raw=7.2),
        MovieRecord(tmdb_id=11, title="Quiet Drama", year=2005,
                    genres=["Drama"], director="Dir K",
                    cast=["Alice Kim", "Ravi Menon"],
                    plot="A slow quiet drama about two estranged brothers reuniting.",
                    regions=["IN"], rating_raw=7.6),
        MovieRecord(tmdb_id=12, title="Night Terror", year=2018,
                    genres=["Horror", "Mystery"], director="Dir L",
                    cast=[REPEAT_STAR, "Priya Nair"],
                    plot="A detective investigates murders that only happen at night.",
                    regions=["US", "KR"], rating_raw=6.9),
        MovieRecord(tmdb_id=13, title="Kerala Monsoon", year=2017,
                    genres=["Drama", "Romance"], director="Dir M",
                    cast=["Ravi Menon", "Priya Nair"],
                    plot="A tender monsoon romance set among the backwaters of Kerala.",
                    regions=["IN"], rating_raw=8.1),
    ]


@pytest.fixture()
def retriever(tmp_path: Path):
    """A HybridRetriever (final_k=10) over a stub index of the synthetic corpus."""
    store = TraceStore(tmp_path / "corpus.sqlite")
    try:
        for movie in _corpus():
            store.write_movie(movie)
        embedder = StubEmbedder(dim=16)
        result = build_index(
            store, embedder, dataset_version="t17",
            out_root=tmp_path / "index", cache=None, flip_pointer=False,
        )
        loaded = load_index(result.out_dir)
        yield HybridRetriever.from_store(loaded, embedder, store, final_k=10)
    finally:
        store.close()


# -- loader (no faiss needed) -------------------------------------------------


def test_load_labels_parses_example_set() -> None:
    labels = load_labels(FIXTURE)
    assert [lab.query_id for lab in labels] == ["q1", "q2", "q3", "q4", "q5", "q6"]
    by_id = {lab.query_id: lab for lab in labels}
    # Every category is a known one.
    assert all(lab.category in CATEGORIES for lab in labels)
    # Optional parsed fields load through to a ParsedQuery.
    assert by_id["q3"].parsed.exclude_actors == [REPEAT_STAR]
    assert by_id["q4"].parsed.region == "IN"
    # Missing parsed -> empty ParsedQuery; ood row -> no relevant ids.
    assert by_id["q1"].parsed == ParsedQuery()
    assert by_id["q6"].relevant_tmdb_ids == []


def test_load_labels_rejects_unknown_category(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"query_id": "x", "query": "q", "category": "not_a_category", '
        '"relevant_tmdb_ids": [1]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_labels(bad)


def test_load_labels_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "labels.jsonl"
    p.write_text(
        '\n{"query_id": "a", "query": "q", "category": "short", '
        '"relevant_tmdb_ids": [2]}\n\n',
        encoding="utf-8",
    )
    labels = load_labels(p)
    assert len(labels) == 1 and labels[0].query_id == "a"


# -- harness metrics (need faiss) ---------------------------------------------


@requires_faiss
def test_known_item_ranks_are_deterministic(retriever: HybridRetriever) -> None:
    """The synthetic setup pins each known-item query to a fixed first rank."""
    labels = load_labels(FIXTURE)
    report = evaluate(retriever, labels)
    by_id = {q.query_id: q for q in report.queries}
    # Hand-observed deterministic first-relevant ranks (see module docstring).
    assert by_id["q1"].first_relevant_rank == EXPECTED_FIRST_RANK["q1"]
    assert by_id["q2"].first_relevant_rank == EXPECTED_FIRST_RANK["q2"]
    assert by_id["q3"].first_relevant_rank == EXPECTED_FIRST_RANK["q3"]
    assert by_id["q4"].first_relevant_rank == EXPECTED_FIRST_RANK["q4"]
    assert by_id["q5"].first_relevant_rank == EXPECTED_FIRST_RANK["q5"]


@requires_faiss
def test_hit_and_recall_match_hand_computed(retriever: HybridRetriever) -> None:
    labels = load_labels(FIXTURE)
    report = evaluate(retriever, labels)
    by_id = {q.query_id: q for q in report.queries}
    for qid, rank in EXPECTED_FIRST_RANK.items():
        q = by_id[qid]
        # Single relevant id: recall@k is 1.0 once k reaches its rank, else 0.0;
        # hit@k mirrors that. MRR term is 1/rank.
        for k in (1, 3, 5, 10):
            assert q.hit_at[k] is (k >= rank), f"{qid} hit@{k}"
            assert q.recall_at[k] == (1.0 if k >= rank else 0.0), f"{qid} recall@{k}"
        assert q.reciprocal_rank == pytest.approx(1.0 / rank)


@requires_faiss
def test_overall_mrr_matches_hand_computed(retriever: HybridRetriever) -> None:
    labels = load_labels(FIXTURE)
    report = evaluate(retriever, labels)
    # MRR averages over the 5 answerable queries (q6 is ood -> excluded).
    expected_mrr = sum(1.0 / r for r in EXPECTED_FIRST_RANK.values()) / len(EXPECTED_FIRST_RANK)
    assert report.overall.n == 6
    assert report.overall.n_answerable == 5
    assert report.overall.mrr == pytest.approx(expected_mrr)


@requires_faiss
def test_exclusion_precision_is_one(retriever: HybridRetriever) -> None:
    """q3 excludes 'Repeat Star': no returned candidate may carry that actor."""
    labels = load_labels(FIXTURE)
    report = evaluate(retriever, labels)
    q3 = next(q for q in report.queries if q.query_id == "q3")
    assert q3.has_exclusions is True
    assert q3.exclusion_clean is True
    corpus = {m.tmdb_id: m for m in _corpus()}
    scored = retriever.retrieve("detective murder mystery", parsed=ParsedQuery())
    # Non-vacuous: 'Repeat Star' really does appear without the exclusion.
    assert any(REPEAT_STAR in corpus[s.tmdb_id].cast for s in scored)
    # Per-category summary reports exclusion precision 1.00.
    neg = report.per_category["negation_exclusion"]
    assert neg.exclusion_precision == 1.0
    assert neg.n_exclusion_queries == 1


@requires_faiss
def test_ood_query_is_handled(retriever: HybridRetriever) -> None:
    """An ood query with no relevant ids: recall/MRR undefined, no crash."""
    labels = load_labels(FIXTURE)
    report = evaluate(retriever, labels)
    q6 = next(q for q in report.queries if q.query_id == "q6")
    assert q6.num_relevant == 0
    assert q6.reciprocal_rank is None
    assert all(v is None for v in q6.recall_at.values())
    ood = report.per_category["ood_unanswerable"]
    # A fallback (zero candidates) is the CORRECT outcome for ood.
    assert ood.fallback_is_correct is True
    assert ood.n_answerable == 0


@requires_faiss
def test_fallback_rate_when_filters_remove_everything(retriever: HybridRetriever) -> None:
    """A query whose filters exclude every movie returns zero candidates."""
    labels = [
        LabeledQuery(
            query_id="z", query="detective", category="numeric_filter",
            relevant_tmdb_ids=[1], parsed=ParsedQuery(min_year=3000),
        )
    ]
    report = evaluate(retriever, labels)
    q = report.queries[0]
    assert q.fell_back is True
    assert q.num_returned == 0
    assert report.overall.fallback_rate == 1.0


@requires_faiss
def test_format_report_is_ascii(retriever: HybridRetriever) -> None:
    labels = load_labels(FIXTURE)
    report = evaluate(retriever, labels)
    text = format_report(report)
    assert text.isascii()
    assert "overall" in text
    assert "MRR" in text
    assert "exclusion_precision" in text


# Deterministic first-relevant ranks for the synthetic corpus. RRF fuses by RANK
# (not score), so the stub embedder's random dense ordering, not the rare BM25
# token, decides where each single relevant id lands - but the corpus + stub are
# fully deterministic, so these are fixed constants. Hand-derived metrics:
#   recall@k = hit@k = 1 once k reaches the rank; reciprocal rank = 1/rank.
#
# StubEmbedder seeds its RNG with sha256(composed text), so these constants are
# an artifact of the exact string ``compose_text`` emits and MOVE whenever that
# string changes - they say nothing about real retrieval quality. Last re-observed
# for the #83 plot-before-cast composition (they were 7/5/4/2/1 before it).
EXPECTED_FIRST_RANK: dict[str, int] = {
    "q1": 5,
    "q2": 5,
    "q3": 1,
    "q4": 1,
    "q5": 3,
}
