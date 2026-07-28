"""Tests for the security / cost controls and degradation ladder (ticket #23).

All deterministic, all offline: no network, no live LLM, no wall clock. The L2
tests drive a stub retriever returning known candidates; the L3 test points the
health check at a temp pointer file; the budget test injects a fixed clock and
store so accumulation is reproducible.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from imdb_chatbot.guards import (
    BudgetTracker,
    count_tokens,
    enforce_context_cap,
    enforce_input_caps,
    health_check,
    honest_exit_message,
    llm_call_params,
    llm_free_answer,
    record_degradation,
    session_turn_guard,
    should_enter_l2,
    topic_gate,
)
from imdb_chatbot.guards.degradation import L2_BANNER, L2_FLAG, L3_FLAG
from imdb_chatbot.guards.limits import INPUT_CAP_FLAG
from imdb_chatbot.schemas import MovieRecord, ScoredMovie

# -- input caps ---------------------------------------------------------------


def test_oversize_chars_truncates_not_errors() -> None:
    text = "a" * 900  # one long token, well over the 500-char cap
    result = enforce_input_caps(text)
    assert result.truncated is True
    assert INPUT_CAP_FLAG in result.flags
    assert result.notice  # a friendly notice, not an exception
    assert len(result.text) <= 500


def test_oversize_tokens_truncates_under_char_cap() -> None:
    # 200 single-char words -> 200 tokens but only ~399 chars: trips the TOKEN
    # cap while staying under the 500-char cap.
    text = " ".join("x" for _ in range(200))
    assert len(text) < 500
    result = enforce_input_caps(text)
    assert result.truncated is True
    assert INPUT_CAP_FLAG in result.flags
    assert count_tokens(result.text) <= 150


def test_within_caps_untouched() -> None:
    text = "a cozy mystery movie set in the 1990s"
    result = enforce_input_caps(text)
    assert result.truncated is False
    assert result.flags == []
    assert result.text == text


def test_context_cap_truncates() -> None:
    text = " ".join("word" for _ in range(2500))
    result = enforce_context_cap(text)
    assert result.truncated is True
    assert count_tokens(result.text) <= 2000


def test_llm_call_params_force_max_tokens() -> None:
    # max_tokens=400 is set on every call, even overriding a caller value.
    assert llm_call_params()["max_tokens"] == 400
    assert llm_call_params({"temperature": 0.7})["temperature"] == 0.7
    assert llm_call_params({"max_tokens": 9999})["max_tokens"] == 400


def test_session_turn_cap_polite_refusal() -> None:
    ok = session_turn_guard(29)
    assert ok.allowed is True
    assert ok.turn_number == 30

    over = session_turn_guard(30)
    assert over.allowed is False
    assert over.message and "new session" in over.message.lower()


# -- topic gate ---------------------------------------------------------------


def test_offtopic_query_fixed_refusal() -> None:
    result = topic_gate("what is the weather in Paris today")
    assert result.on_topic is False
    assert result.refusal is not None
    # A fixed, deterministic refusal (same string every time).
    assert result.refusal == topic_gate("how do I write python code").refusal


def test_ontopic_movie_query_passes() -> None:
    assert topic_gate("recommend a good thriller movie").on_topic is True
    # Movie context wins even when an off-topic word appears.
    assert topic_gate("a film about cooking in Italy").on_topic is True


# -- L1: flag recording -------------------------------------------------------


def test_record_degradation_is_idempotent() -> None:
    flags = record_degradation([], "rewrite_skipped")
    assert flags == ["rewrite_skipped"]
    assert record_degradation(flags, "rewrite_skipped") == ["rewrite_skipped"]


# -- L2: LLM-free deterministic mode ------------------------------------------


class _StubRetriever:
    """A deterministic, LLM-free stand-in for HybridRetriever."""

    def __init__(self, candidates: list[ScoredMovie], movies: list[MovieRecord]) -> None:
        self._candidates = candidates
        self.movies_by_id = {m.tmdb_id: m for m in movies}
        self.calls = 0

    def retrieve(self, query, parsed=None, shown_movies=None):
        self.calls += 1
        return list(self._candidates)


def _l2_fixtures() -> _StubRetriever:
    movies = [
        MovieRecord(
            tmdb_id=1,
            title="Midnight Detective",
            year=2001,
            genres=["Thriller", "Mystery"],
            rating_raw=7.5,
            poster_url="https://img.example/1.jpg",
        ),
        MovieRecord(
            tmdb_id=2,
            title="Seoul Mystery",
            year=2019,
            genres=["Mystery", "Drama"],
            rating_raw=8.2,
        ),
    ]
    candidates = [
        ScoredMovie(tmdb_id=1, title="Midnight Detective", year=2001, rank=1, rrf_score=0.9),
        ScoredMovie(tmdb_id=2, title="Seoul Mystery", year=2019, rank=2, rrf_score=0.5),
    ]
    return _StubRetriever(candidates, movies)


def test_l2_triggers_on_slots_failing_or_budget() -> None:
    assert should_enter_l2(all_slots_failing=True) is True
    assert should_enter_l2(budget_exhausted=True) is True
    assert should_enter_l2(openrouter_down=True) is True
    assert should_enter_l2() is False


def test_l2_renders_metadata_cards_banner_no_prose() -> None:
    retriever = _l2_fixtures()
    result = llm_free_answer("murder mystery at night", retriever)

    # Banner + degradation flag, and NO generated prose anywhere.
    assert result.banner == L2_BANNER
    assert L2_FLAG in result.flags
    assert result.recommendations.prose == ""
    assert all(pick.reason == "" for pick in result.recommendations.picks)

    # Metadata cards carry title/year/genre/rating/poster from the live index.
    assert [c.title for c in result.cards] == ["Midnight Detective", "Seoul Mystery"]
    first = result.cards[0]
    assert first.year == 2001
    assert first.genres == ["Thriller", "Mystery"]
    assert first.rating == 7.5
    assert first.poster_url == "https://img.example/1.jpg"


# -- L3: startup health check refuses to serve --------------------------------


def _write_pointer(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_l3_missing_pointer_refuses_without_raising(tmp_path: Path) -> None:
    result = health_check(live_index_path=tmp_path / "nope.json")
    assert result.healthy is False
    assert result.message == honest_exit_message()
    assert L3_FLAG in result.flags


def test_l3_null_active_pointer_refuses(tmp_path: Path) -> None:
    ptr = tmp_path / "live_index.json"
    _write_pointer(ptr, {"active": None, "path": None})
    result = health_check(live_index_path=ptr)
    assert result.healthy is False
    assert result.message == honest_exit_message()


def test_l3_corrupt_pointer_refuses(tmp_path: Path) -> None:
    ptr = tmp_path / "live_index.json"
    ptr.write_text("{ not valid json", encoding="utf-8")
    result = health_check(live_index_path=ptr)
    assert result.healthy is False
    assert result.checks["pointer"] is False


def test_l3_pointer_ok_but_missing_artifacts_refuses(tmp_path: Path) -> None:
    ptr = tmp_path / "live_index.json"
    _write_pointer(ptr, {"active": "v1", "path": str(tmp_path / "index_v1")})
    result = health_check(live_index_path=ptr)
    assert result.healthy is False
    assert result.checks["pointer"] is True
    assert result.checks["index"] is False


def test_l3_healthy_when_all_present(tmp_path: Path) -> None:
    index_dir = tmp_path / "index_v1"
    index_dir.mkdir()
    for name in ("index.faiss", "bm25.pkl", "sidecar.json"):
        (index_dir / name).write_text("x", encoding="utf-8")
    ptr = tmp_path / "live_index.json"
    _write_pointer(ptr, {"active": "v1", "path": str(index_dir)})

    opened = {"count": 0}

    class _Handle:
        def close(self):
            opened["count"] += 1

    result = health_check(live_index_path=ptr, db_opener=lambda: _Handle())
    assert result.healthy is True
    assert result.checks == {"pointer": True, "index": True, "db": True}
    assert opened["count"] == 1


def test_l3_broken_db_refuses(tmp_path: Path) -> None:
    index_dir = tmp_path / "index_v1"
    index_dir.mkdir()
    for name in ("index.faiss", "bm25.pkl", "sidecar.json"):
        (index_dir / name).write_text("x", encoding="utf-8")
    ptr = tmp_path / "live_index.json"
    _write_pointer(ptr, {"active": "v1", "path": str(index_dir)})

    def _broken():
        raise OSError("database is locked")

    result = health_check(live_index_path=ptr, db_opener=_broken)
    assert result.healthy is False
    assert result.checks["index"] is True
    assert result.checks["db"] is False


# -- budget: crossing $3/day flips to L2 --------------------------------------


def test_budget_accumulation_flips_to_l2() -> None:
    day = date(2026, 7, 25)
    tracker = BudgetTracker(budget_usd=3.0, clock=lambda: day, store={})

    tracker.record(1.0)
    tracker.record(1.5)
    assert tracker.spent_today() == 2.5
    assert tracker.exhausted() is False
    assert tracker.mode() == "normal"

    # Crossing $3 flips the mode to L2.
    tracker.record(0.8)  # total 3.3 > 3.0
    assert tracker.exhausted() is True
    assert tracker.mode() == "L2"


def test_budget_resets_per_day() -> None:
    store: dict[date, float] = {}
    clock = {"d": date(2026, 7, 25)}
    tracker = BudgetTracker(budget_usd=3.0, clock=lambda: clock["d"], store=store)

    tracker.record(4.0)
    assert tracker.mode() == "L2"

    # New day: fresh bucket, back to normal.
    clock["d"] = date(2026, 7, 26)
    assert tracker.spent_today() == 0.0
    assert tracker.mode() == "normal"


def test_budget_from_config_default() -> None:
    # Default budget comes from limits.yaml (3.0) when not overridden.
    tracker = BudgetTracker(clock=lambda: date(2026, 7, 25), store={})
    assert tracker.budget_usd == 3.0
