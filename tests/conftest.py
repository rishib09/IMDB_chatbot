"""Shared fixtures: resource discovery for the live test tier (ticket #69).

The live tier (``pytest -m live``) runs real turns against the real model,
index and corpus. The fixtures here resolve those resources - overridable via
``IMDB_TEST_CORPUS`` and ``IMDB_TEST_INDEX`` so the suite runs from any
checkout - and skip with the missing piece named when one is absent. The same
tests therefore work locally (everything present), in CI (key but no index:
index-dependent tests self-skip), and on a fresh clone (all live tests skip).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Never let the OpenRouter key (or any auth material) into a cassette."""
    return {"filter_headers": ["authorization", "x-api-key", "cookie"]}


@pytest.fixture(scope="session")
def live_corpus_path() -> Path:
    path = Path(os.environ.get("IMDB_TEST_CORPUS") or PROJECT_ROOT / "data" / "corpus.sqlite")
    if not path.is_file():
        pytest.skip(f"corpus not found at {path}; set IMDB_TEST_CORPUS to a corpus.sqlite")
    return path


@pytest.fixture(scope="session")
def live_index_dir() -> Path:
    override = os.environ.get("IMDB_TEST_INDEX")
    if override:
        path = Path(override)
    else:
        pointer_file = PROJECT_ROOT / "config" / "live_index.json"
        pointer = json.loads(pointer_file.read_text(encoding="utf-8"))
        if not pointer.get("active"):
            pytest.skip(
                "no active index in config/live_index.json; set IMDB_TEST_INDEX to an index dir"
            )
        path = Path(pointer["path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_dir():
        pytest.skip(f"index dir not found at {path}; set IMDB_TEST_INDEX to an index dir")
    return path


@pytest.fixture(scope="session")
def openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        pytest.skip("OPENROUTER_API_KEY is not set; run under dotenvx (see README)")
    if key.startswith("encrypted:"):
        pytest.skip("OPENROUTER_API_KEY is dotenvx-encrypted; run under dotenvx (see README)")
    return key


@pytest.fixture(scope="session")
def live_resources(live_corpus_path: Path, live_index_dir: Path, openrouter_key: str):
    """The real resource bundle (index + embedder + corpus + models), loaded once.

    ``load_live_resources`` reads the index pointer from ``config/``; here the
    pointer is redirected to the fixture-resolved directory so the tier runs
    from any checkout (the committed pointer is null on a fresh clone).
    """
    from imdb_chatbot import config as config_module
    from imdb_chatbot.dashboard.live import load_live_resources

    pointer = {"active": live_index_dir.name, "path": str(live_index_dir)}
    patcher = pytest.MonkeyPatch()
    patcher.setattr(config_module, "load_live_index", lambda: pointer)
    try:
        yield load_live_resources(corpus_path=live_corpus_path)
    finally:
        patcher.undo()
