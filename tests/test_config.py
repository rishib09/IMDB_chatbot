"""Scaffold smoke tests: the versioned config artifacts load and parse."""

from imdb_chatbot import config


def test_models_config_has_all_slots():
    cfg = config.load_models_config()
    assert cfg["secret"] == "OPENROUTER_API_KEY"
    for slot in ("rewriter", "extractor", "generator", "judge"):
        assert slot in cfg["slots"], f"missing model slot: {slot}"
    # decision #8: the phantom gemma-3-9b slug must not reappear
    assert "gemma-3-9b" not in cfg["slots"]["extractor"]["default"]


def test_limits_config_presets():
    cfg = config.load_limits_config()
    assert cfg["per_query"]["max_tokens"] == 400
    assert cfg["per_turn"]["max_retries_per_slot"] == 2
    assert cfg["per_day"]["production_budget_usd"] == 3.0


def test_live_index_pointer_loads():
    ptr = config.load_live_index()
    assert "active" in ptr


def test_missing_required_secret_raises():
    import pytest

    with pytest.raises(RuntimeError):
        config.get_secret("DEFINITELY_NOT_SET_XYZ", required=True)
