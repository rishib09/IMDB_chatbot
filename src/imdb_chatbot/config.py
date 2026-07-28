"""Typed configuration + secret access.

Reads the versioned artifacts under ``config/`` (models.yaml, limits.yaml,
live_index.json) and resolves runtime secrets from the environment. Local
development may keep secrets in a git-ignored ``.env`` at the repo root; in CI
and on Hugging Face Spaces they arrive as platform environment variables.

No secret values are ever hard-coded here — only their names.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

# Load local dev secrets if a .env exists. No-op in CI / Spaces (they inject env
# vars directly), and a missing file is not an error.
load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_json(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_models_config() -> dict[str, Any]:
    """Model-slot configuration (OpenRouter). Versioned artifact `model_config`."""
    return _load_yaml("models.yaml")


@lru_cache(maxsize=1)
def load_limits_config() -> dict[str, Any]:
    """Security / cost limit ladder (§5.6)."""
    return _load_yaml("limits.yaml")


@lru_cache(maxsize=1)
def load_live_index() -> dict[str, Any]:
    """Pointer to the live index artifact (flipped at index-build time)."""
    return _load_json("live_index.json")


@lru_cache(maxsize=1)
def load_persona() -> dict[str, Any]:
    """Chatbot persona (Maya). Versioned prompt artifact `persona` (ticket #44)."""
    return _load_yaml("persona.yaml")


def get_secret(name: str, *, required: bool = True) -> str | None:
    """Resolve a secret from the environment by name.

    Never stores or logs the value. Raises if a required secret is absent so
    startup fails loudly rather than serving broken (cf. the L3 health check).
    """
    value = os.environ.get(name)
    if required and not value:
        raise RuntimeError(
            f"Missing required secret {name!r}. Set it in a local .env "
            f"(see .env.example) or as a platform environment variable."
        )
    return value
