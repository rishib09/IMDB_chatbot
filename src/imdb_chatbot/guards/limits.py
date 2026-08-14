"""Input / context / session caps and the LLM-call token backstop (section 5.6).

Every function here is pure and injectable: the caps default to the values in
``config/limits.yaml`` (loaded via ``config.load_limits_config``) but each entry
point takes an explicit override so tests drive the boundaries without touching
config on disk.

The governing principle: an oversized input is TRUNCATED with a friendly notice
and an ``O1_input_cap`` flag - it is never an error. The app degrades toward a
still-useful, still-deterministic turn instead of raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import partial

from ..config import load_limits_config

# The single flag emitted whenever an input cap truncates something.
INPUT_CAP_FLAG = "O1_input_cap"

# Fallback defaults, only used if a key is somehow missing from limits.yaml.
DEFAULT_INPUT_MAX_CHARS = 500
DEFAULT_INPUT_MAX_TOKENS = 150
DEFAULT_CONTEXT_MAX_TOKENS = 2000
DEFAULT_MAX_TOKENS = 400
DEFAULT_MAX_TURNS = 30

_TRUNCATE_NOTICE = (
    "Your message was a bit long, so I trimmed it to the first {kept} "
    "before searching. Feel free to rephrase more briefly."
)
_SESSION_LIMIT_NOTICE = (
    "This session has reached its {max_turns}-turn limit. Please start a new "
    "session to keep chatting - your next message will begin a fresh one."
)

# Token counting is a deterministic heuristic (no tokenizer dependency): a token
# is a maximal run of word characters or a single non-space punctuation char.
# Good enough to enforce a budget and identical across platforms.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def _limits() -> dict:
    try:
        return load_limits_config()
    except Exception:  # noqa: BLE001 - never let config I/O turn a cap into a crash
        return {}


def _limit(section: str, key: str, fallback: int) -> int:
    """One int out of one section of limits.yaml, with a hard-coded fallback."""
    return int(_limits().get(section, {}).get(key, fallback))


_per_query = partial(_limit, "per_query")
_per_session = partial(_limit, "per_session")


def count_tokens(text: str) -> int:
    """Deterministic token count used by every token cap. No external tokenizer."""
    return len(_TOKEN_RE.findall(text))


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


@dataclass
class CapResult:
    """The outcome of enforcing a cap: the (possibly trimmed) text plus signals."""

    text: str
    truncated: bool = False
    flags: list[str] = field(default_factory=list)
    notice: str | None = None


def enforce_input_caps(
    text: str,
    *,
    max_chars: int | None = None,
    max_tokens: int | None = None,
) -> CapResult:
    """Enforce the per-query input caps (<=500 chars AND <=150 tokens).

    Oversize input is truncated (chars first, then tokens) to the tighter of the
    two budgets, an ``O1_input_cap`` flag is attached, and a friendly notice is
    returned. This is a graceful degrade, NOT an error - the caller keeps serving.
    """
    max_chars = max_chars if max_chars is not None else _per_query("input_max_chars", DEFAULT_INPUT_MAX_CHARS)
    max_tokens = (
        max_tokens if max_tokens is not None else _per_query("input_max_tokens", DEFAULT_INPUT_MAX_TOKENS)
    )

    truncated = False
    result = text

    # Char cap first (UI-facing hard bound).
    if len(result) > max_chars:
        result = result[:max_chars]
        truncated = True

    # Token cap second (authoritative server-side bound). Rebuild from the kept
    # tokens so we never emit a dangling partial token.
    toks = _tokens(result)
    if len(toks) > max_tokens:
        result = " ".join(toks[:max_tokens])
        truncated = True

    if not truncated:
        return CapResult(text=result)

    kept = f"{len(_tokens(result))} tokens"
    return CapResult(
        text=result,
        truncated=True,
        flags=[INPUT_CAP_FLAG],
        notice=_TRUNCATE_NOTICE.format(kept=kept),
    )


def enforce_context_cap(text: str, *, max_tokens: int | None = None) -> CapResult:
    """Cap the assembled retrieval context at <=2000 tokens (truncate, never error)."""
    max_tokens = (
        max_tokens if max_tokens is not None else _per_query("context_max_tokens", DEFAULT_CONTEXT_MAX_TOKENS)
    )
    toks = _tokens(text)
    if len(toks) <= max_tokens:
        return CapResult(text=text)
    return CapResult(
        text=" ".join(toks[:max_tokens]),
        truncated=True,
        flags=[INPUT_CAP_FLAG],
        notice="Context was trimmed to fit the model's window.",
    )


def llm_call_params(overrides: dict | None = None, *, max_tokens: int | None = None) -> dict:
    """Return the kwargs that MUST accompany every LLM call, with ``max_tokens`` set.

    ``max_tokens`` defaults to the limits.yaml value (400) - the injection
    backstop that caps completion length on every single call. Callers merge the
    returned dict into their model-invocation kwargs; an explicit ``max_tokens``
    passed by a caller is always overridden to enforce the ceiling.
    """
    max_tokens = max_tokens if max_tokens is not None else _per_query("max_tokens", DEFAULT_MAX_TOKENS)
    params = dict(overrides or {})
    params["max_tokens"] = max_tokens
    return params


@dataclass
class SessionTurnDecision:
    """Whether a turn may proceed under the per-session turn budget."""

    allowed: bool
    turn_number: int
    max_turns: int
    message: str | None = None


def session_turn_guard(turns_used: int, *, max_turns: int | None = None) -> SessionTurnDecision:
    """Enforce the per-session turn cap (<=30 turns).

    ``turns_used`` is how many turns the session has already served. If that has
    reached the cap the next turn is refused with a polite "start a new session"
    message rather than an error.
    """
    max_turns = max_turns if max_turns is not None else _per_session("max_turns", DEFAULT_MAX_TURNS)
    if turns_used >= max_turns:
        return SessionTurnDecision(
            allowed=False,
            turn_number=turns_used,
            max_turns=max_turns,
            message=_SESSION_LIMIT_NOTICE.format(max_turns=max_turns),
        )
    return SessionTurnDecision(allowed=True, turn_number=turns_used + 1, max_turns=max_turns)
