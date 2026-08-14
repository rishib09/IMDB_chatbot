"""Follow-up turn routing for the refinement loop (ticket #54).

After results are on screen, each new turn is interpreted broadly. Greeting /
meta / chit-chat are still handled by the deterministic ``persona`` gate; this
module decides, for a search-like follow-up, whether the user is:

- ``REFINE``  - narrowing the current results (accumulate this turn's
  constraints onto the standing ones), the default;
- ``REPLACE`` - starting a brand-new, unrelated search (reset the standing
  constraints);
- ``CLARIFY`` - vaguely rejecting the results with no new constraint ("no, not
  these"), so we ask ONE targeted question instead of searching.

The classifier is INJECTED (``FollowupClassifier``) so the LLM lives at the edge
and tests use a fake. A bare-negation regex short-circuits to ``CLARIFY`` before
the LLM, and anything the classifier can't place defaults to ``REFINE`` - the
user's model is "cover the broad space, then narrow through interaction", so
when unsure we keep narrowing rather than throwing away context.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import StrEnum


class FollowupKind(StrEnum):
    REFINE = "refine"
    REPLACE = "replace"
    CLARIFY = "clarify"


# (query, last recommended titles) -> one of the FollowupKind values (as text).
FollowupClassifier = Callable[[str, list[str]], str]

# A whole-message rejection with no new constraint -> ask a clarifying question.
_BARE_NEGATION_RE = re.compile(
    r"^(no+|nope|nah|meh|not (these|this|it|them)|none( of these)?|"
    r"do(n'?t| not) (like|want) (these|this|them|it)|something else|try again)\.?!?$"
)

_CLARIFY_QUESTION = (
    "No problem - let me narrow it down. Are you after a particular region, a "
    "specific genre, or a certain actor or director?"
)


def clarify_question() -> str:
    """The single targeted question asked on a vague negation."""
    return _CLARIFY_QUESTION


def route_followup(
    query: str,
    last_titles: list[str],
    classifier: FollowupClassifier,
) -> FollowupKind:
    """Classify a follow-up turn (prior results exist) as REFINE / REPLACE / CLARIFY."""
    normalized = (query or "").strip().casefold()
    if _BARE_NEGATION_RE.match(normalized):
        return FollowupKind.CLARIFY

    raw = (classifier(query, last_titles) or "").casefold()
    for kind in (FollowupKind.CLARIFY, FollowupKind.REPLACE, FollowupKind.REFINE):
        if kind.value in raw:
            return kind
    # Unsure -> keep narrowing rather than discard the conversation's context.
    return FollowupKind.REFINE
