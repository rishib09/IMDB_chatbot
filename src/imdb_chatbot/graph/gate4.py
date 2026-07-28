"""Gate-4: deterministic output validation (PRD section 7.5).

The final gate before a generated answer reaches the user. It is PURE and
DETERMINISTIC - given a ``RecommendationSet``, the retrieval candidates, and the
excluded actors from the ``ParsedQuery``, it either passes or returns a single
machine-readable violation reason. No LLM, no network, no randomness, so it is
trivially unit-testable in isolation (see tests/test_gate4.py).

Three families of check, in priority order:

- (a) EXCLUDED ACTORS - a name the user asked to exclude must not appear in any
  pick (title/reason) nor anywhere in the prose (case-insensitive substring).
- (b) TITLE EXISTENCE (G1 anti-hallucination) - every recommended title must
  correspond to a retrieval candidate. This catches a model inventing a film
  that was never retrieved.
- (c) FACT GROUNDING - facts stated in the free prose must match the candidate
  DB records, catching training-knowledge leakage:
    * every 4-digit year mentioned in the prose must be the year of some
      candidate (or pick);
    * a "directed by <Name>" claim must match a candidate's director;
    * a "starring/stars <Name>" claim must match a candidate's cast.

LIMITS (documented deliberately - free-prose grounding is NOT fully verifiable):
Candidate objects are duck-typed. ``ScoredMovie`` (what the live graph carries)
exposes only ``title`` and ``year``, so for graph turns only the year check and
the title-existence check have signal; director/cast grounding is a no-op when no
candidate carries that metadata. When candidates DO carry ``director``/``cast``
(e.g. a ``MovieRecord``), those claims are grounded too. We only inspect names in
the narrow, regex-anchored positions ("directed by X", "starring X"); arbitrary
prose that asserts a fact without those anchors is not checked, because reliably
extracting free-form entity claims is beyond a deterministic gate and would trade
false negatives for false positives. Gate-4 is a safety net, not a proof system.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..schemas import RecommendationSet

# 4-digit years 1888..2029-ish; we only anchor on the 19xx/20xx shape.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# A person name following an anchor phrase: 1-4 capitalized tokens (allowing
# internal apostrophes/hyphens/periods, e.g. "Joel Coen", "J.J. Abrams").
_NAME = r"[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3}"
_DIRECTED_BY_RE = re.compile(rf"directed by (?:the )?({_NAME})")
_STARRING_RE = re.compile(rf"(?:starring|stars(?:\s+the)?)\s+({_NAME})")


@dataclass(frozen=True)
class Gate4Result:
    """Outcome of Gate-4: ``ok`` plus a machine-readable ``reason`` on failure.

    ``reason`` is empty when ``ok`` is True. On failure it is a stable, colon-
    delimited token (e.g. ``"excluded_actor:tom cruise"``) suitable both for
    machine routing and for appending to the regeneration prompt.
    """

    ok: bool
    reason: str = ""


def _titles(response: RecommendationSet) -> list[str]:
    return [p.title for p in response.picks]


def _norm(text: str) -> str:
    return text.strip().casefold()


def _norm_name(text: str) -> str:
    # A regex-captured name can pick up a trailing sentence period or comma
    # (e.g. "Christopher Nolan."); strip surrounding punctuation before matching.
    return _norm(text).strip(".,;:!?\"'")


def check_excluded_actors(
    response: RecommendationSet, exclude_actors: Sequence[str]
) -> str | None:
    """(a) No excluded actor may appear in a pick or in the prose.

    Case-insensitive substring match on the raw actor name against pick titles,
    pick reasons, and the prose. Returns a reason token or ``None``.
    """
    haystacks = [response.prose]
    for pick in response.picks:
        haystacks.append(pick.title)
        haystacks.append(pick.reason)
    blob = _norm("\n".join(h for h in haystacks if h))
    for actor in exclude_actors:
        name = _norm(actor)
        if name and name in blob:
            return f"excluded_actor:{name}"
    return None


def check_titles_exist(
    response: RecommendationSet, candidates: Sequence[Any]
) -> str | None:
    """(b) Every recommended title must exist among the candidates (G1)."""
    allowed = {_norm(getattr(c, "title", "")) for c in candidates}
    for title in _titles(response):
        if _norm(title) not in allowed:
            return f"hallucinated_title:{_norm(title)}"
    return None


def check_fact_grounding(
    response: RecommendationSet, candidates: Sequence[Any]
) -> str | None:
    """(c) Years / director / cast asserted in the prose must match a candidate.

    Grounds only what the candidate metadata supports (see module LIMITS). Years
    are always checked; director/cast are checked only when some candidate
    carries that field.
    """
    prose = response.prose or ""

    # Years: allowed set = candidate years plus the years of the picks
    # themselves (a pick's own year is authoritative for that pick).
    allowed_years = {getattr(c, "year", None) for c in candidates}
    allowed_years |= {p.year for p in response.picks}
    allowed_years.discard(None)
    for match in _YEAR_RE.findall(prose):
        if int(match) not in allowed_years:
            return f"ungrounded_year:{match}"

    # Directors: only if at least one candidate exposes a non-empty director.
    directors = {_norm(d) for c in candidates if (d := getattr(c, "director", None))}
    if directors:
        for claimed in _DIRECTED_BY_RE.findall(prose):
            name = _norm_name(claimed)
            if name not in directors:
                return f"ungrounded_director:{name}"

    # Cast: only if at least one candidate exposes a non-empty cast list.
    cast_names: set[str] = set()
    for c in candidates:
        for member in getattr(c, "cast", []) or []:
            cast_names.add(_norm(member))
    if cast_names:
        for claimed in _STARRING_RE.findall(prose):
            name = _norm_name(claimed)
            if name not in cast_names:
                return f"ungrounded_cast:{name}"

    return None


def run_gate4(
    response: RecommendationSet,
    candidates: Sequence[Any],
    exclude_actors: Sequence[str] = (),
) -> Gate4Result:
    """Run all Gate-4 checks and return the first violation (or a pass).

    Check order is (a) excluded actors -> (b) title existence -> (c) fact
    grounding; the first failing check short-circuits so ``reason`` names the
    most user-facing violation.
    """
    for check in (
        lambda: check_excluded_actors(response, exclude_actors),
        lambda: check_titles_exist(response, candidates),
        lambda: check_fact_grounding(response, candidates),
    ):
        reason = check()
        if reason is not None:
            return Gate4Result(ok=False, reason=reason)
    return Gate4Result(ok=True)
