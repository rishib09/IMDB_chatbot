"""Pure, Streamlit-free render-data prep for the Chat page.

These helpers turn a ``RecommendationSet`` into plain dicts/lists that the
Streamlit code in ``app.py`` renders as poster cards or fallback quick-reply
buttons. Keeping them free of any ``streamlit`` import means the card mapping
and the fallback branch can be unit-tested without spinning up a server.

Card fields come STRICTLY from the structured ``MovieRecommendation`` fields -
never from parsing the model's prose.
"""

from __future__ import annotations

from ..schemas import RecommendationSet

# Shown when a recommendation has no poster (or a blank one). A data URI so the
# card never triggers a network fetch and never crashes on a broken URL.
PLACEHOLDER_POSTER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='200' height='300'>"
    "<rect width='200' height='300' fill='%23303030'/>"
    "<text x='100' y='150' fill='%23cccccc' font-size='16' "
    "text-anchor='middle' dominant-baseline='middle'>No poster</text></svg>"
)

# Deterministic relax-a-constraint quick replies for the empty/fallback turn.
# Each is (button label, the follow-up query text sent on click).
RELAX_OPTIONS: list[dict[str, str]] = [
    {"label": "Widen the year range", "query": "Widen the year range."},
    {"label": "Drop an excluded genre", "query": "Drop an excluded genre."},
    {"label": "Lower the minimum rating", "query": "Lower the minimum rating."},
    {"label": "Any region", "query": "Include any region."},
]


def poster_src(url: object) -> str:
    """Return a usable image src: the given URL, or a placeholder.

    Accepts ``None``, an empty/whitespace string, or an ``HttpUrl``/``str``.
    Anything falsy or blank yields ``PLACEHOLDER_POSTER`` so ``st.image`` never
    receives an empty value and the card never crashes.
    """
    if url is None:
        return PLACEHOLDER_POSTER
    text = str(url).strip()
    return text or PLACEHOLDER_POSTER


def recommendation_cards(rec: RecommendationSet) -> list[dict]:
    """Map a ``RecommendationSet`` to a list of card dicts.

    Each card has the keys ``title``, ``year``, ``reason`` and ``poster``,
    all sourced from the structured ``MovieRecommendation`` fields. A fallback
    or empty set (no picks) yields an empty list.
    """
    return [
        {
            "title": pick.title,
            "year": pick.year,
            "reason": pick.reason,
            "poster": poster_src(pick.poster_url),
        }
        for pick in rec.picks
    ]


def is_fallback(rec: RecommendationSet) -> bool:
    """True when the turn produced no picks (the relax-a-constraint path)."""
    return not rec.picks


def relax_options() -> list[dict[str, str]]:
    """Deterministic relax-a-constraint quick replies for a fallback turn.

    Returns a fresh copy so callers cannot mutate the module-level constant.
    """
    return [dict(option) for option in RELAX_OPTIONS]
