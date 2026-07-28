"""Topic gate: a cheap on-topic check that runs BEFORE rewrite (section 5.6).

The default path is fully deterministic - a keyword / heuristic classifier over
a movie-domain vocabulary and an off-topic vocabulary, with NO live LLM. An
optional cheap-LLM classifier can be injected for the ambiguous middle, but tests
(and the safe default) never call it.

An off-topic query gets a single fixed movies-only refusal string; the caller
short-circuits the turn there and spends nothing on retrieval or generation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

REFUSAL = (
    "I can only help with movies - finding films, recommendations, and details "
    "about them. Ask me for something to watch and I'm all yours."
)

# Movie-domain vocabulary: presence of any of these keeps a query on-topic even
# if an off-topic word also appears (e.g. "a movie about cooking").
_MOVIE_TERMS = {
    "movie", "movies", "film", "films", "cinema", "flick", "flicks",
    "watch", "watching", "rewatch", "stream", "streaming",
    "recommend", "recommendation", "recommendations", "suggest", "suggestion",
    "director", "directed", "actor", "actress", "cast", "starring", "stars",
    "genre", "plot", "sequel", "trilogy", "franchise", "screenplay",
    "show", "series", "documentary", "animated", "animation", "blockbuster",
    "rated", "rating", "imdb", "oscar", "oscars",
    # Genre words (mirror the retriever's vocabulary).
    "action", "adventure", "comedy", "crime", "drama", "fantasy", "horror",
    "mystery", "romance", "romcom", "sci-fi", "scifi", "thriller", "war",
    "western", "noir", "heist", "superhero", "slasher",
}

# Clearly off-topic vocabulary spanning the common abuse categories: coding,
# finance, weather, cooking, health, general knowledge, homework, etc.
_OFFTOPIC_TERMS = {
    "weather", "forecast", "temperature", "rain", "snow",
    "stock", "stocks", "crypto", "bitcoin", "ethereum", "invest", "investment",
    "recipe", "recipes", "cook", "cooking", "bake", "ingredient", "ingredients",
    "python", "javascript", "code", "coding", "program", "function", "regex",
    "sql", "bug", "compile", "algorithm",
    "medicine", "medical", "symptom", "symptoms", "diagnose", "disease", "dosage",
    "president", "election", "politics", "senator", "congress",
    "homework", "essay", "translate", "translation", "capital",
    "math", "equation", "integral", "derivative", "calculus",
    "flight", "hotel", "directions", "restaurant",
}

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")

TopicClassifier = Callable[[str], bool]


@dataclass
class TopicResult:
    """Outcome of the topic gate: whether to proceed, and the refusal if not."""

    on_topic: bool
    reason: str
    refusal: str | None = None


def _words(query: str) -> set[str]:
    return set(_WORD_RE.findall(query.lower()))


def topic_gate(query: str, *, llm_classifier: TopicClassifier | None = None) -> TopicResult:
    """Classify a query as on- or off-topic BEFORE any rewrite / retrieval.

    Deterministic rules, cheapest first:

    1. Any movie-domain term present -> on-topic (movie context wins).
    2. Any off-topic term present (and no movie term) -> off-topic refusal.
    3. Otherwise ambiguous: on-topic by default, unless an injected cheap-LLM
       ``llm_classifier`` is provided and votes off-topic. Tests never inject one.
    """
    words = _words(query)

    if words & _MOVIE_TERMS:
        return TopicResult(on_topic=True, reason="movie_term")

    if words & _OFFTOPIC_TERMS:
        return TopicResult(on_topic=False, reason="offtopic_term", refusal=REFUSAL)

    if llm_classifier is not None:
        try:
            vote_on_topic = bool(llm_classifier(query))
        except Exception:  # noqa: BLE001 - a flaky cheap check must not block the turn
            vote_on_topic = True  # fail open: never let the optional check crash a turn
        if not vote_on_topic:
            return TopicResult(on_topic=False, reason="llm_offtopic", refusal=REFUSAL)

    # Ambiguous and no signal either way: default to serving (fail open toward
    # helpfulness; the downstream caps still bound cost).
    return TopicResult(on_topic=True, reason="default_allow")
