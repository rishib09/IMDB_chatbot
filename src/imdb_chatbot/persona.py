"""Maya - the chatbot persona - and the intent gate (ticket #44).

Two responsibilities:

1. **Identity.** ``generator_system_prompt`` returns Maya's system prompt so real
   recommendations speak in her voice (wired into the generator slot).

2. **Intent routing.** ``classify_intent`` is a deterministic, dependency-free
   classifier that decides whether a message is small talk, a meta/help question,
   or a genuine search. Non-search intents are answered by ``persona_reply`` with
   a fixed persona message - NO retrieval and NO LLM - so "hi" or "what is the
   purpose of this chatbot?" get Maya's introduction instead of five films whose
   titles happen to contain "Hi".

The classifier is intentionally conservative: a greeting is recognised only when
the WHOLE message is greeting/thanks words (so "high school movies" stays a
search), and meta intent matches a small set of unambiguous phrases (so "help me
find a horror movie" stays a search while a bare "help" is meta).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from .config import load_persona
from .schemas import RecommendationSet


class Intent(str, Enum):
    """What the user's message is asking for."""

    GREETING = "greeting"  # small talk: hi / hello / thanks / bye
    META = "meta"  # who/what are you, purpose, capabilities, help
    CHITCHAT = "chitchat"  # conversational (how are you, tell me a joke) -> LLM reply
    MOVIE_QUESTION = "movie_question"  # factual question about a film -> answer from corpus
    SEARCH = "search"  # a genuine movie request -> run the graph


# Whole-message greeting/thanks vocabulary. A message counts as GREETING only if
# every one of its words is in here (keeps "high school movies" a search).
_GREETING_WORDS = {
    "hi",
    "hii",
    "hiya",
    "hey",
    "heya",
    "hello",
    "helo",
    "yo",
    "howdy",
    "sup",
    "greetings",
    "thanks",
    "thank",
    "thankyou",
    "thx",
    "ty",
    "cheers",
    "bye",
    "goodbye",
    "morning",
    "afternoon",
    "evening",
    "good",
    "there",
    "you",
}

# Unambiguous meta / help phrases (substring match on the normalized message).
_META_PHRASES = (
    "who are you",
    "what are you",
    "who is this",
    "what is this",
    "what can you do",
    "what do you do",
    "what can i ask",
    "what can i do",
    "your purpose",
    "the purpose",
    "what is the purpose",
    "how do you work",
    "how does this work",
    "how do i use",
    "what is maya",
    "who is maya",
    "tell me about yourself",
    "about yourself",
)

# Bare messages that are meta on their own.
_META_EXACT = {"help", "?", "??"}

# Conversational / small-talk questions that are NOT movie searches. Matched as
# substrings on the normalized message; these get a smart LLM reply (Maya) rather
# than being run through retrieval. Deliberately a curated list - the offer in
# the reply lets the user redirect if we misjudge.
_CHITCHAT_PHRASES = (
    "how are you",
    "how are u",
    "how r you",
    "how are things",
    "how is it going",
    "how s it going",
    "hows it going",
    "how is everything",
    "how do you do",
    "how have you been",
    "how you doing",
    "how is your day",
    "hows your day",
    "how s your day",
    "whats up",
    "what s up",
    "wassup",
    "hows life",
    "how s life",
    "how is life",
    "tell me a joke",
    "say something funny",
    "do you like movies",
    "whats your favorite",
    "what s your favorite",
    "favourite movie",
    "favorite movie",
    "are you a robot",
    "are you human",
    "are you real",
    "are you ok",
    "whats new",
    "what s new",
    "nice to meet you",
    "good to see you",
    "how is your",
)

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")

# Factual questions ABOUT a specific film (cast, plot, director, "heard of X").
# Deliberately specific so recommendation requests ("what's a good thriller")
# are NOT caught. Matched on the normalized (punctuation-stripped) message.
_MOVIE_QUESTION_RES = [
    re.compile(r"\bwho (is|are|was) (the )?(cast|actors?|stars?|director|writer|producer)\b"),
    re.compile(r"\bwho (directed|wrote|produced|stars? in|acted in|plays|played)\b"),
    re.compile(
        r"\bwhat (is|s|are) (the )?(plot|cast|story|synopsis|runtime|rating|genre|release)\b"
    ),
    re.compile(r"\bwhat (is|s) .+ about\b"),
    re.compile(r"\bwhen (was|did) .+ (released|come out|made)\b"),
    re.compile(r"\b(have|did) you (ever )?(hear|heard|see|seen) (of|about)\b"),
    re.compile(r"\bdo you know (the movie|the film|about the movie|about the film)\b"),
    re.compile(r"\btell me about the (movie|film)\b"),
]


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    lowered = (text or "").casefold()
    lowered = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", lowered).strip()


def classify_intent(query: str) -> Intent:
    """Classify a raw user message as GREETING, META, or SEARCH (deterministic)."""
    normalized = _normalize(query)
    if not normalized:
        # An empty / punctuation-only message: treat as a greeting so Maya
        # introduces herself rather than searching for nothing.
        return Intent.GREETING

    if normalized in _META_EXACT:
        return Intent.META

    words = normalized.split()
    if words and all(word in _GREETING_WORDS for word in words):
        return Intent.GREETING

    if any(phrase in normalized for phrase in _META_PHRASES):
        return Intent.META

    if any(phrase in normalized for phrase in _CHITCHAT_PHRASES):
        return Intent.CHITCHAT

    if any(pattern.search(normalized) for pattern in _MOVIE_QUESTION_RES):
        return Intent.MOVIE_QUESTION

    return Intent.SEARCH


def persona_reply(intent: Intent, persona: dict[str, Any] | None = None) -> RecommendationSet:
    """The deterministic, LLM-free reply for a non-search intent.

    Returns an empty-picks ``RecommendationSet`` whose prose is Maya's greeting
    (GREETING) or purpose explanation (META). Raises for SEARCH - callers must
    route search intents to the graph, not here.
    """
    persona = persona or load_persona()
    if intent is Intent.GREETING:
        prose = str(persona.get("greeting", "")).strip()
    elif intent is Intent.META:
        prose = str(persona.get("purpose", "")).strip()
    else:
        raise ValueError(f"persona_reply is only for non-search intents, got {intent}")
    return RecommendationSet(picks=[], prose=prose)


def generator_system_prompt(persona: dict[str, Any] | None = None) -> str:
    """Maya's system prompt, prepended to the generator slot."""
    persona = persona or load_persona()
    return str(persona.get("system_prompt", "")).strip()


def chat_system_prompt(persona: dict[str, Any] | None = None) -> str:
    """System prompt for the cheap LLM that answers CHITCHAT messages."""
    persona = persona or load_persona()
    return str(persona.get("chat_system_prompt", "")).strip()


def chitchat_fallback(persona: dict[str, Any] | None = None) -> str:
    """Deterministic conversational reply used when no LLM is available."""
    persona = persona or load_persona()
    return str(persona.get("chitchat", "")).strip()


def persona_version(persona: dict[str, Any] | None = None) -> str:
    """The persona artifact version (recorded as the trace prompt_version)."""
    persona = persona or load_persona()
    return str(persona.get("version", "persona-v1"))
