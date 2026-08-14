"""Adversary test for the one text normalizer (ticket #77).

The assumption under attack: *"normalization is idempotent and all five former
call sites agree"* - the premise that justified collapsing five hand-rolled
normalizers into ``imdb_chatbot.text.normalize_text``.

Each of the five originals is reproduced verbatim below as an ORACLE. A
generator hammers all six implementations with adversarial text (mixed case,
Unicode case-folding specials like ``ss``/``KELVIN SIGN``, combining marks,
non-breaking and line separators, emoji, punctuation-only and empty strings)
plus a random fuzz corpus, and the test asserts exactly which oracles agree.

The finding it encodes: THREE agree and were merged; TWO do not, and the test
names the input that breaks each. If someone later "finishes the job" by routing
``gate4`` or ``durable`` through ``normalize_text``, this test fails with the
counter-example rather than silently corrupting violation reasons or orphaning
durable-memory files on disk.
"""

from __future__ import annotations

import random
import re

from imdb_chatbot.movie_info import normalize_title
from imdb_chatbot.retrieval.retrieve import _normalize_name
from imdb_chatbot.text import normalize_text

# -- the five originals, reproduced verbatim as oracles -----------------------

_PERSONA_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_PERSONA_WS_RE = re.compile(r"\s+")
_TITLE_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def _oracle_persona(text: str) -> str:
    """``persona._normalize`` before #77."""
    lowered = (text or "").casefold()
    lowered = _PERSONA_PUNCT_RE.sub(" ", lowered)
    return _PERSONA_WS_RE.sub(" ", lowered).strip()


def _oracle_title(text: str) -> str:
    """``movie_info.normalize_title`` before #77."""
    return " ".join(_TITLE_PUNCT_RE.sub(" ", (text or "").casefold()).split())


def _oracle_name(text: str) -> str:
    """``retrieval.retrieve._normalize_name`` before #77 (byte-identical to above)."""
    return " ".join(_TITLE_PUNCT_RE.sub(" ", (text or "").casefold()).split())


def _oracle_gate4(text: str) -> str:
    """``graph.gate4._norm`` - punctuation-PRESERVING, deliberately not merged."""
    return text.strip().casefold()


def _oracle_user_id(text: str) -> str:
    """``memory.durable.normalize_user_id`` - ``.lower()``, deliberately not merged."""
    return text.strip().lower() or "rishi"


# -- the corpus ---------------------------------------------------------------

# Hand-picked inputs that have historically broken naive normalizers.
ADVERSARIAL = [
    "",
    " ",
    "...",
    "!?!",
    "Spider-Man: No Way Home",
    "Bong Joon-ho",
    "J.J. Abrams",
    "  leading and trailing  ",
    "tabs\tand\nnewlines",
    "vertical\x0btab\x0cform\rfeed",
    "non\xa0breaking\xa0space",
    "line separator",
    "Straße",  # casefold("ß") == "ss"; lower("ß") == "ß"
    "ẞSZETT",  # capital sharp S
    "İstanbul",  # dotted capital I
    "Kelvin",  # KELVIN SIGN casefolds to "k"
    "café",  # combining acute accent
    "Café",
    "\U0001f600 emoji \U0001f600",
    "1999",
    "R2-D2",
    "don't stop",
    "_underscore_",
    "ALLCAPS",
    "M*A*S*H",
    "你好",  # non-Latin script: folds away entirely
]

_FUZZ_ALPHABET = list("abAB09 -_.:'\"\t\n\xa0 ßİ́\U0001f600ijKẞé")


def _fuzz_corpus(n: int = 4000, seed: int = 77) -> list[str]:
    """A deterministic random corpus over the nastiest characters we know of."""
    rng = random.Random(seed)
    return [
        "".join(rng.choice(_FUZZ_ALPHABET) for _ in range(rng.randint(0, 14)))
        for _ in range(n)
    ]


def _corpus() -> list[str]:
    return ADVERSARIAL + _fuzz_corpus()


# -- the invariants -----------------------------------------------------------


def test_normalization_is_idempotent() -> None:
    """normalize(normalize(x)) == normalize(x) for every input we can throw at it.

    A normalizer that is not a fixed point silently makes cache keys and index
    keys depend on how many times the value has been round-tripped.
    """
    for text in _corpus():
        once = normalize_text(text)
        assert normalize_text(once) == once, f"not idempotent for {text!r}"


def test_normalized_output_has_no_punctuation_or_stray_whitespace() -> None:
    """The post-condition every caller relies on: bare tokens, single-spaced."""
    for text in _corpus():
        out = normalize_text(text)
        assert out == out.strip()
        assert "  " not in out
        assert re.fullmatch(r"[a-z0-9]+(?: [a-z0-9]+)*|", out), f"{text!r} -> {out!r}"


def test_the_three_merged_call_sites_agree_with_the_unified_helper() -> None:
    """persona / movie_info / retrieve normalized identically - hence one helper.

    Attacks the merge itself: if ``normalize_text`` ever drifts from any of the
    three implementations it replaced, this names the input that diverged.
    """
    for text in _corpus():
        expected = normalize_text(text)
        assert _oracle_persona(text) == expected, f"persona diverged on {text!r}"
        assert _oracle_title(text) == expected, f"movie_info diverged on {text!r}"
        assert _oracle_name(text) == expected, f"retrieve diverged on {text!r}"


def test_live_call_sites_are_the_unified_helper() -> None:
    """The exported names really are one implementation, not three copies again."""
    assert normalize_title is normalize_text
    assert _normalize_name is normalize_text


def test_gate4_normalizer_must_not_be_merged() -> None:
    """gate4 PRESERVES punctuation - merging it would rewrite violation reasons.

    ``check_titles_exist`` emits ``hallucinated_title:<_norm(title)>`` as a
    machine-readable token and compares it against raw candidate titles. Folding
    punctuation would turn "spider-man" into "spider man" on both sides of a
    contract other systems read.
    """
    breaker = "Spider-Man: No Way Home"
    assert _oracle_gate4(breaker) == "spider-man: no way home"
    assert normalize_text(breaker) == "spider man no way home"
    assert _oracle_gate4(breaker) != normalize_text(breaker)

    # And it is not even idempotent-compatible: gate4 keeps interior whitespace.
    assert _oracle_gate4("a  b") == "a  b" != normalize_text("a  b")


def test_user_id_normalizer_must_not_be_merged() -> None:
    """durable user ids are FILENAMES: casefolding or folding punctuation orphans data.

    ``normalize_user_id`` output becomes ``<user_id>.jsonl`` on disk and the
    identity key inside every stored triple. ``casefold`` and punctuation folding
    both change that key for real inputs, which would silently strand a user's
    existing durable memory.
    """
    # casefold("ß") == "ss" but lower("ß") == "ß": same person, two files.
    assert _oracle_user_id("Straße") == "straße"
    assert normalize_text("Straße") == "strasse"

    # Punctuation folding collapses distinct ids into one file.
    assert _oracle_user_id("rishi.b") == "rishi.b"
    assert normalize_text("rishi.b") == "rishi b"

    # The empty-input default is part of the contract too.
    assert _oracle_user_id("   ") == "rishi"
    assert normalize_text("   ") == ""
