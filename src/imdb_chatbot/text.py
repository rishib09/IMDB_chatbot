"""The one text normalizer (ticket #77).

Titles, person names and user messages are all matched the same way: casefold,
fold every non-alphanumeric run to a single space, and trim. Five near-copies of
this used to live in ``persona``, ``movie_info``, ``retrieval.retrieve``,
``graph.gate4`` and ``memory.durable``; three of them were byte-equivalent and
now share this implementation.

Two deliberately did NOT move:

- ``gate4._norm`` only casefolds and strips - it must PRESERVE punctuation,
  because its output is embedded verbatim in machine-readable violation reasons
  (``hallucinated_title:spider-man``) and compared against raw candidate titles.
- ``durable.normalize_user_id`` uses ``.lower()`` and keeps punctuation, because
  its output is a FILENAME (``<user_id>.jsonl``) and the identity key stored in
  every triple; folding it would orphan existing durable memory on disk.

``tests/test_text.py`` pins both divergences so neither is "unified" by accident.
"""

from __future__ import annotations

import re

# One run of anything that is not an ASCII letter or digit -> one space.
_FOLD_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str | None) -> str:
    """Casefold, fold punctuation to spaces, collapse whitespace, trim.

    Folding punctuation is what makes matching robust to hyphen / space / period
    variants: "Bong Joon-ho" and "Bong Joon Ho" both normalize to "bong joon ho",
    "J.J. Abrams" to "j j abrams", and "Spider-Man: No Way Home" to
    "spider man no way home". ``None`` and non-alphanumeric input normalize to
    the empty string, and the result is idempotent under a second application.
    """
    return " ".join(_FOLD_RE.sub(" ", (text or "").casefold()).split())
