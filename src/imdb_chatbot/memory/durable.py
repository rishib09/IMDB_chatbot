"""Durable cross-session memory: passive prefs + widgets + minimal precedence.

This is PRD memory tier 3 (section 6): the bot remembers a user ACROSS sessions
via EXPLICIT feedback, not by inferring silently. The unit of memory is a
``Triple`` - ``(user_id, relation, object)`` with provenance and a timestamp -
appended to an append-only log. Widget clicks (thumbs-up/down, "Seen it", "Not
interested") are the only writers; nothing is inferred behind the user's back.

The layering (the "rungs" of the passive ladder):

- rung 2 (passive injection): at session start, durable triples are loaded and
  folded into the session's constraint state - durable EXCLUDED/DISLIKED become
  standing filters, LIKED become preference hints, WATCHED/REJECTED join
  ``shown_movies`` so they never repeat. See ``known_preferences``.
- rung 3 (minimal precedence): a LIVE request always outranks a remembered
  preference. If this turn's parsed query contradicts a durable EXCLUDED (durable
  EXCLUDED(horror) but the user now asks for horror), the exclusion is suspended
  FOR THIS TURN so the bot complies - but the durable triple is kept unchanged
  (the user did not un-remember their standing dislike). No clarifying question
  is asked here; the interactive-question rung is a later ticket. See
  ``resolve_collision``.

Two stores implement the ``MemoryStore`` protocol:

- ``LocalJsonlStore`` - append-only JSONL per user under a directory. Fully
  deterministic, no network - used by ALL tests.
- ``HFDatasetStore`` - syncs the same per-user JSONL to a Hugging Face Dataset
  repo. ``huggingface_hub`` is lazy-imported inside the methods and a clear error
  is raised without a token. NEVER exercised by tests (no network).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..schemas import MovieRecord
from .session import ConversationState

# The closed relation vocabulary. Kept small and explicit (PRD section 6): each
# relation maps to exactly one passive-layer effect in ``known_preferences``.
Relation = Literal["LIKED", "DISLIKED", "WATCHED", "REJECTED", "EXCLUDED"]

# Widget affordances the UI exposes, mapped 1:1 to a relation below.
WidgetKind = Literal["thumbs_up", "thumbs_down", "seen_it", "not_interested"]

# Retention cap per user: keep at most this many triples, dropping the OLDEST
# first. Triples persist indefinitely otherwise (no decay / rolling window in
# this ticket - that is a later refinement).
MAX_TRIPLES_PER_USER = 200

# thumbs-up -> LIKED, thumbs-down -> DISLIKED, "Seen it" -> WATCHED,
# "Not interested" -> REJECTED (per ticket spec).
_WIDGET_RELATION: dict[str, Relation] = {
    "thumbs_up": "LIKED",
    "thumbs_down": "DISLIKED",
    "seen_it": "WATCHED",
    "not_interested": "REJECTED",
}


def normalize_user_id(name: str | None) -> str:
    """Name-field identity: ``user_id`` is a lowercased, trimmed name.

    Defaults to ``"rishi"`` when nothing usable is supplied.
    """
    if name is None:
        return "rishi"
    cleaned = name.strip().lower()
    return cleaned or "rishi"


class Triple(BaseModel):
    """One durable memory fact: ``(user_id, relation, object)`` + provenance/ts.

    ``object`` is a free-form string: a movie id (as text) for WATCHED/REJECTED,
    or a genre / actor / title for EXCLUDED / DISLIKED / LIKED. ``provenance``
    records how the triple was created (e.g. ``"widget:thumbs_up"``) so the audit
    trail shows the fact came from explicit user feedback.
    """

    user_id: str
    relation: Relation
    object: str
    provenance: str = "unknown"
    ts: float = Field(default_factory=lambda: time.time())


@runtime_checkable
class MemoryStore(Protocol):
    """Append-only durable store keyed by ``user_id``.

    Implementations persist triples per user and expose the whole user set. The
    contract is deliberately tiny (load / append / all_users): the passive layer
    only ever needs to read a user's full history and append one triple at a time.
    """

    def load(self, user_id: str) -> list[Triple]:
        """Return all durable triples for ``user_id`` (oldest first)."""
        ...

    def append(self, user_id: str, triple: Triple) -> None:
        """Append one triple to ``user_id``'s durable log."""
        ...

    def all_users(self) -> list[str]:
        """Return every ``user_id`` that has at least one triple."""
        ...


class LocalJsonlStore:
    """Append-only JSONL store: one ``<dir>/<user_id>.jsonl`` file per user.

    Deterministic and network-free - the store every test uses. Each line is one
    serialized ``Triple``. Retention (``MAX_TRIPLES_PER_USER``) is enforced on
    append by rewriting the file with the newest triples when the cap is hit.
    """

    def __init__(self, dir: str | Path) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str) -> Path:
        return self.dir / f"{normalize_user_id(user_id)}.jsonl"

    def load(self, user_id: str) -> list[Triple]:
        path = self._path(user_id)
        if not path.exists():
            return []
        triples: list[Triple] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            triples.append(Triple.model_validate_json(line))
        return triples

    def append(self, user_id: str, triple: Triple) -> None:
        path = self._path(user_id)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(triple.model_dump_json() + "\n")
        self._enforce_retention(user_id)

    def all_users(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.jsonl"))

    def _enforce_retention(self, user_id: str) -> None:
        """Cap the log at ``MAX_TRIPLES_PER_USER``, dropping the oldest first."""
        path = self._path(user_id)
        if not path.exists():
            return
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) <= MAX_TRIPLES_PER_USER:
            return
        kept = lines[-MAX_TRIPLES_PER_USER:]
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")


class HFDatasetStore:
    """Hugging Face Dataset-backed store: per-user JSONL pushed/pulled from a repo.

    Mirrors ``LocalJsonlStore`` semantics but syncs each user's JSONL to a HF
    Dataset repo. ``huggingface_hub`` is imported lazily inside each method so the
    dependency is only needed when this store is actually used, and a clear error
    is raised when no token is available. This store is NEVER exercised by tests
    (it would require network + credentials).
    """

    def __init__(self, repo_id: str, token: str | None = None) -> None:
        self.repo_id = repo_id
        self.token = token

    def _require_token(self) -> str:
        if not self.token:
            raise RuntimeError(
                "HFDatasetStore requires a Hugging Face token. Pass token=... or set "
                "HF_TOKEN in the environment (see config.get_secret). Use LocalJsonlStore "
                "for local / offline / test runs."
            )
        return self.token

    def _api(self):  # pragma: no cover - network path, never hit by tests
        token = self._require_token()
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed; install it to use HFDatasetStore."
            ) from exc
        return HfApi(token=token)

    def _download(self, user_id: str) -> str | None:  # pragma: no cover - network path
        token = self._require_token()
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed; install it to use HFDatasetStore."
            ) from exc
        from huggingface_hub.utils import EntryNotFoundError

        try:
            local = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"{normalize_user_id(user_id)}.jsonl",
                repo_type="dataset",
                token=token,
            )
        except EntryNotFoundError:
            return None
        return Path(local).read_text(encoding="utf-8")

    def load(self, user_id: str) -> list[Triple]:  # pragma: no cover - network path
        text = self._download(user_id)
        if not text:
            return []
        triples: list[Triple] = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                triples.append(Triple.model_validate_json(line))
        return triples

    def append(self, user_id: str, triple: Triple) -> None:  # pragma: no cover - network
        existing = self.load(user_id)
        existing.append(triple)
        if len(existing) > MAX_TRIPLES_PER_USER:
            existing = existing[-MAX_TRIPLES_PER_USER:]
        payload = "\n".join(t.model_dump_json() for t in existing) + "\n"
        api = self._api()
        api.upload_file(
            path_or_fileobj=payload.encode("utf-8"),
            path_in_repo=f"{normalize_user_id(user_id)}.jsonl",
            repo_id=self.repo_id,
            repo_type="dataset",
        )

    def all_users(self) -> list[str]:  # pragma: no cover - network path
        api = self._api()
        files = api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        return sorted(Path(f).stem for f in files if f.endswith(".jsonl"))


def widget_to_triple(user_id: str, kind: str, movie: MovieRecord) -> Triple:
    """Map a widget click to the durable ``Triple`` it records.

    thumbs-up -> LIKED, thumbs-down -> DISLIKED, "Seen it" -> WATCHED,
    "Not interested" -> REJECTED. WATCHED / REJECTED record the movie's
    ``tmdb_id`` (a no-repeat / shown fact); LIKED / DISLIKED record the title
    (a taste fact). Provenance is stamped ``widget:<kind>``.
    """
    if kind not in _WIDGET_RELATION:
        raise ValueError(
            f"Unknown widget kind {kind!r}; expected one of {sorted(_WIDGET_RELATION)}."
        )
    relation = _WIDGET_RELATION[kind]
    obj = str(movie.tmdb_id) if relation in ("WATCHED", "REJECTED") else movie.title
    return Triple(
        user_id=normalize_user_id(user_id),
        relation=relation,
        object=obj,
        provenance=f"widget:{kind}",
    )


class KnownPreferences(BaseModel):
    """The passive-layer inputs derived from a user's durable triples.

    A ``ConversationState``-ish bundle: everything ``known_preferences`` extracts
    from durable memory so it can seed a fresh session. ``exclude_genres`` come
    from durable EXCLUDED (and DISLIKED genre-like objects); ``shown_movies`` from
    WATCHED / REJECTED movie ids; ``liked`` are LIKED taste hints. It keeps the
    raw ``triples`` too so ``resolve_collision`` can reason about provenance.
    """

    user_id: str
    exclude_genres: list[str] = []
    exclude_actors: list[str] = []
    liked: list[str] = []
    shown_movies: set[int] = Field(default_factory=set)
    triples: list[Triple] = []

    def apply_to(self, conversation: ConversationState) -> None:
        """Fold these durable preferences into a session's constraint state."""
        conversation.add_exclusions(
            actors=self.exclude_actors,
            genres=self.exclude_genres,
        )
        conversation.mark_shown(self.shown_movies)
        if self.liked:
            merged = [*conversation.preferences.get("liked", []), *self.liked]
            conversation.preferences["liked"] = list(dict.fromkeys(merged))


def known_preferences(user_id: str, store: MemoryStore) -> KnownPreferences:
    """Load a user's durable triples and produce the passive-layer inputs.

    Fed into a session at start (``KnownPreferences.apply_to``). The relation ->
    effect mapping is the passive rung of the ladder:

    - EXCLUDED  -> a standing genre/actor filter,
    - DISLIKED  -> also a standing exclusion (a strong negative taste signal),
    - LIKED     -> a preference hint (surfaced to the generator, not a hard gate),
    - WATCHED   -> a ``shown`` movie (already seen, do not repeat),
    - REJECTED  -> a ``shown`` movie (explicitly turned down, do not repeat).
    """
    uid = normalize_user_id(user_id)
    triples = store.load(uid)

    def objects(*relations: str) -> list[str]:
        """The triple objects for those relations, de-duplicated, oldest first."""
        return list(dict.fromkeys(t.object for t in triples if t.relation in relations))

    shown = {
        movie_id
        for t in triples
        if t.relation in ("WATCHED", "REJECTED")
        if (movie_id := _as_movie_id(t.object)) is not None
    }

    return KnownPreferences(
        user_id=uid,
        exclude_genres=objects("EXCLUDED", "DISLIKED"),
        liked=objects("LIKED"),
        shown_movies=shown,
        triples=list(triples),
    )


def _as_movie_id(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_collision(
    durable_exclusions: Iterable[str],
    requested: Iterable[str],
) -> list[str]:
    """Minimal precedence: a live request outranks a remembered exclusion.

    Given the durable EXCLUDED objects and what the live parsed query asks FOR
    (e.g. its requested genres), return the exclusions that are EFFECTIVE for
    this turn: any durable exclusion the user is now explicitly asking for is
    suspended (dropped) so the bot complies. The comparison is case-insensitive.

    This is turn-local only: it returns the effective set to hand to retrieval and
    NEVER mutates the store, so the durable triple survives unchanged for future
    turns. No clarifying question is asked (that is a later ticket).
    """
    requested_norm = {item.strip().lower() for item in requested if item and item.strip()}
    effective: list[str] = []
    for exclusion in durable_exclusions:
        if exclusion and exclusion.strip().lower() in requested_norm:
            # The user is explicitly asking for this now - comply this turn.
            continue
        effective.append(exclusion)
    return effective
