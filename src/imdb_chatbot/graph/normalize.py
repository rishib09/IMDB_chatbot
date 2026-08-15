"""Normalize the extractor's ``ParsedQuery`` against what the corpus can satisfy (#84).

The cheap extractor emits ``region='korean'`` (the corpus stores ISO codes),
``genres=['thriller', 'revenge']`` (the corpus stores ``'Thriller'``; "revenge"
is not a TMDB genre) and ``similar_to='gritty korean'`` (not a title). Any one
of those, applied as a filter, silently empties the candidate pool.

Principle: **an unrecognized constraint is DROPPED, never applied.** A
constraint the corpus cannot satisfy is strictly worse than no constraint.
The vocabulary is read from the corpus itself (its region codes, its genre
labels, its titles); only the demonym -> ISO map is a hand-typed table.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..movie_info import TitleLookup, build_title_lookup
from ..schemas import ParsedQuery
from ..text import normalize_text

# Nationality / country words -> ISO 3166-1 alpha-2, for the corpus's significant
# regions. Keys are ``normalize_text`` form. A bare code ("KR") needs no entry.
_DEMONYMS: dict[str, tuple[str, ...]] = {
    "US": ("american", "america", "usa", "united states", "hollywood"),
    "GB": ("british", "britain", "uk", "united kingdom", "england", "english", "scottish"),
    "KR": ("korean", "korea", "south korea", "south korean"),
    "FR": ("french", "france"),
    "IT": ("italian", "italy"),
    "DE": ("german", "germany"),
    "ES": ("spanish", "spain"),
    "CA": ("canadian", "canada"),
    "CH": ("swiss", "switzerland"),
    "AT": ("austrian", "austria"),
    "BE": ("belgian", "belgium"),
    "IE": ("irish", "ireland"),
    "AU": ("australian", "australia"),
    "AR": ("argentine", "argentinian", "argentina"),
    "MX": ("mexican", "mexico"),
    "NL": ("dutch", "netherlands", "holland"),
    "JP": ("japanese", "japan"),
    "BR": ("brazilian", "brazil"),
    "DK": ("danish", "denmark"),
    "HK": ("hong kong", "hongkong"),
    "RU": ("russian", "russia", "soviet"),
    "PT": ("portuguese", "portugal"),
    "SE": ("swedish", "sweden"),
    "GR": ("greek", "greece"),
    "CN": ("chinese", "china"),
    "IN": ("indian", "india", "bollywood", "hindi"),
    "TR": ("turkish", "turkey"),
    "IR": ("iranian", "iran", "persian"),
    "PL": ("polish", "poland"),
    "NO": ("norwegian", "norway"),
    "FI": ("finnish", "finland"),
    "IL": ("israeli", "israel"),
    "NZ": ("new zealand", "kiwi"),
    "TW": ("taiwanese", "taiwan"),
    "ZA": ("south african", "south africa"),
}
REGION_WORDS: dict[str, str] = {w: code for code, words in _DEMONYMS.items() for w in words}

# Colloquial genre names -> the corpus's TMDB label (``normalize_text`` form).
_GENRE_ALIASES = {"sci fi": "science fiction", "scifi": "science fiction"}


@dataclass(frozen=True)
class CorpusVocab:
    """What the corpus can actually filter on: region codes, genre labels, titles."""

    regions: frozenset[str]
    genres: dict[str, str]  # normalize_text(label) -> canonical label
    title_lookup: TitleLookup

    @classmethod
    def from_movies(cls, movies: Iterable[Any]) -> CorpusVocab:
        movies = list(movies)
        return cls(
            regions=frozenset(r for m in movies for r in m.regions),
            genres={normalize_text(g): g for m in movies for g in m.genres},
            # No substring tier: "gritty korean" must not resolve to a film "KORE".
            title_lookup=build_title_lookup(movies, substring=False),
        )

    def region(self, value: str | None) -> str | None:
        if value is None:
            return None
        key = normalize_text(value)
        code = REGION_WORDS.get(key, key.upper())
        return code if code in self.regions else None

    def genre_list(self, values: Iterable[str]) -> list[str]:
        keys = [normalize_text(v) for v in values]
        found = [self.genres.get(_GENRE_ALIASES.get(k, k)) for k in keys]
        return list(dict.fromkeys(g for g in found if g))

    def similar_to(self, value: str | None) -> str | None:
        return value if value and self.title_lookup(value) is not None else None


def normalize_parsed(parsed: ParsedQuery, vocab: CorpusVocab) -> ParsedQuery:
    """Map every corpus-facing constraint onto the corpus vocabulary; drop the rest."""
    return parsed.model_copy(
        update={
            "region": vocab.region(parsed.region),
            "genres": vocab.genre_list(parsed.genres),
            "exclude_genres": vocab.genre_list(parsed.exclude_genres),
            "similar_to": vocab.similar_to(parsed.similar_to),
        }
    )
