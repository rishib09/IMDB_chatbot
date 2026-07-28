"""Per-region certificate normalization - the region adapter registry.

Each supported region contributes ONE YAML file under ``config/certificates/``
(e.g. ``us.yaml``, ``in.yaml``). The core transform in ``tmdb.py`` never learns
region-specific ratings: it asks this registry to normalize a raw certificate
for a region code. Adding a region is therefore a new YAML plus registering its
code here - no change to the mapping logic itself.

A YAML file looks like::

    system: MPAA
    map:
      G: ALL
      "PG-13": TEEN

Normalization contract:

- Known raw certificate -> its bucket ({ALL, TEEN, MATURE, ADULT}).
- Unknown/blank certificate -> ``None`` plus a logged warning (never a hard
  failure). The certificate_system is still reported so callers know which
  ratings body governed the (unmapped) row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from ..config import CONFIG_DIR

logger = logging.getLogger(__name__)

CERT_DIR = CONFIG_DIR / "certificates"

# Region code (ISO 3166-1 alpha-2) -> YAML filename. Adding a region means a new
# YAML and a new entry here; the transform code stays untouched.
REGION_YAML: dict[str, str] = {
    "US": "us.yaml",
    "IN": "in.yaml",
}


@dataclass(frozen=True)
class CertificateMap:
    """One region's rating system and its raw -> bucket lookup."""

    region: str
    system: str
    buckets: dict[str, str]

    def normalize(self, raw: str | None) -> str | None:
        """Return the bucket for ``raw``, or ``None`` (with a warning) if unknown."""
        if raw is None:
            return None
        key = raw.strip()
        if not key:
            return None
        bucket = self.buckets.get(key)
        if bucket is None:
            logger.warning(
                "Unknown %s certificate %r for region %s; certificate_norm=None",
                self.system,
                raw,
                self.region,
            )
        return bucket


@cache
def load_certificate_map(region: str) -> CertificateMap | None:
    """Load and cache the certificate adapter for ``region`` (case-insensitive).

    Returns ``None`` for a region with no registered YAML - the caller then
    leaves certificate fields unset rather than failing.
    """
    code = region.upper()
    filename = REGION_YAML.get(code)
    if filename is None:
        return None
    path: Path = CERT_DIR / filename
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return CertificateMap(
        region=code,
        system=str(doc.get("system", "")),
        buckets=dict(doc.get("map", {})),
    )


def registered_regions() -> list[str]:
    """The region codes with a certificate adapter available."""
    return sorted(REGION_YAML)
