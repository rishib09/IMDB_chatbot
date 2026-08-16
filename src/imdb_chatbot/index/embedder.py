"""Text embedders and L2 normalization.

Three implementations sit behind one ``Embedder`` protocol:

- ``StubEmbedder``: deterministic, dependency-free (hash -> seeded vector). Every
  unit test uses it, so tests never need torch or a network.
- ``SentenceTransformerEmbedder``: the local model. ``sentence_transformers`` is
  imported LAZILY inside ``__init__`` so merely importing this module never pulls
  torch; you only pay for it when you actually construct the real embedder.
- ``OpenRouterEmbedder``: a hosted model (OpenAI ``text-embedding-3-*``) reached
  through OpenRouter's OpenAI-compatible ``/embeddings`` endpoint, on the SAME
  ``OPENROUTER_API_KEY`` the chat slots use. No second secret (PRD §5.1, §10).

Which one runs is config, not code: ``build_embedder`` resolves a named profile
out of ``config/models.yaml``. The local model stays the default and the fallback
for every profile, because a hosted embedder is a network dependency and L2
("LLM-free deterministic retrieval") must survive an outage.

Cosine similarity is done as inner product over L2-normalized vectors, so
``l2_normalize`` is applied to document vectors at build time AND to the query
vector at search time. Forgetting the query-side normalization is the classic
retrieval bug that a named regression test guards against.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Turns text into float32 vectors.

    ``encode`` returns a ``(len(texts), dim)`` float32 array. ``name`` is a
    filesystem-safe identifier that feeds the index version stamp, and it is also
    part of the embedding cache key, so different embedders never collide.
    """

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row so inner product equals cosine similarity.

    Zero vectors are left as zeros (their norm is replaced by 1 to avoid a
    divide-by-zero) rather than producing NaNs.
    """
    mat = np.asarray(vectors, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (mat / norms).astype(np.float32)


class StubEmbedder:
    """Deterministic, dependency-free embedder for tests.

    Each text is hashed with SHA-256; the digest seeds a NumPy generator that
    draws a ``dim``-length vector. Same text -> same vector, always, with no
    model download and no torch. The output is NOT normalized here; the build
    normalizes it (matching how the real embedder is treated).
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return f"stub{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big", signed=False)
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(self._dim).astype(np.float32)
        return out


class SentenceTransformerEmbedder:
    """The real embedder, backed by ``sentence-transformers`` (torch).

    ``sentence_transformers`` is imported lazily in ``__init__`` so importing this
    module stays torch-free. A clear error is raised if the ``embed`` extra is not
    installed.
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "SentenceTransformerEmbedder requires the 'embed' extra. "
                "Install it with: pip install -e '.[embed]'"
            ) from exc

        self._model_id = model
        self._model = SentenceTransformer(model)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        # Keep it filesystem-safe: drop any org prefix like "sentence-transformers/".
        return self._model_id.split("/")[-1]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_tokens(self) -> int:
        """Hard input window - everything past it is dropped, not compressed.

        256 for all-MiniLM-L6-v2. ``index.build.over_budget`` reads this to
        report which movies the window truncates (ticket #83).
        """
        return int(self._model.max_seq_length)

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Tokens per text as the window counts them (special tokens included)."""
        return [len(ids) for ids in self._model.tokenizer(texts)["input_ids"]]

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=False
        )
        return np.asarray(vectors, dtype=np.float32)


# Native output width of each hosted model. A profile may ask for FEWER
# dimensions (Matryoshka truncation); asking for more is a config error.
NATIVE_DIMS: dict[str, int] = {
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-3-small": 1536,
}

# Transient: worth an exponential-backoff retry. Anything else (401, 400, an
# unknown model slug) is a configuration error - retrying only burns time, so it
# fails immediately and loudly rather than half-filling an index.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenRouterEmbedder:
    """OpenAI embeddings served through OpenRouter's OpenAI-compatible endpoint.

    ``POST {base_url}/embeddings`` with ``{"model": ..., "input": [...],
    "dimensions": N}``. Billing is per INPUT token, so ``dimensions`` is free to
    choose and only trades index size against fidelity.

    ``name`` embeds the dimension count on purpose. It is the embedding cache's
    namespace and the index version stamp, so two profiles that differ only in
    ``dimensions`` MUST NOT share a name - otherwise a 512-d vector is read back
    into a 1024-d slot and the index is silently corrupt.

    ``httpx`` is imported lazily inside ``__init__`` (matching ``TMDBClient``) so
    importing this module stays free.
    """

    def __init__(
        self,
        model: str = "openai/text-embedding-3-large",
        *,
        dimensions: int | None = None,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        batch_size: int = 64,
        timeout: float = 120.0,
        retries: int = 4,
        backoff: float = 1.0,
        client: Any = None,
    ) -> None:
        import httpx

        native = NATIVE_DIMS.get(model)
        if dimensions is None:
            if native is None:
                raise RuntimeError(
                    f"Unknown embedding model {model!r}: set an explicit "
                    "'dimensions' in its config profile."
                )
            dimensions = native
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        if native is not None and dimensions > native:
            raise ValueError(
                f"{model} outputs at most {native} dimensions, asked for {dimensions}"
            )

        if api_key is None and client is None:
            from ..config import get_secret

            api_key = get_secret("OPENROUTER_API_KEY")

        self._model = model
        self._dim = int(dimensions)
        self._batch_size = max(1, int(batch_size))
        self._retries = max(1, int(retries))
        self._backoff = backoff
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        # Input tokens billed so far, as reported by the API. Lets a build report
        # what it actually cost instead of an estimate.
        self.input_tokens = 0

    @property
    def name(self) -> str:
        # Filesystem-safe, and dimension-scoped so the cache namespace is unique.
        return f"{self._model.split('/')[-1]}-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _post_batch(self, batch: list[str]) -> list[list[float]]:
        """One /embeddings call, retrying transient failures with backoff."""
        import httpx

        payload = {"model": self._model, "input": batch, "dimensions": self._dim}
        delay = self._backoff
        last: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                resp = self._client.post("/embeddings", json=payload)
                resp.raise_for_status()
                body = resp.json()
                self.input_tokens += int(body.get("usage", {}).get("prompt_tokens", 0))
                # OpenAI does not guarantee response order; "index" does.
                rows = sorted(body["data"], key=lambda row: row["index"])
                return [row["embedding"] for row in rows]
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRYABLE_STATUS:
                    raise
                last = exc
            except httpx.RequestError as exc:
                last = exc
            if attempt < self._retries:
                logger.warning(
                    "embeddings batch failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    self._retries,
                    delay,
                    last,
                )
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(
            f"OpenRouter embeddings failed after {self._retries} attempts: {last}"
        ) from last

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` in batches -> ``(len(texts), dim)`` float32.

        The returned width is checked against ``dim`` rather than trusted: a
        provider that silently ignores ``dimensions`` would otherwise produce an
        index whose vectors do not match the FAISS dimension it was built for.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors = self._post_batch(batch)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"asked for {len(batch)} embeddings, got {len(vectors)}"
                )
            for offset, vector in enumerate(vectors):
                if len(vector) != self._dim:
                    raise RuntimeError(
                        f"{self._model} returned {len(vector)} dimensions, "
                        f"expected {self._dim} - the provider ignored 'dimensions'."
                    )
                out[start + offset] = vector
        return out

    def close(self) -> None:
        self._client.close()


def build_embedder(
    profile: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> Embedder:
    """Construct the embedder named by ``profile`` in ``config/models.yaml``.

    ``profile`` defaults to ``embedder.default``. An unknown profile name, or a
    hosted profile that cannot be constructed (no key, no network), degrades to
    the configured local fallback with a warning rather than failing the build -
    the local model is retained regardless of which profile wins the sweep.
    """
    from ..config import load_models_config

    cfg = cfg or load_models_config()
    section = cfg.get("embedder") or {}
    profiles: dict[str, dict[str, Any]] = section.get("profiles") or {}
    fallback = section.get("fallback", "local")
    wanted = profile or section.get("default") or fallback

    spec = profiles.get(wanted)
    if spec is None:
        logger.warning(
            "unknown embedder profile %r; falling back to %r", wanted, fallback
        )
        spec = profiles.get(fallback)
    if spec is None:
        raise RuntimeError(
            f"No embedder profile {wanted!r} and no usable fallback {fallback!r} "
            "in config/models.yaml."
        )

    try:
        return _embedder_from_spec(spec)
    except Exception as exc:  # any construction failure degrades to the fallback
        if wanted == fallback:
            raise
        logger.warning(
            "embedder profile %r unavailable (%s); falling back to %r",
            wanted,
            exc,
            fallback,
        )
        return _embedder_from_spec(profiles[fallback])


def _embedder_from_spec(spec: dict[str, Any]) -> Embedder:
    """One profile dict -> a constructed embedder. ``kind`` picks the class."""
    kind = spec.get("kind", "sentence_transformer")
    if kind == "sentence_transformer":
        return SentenceTransformerEmbedder(spec["model"])
    if kind == "openrouter":
        return OpenRouterEmbedder(
            spec["model"],
            dimensions=spec.get("dimensions"),
            batch_size=spec.get("batch_size", 64),
        )
    if kind == "stub":
        return StubEmbedder(dim=spec.get("dimensions", 8))
    raise ValueError(f"Unknown embedder kind: {kind!r}")
