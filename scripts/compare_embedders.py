"""Compare embedder profiles on a SLICE of the corpus (ticket #76 / B11).

Builds one index per profile over the same slice, runs the same probe queries
through the same ``HybridRetriever``, and prints hit rate, index size, build
time, and query latency side by side.

This is a scaffold, not the sweep. The probes are hand-written and the metric is
a crude hit@k, because B3 (#68) - the golden eval set - does not exist yet. When
it does, point ``imdb_chatbot.eval`` at the built index dirs instead and delete
the probe list here.

**Cost control.** ``--slice`` is REQUIRED and there is no "all" value: a hosted
profile bills per input token, and the full 46k corpus is a ~$1.36 operation
that a comparison run must never trigger by accident. The slice is the top-N
movies by ``vote_count`` so hand-written probes about well-known films are
actually answerable.

Usage (hosted profiles need the key, so run under dotenvx):

    npx @dotenvx/dotenvx run -f .env -- .venv/Scripts/python.exe \
        scripts/compare_embedders.py --slice 300 \
        --profiles local,openai-3-large-1024

    .venv/Scripts/python.exe scripts/compare_embedders.py \
        --slice 50 --profiles local        # offline, no spend
"""

from __future__ import annotations

import argparse
import logging
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from imdb_chatbot.config import load_models_config
from imdb_chatbot.index.build import build_index, load_index
from imdb_chatbot.index.cache import EmbeddingCache
from imdb_chatbot.index.embedder import Embedder, build_embedder
from imdb_chatbot.retrieval.retrieve import HybridRetriever
from imdb_chatbot.schemas import MovieRecord
from imdb_chatbot.store import TraceStore

logger = logging.getLogger("compare_embedders")

# Hand-written probes: (query, titles that should surface). Deliberately weighted
# toward the vibe / similar-to queries where a better bi-encoder is *expected* to
# help - the ticket's own expectation-setting says metadata and ranking failures
# will not move. Replace wholesale with B3's golden set.
PROBES: list[tuple[str, list[str]]] = [
    ("gritty korean revenge thriller", ["Oldboy", "I Saw the Devil", "The Chaser"]),
    ("mind-bending sci-fi about memory and dreams", ["Inception", "Memento", "Paprika"]),
    ("animated film about growing up", ["Toy Story", "Inside Out", "Up"]),
    ("heist crew pulls off an impossible robbery", ["Ocean's Eleven", "Heat", "The Italian Job"]),
    ("slow-burn haunted house horror", ["The Shining", "Hereditary", "The Conjuring"]),
    ("courtroom drama about a wrongful conviction", ["12 Angry Men", "A Few Good Men"]),
    ("wartime survival story", ["Dunkirk", "1917", "Saving Private Ryan"]),
    ("romantic comedy set in New York", ["When Harry Met Sally...", "Annie Hall"]),
]


@dataclass
class ProfileResult:
    """Everything measured for one embedder profile on the slice."""

    profile: str
    embedder_name: str
    dim: int
    build_seconds: float
    index_bytes: int
    hit_rate: float
    hits: int
    probes: int
    latency_p50_ms: float
    latency_p95_ms: float
    input_tokens: int
    cost_usd: float
    extrapolated_corpus_mb: float
    extrapolated_corpus_usd: float


def top_by_votes(store: TraceStore, n: int) -> list[MovieRecord]:
    """The ``n`` most-voted movies. Popular titles make the probes answerable."""
    movies = store.read_all_movies()
    movies.sort(key=lambda m: (-(m.vote_count or 0), m.tmdb_id))
    return movies[:n]


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def faiss_bytes_for(count: int, dim: int) -> int:
    """A flat float32 index is exactly n x dim x 4 bytes of vectors."""
    return count * dim * 4


def run_probes(
    retriever: HybridRetriever,
    probes: list[tuple[str, list[str]]],
) -> tuple[int, list[float]]:
    """Run every probe; return (hits, per-query latencies in ms).

    A probe hits if ANY expected title appears in the returned candidates. Crude
    on purpose - it is a smoke signal for a slice, not a Recall@5 measurement.
    """
    hits = 0
    latencies: list[float] = []
    for query, expected in probes:
        start = time.perf_counter()
        results = retriever.retrieve(query)
        latencies.append((time.perf_counter() - start) * 1000.0)
        titles = {r.title.casefold() for r in results}
        if any(want.casefold() in titles for want in expected):
            hits += 1
    return hits, latencies


def measure(
    profile: str,
    slice_movies: list[MovieRecord],
    *,
    work_root: Path,
    cache_path: Path | None,
    corpus_size: int,
    price_per_mtok: float,
) -> ProfileResult:
    """Build an index for one profile over the slice and probe it."""
    embedder: Embedder = build_embedder(profile)

    slice_db = work_root / f"{profile}_slice.sqlite"
    slice_store = TraceStore(slice_db)
    cache = EmbeddingCache(cache_path) if cache_path else None
    try:
        for movie in slice_movies:
            slice_store.write_movie(movie)

        started = time.perf_counter()
        result = build_index(
            slice_store,
            embedder,
            dataset_version=f"slice{len(slice_movies)}",
            out_root=work_root / "index",
            cache=cache,
            flip_pointer=False,
        )
        build_seconds = time.perf_counter() - started

        loaded = load_index(result.out_dir)
        retriever = HybridRetriever.from_store(loaded, embedder, slice_store)
        hits, latencies = run_probes(retriever, PROBES)
        if retriever.dense_failures:
            logger.warning(
                "%s: %d probe(s) ran sparse-only (dense side failed)",
                profile,
                retriever.dense_failures,
            )
    finally:
        if cache is not None:
            cache.close()
        slice_store.close()
        slice_db.unlink(missing_ok=True)

    # Only a hosted embedder reports billed tokens; a local one stays at 0.
    input_tokens = int(getattr(embedder, "input_tokens", 0))
    cost = input_tokens / 1_000_000 * price_per_mtok
    per_movie_tokens = input_tokens / len(slice_movies) if slice_movies else 0.0

    ordered = sorted(latencies)
    return ProfileResult(
        profile=profile,
        embedder_name=embedder.name,
        dim=embedder.dim,
        build_seconds=build_seconds,
        index_bytes=dir_bytes(result.out_dir),
        hit_rate=hits / len(PROBES) if PROBES else 0.0,
        hits=hits,
        probes=len(PROBES),
        latency_p50_ms=statistics.median(ordered) if ordered else 0.0,
        latency_p95_ms=ordered[max(0, round(0.95 * len(ordered)) - 1)] if ordered else 0.0,
        input_tokens=input_tokens,
        cost_usd=cost,
        extrapolated_corpus_mb=faiss_bytes_for(corpus_size, embedder.dim) / 1e6,
        extrapolated_corpus_usd=per_movie_tokens * corpus_size / 1_000_000 * price_per_mtok,
    )


def _price_for(profile: str, cfg: dict) -> float:
    """USD per 1M input tokens for a profile's model (0.0 for a local model)."""
    spec = (cfg.get("embedder") or {}).get("profiles", {}).get(profile) or {}
    if spec.get("kind") != "openrouter":
        return 0.0
    return float(cfg.get("pricing", {}).get(spec.get("model"), {}).get("input", 0.0))


def format_table(results: list[ProfileResult], slice_size: int, corpus_size: int) -> str:
    header = (
        f"{'profile':<22} {'dim':>5} {'hit':>7} {'build s':>8} "
        f"{'idx MB':>7} {'p50 ms':>7} {'p95 ms':>7} {'tokens':>9} {'USD':>8}"
    )
    lines = [
        f"slice={slice_size} movies (top by vote_count), probes={len(PROBES)}",
        header,
        "-" * len(header),
    ]
    for r in results:
        lines.append(
            f"{r.profile:<22} {r.dim:>5} {r.hits:>3}/{r.probes:<3} "
            f"{r.build_seconds:>8.1f} {r.index_bytes / 1e6:>7.2f} "
            f"{r.latency_p50_ms:>7.1f} {r.latency_p95_ms:>7.1f} "
            f"{r.input_tokens:>9,} {r.cost_usd:>8.4f}"
        )
    lines += ["", f"Extrapolated to the full corpus ({corpus_size:,} movies):"]
    for r in results:
        lines.append(
            f"  {r.profile:<22} vectors {r.extrapolated_corpus_mb:>7.1f} MB   "
            f"one-time embed ${r.extrapolated_corpus_usd:.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare_embedders")
    parser.add_argument(
        "--slice",
        type=int,
        required=True,
        help="How many movies to index (top N by vote_count). REQUIRED - a hosted "
        "profile bills per token, so the whole corpus is never the default.",
    )
    parser.add_argument(
        "--profiles",
        default="local",
        help="Comma-separated embedder profiles from config/models.yaml.",
    )
    parser.add_argument(
        "--db",
        default="data/corpus.sqlite",
        help="Corpus SQLite store to slice (default: data/corpus.sqlite).",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Embedding cache path. Omit for a cold run (every vector is paid for).",
    )
    parser.add_argument(
        "--keep",
        default=None,
        help="Directory to keep the built indexes in (default: a temp dir, deleted).",
    )
    parser.add_argument(
        "--max-slice",
        type=int,
        default=2000,
        help="Refuse a slice larger than this without --yes-i-mean-it (default: 2000).",
    )
    parser.add_argument(
        "--yes-i-mean-it",
        action="store_true",
        help="Allow a slice above --max-slice.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.slice <= 0:
        parser.error("--slice must be positive")
    if args.slice > args.max_slice and not args.yes_i_mean_it:
        parser.error(
            f"--slice {args.slice} exceeds --max-slice {args.max_slice}. This is the "
            "cost guard: pass --yes-i-mean-it if you really want to embed that many."
        )

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    if not profiles:
        parser.error("--profiles is empty")

    cfg = load_models_config()
    store = TraceStore(args.db)
    try:
        corpus_size = len(store.read_all_movies())
        slice_movies = top_by_votes(store, args.slice)
    finally:
        store.close()

    if not slice_movies:
        print("corpus is empty - nothing to compare", file=sys.stderr)
        return 1

    work_root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="embcmp_"))
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        results = [
            measure(
                profile,
                slice_movies,
                work_root=work_root,
                cache_path=Path(args.cache) if args.cache else None,
                corpus_size=corpus_size,
                price_per_mtok=_price_for(profile, cfg),
            )
            for profile in profiles
        ]
    finally:
        if not args.keep:
            shutil.rmtree(work_root, ignore_errors=True)

    print()
    print(format_table(results, len(slice_movies), corpus_size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
