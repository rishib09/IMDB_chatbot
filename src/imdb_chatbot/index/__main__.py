"""CLI entry: ``python -m imdb_chatbot.index --db data/corpus.sqlite``.

Builds the real, versioned FAISS + BM25 index from a corpus DB and flips the
live-index pointer. Uses the real ``SentenceTransformerEmbedder`` by default
(requires the ``embed`` extra), or the dependency-free ``StubEmbedder`` with
``--stub`` for smoke tests.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..store import TraceStore
from .build import (
    CHUNK_POLICY_V1,
    DEFAULT_CACHE_PATH,
    DEFAULT_INDEX_ROOT,
    build_index,
)
from .cache import EmbeddingCache
from .embedder import Embedder, SentenceTransformerEmbedder, StubEmbedder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imdb_chatbot.index")
    parser.add_argument(
        "--db",
        default="data/corpus.sqlite",
        help="Path to the corpus SQLite store to index (default: data/corpus.sqlite).",
    )
    parser.add_argument(
        "--dataset-version",
        default="dev",
        help="Dataset version tag; part of the index version stamp (default: dev).",
    )
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_INDEX_ROOT),
        help="Root directory for versioned index artifacts (default: data/index).",
    )
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE_PATH),
        help="Embedding cache SQLite path (default: data/embedding_cache.sqlite).",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model id (ignored with --stub).",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use the dependency-free StubEmbedder (no torch) instead of the real model.",
    )
    parser.add_argument(
        "--stub-dim",
        type=int,
        default=8,
        help="Dimension for the StubEmbedder (default: 8).",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Build artifacts but do NOT flip the live-index pointer.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    embedder: Embedder = (
        StubEmbedder(dim=args.stub_dim) if args.stub else SentenceTransformerEmbedder(args.model)
    )

    store = TraceStore(args.db)
    cache = EmbeddingCache(args.cache)
    try:
        result = build_index(
            store,
            embedder,
            dataset_version=args.dataset_version,
            chunk_policy=CHUNK_POLICY_V1,
            out_root=Path(args.out_root),
            cache=cache,
            flip_pointer=not args.no_flip,
        )
    finally:
        cache.close()
        store.close()

    print(
        f"built version={result.version} count={result.count} dim={result.dim} "
        f"out_dir={result.out_dir} flipped={not args.no_flip}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
