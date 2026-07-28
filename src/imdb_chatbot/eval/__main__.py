"""CLI entry: ``python -m imdb_chatbot.eval --db data/corpus.sqlite --labels labels.jsonl``.

Loads the live FAISS + BM25 index (via the ``config/live_index.json`` pointer, or
``--index-dir``) plus the corpus store, builds a ``HybridRetriever``, runs the
anchored eval harness over the labeled set, and prints the metric table. This is
how the harness runs against the REAL corpus once the anchored labels exist.

Uses the real ``SentenceTransformerEmbedder`` by default (requires the ``embed``
extra); pass ``--stub`` for a dependency-free smoke run against a stub index.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import CONFIG_DIR
from ..index.build import load_index
from ..index.embedder import Embedder, SentenceTransformerEmbedder, StubEmbedder
from ..retrieval.retrieve import HybridRetriever
from ..store import TraceStore
from .harness import K_VALUES, evaluate, format_report
from .labels import load_labels

LIVE_INDEX_PATH = CONFIG_DIR / "live_index.json"

# Return >=10 candidates so Recall@10 / Hit@10 are measurable.
DEFAULT_FINAL_K = 10


def _resolve_index_dir(explicit: str | None) -> Path:
    """Resolve the index directory: an explicit path, else the live pointer."""
    if explicit:
        return Path(explicit)
    import json

    if not LIVE_INDEX_PATH.exists():
        raise SystemExit(
            f"No --index-dir given and no live pointer at {LIVE_INDEX_PATH}. "
            "Build an index first (python -m imdb_chatbot.index) or pass --index-dir."
        )
    pointer = json.loads(LIVE_INDEX_PATH.read_text(encoding="utf-8"))
    path = pointer.get("path")
    if not path:
        raise SystemExit(
            f"Live pointer {LIVE_INDEX_PATH} has no 'path' (active={pointer.get('active')!r}). "
            "Build an index first or pass --index-dir."
        )
    return Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imdb_chatbot.eval")
    parser.add_argument(
        "--db",
        default="data/corpus.sqlite",
        help="Path to the corpus SQLite store (default: data/corpus.sqlite).",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to the labeled-set JSONL file.",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Index version directory (default: read from config/live_index.json).",
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
        "--final-k",
        type=int,
        default=DEFAULT_FINAL_K,
        help="Candidates returned per query; must be >=10 for Recall@10 (default: 10).",
    )
    args = parser.parse_args(argv)

    index_dir = _resolve_index_dir(args.index_dir)
    labels = load_labels(args.labels)

    embedder: Embedder = (
        StubEmbedder(dim=args.stub_dim) if args.stub else SentenceTransformerEmbedder(args.model)
    )

    store = TraceStore(args.db)
    try:
        loaded = load_index(index_dir)
        retriever = HybridRetriever.from_store(
            loaded, embedder, store, final_k=args.final_k
        )
        report = evaluate(retriever, labels, k_values=K_VALUES)
    finally:
        store.close()

    print(f"index_dir={index_dir} labels={args.labels} n_queries={len(labels)}")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
