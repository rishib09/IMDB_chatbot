"""CLI entry: ``python -m imdb_chatbot.eval --db data/corpus.sqlite``.

Loads the live FAISS + BM25 index (via the ``config/live_index.json`` pointer, or
``--index-dir``) plus the corpus store, builds a ``HybridRetriever``, runs the
anchored eval harness over the golden set (``eval/labels.jsonl`` by default), and
prints the metric table. This is how the harness runs against the REAL corpus.

Two tiers (ticket #68), selected with ``--tier``:

- ``anchored`` (default) - the labeled constraints are handed to the retriever.
  Deterministic, free, and measures retrieval alone.
- ``extract`` - the constraints come from the real LLM extractor plus its
  production guards (corpus vocabulary, person-role correction, US region
  default). One model call per row; the delta against the anchored tier is the
  extraction regression signal. Needs ``OPENROUTER_API_KEY`` (run under dotenvx).

Uses the real ``SentenceTransformerEmbedder`` by default (requires the ``embed``
extra); pass ``--stub`` for a dependency-free smoke run against a stub index.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from ..config import CONFIG_DIR, PROJECT_ROOT
from ..index.build import load_index
from ..index.embedder import (
    Embedder,
    SentenceTransformerEmbedder,
    StubEmbedder,
    build_embedder,
)
from ..retrieval.retrieve import HybridRetriever
from ..schemas import ParsedQuery
from ..store import TraceStore
from .harness import K_VALUES, ParseFn, evaluate, format_report
from .labels import LabeledQuery, load_labels

LIVE_INDEX_PATH = CONFIG_DIR / "live_index.json"
DEFAULT_LABELS = PROJECT_ROOT / "eval" / "labels.jsonl"

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


def _extract_tier(retriever: HybridRetriever, meter) -> ParseFn:
    """Tier 2's parse hook: the REAL extractor plus the graph's deterministic guards.

    Mirrors the ``extract`` node (``graph/build.py``) exactly - corpus-vocabulary
    normalization (#84), person-role correction, then the US region default (#53)
    - so what the eval measures is what production does, including the regex
    fallback the node takes when the model's JSON does not parse.
    """
    from ..graph.build import apply_region_default, correct_person_role, regex_extract
    from ..graph.models import build_models
    from ..graph.normalize import CorpusVocab, normalize_parsed

    models = build_models(meter=meter)
    vocab = CorpusVocab.from_movies(retriever.movies_by_id.values())

    def parse(label: LabeledQuery) -> ParsedQuery:
        try:
            parsed = models.extract(label.query)
        except Exception:  # noqa: BLE001 - production degrades here, so does the eval
            parsed = regex_extract(label.query)
        parsed = normalize_parsed(parsed, vocab)
        return apply_region_default(correct_person_role(parsed, label.query), label.query)

    return parse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imdb_chatbot.eval")
    parser.add_argument(
        "--db",
        default="data/corpus.sqlite",
        help="Path to the corpus SQLite store (default: data/corpus.sqlite).",
    )
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS),
        help=f"Path to the labeled-set JSONL file (default: {DEFAULT_LABELS}).",
    )
    parser.add_argument(
        "--tier",
        choices=("anchored", "extract"),
        default="anchored",
        help=(
            "anchored: hand the labeled constraints to the retriever (free). "
            "extract: derive them with the real LLM extractor (one call per row)."
        ),
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Also write the full report (per-query rows included) to this JSON file.",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Index version directory (default: read from config/live_index.json).",
    )
    parser.add_argument(
        "--embedder",
        default=None,
        help=(
            "Embedder profile from config/models.yaml. MUST match the profile the "
            "index was built with, or the query vectors are meaningless."
        ),
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model id (ignored with --embedder / --stub).",
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

    if args.stub:
        embedder: Embedder = StubEmbedder(dim=args.stub_dim)
    elif args.embedder:
        embedder = build_embedder(args.embedder)
    else:
        embedder = SentenceTransformerEmbedder(args.model)

    store = TraceStore(args.db)
    cost_usd = 0.0
    try:
        loaded = load_index(index_dir)
        retriever = HybridRetriever.from_store(
            loaded, embedder, store, final_k=args.final_k
        )
        parse: ParseFn | None = None
        meter = None
        if args.tier == "extract":
            from ..graph.usage import UsageMeter, estimate_cost

            meter = UsageMeter()
            parse = _extract_tier(retriever, meter)
        report = evaluate(retriever, labels, k_values=K_VALUES, parse=parse)
        if meter is not None:
            cost_usd = estimate_cost(meter)
    finally:
        store.close()

    print(
        f"tier={args.tier} index_dir={index_dir} labels={args.labels} "
        f"n_queries={len(labels)}"
    )
    print(format_report(report))
    if args.tier == "extract":
        print(f"\nextractor cost: ${cost_usd:.4f} for {len(labels)} calls")
    if args.json_out:
        import json

        payload = report.to_dict()
        payload["queries"] = [asdict(q) for q in report.queries]
        payload["tier"] = args.tier
        payload["cost_usd"] = cost_usd
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
