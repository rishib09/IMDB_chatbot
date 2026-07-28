"""CLI entry point: ``python -m imdb_chatbot.demo``.

Walks one full harness cycle over a throwaway temp trace store and prints the
seeded-failure -> code -> fix -> gate PROMOTE -> ledger-row narrative. No
network, no LLM, no persistent side effects (the temp DB is removed on exit).
Pass ``--regress`` to watch the gate REJECT a regressing candidate instead.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from ..store import TraceStore
from .loop import narrate, run_loop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m imdb_chatbot.demo",
        description="Walk one end-to-end reliability loop (seed -> detect -> gate -> ledger).",
    )
    parser.add_argument(
        "--regress",
        action="store_true",
        help="Use a regressing candidate so the gate REJECTS (no ledger row).",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        store = TraceStore(Path(tmp) / "demo_traces.sqlite")
        try:
            result = run_loop(store, regress=args.regress)
        finally:
            store.close()

    print(narrate(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
