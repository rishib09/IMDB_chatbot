"""End-to-end loop demo (ticket #26).

A deterministic, no-network walk through ONE full harness cycle:
seed a failure -> detect its taxonomy code -> represent a candidate fix ->
run the promotion gate -> record a ChangeRecord in the ledger. See
``loop.run_loop`` for the machine-readable result and ``loop.narrate`` /
the ``python -m imdb_chatbot.demo`` CLI for the human-readable narrative.
"""

from __future__ import annotations

from .loop import LoopResult, narrate, run_loop, seed_failure_trace

__all__ = ["LoopResult", "narrate", "run_loop", "seed_failure_trace"]
