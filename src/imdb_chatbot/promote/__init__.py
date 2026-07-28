"""Promotion protocol (ticket #25).

Any behavioral change to the harness (a new prompt, model slot, index, chunk
policy, threshold, or filter) passes ONE promotion gate (PRD section 8.4) and, on
promotion, writes an auditable ``ChangeRecord`` to the change ledger. The
dashboard visualizes that ledger.

- ``gate`` - the pure, deterministic promotion decision.
- ``ledger`` - emit a ``ChangeRecord`` on promotion (never hand-authored).
"""

from __future__ import annotations

from .gate import CandidateResult, PromotionDecision, evaluate_promotion
from .ledger import emit_change_record, promote_and_record

__all__ = [
    "CandidateResult",
    "PromotionDecision",
    "emit_change_record",
    "evaluate_promotion",
    "promote_and_record",
]
