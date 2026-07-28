"""Per-day production spend counter (section 5.6 / decision #7).

Sums ``TurnTrace.cost_usd`` per calendar day. When the running total for the day
EXCEEDS ``production_budget_usd`` (3.0) the app enters L2 (LLM-free mode) - a
degrade, never an outage. The clock and the per-day store are both injectable so
tests are fully deterministic with no wall-clock or persistence dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

from ..config import load_limits_config

DEFAULT_BUDGET_USD = 3.0

# Mode strings shared with the degradation ladder.
MODE_NORMAL = "normal"
MODE_L2 = "L2"


def _default_budget() -> float:
    try:
        return float(load_limits_config()["per_day"]["production_budget_usd"])
    except Exception:  # noqa: BLE001 - fall back to the known preset
        return DEFAULT_BUDGET_USD


def _default_clock() -> date:
    return datetime.now(UTC).date()


class BudgetTracker:
    """A per-day ``$`` counter that flips to L2 once the daily budget is exceeded.

    Pure and injectable: pass ``clock`` (a zero-arg callable returning a
    ``date``) and ``store`` (a ``{date: float}`` mapping) to make accumulation
    deterministic in tests. Spend is bucketed by the clock's date, so a new day
    starts fresh automatically.
    """

    def __init__(
        self,
        *,
        budget_usd: float | None = None,
        clock: Callable[[], date] | None = None,
        store: dict[date, float] | None = None,
    ) -> None:
        self.budget_usd = budget_usd if budget_usd is not None else _default_budget()
        self._clock = clock or _default_clock
        self._by_day: dict[date, float] = store if store is not None else {}

    def record(self, cost_usd: float) -> float:
        """Add one turn's cost to today's bucket; return the new daily total."""
        today = self._clock()
        self._by_day[today] = self._by_day.get(today, 0.0) + float(cost_usd)
        return self._by_day[today]

    def spent_today(self) -> float:
        """Total ``$`` recorded for the clock's current day."""
        return self._by_day.get(self._clock(), 0.0)

    def exhausted(self) -> bool:
        """True once today's spend EXCEEDS the daily budget (crossing $3 trips L2)."""
        return self.spent_today() > self.budget_usd

    def mode(self) -> str:
        """Current serving mode: ``L2`` if the budget is exhausted, else ``normal``."""
        return MODE_L2 if self.exhausted() else MODE_NORMAL

    def record_and_mode(self, cost_usd: float) -> str:
        """Record a cost and return the resulting serving mode in one call."""
        self.record(cost_usd)
        return self.mode()
