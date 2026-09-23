"""#110's own-supply half: the plan's projected sells re-enter the price
path its successor is priced on.

The first pass prices the day's objective on a path with no `our_sells`;
the mix it picks then projects sells (`Column.produce` x the LP's λ)
that the walk never saw. This module turns that mix into the
`forecast(our_sells=)` shape, and `manager.observe` hands the NEXT day's
`_forecast` the plan committed today — a day-over-day fixed point, not a
second solve inside the budget (the budget is the hour-0 constraint and
a re-solve was measured at full-solve cost).

`Column.produce` is (days, N_RESOURCE) in RESOURCE space; product names
map through RESOURCE_ID (R002 — one vocabulary, no parallel index).
Absolute steps (`(day + d) * 24`) because the forecast's walk counts
turns from the observation.
"""

from __future__ import annotations

import numpy as np

from agent.belief.market import PRODUCTS
from agent.world.model import RESOURCE_ID


def plan_supply_sells(produce: np.ndarray, lam: np.ndarray,
                      days: int, first_day: int = 0) -> dict:
    """The chosen mix's projected sells as `forecast(our_sells=)` reads.

    `produce`: per-column (days, N_RESOURCE) outputs stacked to
    (n, days, N_RESOURCE); `lam`: the LP's fractional mix. Expected sells
    per day are `Σ_j lam_j · produce_j[d, good]`; fractions round — a
    fraction of a melon does not move a 3-units-per-coin ladder.
    """
    out: dict[int, dict[str, int]] = {}
    if produce is None or not len(produce) or not len(lam):
        return out
    prod = np.asarray(produce, dtype=np.float64)[:, :days, :]
    weights = np.asarray(lam, dtype=np.float64)[: prod.shape[0]]
    expected = np.tensordot(weights, prod, axes=(0, 0))
    for d in range(min(days, expected.shape[0])):
        step = (first_day + d) * 24
        for name in PRODUCTS:
            units = int(round(float(expected[d, RESOURCE_ID[name]])))
            if units > 0:
                out.setdefault(step, {})[name] = units
    return out


def plan_supply_sells_from_pool(pool, lam: np.ndarray,
                                days: int, first_day: int) -> dict:
    """Same, from the master's pool (columns carry `produce`; idle ones
    carry None and drop out — they project zero sells anyway)."""
    if pool is None or not len(lam):
        return {}
    cols = [c for c in pool[: len(lam)] if c.produce is not None]
    if not cols:
        return {}
    # One stack, then clip to the longest produce any column carries —
    # faster than a per-column slice-then-stack (measured 0.8 ms vs 2.5
    # on a 650-column pool).
    produce = np.stack([np.asarray(c.produce, dtype=np.float64)
                        for c in cols])
    produce = produce[:, :days, :]
    weights = np.asarray(lam, dtype=np.float64)[: len(pool)][: len(produce)]
    return plan_supply_sells(produce, weights, days, first_day)
