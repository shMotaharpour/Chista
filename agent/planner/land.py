"""The land-purchase outer loop (issue #13 §3).

`BUY_LAND` is **prefix-locked**: quadrants open in a fixed order at fixed prices
and a short purse refuses silently (F042), so the decision is not *which* land
but *when* to take the next step of the prefix. That is one dimension, small
enough to enumerate exactly — solve the master (plus rounding) once per
candidate, keep the best — and the issue is explicit that land does **not** go
into the LP: an outer loop over a handful of days is cheaper, exact and
explainable.

`best_land()` takes the master as an injected callable, which is what makes this
module testable while #12 is unbuilt: the harness, the candidate set, the prefix
arithmetic and the cadence cap are all here, and the solve is the caller's.

**Not measured here (R005 — a named TODO):** the issue's budget is "400 ms at
day 0" for the whole enumeration, i.e. ~45 master solves. That number depends on
a real solve, so it stays a TODO(#12); what is enforced today is the *shape* of
the budget — the candidate count is knowable before any solve (`len(candidates)`)
and `max_solves` refuses to run a set that cannot fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.world.rules import LAND_PRICES

# F029: 30 days in a season.
DAYS = 30
# The issue's cadence cap: land is strategic, not daily.
MAX_EVALUATIONS = 3


@dataclass(frozen=True)
class Candidate:
    """One prefix step taken on one day (or never)."""

    quadrants: int            # quadrants owned AFTER the purchase
    day: int | None           # the purchase day; None = buy no more land
    cost: int                 # the cumulative price of the prefix

    def describe(self) -> str:
        if self.day is None:
            return f"{self.quadrants} quadrants, no purchase"
        return f"{self.quadrants} quadrants by day {self.day} ({self.cost})"


def prefix_cost(quadrants: int) -> int:
    """The engine's own prefix price for owning `quadrants` quadrants (R002).

    NW is owned from the start (F042), so `quadrants` counts it: 1 quadrant
    costs nothing, and each further step is the next entry of `LAND_PRICES`.
    """
    if quadrants < 1:
        raise ValueError("a farm always owns NW (F042)")
    steps = min(quadrants - 1, len(LAND_PRICES))
    return int(sum(LAND_PRICES[:steps]))


def candidates(days: int = DAYS, grid: int = 2) -> list[Candidate]:
    """The 4-way prefix × purchase-day grid, coarsely sampled (issue #13 §3).

    The issue's coarse grid is every second day, which keeps the enumeration at
    a handful of dozen solves; `grid` is that step and is the knob the budget
    ownership (#12) turns once a real solve has been timed.
    """
    out = [Candidate(quadrants=1, day=None, cost=0)]
    for quadrants in range(2, len(LAND_PRICES) + 2):
        for day in range(0, days, max(1, grid)):
            out.append(Candidate(quadrants=quadrants, day=day,
                                 cost=prefix_cost(quadrants)))
    return out


@dataclass(frozen=True)
class LandResult:
    """The best candidate found, and what the search cost to find it."""

    best: Candidate
    value: float
    solves: int
    considered: int

    def describe(self) -> str:
        return (f"best {self.best.describe()} value {self.value:.1f} "
                f"({self.solves} solves of {self.considered})")


def best_land(solve: Callable[[Candidate], float], *, days: int = DAYS,
              grid: int = 2, day: int | None = None,
              max_solves: int | None = None) -> LandResult:
    """Enumerate the prefix × day grid and keep the best total.

    `solve(candidate)` runs the master plus rounding for that land decision and
    returns the season's value; it is the only thing this module needs from
    #12. `day` is the earliest purchase day still worth considering — the daily
    re-check drops the days already behind us rather than pretending they are
    still available.
    """
    grid_candidates = [c for c in candidates(days=days, grid=grid)
                       if c.day is None or day is None or c.day >= day]
    if max_solves is not None and len(grid_candidates) > max_solves:
        raise ValueError(
            f"{len(grid_candidates)} candidates exceed the {max_solves}-solve "
            "budget: coarsen the grid (issue #13 §3 caps the enumeration)")
    best: Candidate | None = None
    best_value = float("-inf")
    solves = 0
    for candidate in grid_candidates:
        value = float(solve(candidate))
        solves += 1
        if value > best_value:
            best, best_value = candidate, value
    assert best is not None                       # candidates() is never empty
    return LandResult(best=best, value=best_value, solves=solves,
                      considered=len(grid_candidates))


class LandPlanner:
    """When to spend a master solve on land (the issue's cadence cap).

    One evaluation at day 0, then at most `max_evaluations - 1` more for the
    rest of the season, and only on days where the next prefix step is actually
    affordable — "affordable" being the engine's own price for that step (F042),
    so the threshold needs no measurement of its own (R005).
    """

    def __init__(self, solve: Callable[[Candidate], float], *,
                 days: int = DAYS, grid: int = 2,
                 max_evaluations: int = MAX_EVALUATIONS,
                 max_solves: int | None = None) -> None:
        self.solve = solve
        self.days = days
        self.grid = grid
        self.max_evaluations = max_evaluations
        self.max_solves = max_solves
        self.evaluations = 0
        self.result: LandResult | None = None
        self.history: list[LandResult] = []

    def next_step_cost(self, quadrants: int) -> int:
        """The price of the next prefix step, or 0 when the prefix is complete."""
        if quadrants >= len(LAND_PRICES) + 1:
            return 0
        return prefix_cost(quadrants + 1) - prefix_cost(quadrants)

    def consider(self, day: int, cash: float, quadrants: int) -> LandResult | None:
        """Evaluate the land decision if this day is worth one (else None).

        Skipped when: the evaluation cap is spent, the prefix is complete, or
        (after day 0) the next step is not affordable. Day 0 always evaluates —
        the issue asks for one evaluation there whatever the purse says.
        """
        if self.evaluations >= self.max_evaluations:
            return None
        if quadrants >= len(LAND_PRICES) + 1:
            return None
        if day != 0 and cash < self.next_step_cost(quadrants):
            return None
        result = best_land(self.solve, days=self.days, grid=self.grid,
                           day=day, max_solves=self.max_solves)
        self.evaluations += 1
        self.result = result
        self.history.append(result)
        return result