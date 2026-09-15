"""planner — the Walras–Wolfe layers between the contractor and the engine.

Issue #13 built the half of the master that does not need the master:

- `columns` — the master's fractional λ, rounded into one plan per tile, with
  the coupling rows checked (and demoted) before anything reaches the engine;
- `repair` — F047's repair checklist, applied to a real day plan;
- `land` — the prefix-locked land decision as an outer enumeration over the
  master, injected as a callable.

The two numbers issue #13 asks for that need the LP itself (the integrality gap
and the land enumeration's 400 ms) are named TODOs on #12; see each module's
docstring and `planner/DESIGN.md`.
"""

from planner.columns import (Choice, ClassMix, DAYS, Plan, ROW_NAMES,
                             Violation, assign_tiles, counts,
                             demote_to_feasible, plan_from_board,
                             rounded_value, row_use, violations)
from planner.land import (Candidate, LandPlanner, LandResult, best_land,
                          candidates, prefix_cost)
from planner.repair import (Drop, RepairResult, order_cost, repair_day)

__all__ = [
    "Choice", "ClassMix", "Plan", "Violation", "ROW_NAMES", "DAYS",
    "assign_tiles", "counts", "row_use", "violations", "rounded_value",
    "demote_to_feasible", "plan_from_board",
    "Candidate", "LandPlanner", "LandResult", "best_land", "candidates",
    "prefix_cost",
    "Drop", "RepairResult", "repair_day", "order_cost",
]
