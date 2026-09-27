"""planner — the Walras–Wolfe layers between the contractor and the engine.

Issue #13 built the half of the master that does not need the master:

- `columns` — the master's fractional λ, rounded into one plan per tile, with
  the coupling rows checked (and demoted) before anything reaches the engine;
- `repair` — F047's repair checklist, applied to a real day plan;
- `land` — the prefix-locked land decision as an outer enumeration over the
  master, injected as a callable.

Issue #12 added the master itself:

- `master` — the restricted master LP over the tile contractor's columns, its
  coupling duals, and the damped tâtonnement that turns the duals into the
  prices the contractor prices against.

The two numbers issue #13 asks for that need the LP itself stay OPEN: the
integrality gap (`LP bound − rounded value`) and the land enumeration's 400 ms
budget are named TODOs on `columns.py` / `land.py` — the LP now exists, so they
are measurable, but neither is computed here and no run claims them.
"""

from agent.planner.columns import (Choice, ClassMix, DAYS, Plan, ROW_NAMES,
                             Violation, assign_tiles, counts,
                             demote_to_feasible, plan_from_board,
                             rounded_value, row_use, violations)
from agent.planner.land import (Candidate, LandPlanner, LandResult, best_land,
                          candidates, prefix_cost)
from agent.planner.master import (COUPLING_IDS, MARKET_IDS,
                            MasterResult, CouplingSupply,
                            equilibrate, published_duals, supply_from_obs)
from agent.planner.repair import (Drop, RepairResult, land_step_price, order_cost,
                            repair_day)

__all__ = [
    "Choice", "ClassMix", "Plan", "Violation", "ROW_NAMES", "DAYS",
    "assign_tiles", "counts", "row_use", "violations", "rounded_value",
    "demote_to_feasible", "plan_from_board",
    "Candidate", "LandPlanner", "LandResult", "best_land", "candidates",
    "prefix_cost",
    "COUPLING_IDS", "MARKET_IDS",
    "MasterResult", "CouplingSupply", "equilibrate",
    "published_duals", "supply_from_obs",
    "Drop", "RepairResult", "repair_day", "order_cost", "land_step_price",
]
