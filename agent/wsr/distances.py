"""Grid distance and the worker warehouse-entry-cell placement rule.

The grid has no obstacles and movement is always exactly one cell per
turn, 4-directional — so travel time between any two cells is simply the
Manhattan distance (see docs/problem-formulation.md, section 3: this is
what lets the whole problem be modeled like a standard VRPTW without
per-turn position variables).
"""
from __future__ import annotations

from agent.world.board import manhattan
from .models import WAREHOUSE_ENTRY_CELLS, WAREHOUSE_ENTRY_ORDER, Cell


def distance_to_nearest_entry(cell: Cell) -> int:
    """Distance from `cell` to the closest of the 4 shed-adjacent cells —
    used for `PICKUP`/`DROP` tasks, whose exact entry cell is flexible."""
    return min(manhattan(cell, entry) for entry in WAREHOUSE_ENTRY_CELLS.values())


def assign_entry_cells(worker_starts: list[tuple[int, int]]) -> dict[int, str]:
    """The initial-placement rule: each worker enters through whichever of
    the 4 warehouse cells currently has the fewest workers, ties broken by
    the fixed NW -> NE -> SW -> SE priority order. Workers are processed in
    order of (start_time, worker_index) — i.e. earlier starters go first,
    and workers entering at the very same time step are placed in
    ascending worker-index order. Entry counts accumulate permanently
    (a worker occupies its entry cell from its start time onward for the
    purpose of counting later arrivals), matching the worked example in
    docs/problem-formulation.md.

    `worker_starts`: list of (worker_index, start_time) for active workers.
    Returns {worker_index: entry_cell_name}.
    """
    counts = {name: 0 for name in WAREHOUSE_ENTRY_ORDER}
    assignment: dict[int, str] = {}
    for worker_index, start_time in sorted(worker_starts, key=lambda pair: (pair[1], pair[0])):
        chosen = min(WAREHOUSE_ENTRY_ORDER, key=lambda name: counts[name])
        assignment[worker_index] = chosen
        counts[chosen] += 1
    return assignment
