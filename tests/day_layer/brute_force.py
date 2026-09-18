"""A brute-force optimal-cost finder, completely independent of the CP-SAT
formulation in day/solvers/cpsat_solver.py -- plain Python
enumeration + a from-scratch longest-path feasibility check. This is the
ground truth tests/test_brute_force_cross_check.py compares CP-SAT
against: it exists specifically to catch bugs in the *model itself*
(section 3 of docs/problem-formulation.en.md), not just in its CP-SAT
implementation.

Only tractable for small instances (a handful of tasks and candidate
workers) -- it enumerates every worker subset, every assignment of tasks
to workers, and every per-worker task ordering. Flexible (shed) cell
choices for PICKUP/DROP are *not* enumerated here: give such tasks a
fixed cell (any of the 4 shed cells is a valid concrete choice) in
instances meant for this checker.
"""
from __future__ import annotations

import itertools
from typing import Optional

from agent.wsr.distances import assign_entry_cells, manhattan
from agent.world.rules import hire_cost
from agent.wsr.models import WAREHOUSE_ENTRY_CELLS, Instance, Item, Action, MinorTask

_PRODUCE_ACTIONS = frozenset({Action.PICKUP, Action.HARVEST, Action.COLLECT_FERTILIZER})
_CONSUME_ACTIONS = frozenset({Action.PLACE, Action.FEED, Action.FERTILIZE})


def _earliest_times(
    n: int, edges: list[tuple[int, int, int]], releases: dict[int, int]
) -> Optional[list[int]]:
    """Longest-path relaxation: tau[v] >= tau[u] + w for every (u, v, w) in
    edges, tau[i] >= releases[i] for every release. Returns None if the
    edge set contains a positive cycle (this particular ordering/
    assignment is not realizable)."""
    tau = [releases.get(i, 0) for i in range(n)]
    for _ in range(n + 1):
        changed = False
        for u, v, w in edges:
            candidate = tau[u] + w
            if candidate > tau[v]:
                tau[v] = candidate
                changed = True
        if not changed:
            return tau
    return None  # didn't converge in n+1 rounds -> a positive cycle exists


def _feasible_for_fixed_plan(
    instance: Instance,
    tasks: list[MinorTask],
    task_worker: list[int],
    per_worker_order: dict[int, tuple[int, ...]],
    entry_cell_of_worker: dict[int, tuple[int, int]],
) -> bool:
    n = len(tasks)
    task_index = {t.id: i for i, t in enumerate(tasks)}
    edges: list[tuple[int, int, int]] = []
    releases: dict[int, int] = {}

    for w, order in per_worker_order.items():
        worker = next(worker for worker in instance.workers if worker.index == w)
        for pos, i in enumerate(order):
            cell = tasks[i].cell
            if cell is None:
                return False  # brute force requires fixed cells; see module docstring
            if pos == 0:
                # +1: entering the grid at earliest_start only means the
                # worker is standing at its entry cell, not that it has
                # already spent a turn there -- see docs constraint 3.
                releases[i] = worker.earliest_start + 1 + manhattan(entry_cell_of_worker[w], cell)
            else:
                prev = order[pos - 1]
                edges.append((prev, i, 1 + manhattan(tasks[prev].cell, cell)))  # type: ignore[arg-type]

    for pred_id, succ_id in instance.precedence:
        edges.append((task_index[pred_id], task_index[succ_id], 1))

    tau = _earliest_times(n, edges, releases)
    if tau is None:
        return False
    if any(t > instance.horizon for t in tau):
        return False

    # Per-worker inventory, walked in the (now fully known) execution order;
    # DROP genuinely resets the balance here, since order is fixed.
    for w, order in per_worker_order.items():
        balance: dict[Item, int] = {}
        for i in order:
            task = tasks[i]
            if task.action == Action.DROP:
                balance.clear()
                continue
            if task.item is None:
                continue
            if task.action in _PRODUCE_ACTIONS:
                balance[task.item] = balance.get(task.item, 0) + task.qty
            elif task.action in _CONSUME_ACTIONS:
                balance[task.item] = balance.get(task.item, 0) - task.qty
                if balance[task.item] < 0:
                    return False

    # clct: every HARVEST must be followed (same worker) by a shed-adjacent
    # DROP within the deadline.
    entry_cells = set(WAREHOUSE_ENTRY_CELLS.values())
    for w, order in per_worker_order.items():
        harvest_positions = [pos for pos, i in enumerate(order) if tasks[i].action == Action.HARVEST]
        if not harvest_positions:
            continue
        last_harvest_pos = max(harvest_positions)
        ok = False
        for pos in range(last_harvest_pos + 1, len(order)):
            i = order[pos]
            if tasks[i].action == Action.DROP and tasks[i].cell in entry_cells and tau[i] <= instance.clct_deadline:
                ok = True
                break
        if not ok:
            return False

    return True


def brute_force_optimal_cost(instance: Instance) -> Optional[int]:
    """Returns the true minimum Σ fib(w)·u_w, or None if infeasible."""
    tasks = instance.minor_tasks
    n = len(tasks)
    task_index = {t.id: i for i, t in enumerate(tasks)}
    workers = sorted(instance.workers, key=lambda w: w.index)
    k = len(workers)

    # Group tasks that must share one worker into single "units".
    unit_of = [None] * n
    units: list[list[int]] = []
    for group in instance.single_worker_groups:
        idxs = [task_index[tid] for tid in group]
        uid = len(units)
        units.append(idxs)
        for i in idxs:
            unit_of[i] = uid
    for i in range(n):
        if unit_of[i] is None:
            unit_of[i] = len(units)
            units.append([i])
    n_units = len(units)

    # Hiring is append-only (engine): using hand m means hands 0..m were
    # all hired and paid. Only prefix masks are real workforces; a
    # non-prefix subset would undercount its own cost.
    def _is_prefix(mask: int) -> bool:
        if mask == 0:
            return True
        highest = mask.bit_length() - 1
        return mask == (1 << (highest + 1)) - 1

    subsets = [m for m in range(1, 1 << k) if _is_prefix(m)] if n > 0 else [0]
    subsets.sort(key=lambda mask: sum(hire_cost(workers[w].index) for w in range(k) if mask & (1 << w)))

    for mask in subsets:
        active = [w for w in range(k) if mask & (1 << w)]
        if n == 0:
            return 0
        if not active:
            continue
        cost = sum(hire_cost(workers[w].index) for w in active)
        entry_cell_of_worker = assign_entry_cells([(workers[w].index, workers[w].earliest_start) for w in active])
        entry_cell_of_worker = {
            w: WAREHOUSE_ENTRY_CELLS[entry_cell_of_worker[workers[w].index]] for w in active
        }

        for unit_assignment in itertools.product(active, repeat=n_units):
            task_worker = [None] * n
            for uid, w in enumerate(unit_assignment):
                for i in units[uid]:
                    task_worker[i] = w
            per_worker_tasks: dict[int, list[int]] = {w: [] for w in active}
            for i, w in enumerate(task_worker):
                per_worker_tasks[w].append(i)

            worker_list = [w for w in active if per_worker_tasks[w]]
            order_choices = [list(itertools.permutations(per_worker_tasks[w])) for w in worker_list]
            for combo in itertools.product(*order_choices):
                per_worker_order = dict(zip(worker_list, combo))
                if _feasible_for_fixed_plan(instance, tasks, task_worker, per_worker_order, entry_cell_of_worker):
                    return cost
    return None
