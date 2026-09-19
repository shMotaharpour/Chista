"""or-opt for the day: try the column orders, keep the one that fits the most.

The greedy commits to an order as it goes and never looks back, so a tile it walked away from is
a tile it pays for twice. This scores an ORDER of the targets - the same rules the greedy uses,
applied to a given sequence - and searches the orders of whole columns, which is what makes a tile
finish before the worker leaves it.
"""
from __future__ import annotations

import itertools
from typing import Sequence

from agent.world.board import manhattan
from agent.world.rules import SHED_ACCESS
from agent.world.model import UnitAction


def _column_of(task_id: str) -> str:
    """The column a task belongs to: `d3_water` -> `d3`."""
    return task_id.split("_", 1)[0]


def score_order(order: Sequence[str], *, start_cell, earliest_start: int,
                tasks_by_id, target_preds, target_needs_item, item_to_pickups,
                shed_times, horizon: int = 24) -> tuple[int, int, dict]:
    """Schedule `order` under the day's rules -> (tasks placed, makespan, exec times).

    The rules are the ones the engine enforces, and they are why an order can fail: a predecessor
    must come first, a good must be in the shed before it is fetched, and the day has `horizon`
    turns. Fetching is what makes this more than a permutation count - a fetch is a walk to a door
    and back, and the two consumers of one item share it.
    """
    exec_times: dict[str, int] = {}
    picked: set[str] = set()                 # items this worker has fetched
    pos = start_cell
    t = earliest_start
    placed = 0

    for tid in order:
        item = target_needs_item.get(tid)
        setup = 0
        fetch_from = pos
        if item is not None and item not in picked:
            # the fetch: to the nearest door, then to the task's tile
            door = min(SHED_ACCESS, key=lambda d: manhattan(pos, d))
            task_cell = tasks_by_id[tid].cell or door
            setup = (manhattan(pos, door) + manhattan(door, task_cell)
                     - manhattan(pos, task_cell))
            fetch_from = door
            earliest_fetch = int(shed_times.get(item, 0)) + 1
            t = max(t, earliest_fetch - 1)

        task_cell = tasks_by_id[tid].cell or fetch_from
        t += setup + 1 + manhattan(pos, task_cell)

        ready = t
        for pred in target_preds.get(tid, ()):
            if pred in exec_times:
                ready = max(ready, exec_times[pred] + 1)
        if ready > horizon:
            break                            # this order cannot carry the rest either
        exec_times[tid] = ready
        t = ready
        pos = task_cell
        placed += 1
        if item is not None:
            picked.add(item)

    return placed, t, exec_times


def best_column_order(targets: Sequence[str], *, start_cell, earliest_start: int,
                      tasks_by_id, target_preds, target_needs_item, item_to_pickups,
                      shed_times, horizon: int = 24) -> tuple[list[str], int]:
    """Try every order of the columns; keep the one that places the most, then the shortest day.

    The columns are few (one per tile) and their internal order is fixed by precedence, so the
    search is over the order of the tiles - which is exactly the decision the greedy gets wrong.
    """
    by_column: dict[str, list[str]] = {}
    for tid in targets:
        by_column.setdefault(_column_of(tid), []).append(tid)

    columns = list(by_column)
    best_order, best_key = list(targets), (-1, 10**9)
    for permutation in itertools.permutations(columns):
        order = [tid for column in permutation for tid in by_column[column]]
        placed, makespan, _ = score_order(
            order, start_cell=start_cell, earliest_start=earliest_start,
            tasks_by_id=tasks_by_id, target_preds=target_preds,
            target_needs_item=target_needs_item, item_to_pickups=item_to_pickups,
            shed_times=shed_times, horizon=horizon)
        key = (placed, -makespan)
        if key > best_key:
            best_key, best_order = key, order
    return best_order, best_key[0]
