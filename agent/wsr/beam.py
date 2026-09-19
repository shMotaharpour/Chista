"""Beam search for the day: many routes at once, scored by a layered objective.

The greedy commits to one route as it goes, so a bad early choice is paid for until the end. This
keeps `beam` routes in parallel and prunes to the best each step: a bad choice costs one of them,
not all of them.

A step adds ONE task to every route in the beam, not one turn to one worker. A route is a sequence
of tasks and the clock follows from it, so the search never reasons about idle turns: an idle turn
is only the gap between when a task could start and when it does. That is what makes the step one
set of array operations instead of a schedule simulation.

The objective is layered, because one layer is not enough - a fixed pool of hands can be used by
many different routes, and a search that cannot tell them apart keeps duplicates:

    1. the hands the route puts to work   (money: a hand costs the ladder)
    2. the makespan                       (the day finishes sooner)
    3. the travel                         (a worker does not walk for nothing)

Every quantity below is an array with the beam on axis zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from agent.world.rules import BOARD_SIZE, SHED_ACCESS, TURNS_PER_DAY
from agent.wsr.tasks import DISTANCE, NO_ITEM, SHED_INDEX, TaskArray

Cell = tuple[int, int]
MAX_HANDS = 16

BIG = np.int16(30000)


@dataclass(frozen=True)
class Day:
    """The chains to run today, in the planner's order, with the shed's timetable."""

    chains: tuple[tuple[Cell, tuple[str, ...], str | None], ...]
    available: dict[str, int]
    horizon: int = TURNS_PER_DAY
    units: tuple[Cell, ...] = ()            # the farmer, and any hand already on the field


class Result(NamedTuple):
    """What the search found: the pool it used, the route, and whether the day was carried."""

    pool: int
    route: list[tuple[int, str]]
    complete: bool


def lower_bound(day: Day, tasks: TaskArray) -> int:
    """The fewest hands the day can possibly need, from arithmetic alone.

    Two floors, and the larger wins. A bound that is too high is worse than useless - the search
    would skip a pool size that works, which is how a feasible day gets called infeasible - so
    both are floors and neither is padded:

      work   every task must fit inside the day, and a fetch is not a task of its own: it rides
             along with the task that needs it, so it is counted once per distinct good.
      chain  the longest precedence chain plus the walk to its tile, since that work cannot be
             split between hands however many there are.

    The solver's own bound is not consulted. A bound taken from the solver is circular.
    """
    if tasks.n == 0:
        return 0
    goods = {int(i) for i in tasks.items if int(i) != NO_ITEM}
    by_work = -(-(tasks.n + len(goods)) // day.horizon)

    depth = _chain_depth(tasks)
    reach = int(DISTANCE[SHED_INDEX].min(axis=0)[tasks.cell_index].max()) if tasks.n else 0
    by_chain = -(-(depth + reach) // day.horizon)
    return max(1, by_work, by_chain)


def _chain_depth(tasks: TaskArray) -> int:
    """The longest precedence chain, in tasks - the critical path through the day's work.

    `pred[i, j]` means j precedes i, so a task sits one below its deepest predecessor. The graph is
    acyclic, so repeated relaxation converges and no recursion is needed.
    """
    if tasks.n == 0:
        return 0
    depth = np.ones(tasks.n, dtype=np.int16)
    for _ in range(tasks.n):
        with_pred = tasks.pred.any(axis=1)
        deeper = (depth[None, :] * tasks.pred).max(axis=1) + 1
        updated = np.where(with_pred, deeper, depth).astype(np.int16)
        if (updated == depth).all():
            break
        depth = updated
    return int(depth.max())


def search(day: Day, tasks: TaskArray, *, beam: int = 64,
           hands: int | None = None, max_hands: int = MAX_HANDS) -> Result:
    """The day, searched with `beam` routes in parallel.

    The pool grows from the arithmetic floor until the day fits. When even the largest pool falls
    short the best partial route is returned with `complete=False`: what was built is reported
    rather than discarded, so the caller can keep the part of the day that works.
    """
    if tasks.n == 0:
        return Result(0, [], True)
    lo = lower_bound(day, tasks)
    hi = max_hands if hands is None else int(hands)

    partial: Result | None = None
    for pool in range(lo, hi + 1):
        result = _run(day, tasks, hands=pool, beam=beam)
        if result.complete:
            return result
        if partial is None or len(result.route) > len(partial.route):
            partial = result
    return partial if partial is not None else Result(hi, [], False)


def _run(day: Day, tasks: TaskArray, *, hands: int, beam: int) -> Result:
    """One pool size: search the day, and report how much of it the pool could carry."""
    n = tasks.n
    start_pos = _start_positions(day, hands)                 # (m, 2)
    m = start_pos.shape[0]
    first_hand = len(day.units)                              # workers before this index are units

    done = np.zeros((beam, n), dtype=bool)
    when = np.zeros((beam, n), dtype=np.int16)               # the hour each done task ran
    free = np.zeros((beam, m), dtype=np.int16)
    where = np.tile(start_pos[None, :, :], (beam, 1, 1))
    travel = np.zeros((beam,), dtype=np.int16)
    live = np.ones((beam,), dtype=bool)

    # The best state is remembered as the search goes, because the beam's last generation can be
    # empty - a route that dies at the end would otherwise erase the work it had already placed.
    best = _snapshot(done, when, free, travel, first_hand)
    for _step in range(n):
        expanded = _expand(day, tasks, done, when, free, where, travel, live)
        if expanded is None:
            break
        done, when, free, where, travel, live = _select(expanded, tasks, beam, first_hand)
        if not live.any():
            break
        here = _snapshot(done, when, free, travel, first_hand)
        if _better(here, best):
            best = here

    placed, when_best, done_best = best
    complete = placed == n
    route = [(int(when_best[i]), tasks.ids[i])
             for i in np.argsort(when_best) if done_best[i]]
    return Result(hands, route, complete)


def _snapshot(done, when, free, travel, first_hand):
    """The best route in the beam right now, by the layered objective: work, hands, makespan, walk."""
    placed = done.sum(axis=1)
    hands_used = (free[:, first_hand:] > 0).sum(axis=1)
    makespan = free.max(axis=1)
    row = int(np.lexsort((travel, makespan, hands_used, -placed))[0])
    return int(placed[row]), when[row].copy(), done[row].copy()


def _better(candidate, best) -> bool:
    """Whether `candidate` beats `best`: more work first, then the earliest finish."""
    placed, when, _ = candidate
    best_placed, best_when, _ = best
    if placed != best_placed:
        return placed > best_placed
    return int(when.max()) < int(best_when.max())


def _start_positions(day: Day, hands: int) -> np.ndarray:
    """Who is on the field and where each stands.

    The units already there keep their cells. A hired hand appears on one of the four shed doors -
    the one with the fewest units on it, ties by door order - which is the engine's rule, and the
    reason a hand's first move is not free: it starts at a door, not at the tile it must work.
    """
    from agent.world.rules import spawn_cell
    out = [tuple(c) for c in day.units]
    for _ in range(hands):
        cell = spawn_cell(out)
        out.append((int(cell[0]), int(cell[1])))
    return np.asarray(out, dtype=np.int16)


def _expand(day: Day, tasks: TaskArray, done, when, free, where, travel, live):
    """One task added to every live route: the vectorised step.

    For each route and each task, the earliest hour any worker could finish it: walk from where
    that worker stands, wait for the predecessors, wait for the good. The minimum over workers says
    which worker should take it.
    """
    n = tasks.n
    b, m = free.shape
    rows = np.flatnonzero(live)
    if rows.size == 0:
        return None

    here = _flat(where)                                      # (b, m)
    hop = DISTANCE[here[:, :, None], tasks.cell_index[None, None, :]].astype(np.int16)

    # A fetch happens at a shed door, not on the tile the good is for, and the worker picks the
    # nearest door. Every other task is worked on its own tile.
    is_fetch = _fetch_mask(tasks)
    door = DISTANCE[here[:, :, None], SHED_INDEX[None, None, :]].astype(np.int16).min(axis=-1)
    hop = np.where(is_fetch[None, None, :], door[:, :, None], hop)

    arrive = free[:, :, None] + hop                          # (b, m, n)
    released = _released(when, tasks)                        # (b, n): the predecessors' finish
    ready = tasks.ready(done)                                # (b, n): every predecessor done

    start = np.maximum(arrive, released[:, None, :])
    start = np.maximum(start, tasks.earliest[None, None, :])
    finish = start + np.int16(1)

    # The window and the horizon are facts about the task and the clock, not about which worker
    # takes it, so they are broadcast across the worker axis instead of being written into it.
    legal = (ready & ~done & (finish <= day.horizon).any(axis=1)
             & (start <= tasks.latest[None, None, :]).any(axis=1))
    legal = legal[:, None, :] & (finish <= day.horizon) & (start <= tasks.latest[None, None, :])
    finish = np.where(legal, finish, BIG)

    earliest = finish.min(axis=1)                            # (b, n): the hour it can be done
    worker = finish.argmin(axis=1)                           # (b, n): and by which worker
    return dict(rows=rows, finish=finish, hop=hop, earliest=earliest, worker=worker,
                done=done, when=when, free=free, where=where, travel=travel)


def _select(expanded, tasks: TaskArray, beam: int, first_hand: int):
    """Keep the best `beam` children, and drop the ones a sibling already stands for.

    Children are generated per live route and then deduplicated by what they have done, because
    identical routes reached from different parents are one route: keeping them would spend the
    beam on copies. Survivors are ordered by the layered objective.
    """
    n = tasks.n
    rows, finish, hop = expanded["rows"], expanded["finish"], expanded["hop"]
    done, when, free = expanded["done"], expanded["when"], expanded["free"]
    where, travel = expanded["where"], expanded["travel"]

    # Only the live rows have children, so the candidates are gathered on those rows and the
    # parents are repeated to match: one row of candidates per live route.
    width = min(beam, n)
    live_hour = expanded["earliest"][rows]
    picks = np.argsort(live_hour, axis=1, kind="stable")[:, :width]
    values = np.take_along_axis(live_hour, picks, axis=1)

    parent = np.repeat(rows, width)
    task = picks.ravel()
    hour = values.ravel()
    worker = np.take_along_axis(expanded["worker"][rows], picks, axis=1).ravel()
    ok = hour < BIG

    parent, task, hour, worker = parent[ok], task[ok], hour[ok], worker[ok]
    if parent.size == 0:
        return (_empty_like(done, beam), _empty_like(when, beam), _empty_like(free, beam),
                _empty_like(where, beam), np.zeros((beam,), dtype=np.int16),
                np.zeros((beam,), dtype=bool))

    child_done = done[parent].copy()
    child_done[np.arange(parent.size), task] = True
    child_when = when[parent].copy()
    child_when[np.arange(parent.size), task] = hour
    child_free = free[parent].copy()
    child_free[np.arange(parent.size), worker] = hour
    child_where = where[parent].copy()
    child_where[np.arange(parent.size), worker] = tasks.cells[task]
    child_travel = travel[parent] + hop[parent, worker, task]

    keep = _dedupe(child_done, child_free, child_where)
    child_done, child_when = child_done[keep], child_when[keep]
    child_free, child_where = child_free[keep], child_where[keep]
    child_travel = child_travel[keep]

    hands_used = (child_free[:, first_hand:] > 0).sum(axis=1)
    makespan = child_free.max(axis=1)
    order = np.lexsort((child_travel, makespan, hands_used))[:beam]

    # A beam of fixed width, so a step that produces fewer children leaves the rest of the rows
    # empty rather than shrinking the arrays the next step reads.
    out_done = _empty_like(child_done, beam)
    out_when = _empty_like(child_when, beam)
    out_free = _empty_like(child_free, beam)
    out_where = _empty_like(child_where, beam)
    out_travel = np.zeros((beam,), dtype=np.int16)
    out_live = np.zeros((beam,), dtype=bool)
    k = order.size
    out_done[:k], out_when[:k] = child_done[order], child_when[order]
    out_free[:k], out_where[:k] = child_free[order], child_where[order]
    out_travel[:k], out_live[:k] = child_travel[order], True
    return out_done, out_when, out_free, out_where, out_travel, out_live


def _empty_like(array: np.ndarray, beam: int) -> np.ndarray:
    """A fixed-width buffer with the same trailing shape as `array`."""
    return np.zeros((beam,) + array.shape[1:], dtype=array.dtype)


def _dedupe(done: np.ndarray, free: np.ndarray, where: np.ndarray) -> np.ndarray:
    """The first index of every distinct child, by the whole state and not by its work alone.

    Two routes that have placed the same tasks are still different routes when their workers stand
    in different places and are free at different hours - one may be a step from the next tile and
    the other across the board. Collapsing them on the done flags alone throws away the diversity
    the beam exists for, so the signature carries all three: what is done, when each worker is
    free, and where each stands. Bytes, so whole states compare at once.
    """
    if done.shape[0] <= 1:
        return np.arange(done.shape[0])
    signature = np.hstack([np.packbits(done, axis=1),
                           free.view(np.uint8).reshape(done.shape[0], -1),
                           where.view(np.uint8).reshape(done.shape[0], -1)])
    _, first = np.unique(signature, axis=0, return_index=True)
    return np.sort(first)


def _released(when: np.ndarray, tasks: TaskArray) -> np.ndarray:
    """When each task's predecessors finish, per route.

    `pred[i, j]` means j precedes i, so the answer for task i is the largest `when` among its
    predecessors - one broadcast product over the matrix, and zero where there are none.
    """
    if tasks.n == 0 or not tasks.pred.any():
        return np.zeros_like(when)
    return (when[:, None, :] * tasks.pred[None, :, :]).max(axis=2).astype(np.int16)


def _flat(where: np.ndarray) -> np.ndarray:
    """Cells as distance-matrix rows: `y * size + x`."""
    return (where[:, :, 0].astype(np.int32) * BOARD_SIZE + where[:, :, 1].astype(np.int32))


def _fetch_mask(tasks: TaskArray) -> np.ndarray:
    """Which tasks are fetches, by the action they carry rather than by their name.

    A fetch is the one task that happens at a door instead of on a tile, so the distinction has to
    come from the action - reading it off the task id would break the moment ids are renamed.
    """
    from agent.world.model import UnitAction
    from agent.wsr.tasks import action_codes
    code = action_codes().get(UnitAction.PICKUP, -1)
    return tasks.actions == code
