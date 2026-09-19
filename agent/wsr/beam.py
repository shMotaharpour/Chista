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
    """The chains to run today, in the planner's order, with the shed's timetable.

    `hire_times` is the planner's offer: the hour each hand it will pay for may begin. A hand is
    offered at most one start, and the times are not interchangeable - a hand hired for turn 0 acts
    from hour 1 (F040), and one the market cannot reach until turn 5 is idle before that. The
    engine's own day runs hour 0 to 23, so a hand hired at turn 0 has 23 turns of work in it and a
    hand hired at turn 5 has 18.
    """

    chains: tuple[tuple[Cell, tuple[str, ...], str | None], ...]
    available: dict[str, int]
    horizon: int = TURNS_PER_DAY
    units: tuple[Cell, ...] = ()            # the farmer, and any hand already on the field
    hire_times: tuple[int, ...] = ()        # the hour each offered hand may begin


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

    Two numbers decide how the pool is searched, and they answer two different questions:

      hands      the pool to START at. A caller who says five is saying the day will have five
                 hands whether it needs them or not - so five and one are the same decision and
                 the smaller pools are not worth searching. `None` starts at the arithmetic floor.
      max_hands  the largest pool allowed. Equal to `hands`, it asks one question with no growth:
                 "can five carry this day - yes or no". Larger, it asks "can five, and if not, what
                 can" - and the answer reports the pool that did.

    When no allowed pool carries the day, the best partial route comes back with `complete=False`:
    what was built is reported rather than discarded, so the caller keeps the part of the day that
    works, and the answer never claims a pool the caller did not allow.
    """
    if tasks.n == 0:
        return Result(0, [], True)

    # The arithmetic floor is on the WORKERS a day needs, and the units already on the field are
    # workers, so what has to be hired is the shortfall. Without this the search starts at one hand
    # and stops there, paying the ladder for a day the farmer could have carried alone.
    floor = max(0, lower_bound(day, tasks) - len(day.units))
    start = floor if hands is None else int(hands)
    ceiling = min(int(max_hands), MAX_HANDS)
    if start > ceiling:
        raise ValueError(
            f"start pool {start} is above the ceiling {ceiling}: the caller asked for a pool it "
            f"does not allow")

    partial: Result | None = None
    for pool in range(start, ceiling + 1):
        result = _run(day, tasks, hands=pool, beam=beam)
        if result.complete:
            return result
        if partial is None or len(result.route) > len(partial.route):
            partial = result
    return partial if partial is not None else Result(ceiling, [], False)


def _prefetch_table() -> np.ndarray:
    """The distance from every board cell to the nearest shed door, built once.

    A fetch is a detour through a door, and the door's distance from a task's tile does not depend
    on the state - only the worker's side of the trip does. Precomputing the tile side turns a
    (workers, doors) reduction per step into one lookup per task.
    """
    return DISTANCE[:, SHED_INDEX].min(axis=1).astype(np.int16)


DOOR_FROM: np.ndarray = _prefetch_table()


def _run(day: Day, tasks: TaskArray, *, hands: int, beam: int) -> Result:
    """One pool size: search the day, and report how much of it the pool could carry."""
    n = tasks.n
    start_pos = _start_positions(day, hands)                 # (m, 2)
    first_hand = len(day.units)                              # workers before this index are units

    done = np.zeros((beam, n), dtype=bool)
    when = np.zeros((beam, n), dtype=np.int16)               # the hour each done task ran
    # Every worker starts at its own hour: the farmer at the day's first, a hand at the hour the
    # planner offered it. Starting them all together would hand the search turns the engine will
    # not give, which is how a day gets called feasible that the harness then truncates.
    start_hours = _start_hours(day, hands)
    free = np.tile(start_hours[None, :], (beam, 1)).astype(np.int16)
    where = np.tile(start_pos[None, :, :], (beam, 1, 1))
    travel = np.zeros((beam,), dtype=np.int16)
    live = np.ones((beam,), dtype=bool)

    # The best state is remembered as the search goes, because the beam's last generation can be
    # empty - a route that dies at the end would otherwise erase the work it had already placed.
    best = _snapshot(done, when, free, travel, first_hand, start_hours)
    for _step in range(n):
        expanded = _expand(day, tasks, done, when, free, where, travel, live)
        if expanded is None:
            break
        done, when, free, where, travel, live = _select(
            expanded, tasks, beam, first_hand, start_hours)
        if not live.any():
            break
        here = _snapshot(done, when, free, travel, first_hand, start_hours)
        if _better(here, best):
            best = here

    placed, when_best, done_best = best
    complete = placed == n
    route = [(int(when_best[i]), tasks.ids[i])
             for i in np.argsort(when_best) if done_best[i]]
    return Result(hands, route, complete)


def _snapshot(done, when, free, travel, first_hand, start_hours):
    """The best route in the beam right now, by the layered objective: work, hands, makespan, walk.

    A hand counts as put to work when its clock has moved past the hour it began at - not when its
    clock is merely above zero, which every hand's is from the moment the offer sets its start.
    """
    placed = done.sum(axis=1)
    hands_used = (free[:, first_hand:] > start_hours[first_hand:]).sum(axis=1)
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

    Where a hand stands does not depend on when it may begin, so the position and the start hour
    are read separately: this gives the cells, `_start_hours` the clocks.
    """
    from agent.world.rules import spawn_cell
    out = [tuple(c) for c in day.units]
    for _ in range(hands):
        cell = spawn_cell(out)
        out.append((int(cell[0]), int(cell[1])))
    return np.asarray(out, dtype=np.int16)


def _start_hours(day: Day, hands: int) -> np.ndarray:
    """The hour each worker may first act.

    A unit already on the field acts from the day's first hour. A hand begins at the hour the
    planner offered it, and the offer is not a formality: the engine settles hires in index order
    and a hand hired in turn h acts from h + 1, so a hand the market cannot pay for until turn 5
    does nothing for the first five hours of the day. An offer shorter than the pool means the
    planner did not price that hand, and the engine's own rule for a hire in turn 0 applies.
    """
    hours = [0] * len(day.units)
    for index in range(hands):
        if index < len(day.hire_times):
            hours.append(max(1, int(day.hire_times[index])))
        else:
            hours.append(1)
    return np.asarray(hours, dtype=np.int16)


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
    # A fetch is a detour through a door: out to the nearest one, then on to the tile. The tile
    # side of that trip is a property of the tile alone, so it is looked up rather than measured,
    # and only the worker's side is computed per step.
    is_fetch = _fetch_mask(tasks)
    if is_fetch.any():
        door = DISTANCE[here[:, :, None], SHED_INDEX[None, None, :]].min(axis=-1).astype(np.int16)
        via_door = door[:, :, None] + DOOR_FROM[tasks.cell_index][None, None, :]
        hop = np.where(is_fetch[None, None, :], via_door, hop)

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


def _select(expanded, tasks: TaskArray, beam: int, first_hand: int, start_hours):
    """Keep the best `beam` children, ranked BEFORE they are built.

    A child is a copy of its parent's whole state - what is done, when each worker is free, where
    each stands - so materialising every candidate costs (live x tasks x state) and then throws
    almost all of it away. The hour a child would finish at is already known from the candidate
    arrays, so the shortlist is taken on that and only the survivors are copied. The layered
    objective then orders the survivors, and duplicates among them are dropped.

    The shortlist is a few times the beam wide, not the whole field: the dedupe and the layered
    sort need a choice, but they do not need every task of every live route.
    """
    n = tasks.n
    rows = expanded["rows"]
    done, when, free = expanded["done"], expanded["when"], expanded["free"]
    where, travel, hop = expanded["where"], expanded["travel"], expanded["hop"]
    flat_hour = expanded["earliest"][rows].ravel()           # (live * n)

    legal = np.flatnonzero(flat_hour < BIG)
    if legal.size == 0:
        return (_empty_like(done, beam), _empty_like(when, beam), _empty_like(free, beam),
                _empty_like(where, beam), np.zeros((beam,), dtype=np.int16),
                np.zeros((beam,), dtype=bool))

    budget = min(legal.size, beam * 4)
    if legal.size > budget:
        shortlist = legal[np.argpartition(flat_hour[legal], budget - 1)[:budget]]
    else:
        shortlist = legal

    parent = np.repeat(rows, n)[shortlist]
    task = shortlist % n
    hour = flat_hour[shortlist]
    worker = expanded["worker"][rows].ravel()[shortlist]

    # `hour` is when the action FINISHES. The turn it occupies is the one before that, and that
    # turn is what the route reports - the engine numbers a day's turns 0 to 23, and the farmer
    # acts in the first of them. The worker is free from the turn after.
    child_done = done[parent].copy()
    child_done[np.arange(parent.size), task] = True
    child_when = when[parent].copy()
    child_when[np.arange(parent.size), task] = hour - 1
    child_free = free[parent].copy()
    child_free[np.arange(parent.size), worker] = hour
    child_where = where[parent].copy()
    child_where[np.arange(parent.size), worker] = tasks.cells[task]
    child_travel = travel[parent] + hop[parent, worker, task]

    keep = _dedupe(child_done, child_free, child_where)
    child_done, child_when = child_done[keep], child_when[keep]
    child_free, child_where = child_free[keep], child_where[keep]
    child_travel = child_travel[keep]

    hands_used = (child_free[:, first_hand:] > start_hours[first_hand:]).sum(axis=1)
    makespan = child_free.max(axis=1)
    order = np.lexsort((child_travel, makespan, hands_used))[:beam]

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
    predecessors. The precedence graph is SPARSE - a few edges per task, not a matrix - so this
    walks the edges instead of multiplying the whole matrix: the product would be (batch, n, n)
    and cost n^2 per step for a graph that has only n edges.
    """
    out = np.zeros_like(when)
    if tasks.n == 0:
        return out
    # `when` is the turn a task OCCUPIES, so a successor's earliest start is the turn after it.
    for after, before in tasks.edges:
        np.maximum(out[:, after], when[:, before] + 1, out=out[:, after])
    return out


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
