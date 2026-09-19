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

from agent.world.board import MOVE_DELTA, manhattan
from agent.world.rules import BOARD_SIZE, SHED_ACCESS, TURNS_PER_DAY
from agent.wsr.routing import walk
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
    """What the search found: the pool it used, the route, and whether the day was carried.

    A route entry is `(turn, task, worker)`: the turn the task occupies, its id, and which worker
    does it. The worker matters beyond bookkeeping - a good is carried by one worker, so the fetch
    that brings it and every task that consumes it have to be the same worker's, and this is what
    lets that be checked instead of assumed.
    """

    pool: int
    route: list[tuple[int, str, int]]
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
        # The hands land on the doors that are free WHEN THEY ARE HIRED, and a unit walking off a
        # door in the first turn changes which doors those are. So the positions are settled
        # against the search's own first turn and the day is searched again until they agree -
        # a fixed point, and a cheap one: the search is milliseconds and this converges in two
        # passes or not at all.
        settled = None
        result = _run(day, tasks, hands=pool, beam=beam)
        for _attempt in range(3):
            nxt = _settled_after_first_turn(day, tasks, result)
            if settled is not None and nxt == settled:
                break
            settled = nxt
            result = _run(day, tasks, hands=pool, beam=beam, settled=settled)
        if result.complete:
            return result
        if partial is None or len(result.route) > len(partial.route):
            partial = result
    return partial if partial is not None else Result(ceiling, [], False)





def _run(day: Day, tasks: TaskArray, *, hands: int, beam: int,
         settled=None) -> Result:
    """One pool size: search the day, and report how much of it the pool could carry."""
    n = tasks.n
    start_pos = _start_positions(day, hands, settled)         # (m, 2)
    first_hand = len(day.units)                              # workers before this index are units

    done = np.zeros((beam, n), dtype=bool)
    when = np.zeros((beam, n), dtype=np.int16)               # the turn each done task occupies
    who = np.full((beam, n), -1, dtype=np.int16)             # and the worker that did it
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
    best = _snapshot(done, when, who, free, travel, first_hand, start_hours)
    for _step in range(n):
        expanded = _expand(day, tasks, done, when, who, free, where, travel, live)
        if expanded is None:
            break
        done, when, who, free, where, travel, live = _select(
            expanded, tasks, beam, first_hand, start_hours)
        if not live.any():
            break
        here = _snapshot(done, when, who, free, travel, first_hand, start_hours)
        if _better(here, best):
            best = here

    placed, when_best, who_best, done_best = best
    complete = placed == n
    route = [(int(when_best[i]), tasks.ids[i], int(who_best[i]))
             for i in np.argsort(when_best) if done_best[i]]
    return Result(hands, route, complete)


def _snapshot(done, when, who, free, travel, first_hand, start_hours):
    """The best route in the beam right now, by the layered objective: work, hands, makespan, walk.

    A hand counts as put to work when its clock has moved past the hour it began at - not when its
    clock is merely above zero, which every hand's is from the moment the offer sets its start.
    """
    placed = done.sum(axis=1)
    hands_used = (free[:, first_hand:] > start_hours[first_hand:]).sum(axis=1)
    makespan = free.max(axis=1)
    row = int(np.lexsort((travel, makespan, hands_used, -placed))[0])
    return int(placed[row]), when[row].copy(), who[row].copy(), done[row].copy()


def _better(candidate, best) -> bool:
    """Whether `candidate` beats `best`: more work first, then the earliest finish."""
    placed, when = candidate[0], candidate[1]
    best_placed, best_when = best[0], best[1]
    if placed != best_placed:
        return placed > best_placed
    return int(when.max()) < int(best_when.max())


def _start_positions(day: Day, hands: int, settled=None) -> np.ndarray:
    """Who is on the field and where each stands.

    Two different questions, and conflating them is what put crops on the wrong tiles:

      the units   start the day where they are, so their walks are priced from there and nowhere
                  else. A unit that will walk in the first turn has NOT started halfway.
      the hands   appear on one of the four shed doors - the fewest units on it, ties by door
                  order - and that is counted WHEN THEY ARE HIRED, which is after the first turn's
                  actions. So a unit that walks off its door leaves that door for the next hand.

    `settled` is where the units stand once the first turn is over, and it is used for the hands
    alone. Pricing a unit's walk from `settled` would credit it a move it has not made yet, which
    is exactly the bug: the farmer was started at (3,4) and planted (3,3) while the engine had it
    at (4,4), so the planting landed one tile short and was refused in silence.
    """
    from agent.world.rules import spawn_cell
    out = [(int(c[0]), int(c[1])) for c in day.units]
    occupied = list(out if settled is None else [(int(c[0]), int(c[1])) for c in settled])
    for _ in range(hands):
        cell = spawn_cell(occupied)
        out.append((int(cell[0]), int(cell[1])))
        occupied.append((int(cell[0]), int(cell[1])))
    return np.asarray(out, dtype=np.int16)


def _settled_after_first_turn(day: Day, tasks: TaskArray, result: Result) -> list:
    """Where the units already on the field stand when the first turn is over.

    A unit moves in the first turn only if its first task needs a walk that starts then - the walk
    occupies the turns immediately before the task, so a task at turn t with a walk of w moves
    from turn t-w. This is the compiler's own rule, applied to the route the search just built.
    """

    occupied = [(int(c[0]), int(c[1])) for c in day.units]
    first: dict[int, tuple[int, str]] = {}
    for turn, task_id, worker in result.route:
        worker, turn = int(worker), int(turn)
        if worker >= len(occupied):
            continue
        if worker not in first or turn < first[worker][0]:
            first[worker] = (turn, task_id)

    for worker, (turn, task_id) in first.items():
        row = tasks.ids.index(task_id)
        target = _stand_for(tasks, row, occupied[worker])
        moves = walk(occupied[worker], target)
        if not moves:
            continue
        # The walk occupies the turns immediately before the task, and a unit is free from the
        # first turn of its day. It moves in turn zero only when the walk begins exactly there -
        # a walk that would have to start before the day did is not a walk the day can make, and
        # assuming otherwise is what let the model credit the farmer a move it never took.
        if turn - len(moves) == 0:
            step = moves[0][0]
            if step in MOVE_DELTA:
                dx, dy = MOVE_DELTA[step]
                occupied[worker] = (occupied[worker][0] + int(dx), occupied[worker][1] + int(dy))
    return occupied


def _stand_for(tasks: TaskArray, row: int, at: tuple[int, int]) -> tuple[int, int]:
    """Where a worker must stand to do a task: a door for a fetch, the tile for anything else."""
    if bool(_fetch_mask(tasks)[row]):
        return min(((int(x), int(y)) for x, y in SHED_ACCESS),
                   key=lambda tile: (manhattan(at, tile),
                                     SHED_ACCESS.index((tile[0], tile[1]))))
    return (int(tasks.cells[row][0]), int(tasks.cells[row][1]))


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


def _expand(day: Day, tasks: TaskArray, done, when, who, free, where, travel, live):
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
        # One door, not two minima. The worker walks to a door, picks up, and walks on to the tile
        # - so the trip is dist(worker, door) + dist(door, tile) for the SAME door. Taking the
        # nearest door to the worker and the nearest door to the tile separately can name two
        # different doors, which undercuts the trip and hands the compiler a walk it cannot make.
        to_door = DISTANCE[here[:, :, None], SHED_INDEX[None, None, :]].astype(np.int16)
        chosen = SHED_INDEX[to_door.argmin(axis=-1)]                       # (b, m)
        onward = DISTANCE[chosen[:, :, None], tasks.cell_index[None, None, :]].astype(np.int16)
        via_door = to_door.min(axis=-1)[:, :, None] + onward
        hop = np.where(is_fetch[None, None, :], via_door, hop)

    # A good is carried by ONE worker, so a task that consumes it must be that worker's. The fetch
    # itself may be taken by anyone - that is how the worker becomes the holder. Without this the
    # search happily feeds a tile from a worker whose bag does not have the wheat.
    holder = _holders(done, who, tasks)
    owner = np.where(tasks.items >= 0, holder[:, tasks.items.clip(0)], -1)
    anyone = (tasks.items < 0) | is_fetch
    may = anyone[None, None, :] | (owner[:, None, :] == np.arange(m)[None, :, None])

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
    finish = np.where(legal & may, finish, BIG)

    earliest = finish.min(axis=1)                            # (b, n): the hour it can be done
    worker = finish.argmin(axis=1)                           # (b, n): and by which worker
    return dict(rows=rows, finish=finish, hop=hop, earliest=earliest, worker=worker,
                done=done, when=when, who=who, free=free, where=where, travel=travel)


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
    done, when, who = expanded["done"], expanded["when"], expanded["who"]
    free = expanded["free"]
    where, travel, hop = expanded["where"], expanded["travel"], expanded["hop"]
    flat_hour = expanded["earliest"][rows].ravel()           # (live * n)

    legal = np.flatnonzero(flat_hour < BIG)
    if legal.size == 0:
        return (_empty_like(done, beam), _empty_like(when, beam), _empty_like(who, beam),
                _empty_like(free, beam), _empty_like(where, beam),
                np.zeros((beam,), dtype=np.int16), np.zeros((beam,), dtype=bool))

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
    child_who = who[parent].copy()
    child_who[np.arange(parent.size), task] = worker
    child_when = when[parent].copy()
    child_when[np.arange(parent.size), task] = hour - 1
    child_free = free[parent].copy()
    child_free[np.arange(parent.size), worker] = hour
    child_where = where[parent].copy()
    child_where[np.arange(parent.size), worker] = tasks.cells[task]
    child_travel = travel[parent] + hop[parent, worker, task]

    # Every child array is filtered in ONE place. Filtering them in separate statements is how a
    # new array gets left behind, and a child array out of step with the others reads another
    # route's values - which is what happened to the worker column.
    keep = _dedupe(child_done, child_free, child_where)
    child_done, child_when, child_who, child_free, child_where, child_travel = (
        array[keep] for array in (child_done, child_when, child_who, child_free,
                                  child_where, child_travel))

    hands_used = (child_free[:, first_hand:] > start_hours[first_hand:]).sum(axis=1)
    makespan = child_free.max(axis=1)
    order = np.lexsort((child_travel, makespan, hands_used))[:beam]

    out_done = _empty_like(child_done, beam)
    out_when = _empty_like(child_when, beam)
    out_who = _empty_like(child_who, beam)
    out_free = _empty_like(child_free, beam)
    out_where = _empty_like(child_where, beam)
    out_travel = np.zeros((beam,), dtype=np.int16)
    out_live = np.zeros((beam,), dtype=bool)
    k = order.size
    out_done[:k], out_when[:k] = child_done[order], child_when[order]
    out_who[:k] = child_who[order]
    out_free[:k], out_where[:k] = child_free[order], child_where[order]
    out_travel[:k], out_live[:k] = child_travel[order], True
    return out_done, out_when, out_who, out_free, out_where, out_travel, out_live


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


def _holders(done, who, tasks: TaskArray) -> np.ndarray:
    """Which worker holds each good, per route - read off the fetch that carries it.

    Not tracked as state: a good has exactly one fetch task, so the holder is whoever did it, and
    deriving it keeps the two from ever disagreeing.
    """
    if tasks.fetch_of_good.size == 0:
        return np.full((done.shape[0], 0), -1, dtype=np.int16)
    index = tasks.fetch_of_good
    safe = np.where(index >= 0, index, 0)
    taken = done[:, safe] & (index >= 0)[None, :]
    return np.where(taken, who[:, safe], np.int16(-1))


def _fetch_mask(tasks: TaskArray) -> np.ndarray:
    """Which tasks are fetches, by the action they carry rather than by their name.

    A fetch is the one task that happens at a door instead of on a tile, so the distinction has to
    come from the action - reading it off the task id would break the moment ids are renamed.
    """
    from agent.world.model import UnitAction
    from agent.wsr.tasks import action_codes
    code = action_codes().get(UnitAction.PICKUP, -1)
    return tasks.actions == code
