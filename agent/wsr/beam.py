"""Beam search for the day: many routes at once, scored by a layered objective.

A single route, committed to as it goes, pays for a bad early choice until the end. This keeps
`beam` routes in parallel and prunes to the best each step: a bad choice costs one of them, not all
of them.

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

import time
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from agent.world.board import MOVE_DELTA, SPAWN
from agent.world.rules import BOARD_SIZE, TURNS_PER_DAY
from agent.wsr.routing import walk
from agent.wsr.tasks import DISTANCE, NO_ITEM, SHED_INDEX, TaskArray

Cell = tuple[int, int]

#: What one step of the search is allowed to cost, as `beam x workers x tasks`. A step's arrays are
#: that product wide, so a fixed width makes a hundred tiles cost fifty times a quadrant. This is the
#: product a 16-wide beam has on a 200-task day with 16 workers - a shape measured to lose nothing
#: against 64 on any of the objective's four layers.
STEP_WORK = 16 * 16 * 200
#: The measured ends of the useful range: below 8 a day starts losing tasks (4 placed 47 of 50 where
#: 8 placed all of them), and above 32 nothing improved on any layer.
MIN_BEAM, MAX_BEAM = 8, 64


def beam_for(tasks: TaskArray, workers: int) -> int:
    """The width to search a day of this size with this many workers."""
    return int(min(MAX_BEAM, max(MIN_BEAM, STEP_WORK / max(1, workers * tasks.n))))

BIG = np.int16(30000)

#: Where the farmer stands when a day begins - the shed's first door, which is where the engine
#: respawns it every night. Not an input: the day resets to the farmer there with no hands at all.
FARMER_START: Cell = SPAWN


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
    hire_times: tuple[int, ...] = ()        # the hour each offered hand may begin

    @property
    def units(self) -> tuple[Cell, ...]:
        """The units on the field when the day starts: the farmer, and nobody else.

        Not a field, because it is not a choice. The engine resets every day to the farmer on the
        shed's corner door with no hands, so the planner has nothing to say here - and a hand it
        offers is settled by the engine at the hour of the offer (F040), on the door the spawn rule
        picks, which the search works out for each route it tries.
        """
        return (FARMER_START,)


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
    #: True when a deadline stopped the search before it ran out of tasks to place. A route that is
    #: neither complete nor out of time is the search's own answer: the pool could not carry it.
    out_of_time: bool = False
    #: Where the hands are spawned from, as the search priced them: the cells the units already on
    #: the field occupy when the first turn is over. A route is consistent with THESE and no others -
    #: a unit that walks off its door in turn zero moves every hand after it - so
    #: `compile_route(day, tasks, result, settled=result.settled)` is the one correct way to write
    #: the day down. Always a tuple, never None: a search cut before its fixed point converged priced
    #: the day from where the units stand at the start, and None in the compiler means something
    #: else entirely - derive the positions from the route, which agrees only when it did converge.
    settled: tuple[Cell, ...] = ()

    @property
    def can_improve(self) -> bool:
        """Whether another slice of budget would plausibly find more WORK.

        Work is what the flag is about: a day that was carried has nothing left to place, and a day
        too big for any pool has no better route however long the search runs. Only a route that is
        both incomplete and cut short has more to find. `out_of_time` stays the raw fact - a route
        can be complete and still have crossed its deadline - because a caller that wants to polish a
        carried day's makespan is asking a different question than this one.
        """
        return self.out_of_time and not self.complete
    #: True when the day needs more workers than the caller allowed, so NO pool in range can carry
    #: it. The arithmetic floor is a count of workers and the ceiling is a count of hands, and on a
    #: hundred tiles the first passes the second - which is an answer about the day, not an error.
    infeasible: bool = False


def lower_bound(day: Day, tasks: TaskArray) -> int:
    """The fewest units the day can possibly need: the work, against the turns the day has.

    The work is not the tasks alone. Every task costs a turn, every distinct good a consumer needs
    costs a turn to fetch, and the tiles the day works have to be walked to: to touch T distinct tiles
    a unit moves at least T - 1 times, because it starts standing on one of them. A drop doubles that
    walking - the unit has to come back along the path it went out on - and a day with a deadline for
    its drops is a day whose walking cannot be spent twice.

    The turns are not 24 times the hands either. They are a ladder: the farmer's whole day, then 24
    minus the hire hour for each hand in turn, counted off until the work is covered. A hand hired in
    turn 2 has 22 turns in it, and dividing the work by the horizon pretends otherwise.

    The answer is a POOL, because that is what the caller searches with - the units already on the
    field are counted in and the search takes them back out.
    """
    if tasks.n == 0:
        return 0
    goods = len({int(i) for i in tasks.items if int(i) != NO_ITEM})
    tiles = len(np.unique(tasks.cells, axis=0))
    walking = max(0, tiles - 1)
    if tasks.drop_rows.size:
        # A deadline is a hard window on a task that also has precedence, and it means the unit has
        # to finish at a shed door rather than wherever it stopped. So its path reaches the furthest
        # tile and comes back: at least twice the distance from a door to it. Not a doubling of the
        # tile count, which a route that works the tiles in a loop can beat.
        reach = int(DISTANCE[SHED_INDEX].min(axis=0)[tasks.cell_index].max())
        walking = max(walking, 2 * reach)
    work = tasks.n + goods + walking

    total = day.horizon
    hired = 0
    while total < work and hired < len(day.hire_times):
        total += day.horizon - int(day.hire_times[hired])
        hired += 1
    return len(day.units) + hired


def ceiling_for(day: Day, tasks: TaskArray) -> int:
    """The largest pool worth asking about: the tasks, plus the units already on the field.

    Every unit does at least one task, so a pool larger than the number of tasks cannot be the
    smallest carrying one - which makes this a ceiling by argument rather than by guess. The tighter
    candidates are guesses: a tour of the tiles bounds the BEST walking, and the search's own route
    can walk more than it, so a tour-derived ceiling cuts days that would carry.
    """
    return tasks.n + len(day.units)


def walking_tour(tasks: TaskArray) -> int:
    """A nearest-first walk from the shed over every tile the day works.

    What a hand ends up doing: it comes out of the shed, works the nearest tile it still has, and
    carries on. An upper bound on the walking rather than a floor - a route that plans ahead walks
    less - so it is used to start the search and never to bound it.
    """
    if tasks.n == 0:
        return 0
    remaining = [tuple(int(v) for v in cell) for cell in np.unique(tasks.cells, axis=0)]
    here = FARMER_START
    walked = 0
    while remaining:
        nearest = min(remaining, key=lambda cell: abs(here[0] - cell[0]) + abs(here[1] - cell[1]))
        walked += abs(here[0] - nearest[0]) + abs(here[1] - nearest[1])
        here = nearest
        remaining.remove(nearest)
    return walked


def predicted_pool(day: Day, tasks: TaskArray) -> int:
    """The pool a day of this shape usually needs: the work and the walking to reach it.

    A guess with a floor under it, not a bound. `lower_bound` says what the day cannot need less
    than; this says where the answer usually is, and the search starts here and grows.

    The divisor is the turns ONE hand has - the slowest one, so the guess is not optimistic - and not
    the day's total turns. That sum asks how many hands a perfect division of the work would need,
    which is the same question as the work divided by the day and answers one hand for a day that
    needs fifteen.
    """
    if tasks.n == 0:
        return 0
    goods = len({int(i) for i in tasks.items if int(i) != NO_ITEM})
    starts = [int(hour) for hour in day.hire_times] or [1]
    turns = day.horizon - max(starts)
    if turns <= 0:
        return 0
    return max(1, -(-(tasks.n + goods + walking_tour(tasks)) // turns))


def search(day: Day, tasks: TaskArray, *, beam: int | None = None,
           hands: int | None = None, max_hands: int | None = None,
           budget_s: float | None = None, warm: Result | None = None) -> Result:
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

    `beam` is the width, and None asks for the width the day's size implies (`beam_for`) - a step
    costs `beam x workers x tasks`, so a fixed width is a fixed cost only for a fixed day.

    `warm` is a route from an earlier call on almost this instance, and the search starts from it
    instead of from nothing: a row ON TOP of the beam holds the state that route leaves behind, and
    the snapshot is taken after it, so a search that finds nothing better hands the route back. The
    row is extra, so a seed that no longer fits the day costs the beam nothing. It applies to the
    pool the route was searched with and is ignored at the others.

    `budget_s` bounds the wall clock. The search keeps the best route it has found and returns it
    with `out_of_time=True` rather than running long: the cost of a question grows with the square
    of the day and with every pool the loop tries, so a deadline is the only promise that survives
    a hundred tiles.
    """
    if tasks.n == 0:
        return Result(0, [], True)

    # The arithmetic floor is on the WORKERS a day needs, and the units already on the field are
    # workers, so what has to be hired is the shortfall. Without this the search starts at one hand
    # and stops there, paying the ladder for a day the farmer could have carried alone.
    floor = max(0, lower_bound(day, tasks) - len(day.units))
    if hands is None:
        # A guess above the floor and a hand low. `predicted_pool` counts the walking, which the
        # floor cannot, so it is usually near the answer; being a guess, it is taken low, and the
        # halving grows from it.
        start = max(floor, predicted_pool(day, tasks) - len(day.units) - 1)
    else:
        start = int(hands)
    # A caller's ceiling is theirs to set and may be tighter than the day's own bound - the corpus
    # test asks for no more hands than the game paid, which is exactly that.
    bound = ceiling_for(day, tasks)
    ceiling = bound if max_hands is None else min(int(max_hands), bound)
    deadline = None if budget_s is None else time.perf_counter() + float(budget_s)
    if start > ceiling:
        return Result(ceiling, [], False, infeasible=True)

    def width(pool: int) -> int:
        return beam if beam is not None else beam_for(tasks, len(day.units) + pool)

    def seed(pool: int) -> Result | None:
        # A route is a state for the pool it was searched with: its worker indices are that pool's.
        return warm if warm is not None and warm.pool == pool else None

    if hands is None and ceiling > start:
        return _smallest_pool(day, tasks, width, start, ceiling, deadline, seed)

    partial: Result | None = None
    placed = -1
    for pool in range(start, ceiling + 1):
        result = _fixed_point(day, tasks, width(pool), pool, deadline, seed(pool))
        if result.complete:
            return result
        if partial is None or _better_route(result, partial):
            partial, placed = result, len(result.route)
        elif len(result.route) <= placed:
            # A bigger pool placed no more of the day than a smaller one, so the workers are not
            # what the day is short of and every pool above this one is a search for nothing.
            break
        if result.out_of_time:
            # A larger pool costs more and cannot buy back the time, so the loop stops here.
            break
    return partial if partial is not None else Result(ceiling, [], False)


def _makespan(result: Result) -> int:
    """The turn a route stops at. A drop with an empty bag is written at -1, so it is not one."""
    return max((int(turn) for turn, _task, _worker in result.route if turn >= 0), default=-1) + 1


def _better_route(candidate: Result, best: Result) -> bool:
    """Whether a whole route beats another: carried first, then more work, then the earlier stop.

    The rule the pool loop and the fixed point share. A route that carries the day beats one that
    does not, whatever it placed - and among routes that carry it or fail it alike, the one that
    placed more wins, and the one that stopped earlier wins the tie.
    """
    if candidate.complete != best.complete:
        return candidate.complete
    if len(candidate.route) != len(best.route):
        return len(candidate.route) > len(best.route)
    return _makespan(candidate) < _makespan(best)


def _fixed_point(day: Day, tasks: TaskArray, beam: int, pool: int,
                 deadline: float | None, warm: Result | None = None) -> Result:
    """One pool, searched until the hands stop moving.

    The hands land on the doors that are free WHEN THEY ARE HIRED, and a unit walking off a door in
    the first turn changes which doors those are. So the positions are settled against the search's
    own first turn and the day is searched again until they agree - a fixed point, and a cheap one:
    it converges in two passes or not at all.
    """
    settled = None
    started = time.perf_counter()
    result = _run(day, tasks, hands=pool, beam=beam, deadline=deadline, warm=warm)
    spent = time.perf_counter() - started
    for _attempt in range(3):
        # An attempt costs about what the last one cost, so one that starts with less than that left
        # comes back cut short - non-empty, much worse, and it would REPLACE the work the attempt
        # before it placed. That is not a best-of to be patched up afterwards: only the last attempt
        # is consistent with the positions it priced from, and the compiler re-derives those from the
        # route, so an earlier attempt kept on merit hands back a route priced from positions the day
        # does not have. The reserve is what keeps every returned route both consistent and the most
        # complete one the budget could buy.
        if deadline is not None and time.perf_counter() + spent >= deadline:
            break
        nxt = _settled_after_first_turn(day, tasks, result)
        if settled is not None and nxt == settled:
            break
        settled = nxt
        started = time.perf_counter()
        result = _run(day, tasks, hands=pool, beam=beam, settled=settled, deadline=deadline,
                      warm=warm)
        spent = time.perf_counter() - started
    return result


def _smallest_pool(day: Day, tasks: TaskArray, width, lo: int, hi: int,
                   deadline: float | None, seed) -> Result:
    """The smallest pool that carries the day, by halving.

    The predicate is monotone - a bigger pool is never less able to carry a day - so halving finds
    the smallest carrying pool in about log2 runs instead of one run per pool. It is not free: the
    pools it tries are the LARGE ones, and a run costs more the more workers it has, so it wins when
    the answer is large and loses when it is small. That is a measurement, not a preference, and the
    caller can ask for the scan with `hands=` instead.
    """
    best: Result | None = None
    while lo < hi:
        mid = (lo + hi) // 2
        result = _fixed_point(day, tasks, width(mid), mid, deadline, seed(mid))
        if result.complete:
            best, hi = result, mid
        else:
            best, lo = result, mid + 1
        if result.out_of_time:
            break
    if lo == hi:
        final = _fixed_point(day, tasks, width(lo), lo, deadline, seed(lo))
        if final.complete or best is None:
            return final
        if _better_route(final, best):
            return final
    return best if best is not None else Result(hi, [], False)





def _warm_row(tasks: TaskArray, warm: Result, done, when, who, free, where, travel, count) -> None:
    """Write a previous route's state into the beam's first row.

    A route is a set of `(turn, task, worker)` and the state is what that set implies, so this is a
    translation rather than a re-search. A task the day does not have is skipped, and so is a worker
    the pool does not have: a caller that hands over a route from another instance gets the part of
    it that still applies, not an error.
    """
    row = 0
    column = {task_id: index for index, task_id in enumerate(tasks.ids)}
    per_worker: dict[int, list[tuple[int, int]]] = {}
    for turn, task_id, worker in warm.route:
        index = column.get(task_id)
        worker = int(worker)
        if index is None or worker >= free.shape[1]:
            continue
        turn = int(turn)
        done[row, index] = True
        when[row, index] = turn
        who[row, index] = worker
        free[row, worker] = max(int(free[row, worker]), turn + 1)
        where[row, worker] = tasks.cells[index]
        per_worker.setdefault(worker, []).append((turn, index))

    # The walk the route took, so the warmed row is not credited a travel it did not pay: the
    # objective's last layer would otherwise prefer it to every route the search builds.
    walked = 0
    for worker, entries in per_worker.items():
        at = (int(where[row, worker, 0]), int(where[row, worker, 1]))
        for _turn, index in sorted(entries):
            target = (int(tasks.cells[index][0]), int(tasks.cells[index][1]))
            walked += abs(target[0] - at[0]) + abs(target[1] - at[1])
            at = target
    travel[row] = min(walked, int(np.iinfo(np.int16).max))
    # The counter the warmed row's own placements imply: one per edge whose predecessor it placed.
    np.add.at(count[row], tasks.edge_after, done[row, tasks.edge_before].astype(np.int16))


def _run(day: Day, tasks: TaskArray, *, hands: int, beam: int,
         settled=None, deadline: float | None = None,
         warm: Result | None = None) -> Result:
    """One pool size: search the day, and report how much of it the pool could carry."""
    n = tasks.n
    start_pos = _start_positions(day, hands, settled)         # (m, 2)
    first_hand = len(day.units)                              # workers before this index are units

    # The warmed route gets a row of its own ON TOP of the beam, so a caller handing over a route
    # that no longer fits the day costs the search nothing: the beam below it is as wide as it would
    # have been. The selection still keeps `beam` rows, so from the second generation on the seed
    # competes for a slot like any other route.
    rows = beam + (1 if warm is not None else 0)
    done = np.zeros((rows, n), dtype=bool)
    when = np.zeros((rows, n), dtype=np.int16)               # the turn each done task occupies
    who = np.full((rows, n), -1, dtype=np.int16)             # and the worker that did it
    # Every worker starts at its own hour: the farmer at the day's first, a hand at the hour the
    # planner offered it. Starting them all together would hand the search turns the engine will
    # not give, which is how a day gets called feasible that the harness then truncates.
    hours = start_hours(day, tasks, hands)
    free = np.tile(hours[None, :], (rows, 1)).astype(np.int16)
    where = np.tile(start_pos[None, :, :], (rows, 1, 1))
    travel = np.zeros((rows,), dtype=np.int16)
    live = np.ones((rows,), dtype=bool)
    # How many of each task's predecessors are done, per route. `ready` is a comparison on this,
    # where it used to be a (beam, n) by (n, n) product - a billion multiply-adds over a day.
    count = np.zeros((rows, n), dtype=np.int16)

    if warm is not None:
        _warm_row(tasks, warm, done, when, who, free, where, travel, count)

    # The best state is remembered as the search goes, because the beam's last generation can be
    # empty - a route that dies at the end would otherwise erase the work it had already placed.
    best = _snapshot(done, when, who, free, travel, first_hand, hours)
    cut = False
    for _step in range(n):
        # Every eighth step: a step is a fixed amount of work, so the check cannot pay for itself
        # more often than that, and eight steps is far below the resolution a turn budget needs.
        if deadline is not None and not (_step & 7) and time.perf_counter() >= deadline:
            cut = True
            break
        expanded = _expand(day, tasks, done, when, who, free, where, travel, live, count)
        if expanded is None:
            break
        done, when, who, free, where, travel, live, count = _select(
            expanded, tasks, beam, first_hand, hours)
        if not live.any():
            break
        here = _snapshot(done, when, who, free, travel, first_hand, hours)
        if _better(here, best):
            best = here

    placed, when_best, who_best, done_best = best
    complete = placed == n
    route = [(int(when_best[i]), tasks.ids[i], int(who_best[i]))
             for i in np.argsort(when_best) if done_best[i]]
    # The positions the day was priced from, always concrete: a search that never reached its fixed
    # point priced it from where the units stand when the day begins, and that is what the hands'
    # doors are counted from.
    return Result(hands, route, complete, out_of_time=cut,
                  settled=tuple(day.units) if settled is None else tuple(settled))


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
    """Where a worker must stand to do a task: the tile it works, since no task is a fetch.

    The trip a consumer makes through a door is priced on the consumer and written by the compiler,
    so a task never has a door for a destination.
    """
    return (int(tasks.cells[row][0]), int(tasks.cells[row][1]))


def preload_turns(tasks: TaskArray) -> int:
    """How many pickups the day's walk is shifted by: one per distinct good a task consumes.

    A worker takes what it will use while it stands on the door, so its walk out begins that many
    turns later. Charging every worker the day's own count is a ceiling, not a guess: a worker that
    needs fewer goods leaves the difference idle, and one that needs them all has exactly its room.
    """
    goods = tasks.items[tasks.items >= 0]
    return int(np.unique(goods).size)


def first_arrival(tasks: TaskArray) -> int:
    """The earliest turn any good a task consumes is in the shed."""
    goods = tasks.items >= 0
    if not goods.any():
        return 0
    return int(tasks.earliest[goods].min())


def start_hours(day: Day, tasks: TaskArray, hands: int) -> np.ndarray:
    """The hour each worker's WALK may begin: its own hour, the goods, and the pickups before it."""
    return np.maximum(_start_hours(day, hands), first_arrival(tasks)) + np.int16(preload_turns(tasks))


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


def _expand(day: Day, tasks: TaskArray, done, when, who, free, where, travel, live, count):
    """One task added to every live route: the vectorised step.

    For each route and each task, the earliest hour any worker could finish it: walk from where that
    worker stands, wait for the predecessors, wait for the good. The minimum over workers says which
    worker should take it.

    A task that consumes a good its worker does not yet hold pays for the trip: out to a door, the
    pickup, and on to the tile. The second feeding of the same good on the same worker pays nothing
    extra, because the good is already in that worker's bag - which is the saving a day makes when
    it fetches once and eats twice.
    """
    n = tasks.n
    b, m = free.shape
    rows = np.flatnonzero(live)
    if rows.size == 0:
        return None

    # The frontier: the tasks some live route could place next. Everything past it is illegal for
    # every route, so pricing those columns is pricing a column that is thrown away - and there are
    # few of them only in appearance: a chain's tasks are nearly all open at once on a big day.
    ready_all = count == tasks.pred_count16[None, :]         # (b, n)
    index = np.flatnonzero((ready_all & ~done).any(axis=0))
    if index.size == 0:
        return None
    width = index.size

    here = _flat(where)                                      # (b, m)
    # One gather, not two 1-D differences: Manhattan distance separates exactly, and computing it
    # that way is 1.7x faster on its own - but it is five passes over the same (b, m, w) shape
    # against one, and end to end that made a hundred tiles 15 per cent SLOWER. The table is one
    # dimension more than the arithmetic needs and one pass less than the machine wants.
    # No cast: the table is int16, the width the arithmetic runs in, so the gather is the answer.
    hop = DISTANCE[here[:, :, None], tasks.cell_index[index][None, None, :]]

    # The trip a consumer makes when its good is not in the bag: to a door, the pickup, and on. One
    # door, not two minima - the nearest door to the worker and the nearest to the tile can be
    # different doors, and the compiler would then walk a trip the search never priced.
    # The trip: one turn for the pickup, taken at the door before the day's walk. The worker does
    # not move for it, because it is already standing where the good is when it takes it.
    trip = np.zeros(hop.shape, dtype=bool)
    needs = tasks.items[index] >= 0
    if needs.any():
        carried = _carried(done, who, tasks, m)              # (b, m, goods)
        has = np.where(needs[None, None, :], carried[:, :, tasks.items[index].clip(0)], True)
        trip = ~has

    # The drops. A worker that has already dropped since the harvest a drop banks has an empty bag,
    # so that drop hands over nothing: no trip, no turn, and nobody moves. This is what lets one
    # drop bank several harvests instead of one apiece.
    idle = np.zeros(hop.shape, dtype=bool)
    drop_rows = tasks.drop_rows
    if drop_rows.size:
        # Which of the day's drops are on the frontier, and where they sit in the beam's width.
        local = np.searchsorted(index, drop_rows)
        inside = (local < width) & (index[np.minimum(local, width - 1)] == drop_rows)
        if inside.any():
            last_drop = _last_drop(done, when, who, drop_rows, m)        # (b, m)
            held = when[:, tasks.banks[drop_rows].clip(0)]               # (b, d) each harvest's turn
            idle[:, :, local[inside]] = (
                last_drop[:, :, None] > held[:, None, :])[:, :, inside]

    arrive = free[:, :, None] + hop                          # (b, m, w)
    released = _released(when, tasks)[:, index]              # (b, w): the predecessors' finish
    ready = ready_all[:, index]                              # (b, w): every predecessor done
    earliest_here = tasks.earliest[index]
    latest_here = tasks.latest[index]

    start = np.maximum(arrive, released[:, None, :])
    start = np.maximum(start, earliest_here[None, None, :])
    start = start + trip.astype(np.int16)
    finish = start + np.int16(1)
    # An idle drop is finished the moment its worker is free - it costs nothing and takes no turn -
    # and it is written at turn -1, which is the compiler's signal that there is no op to emit.
    finish = np.where(idle, np.maximum(free[:, :, None], released[:, None, :]), finish)
    finish = np.maximum(finish, np.where(idle, np.int16(0), earliest_here[None, None, :]))

    in_time = finish <= day.horizon
    before_latest = start <= latest_here[None, None, :]
    legal = (ready & ~done[:, index] & in_time.any(axis=1) & before_latest.any(axis=1))
    legal = legal[:, None, :] & in_time & before_latest
    finish = np.where(legal, finish, BIG)

    # On the frontier's own width, not the day's: the selection maps the column it picked back
    # through `index`, so nothing is ever spread to the full list and the step never pays for the
    # columns it did not price.
    earliest = finish.min(axis=1)                            # (b, w): the hour it can be done
    worker = finish.argmin(axis=1)                           # (b, w): and by which worker
    return dict(rows=rows, finish=finish, hop=hop, earliest=earliest, worker=worker,
                done=done, when=when, who=who, free=free, where=where, travel=travel,
                idle=idle, index=index, count=count)


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
    index = expanded["index"]
    width = index.size
    count = expanded["count"]
    rows = expanded["rows"]
    done, when, who = expanded["done"], expanded["when"], expanded["who"]
    free = expanded["free"]
    where, travel, hop = expanded["where"], expanded["travel"], expanded["hop"]
    flat_hour = expanded["earliest"][rows].ravel()           # (live * width)

    legal = np.flatnonzero(flat_hour < BIG)
    if legal.size == 0:
        return (_empty_like(done, beam), _empty_like(when, beam), _empty_like(who, beam),
                _empty_like(free, beam), _empty_like(where, beam),
                np.zeros((beam,), dtype=np.int16), np.zeros((beam,), dtype=bool),
                np.zeros((beam, tasks.n), dtype=np.int16))

    # The shortlist is a few percent of the field - `beam x tasks` candidates - and among candidates
    # whose finish hour ties, which of them it keeps was arbitrary. Too narrow, and the states that
    # would carry the day are cut before the ranking ever sees them: at `beam * 4` a width of 50
    # placed 86 of a real day's 93 tasks, and at `beam * 16` it placed 93.
    budget = min(legal.size, beam * 16)
    if legal.size > budget:
        shortlist = legal[np.argpartition(flat_hour[legal], budget - 1)[:budget]]
    else:
        shortlist = legal

    parent = np.repeat(rows, width)[shortlist]
    column = shortlist % width
    # The frontier's column, mapped back to the day's own index - which is what the state and the
    # task's cell are written by.
    task = index[column]
    hour = flat_hour[shortlist]
    worker = expanded["worker"][rows].ravel()[shortlist]

    # `hour` is when the action FINISHES. The turn it occupies is the one before that, and that
    # turn is what the route reports - the engine numbers a day's turns 0 to 23, and the farmer
    # acts in the first of them. The worker is free from the turn after.
    child_done = done[parent].copy()
    child_done[np.arange(parent.size), task] = True
    # A drop with an empty bag has no turn of its own: it is written at -1, and the worker neither
    # moves nor loses the hour.
    idle_here = expanded["idle"][parent, worker, column]
    child_who = who[parent].copy()
    child_who[np.arange(parent.size), task] = worker
    child_when = when[parent].copy()
    child_when[np.arange(parent.size), task] = np.where(idle_here, -1, hour - 1)
    child_free = free[parent].copy()
    child_free[np.arange(parent.size), worker] = np.where(idle_here, free[parent, worker], hour)
    child_where = where[parent].copy()
    child_where[np.arange(parent.size), worker] = np.where(
        idle_here[:, None], where[parent, worker], tasks.cells[task])
    child_travel = travel[parent] + hop[parent, worker, column]
    # Placing a task advances the tasks it precedes, so the counter for those children moves by one
    # on each edge out of it. Most tasks precede one other, so this is a few hundred additions.
    child_count = count[parent].copy()
    flat, start = tasks.successor_groups
    length = (start[task + 1] - start[task]).astype(np.int32)
    if length.sum():
        child_row = np.repeat(np.arange(parent.size, dtype=np.int32), length)
        offset = np.arange(length.sum(), dtype=np.int32) - np.repeat(
            np.cumsum(length) - length, length)
        np.add.at(child_count, (child_row, flat[np.repeat(start[task], length) + offset]), 1)

    # Every child array is filtered in ONE place. Filtering them in separate statements is how a
    # new array gets left behind, and a child array out of step with the others reads another
    # route's values - which is what happened to the worker column.
    keep = _dedupe(child_done, child_free, child_where)
    child_done, child_when, child_who, child_free, child_where, child_travel, child_count = (
        array[keep] for array in (child_done, child_when, child_who, child_free,
                                  child_where, child_travel, child_count))

    # The tasks this state has already made impossible: unplaced, with a latest hour that has gone
    # by the earliest any worker is free. A route that lost one cannot carry the day, so this leads
    # the ranking - without it the beam keeps the states that look best now and drops the ones that
    # will finish.
    dead = ((tasks.latest[None, :] < child_free.min(axis=1)[:, None]) & ~child_done).sum(axis=1)
    hands_used = (child_free[:, first_hand:] > start_hours[first_hand:]).sum(axis=1)
    makespan = child_free.max(axis=1)
    order = np.lexsort((child_travel, makespan, hands_used, dead))[:beam]

    out_done = _empty_like(child_done, beam)
    out_when = _empty_like(child_when, beam)
    out_who = _empty_like(child_who, beam)
    out_free = _empty_like(child_free, beam)
    out_where = _empty_like(child_where, beam)
    out_travel = np.zeros((beam,), dtype=np.int16)
    out_live = np.zeros((beam,), dtype=bool)
    out_count = np.zeros((beam, tasks.n), dtype=np.int16)
    k = order.size
    out_done[:k], out_when[:k] = child_done[order], child_when[order]
    out_who[:k] = child_who[order]
    out_free[:k], out_where[:k] = child_free[order], child_where[order]
    out_travel[:k], out_live[:k] = child_travel[order], True
    out_count[:k] = child_count[order]
    return out_done, out_when, out_who, out_free, out_where, out_travel, out_live, out_count


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
    # A dictionary over the rows' bytes, not : the two answer the same
    # thing, and the unique sorts a 2-D array of void rows - 1.667 ms a call at a beam of 256, which
    # was 81.8 per cent of the selection and about half the whole search. The rows are few, so
    # walking them in Python costs 0.069 ms for the same answer, twenty-four times less.
    first: dict[bytes, int] = {}
    for index, row in enumerate(signature):
        first.setdefault(row.tobytes(), index)
    # No sort: a dictionary keeps insertion order and the rows are walked in order, so the first
    # occurrence of each distinct row is already in increasing order.
    return np.fromiter(first.values(), dtype=np.int64, count=len(first))


def _last_drop(done, when, who, drop_rows: np.ndarray, workers: int) -> np.ndarray:
    """The turn each worker last dropped at, per route - the bag is empty after it.

    Scattered rather than derived from a matrix: the drops are a handful of rows against the day's
    hundreds, so this is the one place where walking them beats an array the size of the list.
    """
    batch = who.shape[0]
    out = np.full((batch, workers), -1, dtype=np.int16)
    worker = who[:, drop_rows]
    live = done[:, drop_rows] & (worker >= 0)
    if live.any():
        rows = np.broadcast_to(np.arange(batch)[:, None], live.shape)
        np.maximum.at(out, (rows[live], worker[live]), when[:, drop_rows][live])
    return out


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
    # A task may have several predecessors, so the answer for a successor is the LARGEST of their
    # finishes - and the edges are grouped by successor, so that is one segment reduction per
    # successor rather than a scatter per edge. `maximum.at` answered the same thing at 83 per cent
    # of this function's time; it is the slowest way numpy has to write one element at a time.
    order, starts, targets = tasks.edge_groups
    if order.size == 0:
        return out
    values = when[:, tasks.edge_before[order]] + 1
    out[:, targets] = np.maximum.reduceat(values, starts, axis=1)
    return out


def _flat(where: np.ndarray) -> np.ndarray:
    """Cells as distance-matrix rows: `y * size + x`."""
    return (where[:, :, 0].astype(np.int32) * BOARD_SIZE + where[:, :, 1].astype(np.int32))


def _carried(done, who, tasks: TaskArray, workers: int) -> np.ndarray:
    """Whether each worker's bag holds each good, per route - (batch, workers, goods).

    Derived from the route rather than tracked: a worker holds a good exactly when it has done a
    task that consumes it, because the trip that brought it is the same worker's. Two arrays that
    cannot disagree are worth more than one kept in step by hand.
    """
    batch = who.shape[0]
    # A bag holds a good when the worker did a task that NEEDED it - the trip that brought it - or
    # one that YIELDED it, because a harvest puts the crop in the bag it is carried in. No task
    # does both, so one column answers for every row.
    goods = np.where(tasks.yields >= 0, tasks.yields, tasks.items)
    n_goods = max(int(goods.max()) + 1, 1) if goods.size else 1
    flat = np.zeros(batch * workers * n_goods, dtype=bool)
    consuming = np.flatnonzero(goods >= 0)
    if consuming.size == 0:
        return flat.reshape(batch, workers, n_goods)
    # One scatter for the whole batch: (route, worker, good) -> a cell of the bag. Duplicates are
    # harmless - the cell is a boolean and every write sets it the same way.
    worker = who[:, consuming]
    live = done[:, consuming] & (worker >= 0)
    if live.any():
        rows = np.arange(batch)[:, None]
        index = ((rows * workers) + np.where(live, worker, 0)) * n_goods + goods[consuming]
        flat[index[live]] = True
    return flat.reshape(batch, workers, n_goods)


def _fetch_mask(tasks: TaskArray) -> np.ndarray:
    """Which tasks are fetches, by the action they carry rather than by their name.

    A fetch is the one task that happens at a door instead of on a tile, so the distinction has to
    come from the action - reading it off the task id would break the moment ids are renamed.
    """
    from agent.world.model import UnitAction
    from agent.wsr.tasks import action_codes
    code = action_codes().get(UnitAction.PICKUP, -1)
    return tasks.actions == code
