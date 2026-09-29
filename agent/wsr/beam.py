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
from pathlib import Path
from typing import NamedTuple

import numpy as np

from agent.world.board import MOVE_DELTA, SPAWN
from agent.world.rules import BOARD_SIZE, TURNS_PER_DAY
from agent.wsr.routing import nearest_shed, walk
from agent.wsr.tasks import (DISTANCE, ITEM_CODE, NO_ITEM, SHED_INDEX, TaskArray,
                             index_of, spanning_walk)

Cell = tuple[int, int]

DOOR_OF: np.ndarray = np.asarray(
    [index_of(nearest_shed((i // BOARD_SIZE, i % BOARD_SIZE))) for i in range(BOARD_SIZE ** 2)],
    dtype=np.int32)


DROP_WALK: np.ndarray = DISTANCE[np.arange(BOARD_SIZE ** 2), DOOR_OF].astype(np.int16)
DOOR_TO: np.ndarray = DISTANCE[DOOR_OF, :]

DOOR_CELL: np.ndarray = np.stack([DOOR_OF // BOARD_SIZE, DOOR_OF % BOARD_SIZE], axis=1).astype(np.int16)

REFETCH: np.ndarray = (DISTANCE[np.arange(BOARD_SIZE ** 2), DOOR_OF][:, None]
                       + DOOR_TO).astype(np.int16)

ITEM_NAME: dict[int, str] = {code: item.name for item, code in ITEM_CODE.items()}

_TRANS_PATH = Path(__file__).resolve().parents[1] / "artifact" / "spatial_transitions.npy"
SPATIAL_TRANSITIONS: np.ndarray = (
    np.load(_TRANS_PATH) if _TRANS_PATH.exists()
    else np.zeros((BOARD_SIZE ** 2, BOARD_SIZE ** 2), dtype=np.float32)
)

def _stand_for(tasks: TaskArray, row: int, at: tuple[int, int]) -> tuple[int, int]:
    """Where a worker must stand to do a task: the tile it works, since no task is a fetch.

    The trip a consumer makes through a door is priced on the consumer and written by the compiler,
    so a task never has a door for a destination.
    """
    return (int(tasks.cells[row][0]), int(tasks.cells[row][1]))

#: What one step of the search is allowed to cost, as `beam x workers x tasks`. A step's arrays are
#: that product wide, so a fixed width makes a hundred tiles cost fifty times a quadrant. This is the
#: product a 16-wide beam has on a 200-task day with 16 workers - a shape measured to lose nothing
#: against 64 on any of the objective's four layers.
STEP_WORK = 16 * 16 * 200
#: The measured ends of the useful range: below 8 a day starts losing tasks (4 placed 47 of 50 where
#: 8 placed all of them), and above 32 nothing improved on any layer.
MIN_BEAM, MAX_BEAM = 8, 64
#: How many times the pickup charge is re-derived before the conservative day is handed back. The
#: charge only grows (each pass keeps the larger of the two) and is bounded by the day's own
#: distinct-good count, so this is a ceiling on an iteration that has usually settled by the first.
CHARGE_PASSES = 1

#: The rankings the selection keeps its beam under, each with a beam of its own. No single key is
#: right: the earliest finish keeps the most work placed and is blind to an hour that is nearly gone,
#: the least slack sees only the hour, and the deadline count chases the tasks that must land by one.
#: A narrow beam loses work (the mixed day: 50 of 54 tasks at a third of the width, 53 at two
#: thirds, 54 at the full one), so the rankings do not share a width - they each keep one.
RANKINGS: tuple[str, ...] = ("finish", "slack", "bound")


def beam_for(tasks: TaskArray, workers: int) -> int:
    """The width to search a day of this size with this many workers."""
    return int(min(MAX_BEAM, max(MIN_BEAM, STEP_WORK / max(1, workers * tasks.n))))

BIG = np.int16(30000)

#: Where the farmer stands when a day begins - the shed's first door, which is where the engine
#: respawns it every night. Not an input: the day resets to the farmer there with no hands at all.
FARMER_START: Cell = SPAWN


@dataclass(frozen=True)
class Day:
    """The chains to run today, in the planner's order, with the shed's timetable and the hands.

    `hands` is the planner's offer: how many hands it will hire today, besides the farmer who is
    always on the field. Each starts at the earliest hour the engine allows (`rules.hire_hour`: a
    hand hired in turn t acts from t + 1, ten orders a turn) unless the caller passes `hire_times` -
    a recorded day whose hands really began later passes its own hours, and then `hands` is their
    count. Zero hands is a day the farmer walks alone.
    """

    chains: tuple[tuple[Cell, tuple[str, ...], str | None], ...]
    available: dict[str, int]
    horizon: int = TURNS_PER_DAY
    hire_times: tuple[int, ...] = ()        # the hour each hand is AVAILABLE to act

    def __post_init__(self) -> None:
        # ONE source for the day's labour: this tuple. Its length is the count, and
        # each entry is the hour that hand is AVAILABLE — the hour the HOURLY layer
        # says its HIRE settles plus one (F040), not the turn the order is queued in
        # and not a number the engine can be asked for.
        #
        # No hours are derived here on purpose: a default computed inside `Day`
        # hides where the day's hands came from, and the engine's own earliest
        # (`rules.earliest_hire_times`) is an optimistic BOUND that belongs at the
        # call site, where the hourly layer will replace it.
        times = tuple(int(t) for t in self.hire_times)
        if any(t < 0 for t in times):
            raise ValueError(f"hire hours must be >= 0, got {times}")
        object.__setattr__(self, "hire_times", times)

    @property
    def hands(self) -> int:
        """How many hands the day is priced with — the tuple's own length."""
        return len(self.hire_times)

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
    #: The worker-turns the route left unspent, walks included: what more work could be laid on the
    #: same hands. Zero until `search` fills it, which is the only place the pool is known.
    spare: int = 0
    #: True when a deadline stopped the search before it ran out of tasks to place. A route that is
    #: neither complete nor out of time is the search's own answer: the pool could not carry it.
    out_of_time: bool = False
    #: Where each hand is spawned from, as the search priced it: the cells the units already on
    #: the field occupy when the hand's own hire moment comes (F040). wsr computes this from its
    #: own route - it is never handed in, because WHERE a hand lands is the search's answer, while
    #: WHEN it appears is the hourly layer's input (`Day.hire_times`). Always a tuple, never None.
    settled: tuple[Cell, ...] = ()
    #: The door the engine gives each hand, hand by hand, as the search priced it (`_hand_doors`):
    #: the least-occupied shed-access tile at THAT hand's own hire moment, not at the first turn.
    #: This is the one statement of where the hands start - `compile_route`, `check_route` and
    #: `remaining_turns` all write the day from it. Empty only when the pool has no hands.
    doors: tuple[Cell, ...] = ()

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
    state: tuple | None = None


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
    # A spanning tree over the tiles and the doors, not the tile count: a day that works three
    # quadrants pays for the crossings and `tiles - 1` does not.
    walking = spanning_walk(tasks)
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



def remaining_turns(day: Day, tasks: TaskArray, result: Result) -> list[int]:
    """What each worker's day has left once the route is carried out, per worker.

    A worker's capacity is its own day: the horizon less the hour it begins at, so the farmer holds
    one more turn than a hand hired in turn 0 (F040). The route spends a turn per task, a turn per
    tile walked - the walk from where the worker stands to its next task - and the turns its own
    pickups take at its door before the first walk. What is left is the room a manager may lay more
    work into, and it is the same number the compiler leaves as PASS turns: `_run` prices the day
    from these, `compile_route` writes it from the same rule.
    """
    hours = _start_hours(day, result.pool)
    starts = _start_positions(day, result.pool, result.doors)
    per: dict[int, list[tuple[int, str]]] = {}
    for turn, task_id, worker in result.route:
        per.setdefault(int(worker), []).append((int(turn), task_id))

    out: list[int] = []
    for worker in range(len(day.units) + result.pool):
        entries = sorted(per.get(worker, []))
        bag = _bag(tasks, entries)
        here = (int(starts[worker][0]), int(starts[worker][1]))
        # The pickups themselves, and not the wait for the goods to reach the shed: the compiler
        # writes one PICKUP op per good at `max(hour, arrival)` and every turn before that is a PASS,
        # so charging the wait as spent under-reports the room by the gap - measured at one turn on
        # the mixed day, where the farmer begins at hour 0 and the shed opens at hour 1.
        spent = len(bag)
        for _turn, row, fetch in legs(tasks, entries):
            target = leg_target(tasks, row, here)
            spent += len(leg_moves(here, target, fetch)) + 1
            here = target
        out.append(int(day.horizon) - int(hours[worker]) - spent)
    return out


def spare_turns(day: Day, tasks: TaskArray, result: Result) -> int:
    """The worker-turns a route leaves unspent: `remaining_turns`, summed over the pool.

    The hands the pool paid for have the day's turns between them; the route spends one per task, one
    per tile walked, and the turns its own pickups take - a worker that carries goods spends its first
    turns at its door, and those are not room for more work. A manager reads it after `complete` says
    yes, to decide whether to lay more work on the same hands rather than hiring again.
    """
    return sum(remaining_turns(day, tasks, result))


def ceiling_for(day: Day, tasks: TaskArray) -> int:
    """The largest pool worth asking about: the units on the field, plus the hands the day offered.

    `hire_times` is the planner's offer - the hour each hand it will pay for may begin - so a pool
    beyond it is a pool nobody is paying for, and the search has no hour to start those hands on. The
    tasks bound it too, by argument: every unit does at least one task, so a pool larger than the work
    cannot be the smallest carrying one. The smaller of the two wins.

    The offer is the tighter of the two by a wide margin, which is the point: the search used to be
    allowed to hire hands the planner never offered, and answered with pools no one would pay for.
    """
    work_bound = tasks.n + len(day.units)
    offered = len(day.units) + len(day.hire_times)
    return min(work_bound, offered)

def _repair_unplaced(day: Day, tasks: TaskArray, result: Result) -> Result:
    """Fast targeted repair: try to place any remaining unplaced leaf/slack tasks."""
    from agent.wsr.emit import check_route, compile_route

    placed_ids = {t for _, t, _ in result.route}
    unplaced = [tid for tid in tasks.ids if tid not in placed_ids]
    if not unplaced or len(unplaced) > 12:
        return result

    curr_route = list(result.route)
    when_dict = {tid: t for t, tid, w in curr_route if t >= 0}
    by_worker: dict[int, list[tuple[int, str]]] = {}
    for t, tid, w in curr_route:
        by_worker.setdefault(w, []).append((t, tid))

    starts = _start_positions(day, result.pool, result.doors)
    hours = _start_hours(day, result.pool)

    improved = False
    for u_id in unplaced:
        u_idx = tasks.ids.index(u_id)
        if tasks.items[u_idx] >= 0:
            continue  # only repair itemless tasks (water, harvest, care, dig)
        u_cell = tuple(tasks.cells[u_idx])
        preds = [tasks.ids[j] for j in range(tasks.n) if tasks.pred[u_idx, j]]
        if any(p not in when_dict for p in preds):
            continue
        t_min = max([when_dict[p] + 1 for p in preds] + [int(tasks.earliest[u_idx]), 0])
        t_max = min([int(tasks.latest[u_idx]), int(day.horizon) - 1])

        best_cand = None
        best_t = 999
        for w in range(len(day.units) + result.pool):
            w_ops = sorted(by_worker.get(w, []))
            if not w_ops:
                last_t = int(hours[w]) - 1
                last_cell = tuple(starts[w])
            else:
                last_t, last_id = w_ops[-1]
                last_idx = tasks.ids.index(last_id)
                last_cell = tuple(tasks.cells[last_idx])
            dist = abs(last_cell[0] - u_cell[0]) + abs(last_cell[1] - u_cell[1])
            arr_t = last_t + 1 + dist
            t_cand = max(t_min, arr_t)
            if t_cand <= t_max and t_cand < best_t:
                best_t = t_cand
                best_cand = (t_cand, u_id, w)

        if best_cand is not None:
            cand_route = sorted(curr_route + [best_cand])
            cand_res = result._replace(route=cand_route, complete=(len(cand_route) == tasks.n))
            if not check_route(day, tasks, cand_res):
                try:
                    compile_route(day, tasks, cand_res)
                    curr_route = cand_route
                    when_dict[u_id] = best_cand[0]
                    by_worker.setdefault(best_cand[2], []).append((best_cand[0], u_id))
                    improved = True
                except Exception:
                    pass

    if improved:
        return result._replace(route=curr_route, complete=(len(curr_route) == tasks.n))
    return result


_SEARCH_CACHE_RETIRED = True


def clear_search_cache() -> None:
    """Legacy no-op: `_SEARCH_CACHE` is retired — the self-warm memory
    (`agent.wsr.selfwarm`) replaced it, and it stores seeds, not answers.
    Kept as a no-op so existing callers and tests keep working."""


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
    def done(result: Result) -> Result:
        """The answer with its spare capacity on it. `search` is the only place the pool is known,
        and the spare is counted against the hands that pool paid for."""
        if not result.complete:
            result = _repair_unplaced(day, tasks, result)
        return result._replace(spare=spare_turns(day, tasks, result))

    if tasks.n == 0:
        return done(Result(0, [], True))

    # The arithmetic floor is on the WORKERS a day needs, and the units already on the field are
    # workers, so what has to be hired is the shortfall. Without this the search starts at one hand
    # and stops there, paying the ladder for a day the farmer could have carried alone.
    floor = max(0, lower_bound(day, tasks) - len(day.units))
    start = floor if hands is None else int(hands)
    bound = ceiling_for(day, tasks)
    ceiling = bound if max_hands is None else min(int(max_hands), bound)
    if ceiling < floor or start > ceiling:
        return done(Result(ceiling, [], False, infeasible=True))

    # The warm memory stands where `_SEARCH_CACHE` stood (#206) and fixes what
    # made that cache unsound: entries are SEEDS, never answers — the search
    # always runs — and the key is the (category x quadrant) logistic
    # signature plus the hands-per-hour vector, so a stored route is only
    # offered to a day whose logistic problem it fits. When the caller passed
    # their own warm, that takes priority and the memory is not consulted.
    if warm is None and hands is not None:
        from agent.wsr import selfwarm as _warm_mod
        _cands = _warm_mod.candidates_for(day, tasks, hands)
        if _cands:
            warm = _cands[0]

    def width(pool: int) -> int:
        return beam if beam is not None else beam_for(tasks, len(day.units) + pool)

    def seed(pool: int) -> Result | None:
        return warm if warm is not None and warm.pool == pool else None

    BRANCH_IDLE[0] = (len(day.units) + start) >= 2 * lower_bound(day, tasks)

    if ceiling > start:
        # THE MULTIPOOL PASS (the owner's design): every pool in the offer's range
        # searched SIMULTANEOUSLY in one vectorised pass - the smallest pool that
        # carries the day is found, and larger pools switch off the moment a smaller
        # one completes. ~6x cheaper than the halving it replaces, and the minimum is
        # exact rather than probed. The pass finds the pool; the ANSWER is refined by
        # the standard fixed point AT that pool, warm from the pass's route - the
        # charge and the doors are settled by the machinery that owns them.
        per_pool = _range_multipool(day, tasks, width, start, ceiling,
                                    branch_floor=lower_bound(day, tasks))
        carried = sorted(pool for pool, r in per_pool.items() if r.complete)
        if carried:
            # The pass only FINDS the smallest carrying pool; the ANSWER is the
            # standard cold fixed point at that pool - its charge/door machinery
            # owns the route, and a warm seed from the charge-free pass measured
            # worse (the warm row's basin fits a different world).
            pool = carried[0]
            BRANCH_IDLE[0] = (len(day.units) + pool) >= 2 * lower_bound(day, tasks)
            ans = done(_fixed_point(day, tasks, width(pool), pool))
        else:
            pool, _partial = max(per_pool.items(), key=lambda item: len(item[1].route))
            BRANCH_IDLE[0] = (len(day.units) + pool) >= 2 * lower_bound(day, tasks)
            ans = done(_fixed_point(day, tasks, width(pool), pool))
    else:
        BRANCH_IDLE[0] = (len(day.units) + start) >= 2 * lower_bound(day, tasks)
        ans = done(_fixed_point(day, tasks, width(start), start, seed(start)))
    return ans


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
                 warm: Result | None = None) -> Result:
    """One pool, searched until the pickups and the hands' doors stop moving.

    Two things the search prices are properties of the ROUTE it produces rather than of the day: the
    pickups each worker's own bag needs, and the doors the hands land on (a unit that walks off its
    door in the first turn moves every hand hired after it, F040). The doors are settled in `_settle`
    and the pickups are charged here, because the charge changes which route the search finds and the
    route changes the charge.

    The charge is a ceiling: a route priced with fewer turns at its door than its own bags need cannot
    be written (the compiler puts the pickups there), so the ceiling may only grow, and it grows to
    whatever the last route asked for. A route that asks for LESS than it was charged is a route with
    turns nobody spends - its workers wait at their doors for pickups they never make - so once the
    ceiling stops moving, the pass is repeated with the route's own bags: priced with exactly the
    pickups it makes, or not returned at all.

    The answer is the day-wide charge's own day, improved only by a route that compiles and beats it
    by the search's own ordering (`_better_route` - carried first, then more work, then the earlier
    stop). The conservative pass is therefore always in hand: a deadline can cut a later pass short,
    and a half-built attempt must never replace the fuller day the first pass found.
    """
    conservative = _settle(day, tasks, beam, pool, warm=warm)
    if conservative.complete:
        # The day-wide charge already carries the day, so the tightening pass has nothing to win on
        # completeness - and it costs a search per pass, which on a real day is the budget the caller
        # gave (F046). The day-wide charge is a ceiling, so the day it found is the day.
        return conservative
    best: Result = conservative
    charge = bags_of(day, tasks, conservative)
    tightened = False
    for _attempt in range(CHARGE_PASSES + 1):
        candidate = _settle(day, tasks, beam, pool, warm=warm, charge=charge)
        if _consistent(day, tasks, candidate) and _better_route(candidate, best):
            best = candidate
        bags = bags_of(day, tasks, candidate)
        grown = [charged | bag for charged, bag in zip(charge, bags)]
        if grown != charge:
            charge = grown
            continue
        if tightened:
            break
        tightened = True
        charge = bags
    return best


def _consistent(day: Day, tasks: TaskArray, result: Result) -> bool:
    """Whether the day that would be compiled is the day that was priced, and one that compiles.

    Two things the compiler derives from the route have to agree with what the search priced:

      the doors    the hands land where the engine puts them at their own hire moment
                   (`_hand_doors`, F040), and a unit that walks off its door moves every hand hired
                   after it. A route priced from doors the day does not have is one the engine
                   silently scrambles rather than refuses, so the doors are checked here by hand. A
                   pool with no hands has no doors, and the farmer's own start is not a choice.
      the rest     the pickups each worker's own bag needs, the walks that carry them, and the turns
                   the tasks were given: `check_route` re-derives all of it from the route and names
                   what the engine's rules would refuse, and the compile itself is the last word -
                   `check_route` does not cover the walk that carries a worker between tasks, and
                   `compile_route` refuses rather than pads a walk that does not fit.

    A route that fails either is not an answer the search may hand back.
    """
    from agent.wsr.emit import check_route, compile_route   # emit imports this module: late

    if tuple(result.doors) != _hand_doors(day, tasks, result, result.pool):
        return False
    if check_route(day, tasks, result):
        return False
    try:
        compile_route(day, tasks, result)
    except ValueError:
        return False
    return True


def _settle(day: Day, tasks: TaskArray, beam: int, pool: int,
            warm: Result | None = None, charge=None) -> Result:
    """One pool at one pickup charge, searched until the hands stop moving.

    The hands land on the doors that are free WHEN THEY ARE HIRED - each hand at its own hour, and a
    unit walking off a door in the first turn changes which doors those are. So the doors are settled
    against the route the search itself produces (`_hand_doors`) and the day is searched again until
    they agree - a fixed point, and a cheap one: it converges in two passes or not at all.
    """
    doors = None
    result = _run(day, tasks, hands=pool, beam=beam, warm=warm, charge=charge)
    for _attempt in range(3):
        nxt = _hand_doors(day, tasks, result, pool)
        if doors is not None and nxt == doors:
            break
        doors = nxt
        result = _run(day, tasks, hands=pool, beam=beam, doors=doors,
                      warm=warm, charge=charge)
    final = _hand_doors(day, tasks, result, pool)
    if tuple(result.doors) != final:
        repriced = _run(day, tasks, hands=pool, beam=beam, doors=final,
                        warm=warm, charge=charge)
        if repriced.complete or len(repriced.route) >= len(result.route):
            result = repriced
        else:
            result = result._replace(doors=final)
    return result







def _range_multipool(day: Day, tasks: TaskArray, width, lo: int, hi: int,
                     branch_floor: int) -> dict[int, Result]:
    """Every pool in [lo, hi] searched SIMULTANEOUSLY in one vectorised pass.

    The beam's rows are grouped by pool; a group's worker columns beyond its own
    pool are masked dead. All groups step in lockstep; each band's best route is
    tracked as the pass goes (a dying pool's last placed count is kept, not read
    off the final arrays), and the smallest pool that carries the day switches
    every larger one off. Returns {pool: best Result} for every pool.
    """
    pools = list(range(lo, hi + 1))
    n_groups = len(pools)
    n = tasks.n
    rankings = _rankings_for(tasks)
    max_pool = hi
    beam = width(max_pool)
    rows = beam * len(rankings)
    m = len(day.units) + max_pool
    total = rows * n_groups

    group_of = np.repeat(np.arange(n_groups), rows)
    pool_of = np.asarray(pools, dtype=np.int16)[group_of]
    band_of = group_of.astype(np.int16)

    max_day = Day(chains=day.chains, available=day.available,
                  hire_times=(tuple(day.hire_times) + (TURNS_PER_DAY,) * max_pool)[:max_pool],
                  horizon=day.horizon)
    n = tasks.n
    done = np.zeros((total, n), dtype=bool)
    when = np.zeros((total, n), dtype=np.int16)
    who = np.full((total, n), -1, dtype=np.int16)
    hours_of = np.zeros((total, m), dtype=np.int16)
    free = np.zeros((total, m), dtype=np.int16)
    for gi, pool in enumerate(pools):
        sl = slice(gi * rows, (gi + 1) * rows)
        hs = start_hours(Day(chains=day.chains, available=day.available,
                             hire_times=(tuple(day.hire_times)
                                         + (TURNS_PER_DAY,) * pool)[:pool]),
                         tasks, pool, None)
        free[sl, :len(hs)] = hs
        hours_of[sl, :len(hs)] = hs
        hours_of[sl, len(hs):] = BIG
        free[sl, len(hs):] = BIG
    where = np.tile(_start_positions(max_day, max_pool, None)[None, :, :], (total, 1, 1))
    travel = np.zeros((total,), dtype=np.int16)
    live = np.ones((total,), dtype=bool)
    count = np.zeros((total, n), dtype=np.int16)
    load = door_load(tasks, None, _start_hours(max_day, max_pool))

    best: list[tuple | None] = [None] * n_groups

    for _step in range(n):
        if not live.any():
            break
        expanded = _expand(max_day, tasks, done, when, who, free, where, travel,
                           live, count, load)
        if expanded is None:
            break
        done, when, who, free, where, travel, live, count = _select(
            expanded, tasks, beam, len(day.units), hours_of, pool_of,
            hours_of, branch_floor, band_of)
        placed = done.sum(axis=1)
        makespan = free.max(axis=1)
        for gi in range(n_groups):
            sl = slice(gi * rows, (gi + 1) * rows)
            rows_band = np.flatnonzero(live[sl])
            if rows_band.size == 0:
                continue
            idx = np.lexsort((makespan[sl][rows_band],))[0]
            r = rows_band[idx]
            cand_placed = int(placed[sl][r])
            cur = best[gi]
            if cur is None or cand_placed > cur[0] or (
                    cand_placed == cur[0] and makespan[sl][r] < cur[1]):
                best[gi] = (cand_placed, int(makespan[sl][r]),
                            done[sl][r].copy(), when[sl][r].copy(),
                            who[sl][r].copy(), free[sl][r].copy(),
                            where[sl][r].copy(), int(travel[sl][r]))
        # Early exit: the smallest completing pool switches off the larger ones.
        for gi in range(n_groups):
            if best[gi] is not None and best[gi][0] == n:
                live[(gi + 1) * rows:] = False
                break

    out: dict[int, Result] = {}
    for gi, pool in enumerate(pools):
        b = best[gi]
        if b is None:
            out[pool] = Result(pool, [], False)
            continue
        placed_n, _mk, d, w, wh, f, wr, tv = b
        route = [(int(w[i]), tasks.ids[i], int(wh[i]))
                 for i in np.argsort(w) if d[i]]
        out[pool] = Result(pool, route, placed_n == n,
                           state=(d, w, wh, f, wr, tv))
    return out


def first_hand_len(day: Day) -> int:
    """The units before the first hired hand: the farmer, and nobody else."""
    return len(day.units)


def _warm_row(tasks: TaskArray, warm: Result, done, when, who, free, where, travel, count) -> None:
    """Write a previous route's state into the beam's first row.

    A route is a set of `(turn, task, worker)` and the state is what that set implies, so this is a
    translation rather than a re-search. A task the day does not have is skipped, and so is a worker
    the pool does not have: a caller that hands over a route from another instance gets the part of
    it that still applies, not an error.
    """
    row = 0
    first_hand = 1
    if warm.state is not None and free.shape[1] >= warm.state[3].size:
        w_doors = tuple(warm.doors) if hasattr(warm, "doors") else ()
        cur_doors = tuple((int(c[0]), int(c[1])) for c in where[0, first_hand:])
        if w_doors and w_doors == cur_doors:
            w_done, w_when, w_who, w_free, w_where, w_travel = warm.state
            n_c = min(tasks.n, w_done.size)
            m_c = min(free.shape[1], w_free.size)
            done[row, :n_c] = w_done[:n_c]
            when[row, :n_c] = w_when[:n_c]
            who[row, :n_c] = w_who[:n_c]
            free[row, :m_c] = w_free[:m_c]
            where[row, :m_c] = w_where[:m_c]
            travel[row] = w_travel
            np.add.at(count[row], tasks.edge_after, done[row, tasks.edge_before].astype(np.int16))
            return

    column = {task_id: index for index, task_id in enumerate(tasks.ids)}
    # Where each worker stands before its first task: the row's own start, taken before the loop
    # writes over it. The walk is measured from here, not from the worker's last tile.
    start = where[row].copy()
    per_worker: dict[int, list[tuple[int, str]]] = {}
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
        per_worker.setdefault(worker, []).append((turn, task_id))

    # The walk the route took, leg by leg as the compiler writes it (`legs`), so the warmed row is
    # neither credited a travel it did not pay nor put on a tile it does not end on - an idle drop
    # has no leg, so it moves nobody.
    walked = 0
    for worker, entries in per_worker.items():
        at = (int(start[worker, 0]), int(start[worker, 1]))
        for _turn, index, fetch in legs(tasks, entries):
            target = leg_target(tasks, index, at)
            walked += sum(1 for op in leg_moves(at, target, fetch) if op[0] in MOVE_DELTA)
            at = target
        where[row, worker] = at
    travel[row] = min(walked, int(np.iinfo(np.int16).max))
    # The counter the warmed row's own placements imply: one per edge whose predecessor it placed.
    np.add.at(count[row], tasks.edge_after, done[row, tasks.edge_before].astype(np.int16))


def _run(day: Day, tasks: TaskArray, *, hands: int, beam: int,
         doors=None,
         warm: Result | None = None, charge=None,
         pool_of: np.ndarray | None = None,
         hours_of: np.ndarray | None = None,
         branch_floor: int = 0,
         band_of: np.ndarray | None = None) -> Result:
    """One pool size: search the day, and report how much of it the pool could carry.

    `doors` is where the hands start. None is the first pass of `_settle`, before any route exists:
    the hands are then placed from where the units stand at the start of the day (`_start_positions`),
    and the answer carries those doors so what it was priced from is never implicit.
    """
    n = tasks.n
    start_pos = _start_positions(day, hands, doors)          # (m, 2)
    first_hand = len(day.units)                              # workers before this index are units

    # The warmed route gets a row of its own ON TOP of the beam, so a caller handing over a route
    # that no longer fits the day costs the search nothing: the beam below it is as wide as it would
    # have been. The selection still keeps `beam` rows, so from the second generation on the seed
    # competes for a slot like any other route.
    rows = beam * len(_rankings_for(tasks)) + (1 if warm is not None else 0)
    done = np.zeros((rows, n), dtype=bool)
    when = np.zeros((rows, n), dtype=np.int16)               # the turn each done task occupies
    who = np.full((rows, n), -1, dtype=np.int16)             # and the worker that did it
    # Every worker starts at its own hour: the farmer at the day's first, a hand at the hour the
    # planner offered it. Starting them all together would hand the search turns the engine will
    # not give, which is how a day gets called feasible that the harness then truncates. The door
    # load is charged before the first walk (`start_hours`), except for door work before the goods
    # land (`door_load`, `_expand`).
    hours = start_hours(day, tasks, hands, charge)
    load = door_load(tasks, charge, _start_hours(day, hands))
    free = np.tile(hours[None, :], (rows, 1)).astype(np.int16)
    where = np.tile(start_pos[None, :, :], (rows, 1, 1))
    if pool_of is not None:
        # Multi-pool pass: columns beyond each row-group's pool are masked dead -
        # free = BIG so no candidate ever wins there; per-row start hours.
        dead = np.arange(free.shape[1])[None, :] >= pool_of[:, None]
        free[dead] = BIG
        if hours_of is not None:
            free[~dead] = hours_of[~dead]
    travel = np.zeros((rows,), dtype=np.int16)
    live = np.ones((rows,), dtype=bool)
    # How many of each task's predecessors are done, per route. `ready` is a comparison on this,
    # where it used to be a (beam, n) by (n, n) product - a billion multiply-adds over a day.
    count = np.zeros((rows, n), dtype=np.int16)

    if warm is not None:
        _warm_row(tasks, warm, done, when, who, free, where, travel, count)

    # The best state is remembered as the search goes, because the beam's last generation can be
    # empty - a route that dies at the end would otherwise erase the work it had already placed.
    best = _snapshot(done, when, who, free, where, travel, first_hand, hours)
    for _step in range(n):
        expanded = _expand(day, tasks, done, when, who, free, where, travel, live, count, load)
        if expanded is None:
            break
        done, when, who, free, where, travel, live, count = _select(
            expanded, tasks, beam, first_hand, hours, pool_of, hours_of,
            branch_floor, band_of)
        if not live.any():
            break
        here = _snapshot(done, when, who, free, where, travel, first_hand, hours)
        if _better(here, best):
            best = here

    placed, when_best, who_best, done_best, state_best = best
    complete = placed == n
    route = [(int(when_best[i]), tasks.ids[i], int(who_best[i]))
             for i in np.argsort(when_best) if done_best[i]]
    return Result(hands, route, complete,
                  doors=tuple((int(c[0]), int(c[1])) for c in start_pos[first_hand:]),
                  state=state_best)


def _snapshot(done, when, who, free, where, travel, first_hand, start_hours):
    """The best route in the beam right now, by the layered objective: work, hands, makespan, walk.

    A hand counts as put to work when its clock has moved past the hour it began at - not when its
    clock is merely above zero, which every hand's is from the moment the offer sets its start.
    """
    placed = done.sum(axis=1)
    hands_used = (free[:, first_hand:] > start_hours[first_hand:]).sum(axis=1)
    makespan = free.max(axis=1)
    row = int(np.lexsort((travel, makespan, hands_used, -placed))[0])
    state = (done[row].copy(), when[row].copy(), who[row].copy(), free[row].copy(), where[row].copy(), int(travel[row]))
    return int(placed[row]), when[row].copy(), who[row].copy(), done[row].copy(), state


def _better(candidate, best) -> bool:
    """Whether `candidate` beats `best`: more work first, then the earliest finish."""
    placed, when = candidate[0], candidate[1]
    best_placed, best_when = best[0], best[1]
    if placed != best_placed:
        return placed > best_placed
    return int(when.max()) < int(best_when.max())


def _stand_after(tasks: TaskArray, route, start, turn: int) -> Cell:
    """Where a unit is after `turn`, read back from its own route.

    The compiler writes every walk to END at its task (`walk_start_turn`), so a unit stays on its
    previous tile until the walk for its next task begins and then moves one step a turn. Reading the
    route with the writer's own rule is what makes this exact rather than a guess about idle turns.
    """
    pos = (int(start[0]), int(start[1]))
    for task_turn, row, fetch in legs(tasks, route):
        target = leg_target(tasks, row, pos)
        moves = leg_moves(pos, target, fetch)
        began = walk_start_turn(task_turn, len(moves))
        if turn < began:
            return pos
        if turn >= task_turn:
            pos = target
            continue
        for step in moves[: turn - began + 1]:
            if step[0] in MOVE_DELTA:          # a refetch's PICKUP is a turn in place
                dx, dy = MOVE_DELTA[step[0]]
                pos = (pos[0] + int(dx), pos[1] + int(dy))
        return pos
    return pos


def _hand_doors(day: Day, tasks: TaskArray, result: Result, hands: int) -> tuple[Cell, ...]:
    """The door each hand lands on, as the engine gives it: at that hand's OWN hire moment.

    The engine settles a turn's hires in `(start_time, index)` order and each one takes the
    least-occupied shed-access tile at that moment (`rules.spawn_cell`, F040). So a hand hired at hour
    five lands from where every unit stands AT HOUR FIVE - not from where they stood when the first
    turn was over, which is what the search used to price it from. `day.hire_times` is the hour each
    offer first acts, so a hand acts from `h` and is hired in the market of turn `h - 1`.
    """
    from agent.world.rules import spawn_cell

    # The hours come from `Day.hire_times` and nowhere else - the same tuple `hands` is the
    # length of, so there is no separate count to disagree with it and zero hires is simply an
    # empty tuple (the farmer is always on the field and is not in this list). No floor and no
    # fallback: rewriting the caller's hour silently is how a day gets priced on hours it does
    # not have. A bad tuple is `Day`'s to refuse, not this function's to mop up.
    hire = [int(h) for h in day.hire_times]
    routes: dict[int, list] = {}
    for turn, task_id, worker in result.route:
        routes.setdefault(int(worker), []).append((int(turn), task_id))
    for worker in routes:
        routes[worker].sort()
    units = [(int(c[0]), int(c[1])) for c in day.units]
    placed: list[tuple[int, Cell]] = []
    for index in sorted(range(hands), key=lambda k: (hire[k], k)):
        at = hire[index] - 1
        occupied = [_stand_after(tasks, routes.get(u, []), units[u], at) for u in range(len(units))]
        # The hands already placed this turn stand on their doors, so they count - at the worker
        # index they have on the field, which is the units' count plus the hand's own index.
        occupied += [_stand_after(tasks, routes.get(len(units) + k, []), cell, at)
                     for k, cell in placed]
        placed.append((index, spawn_cell(occupied)))
    return tuple(cell for _index, cell in sorted(placed))


def _start_positions(day: Day, hands: int, doors=None) -> np.ndarray:
    """Who is on the field and where each stands.

    Two different questions, and conflating them is what put crops on the wrong tiles:

      the units   start the day where they are, so their walks are priced from there and nowhere
                  else. A unit that will walk in the first turn has NOT started halfway.
      the hands   appear on the door the engine gives each at its own hire moment (`_hand_doors`,
                  F040), which the result carries as `doors`.

    Without `doors` - the first pass of the search, before there is a route to read - the hands are
    placed by the spawn rule from where the units stand at the start of the day. Pricing a unit's
    walk from anywhere else credits it a move it has not made yet: the farmer was once started at
    (3,4) and planted (3,3) while the engine had it at (4,4), and the planting was refused in silence.
    """
    from agent.world.rules import spawn_cell
    out = [(int(c[0]), int(c[1])) for c in day.units]
    if doors:
        out.extend((int(c[0]), int(c[1])) for c in doors)
        return np.asarray(out, dtype=np.int16)
    occupied = list(out)
    for _ in range(hands):
        cell = spawn_cell(occupied)
        out.append((int(cell[0]), int(cell[1])))
        occupied.append((int(cell[0]), int(cell[1])))
    return np.asarray(out, dtype=np.int16)


def _bag(tasks: TaskArray, entries: list[tuple[int, str]]) -> dict[int, int]:
    """How much of each good one worker loads at its door (`door_load`, `loads_before`).

    Only the goods used before the worker's first DROP: the engine's DROP empties the whole bag
    (`kaggriculture.py:343-356`), so what is still in it goes to the shed with the harvest, and every
    use after the drop is fetched again (`legs`). A use the worker's own earlier HARVEST or
    COLLECT_FERTILIZER covers is not loaded: the good is already in its bag (`tasks.yield_n` units).
    """
    bag: dict[int, int] = {}
    own: dict[int, int] = {}
    for turn, task_id in sorted(entries):
        row = tasks.ids.index(task_id)
        if int(turn) >= 0 and bool(tasks.is_drop[row]):
            break
        good = int(tasks.items[row])
        if good >= 0:
            if own.get(good, 0) > 0:
                own[good] -= 1
            else:
                bag[good] = bag.get(good, 0) + 1
        made = int(tasks.yields[row])
        if made >= 0:
            own[made] = own.get(made, 0) + int(tasks.yield_n[row])
    return bag


def preload_turns(tasks: TaskArray) -> frozenset[int]:
    """The goods a worker may have to load at its door: every good a task of the day consumes.

    Charging every worker the day's own goods is a ceiling, not a guess: a worker that needs fewer
    leaves the difference idle, and one that needs them all has exactly its room.
    """
    return frozenset(int(g) for g in np.unique(tasks.items[tasks.items >= 0]))


def good_hours(tasks: TaskArray) -> dict[int, int]:
    """The hour each good a task consumes is in the shed - the timetable, per good.

    A PICKUP before its good is in the shed is refused in silence (F047), so each good is loaded at
    its OWN hour: a cow bought in turn 0 is not in the shed at hour 0 however early the wheat is.
    """
    out: dict[int, int] = {}
    for good, hour in zip(tasks.items.tolist(), tasks.earliest.tolist()):
        if good >= 0:
            out[int(good)] = max(out.get(int(good), 0), int(hour))
    return out


def pickup_turns(hour: int, goods, arrival: dict[int, int]) -> list[tuple[int, int]]:
    """The turn each good is loaded at the worker's door: `(turn, good)`, in the order they are taken.

    One turn per good, from the worker's own first hour, each no earlier than its good is in the
    shed - so the goods that are already there go first, and the walk waits only for the last one.
    """
    out: list[tuple[int, int]] = []
    turn = int(hour)
    for good in sorted(goods, key=lambda g: (arrival.get(int(g), 0), int(g))):
        turn = max(turn, arrival.get(int(good), 0))
        out.append((turn, int(good)))
        turn += 1
    return out


def walk_start_turn(turn: int, moves: int) -> int:
    """The turn the compiler writes a walk in: it ENDS at the task's turn.

    The writer (`compile_route`) and the reader that works out where a unit stands at a turn
    (`_stand_after`) both go through here, so "the unit's first op is a move" and
    "the walk starts at turn 0" cannot become two different questions. The rewrite that introduced
    this layer answered the reader's half with a guess at the gap between the walk and the task while
    the writer kept starting walks at the earliest free turn; the two disagreed wherever a first task
    had slack, which moved every hand's door (F040) and made the day the engine ran a different day.
    The pre-rewrite `plan_day` had it right by reading the compiled route's first op; this is that
    reading, spelled once.
    """
    return int(turn) - int(moves)


def first_walk_turn(hour: int, goods, arrival: dict[int, int]) -> int:
    """The turn the compiler writes a worker's first walk in - `compile_route`'s `last + 1`.

    A worker with an empty bag walks from the turn its own day begins; one that carries goods loads
    them at its door first (`pickup_turns`) and walks after the last one - unless its first task is
    door work before its goods land (`loads_before`), which is done first. The compiler writes the ops
    with this rule and the model counts where the units stand after the first turn with it, so the
    day that is written and the day that was priced cannot disagree.
    """
    loads = pickup_turns(hour, goods, arrival)
    return int(hour) if not loads else loads[-1][0] + 1


def grow_charge(hour: int, charged, bag, arrival: dict[int, int]):
    """The larger of two pickup charges, by the turn each starts the worker's walk.

    The charge `_fixed_point` iterates may only grow, and what grows is the delay it puts on the
    walk - not the goods it names. Their union charges a pickup turn neither route makes: a worker
    that loads one good on one pass and another good on the next needs one turn, not two.
    """
    if first_walk_turn(hour, charged, arrival) >= first_walk_turn(hour, bag, arrival):
        return charged
    return bag


def door_work(tasks: TaskArray) -> np.ndarray:
    """The tasks a worker may do before it loads its goods: work on a shed-access tile that needs none.

    A worker that has only worked the four shed-access tiles is still standing at the shed, where any
    PICKUP is taken (`kaggriculture.py:138-139`, `:358-375`), so it can load there whenever it needs
    to. Doing such a task first only pays in the turns before the worker's goods are in the shed -
    goods bought in turn 0 are there from hour 1, and the farmer's hour 0 is free - so it counts as
    door work only at a turn before the day's first good lands (`day_first_good`); at any later
    turn it loads first, as every other task does.
    """
    on_door = np.isin(tasks.cell_index, SHED_INDEX)
    return on_door & (tasks.items < 0) & ~tasks.is_drop


class DoorLoad(NamedTuple):
    """Each worker's door load: what it puts in the bag, the order it is taken in, and when."""

    #: (workers, goods): the goods each worker's load puts in its bag until its first DROP.
    held: np.ndarray
    #: Per worker, the goods in the order they are picked up (`pickup_turns`).
    order: list[list[int]]
    #: The hour each good is in the shed (`good_hours`).
    arrival: dict[int, int]
    #: (n,): the tasks a worker may do before it loads (`door_work`).
    before: np.ndarray
    #: (workers,): the hour the day's first good lands (`day_first_good`); door work before it does
    #: not load.
    first: np.ndarray
    #: (workers,): the hour each worker may first act (`_start_hours`), before any pickup.
    hour: np.ndarray


def start_hours(day: Day, tasks: TaskArray, hands: int, charge=None) -> np.ndarray:
    """The hour each worker's WALK may begin: its own hour, the goods, and the pickups before it.

    `charge` is the goods each worker's own day loads at its door. That is a property of the route
    (which worker does what), so the search iterates it (`_fixed_point`) and this is where the
    iteration lands. Without it every worker is charged the day's whole set of goods: a ceiling that
    is always safe and costs the search the turns nobody uses. The one exception is door work before
    the goods land (`door_work`), priced from the worker's own hour in `_expand`.
    """
    hours = _start_hours(day, hands)
    if charge is None:
        charge = [preload_turns(tasks)] * len(hours)
    arrival = good_hours(tasks)
    return np.asarray([first_walk_turn(h, c, arrival) for h, c in zip(hours, charge)],
                      dtype=np.int16)


def door_load(tasks: TaskArray, charge, hour: np.ndarray) -> DoorLoad:
    """The door load each worker's `charge` names - `preload_turns` for every worker when None.

    `hour` is each worker's own first hour (`_start_hours`). A consumer of a good in the load finds
    it in the bag until the worker's first DROP.
    """
    workers = len(hour)
    goods = np.where(tasks.yields >= 0, tasks.yields, tasks.items)
    n_goods = max(int(goods.max()) + 1, 1) if goods.size else 1
    if charge is None:
        charge = [preload_turns(tasks)] * workers
    arrival = good_hours(tasks)
    held = np.zeros((workers, n_goods), dtype=bool)
    # The day's first good, not the worker's: the compiler knows the worker's bag and not the charge
    # it was searched with, so one number both sides can read (`loads_before`).
    first = np.full(workers, day_first_good(tasks), dtype=np.int16)
    order = []
    for worker, load in enumerate(charge):
        for good in load:
            held[worker, int(good)] = True
        order.append([good for _turn, good in pickup_turns(0, load, arrival)])
    return DoorLoad(held, order, arrival, door_work(tasks), first,
                    np.asarray(hour, dtype=np.int16))


def day_first_good(tasks: TaskArray) -> int:
    """The hour the day's first consumed good is in the shed; door work before it does not load."""
    return min(good_hours(tasks).values(), default=int(BIG))


def loads_before(before: bool, turn: int, first: int) -> bool:
    """Whether a worker that has not loaded yet loads before a task: the one rule, search and writer.

    It does unless the task is door work at a turn before its first good lands (`door_work`).
    """
    return not (before and int(turn) < int(first))


def _load_ready(free: np.ndarray, load: DoorLoad) -> np.ndarray:
    """The turn each worker is free again after loading from `free`: (batch, workers).

    One turn per good, each no earlier than the good is in the shed - `pickup_turns` from `free`.
    """
    out = free.astype(np.int16).copy()
    for worker, goods in enumerate(load.order):
        for good in goods:
            out[:, worker] = np.maximum(out[:, worker], load.arrival.get(good, 0)) + 1
    return out


def done_by_worker(done, who, workers: int) -> np.ndarray:
    """Whether each worker has done any task at all, per route: (batch, workers)."""
    batch = who.shape[0]
    live = done & (who >= 0)
    out = np.zeros(batch * workers, dtype=bool)
    index = np.arange(batch)[:, None] * workers + np.where(who >= 0, who, 0)
    out[index[live]] = True
    return out.reshape(batch, workers)


def _door_first(done, who, when, load: DoorLoad, workers: int) -> np.ndarray:
    """Whether each worker's day so far is door work before its goods landed: (batch, workers).

    Such a worker has not loaded yet: it is still at the shed, and its next task that is not door work
    before the goods land pays the pickups (`_expand`). A worker with no task yet is not in this
    state - its pickups are already in its start (`start_hours`). An idle drop (turn -1) is no leg.
    """
    batch = who.shape[0]
    live = done & (who >= 0) & (when >= 0)
    worker = np.where(who >= 0, who, 0)
    early = load.before[None, :] & (when < load.first[worker])
    any_task = np.zeros(batch * workers, dtype=bool)
    loaded = np.zeros(batch * workers, dtype=bool)
    index = np.arange(batch)[:, None] * workers + worker
    any_task[index[live]] = True
    loaded[index[live & ~early]] = True
    return (any_task & ~loaded).reshape(batch, workers)


def _start_hours(day: Day, hands: int) -> np.ndarray:
    """The hour each worker may first act.

    A unit already on the field acts from the day's first hour. A hand begins at the hour the
    planner offered it, and the offer is not a formality: the engine settles hires in index order
    and a hand hired in turn h acts from h + 1, so a hand the market cannot pay for until turn 5
    does nothing for the first five hours of the day. A hand beyond the offer starts at the
    engine's own hour for it (`rules.hire_hour`).
    """
    from agent.world.rules import hire_hour

    hours = [0] * len(day.units)
    for index in range(hands):
        if index < len(day.hire_times):
            hours.append(max(1, int(day.hire_times[index])))
        else:
            hours.append(hire_hour(index))
    return np.asarray(hours, dtype=np.int16)


#: The take-drop-consume ban's per-call carrier: `_expand` derives the banned
#: (route, worker, column) triples while pricing, before `legal` exists, and
#: the mask is applied once the tie filter has built the final legality.
LEGAL_BAN: list = [None]


def _expand(day: Day, tasks: TaskArray, done, when, who, free, where, travel, live, count,
            load: DoorLoad | None = None):
    """One task added to every live route: the vectorised step.

    For each route and each task, the earliest hour any worker could finish it: walk from where that
    worker stands, wait for the predecessors, wait for the good. The minimum over workers says which
    worker should take it.

    A task that consumes a good its worker does not yet hold pays for the trip: out to a door, the
    pickup, and on to the tile. The second feeding of the same good on the same worker pays nothing
    extra, because the good is already in that worker's bag - which is the saving a day makes when
    it fetches once and eats twice. A good in the worker's door load (`door_load`) is in the bag
    until its first DROP; the load's pickups are paid by the first task that is not `door_work`,
    from the turn the worker is free (`_load_ready`), so work on a shed-access tile can come first.
    """
    b, m = free.shape
    rows = np.flatnonzero(live)
    if rows.size == 0:
        return None
    LEGAL_BAN[0] = None

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
    # Manhattan distance SEPARATES: |wx-cx| + |wy-cy| as int16 arithmetic is
    # 2.3x the table gather (measured on the heavy-day shapes: 0.59 vs 1.34 ms
    # a call) and bit-identical - the table's only job was the abs-diff sum.
    hx = where[:, :, 1].astype(np.int16, copy=False)[:, :, None]
    hy = where[:, :, 0].astype(np.int16, copy=False)[:, :, None]
    cx = tasks.cells[index, 1].astype(np.int16)[None, None, :]
    cy = tasks.cells[index, 0].astype(np.int16)[None, None, :]
    hop = np.abs(hx - cx) + np.abs(hy - cy)
    # A DROP is handed over at the door nearest the worker, not at a door fixed when the day was
    # built (`leg_target`): its walk is the worker's own distance to its nearest door.
    drop_here = tasks.is_drop[index]
    if drop_here.any():
        hop = np.where(drop_here[None, None, :], DROP_WALK[here][:, :, None], hop)

    # Each worker's last DROP, per route. The engine's DROP empties the whole bag, the goods loaded
    # at the door included, so after it a consumer's good is fetched again on the way (`legs`).
    drop_rows = tasks.drop_rows
    last_drop = (_last_drop(done, when, who, drop_rows, m) if drop_rows.size
                 else np.full((b, m), -1, dtype=np.int16))

    # The trip a consumer makes when its good is not in the bag: to a door, the pickup, and on. One
    # door, not two minima - the nearest door to the worker and the nearest to the tile can be
    # different doors, and the compiler would then walk a trip the search never priced.
    # The door load itself is paid before the worker's first task that is not door work
    # (`start_hours`, `door_load`); a trip here is the refetch after a DROP.
    trip = np.zeros(hop.shape, dtype=bool)
    has = np.ones(hop.shape, dtype=bool)
    needs = tasks.items[index] >= 0
    if needs.any():
        carried = _carried(done, who, tasks, m, when, last_drop)  # (b, m, goods)
        if load is not None:
            # The door load is in the bag until the worker's first DROP.
            carried = carried | (load.held[None, :, :carried.shape[2]]
                                 & (last_drop < 0)[:, :, None])
        has = np.where(needs[None, None, :], carried[:, :, tasks.items[index].clip(0)], True)
        trip = ~has
        # After a drop the load at the door is gone, so the trip is a real walk: out to the
        # worker's nearest door, the PICKUP, and on to the tile - the leg `leg_moves` writes.
        refetch = trip & (last_drop >= 0)[:, :, None]
        if refetch.any():
            hop = np.where(refetch, REFETCH[here[:, :, None],
                                            tasks.cell_index[index][None, None, :]], hop)

        # THE OWNER'S BAN (2026-09-25): because the engine's DROP empties the whole bag, a worker's
        # own take of a good may not be followed by that worker's DROP and then a use of the same
        # good - the use would need the mid-day refetch the compiler writes (`legs`), and that is
        # the pattern the owner named as never worth writing. Priced before; now ILLEGAL: a
        # consumer of good g may not land on worker w whose last DROP turn is AFTER w's last take
        # of g. A worker whose take is AFTER its last drop still holds g and is fine.
        n_goods_b = int(max(int(tasks.yields.max()), int(tasks.items.max())) + 1)
        took_turn = np.full((b, m, n_goods_b), -1, dtype=np.int16)
        take_rows = np.flatnonzero(tasks.yields >= 0)            # events that ADD a good
        if take_rows.size:
            w_of = np.where(who[:b, take_rows] >= 0, who[:b, take_rows], 0)
            took_scatter = (((np.arange(b)[:, None] * m
                              + np.minimum(w_of, m - 1)) * n_goods_b
                             + tasks.yields[take_rows][None, :]).ravel())
            turn_of = np.broadcast_to(when[:b, take_rows], (b, take_rows.size)).ravel()
            np.maximum.at(took_turn.reshape(-1), took_scatter, turn_of)
        g_of = tasks.items[index].clip(0)                        # the consumed good per column
        own_take = took_turn[:, :, g_of]
        banned = (needs[None, None, :]
                  & (last_drop >= 0)[:, :, None]
                  & (own_take >= 0)
                  & (last_drop[:, :, None] > own_take))
        if banned.any():
            LEGAL_BAN[0] = banned                                # applied after `legal` is built

    # The drops. A worker that has already dropped since the harvest a drop banks has an empty bag,
    # so that drop hands over nothing: no trip, no turn, and nobody moves. This is what lets one
    # drop bank several harvests instead of one apiece.
    idle = np.zeros(hop.shape, dtype=bool)
    if drop_rows.size:
        # Which of the day's drops are on the frontier, and where they sit in the beam's width.
        local = np.searchsorted(index, drop_rows)
        inside = (local < width) & (index[np.minimum(local, width - 1)] == drop_rows)
        if inside.any():
            held = when[:, tasks.banks[drop_rows].clip(0)]               # (b, d) each harvest's turn
            idle[:, :, local[inside]] = (
                last_drop[:, :, None] > held[:, None, :])[:, :, inside]

    arrive = free[:, :, None] + hop                          # (b, m, w)
    released = _released(when, tasks)[:, index]              # (b, w): the predecessors' finish
    ready = ready_all[:, index]                              # (b, w): every predecessor done
    earliest_here = tasks.earliest[index]
    latest_here = getattr(tasks, "effective_latest", tasks.latest)[index]
    if load is not None and any(load.order):
        # Door work before the goods land (`loads_before`): a worker that has done nothing yet may
        # take it from its own hour, not after its pickups - it is still on the door.
        own = np.maximum(load.hour[None, :], 0)[:, :, None] + hop
        bare = np.maximum(np.maximum(own, released[:, None, :]), earliest_here[None, None, :])
        early = (~done_by_worker(done, who, m)[:, :, None]
                 & load.before[index][None, None, :]
                 & (bare < load.first[None, :, None]))
        arrive = np.where(early, own, arrive)
        # A worker whose day so far is that door work has not loaded yet: its next task that is not
        # door work before the goods land pays the pickups from the turn it is free.
        waiting = _door_first(done, who, when, load, m)[:, :, None]
        if waiting.any():
            bare = np.maximum(np.maximum(arrive, released[:, None, :]), earliest_here[None, None, :])
            still = load.before[index][None, None, :] & (bare < load.first[None, :, None])
            arrive = np.where(waiting & ~still, _load_ready(free, load)[:, :, None] + hop, arrive)

    start = np.maximum(arrive, released[:, None, :])
    start = np.maximum(start, earliest_here[None, None, :])
    # A trip is charged the turn spent fetching and not the walk to the door. The walk is real - the
    # engine makes a PICKUP happen at a door - and pricing it is what makes the search share a bag
    # instead of fetching, but it also takes five days off the corpus: those days were carried with the
    # fetch priced at a turn, and the model's budget is short elsewhere. The issue for the nine days
    # has the measurement; this stays one turn until the budget is right.
    start = start + trip.astype(np.int16)
    finish = start + np.int16(1)
    # An idle drop is finished the moment its worker is free - it costs nothing and takes no turn -
    # and it is written at turn -1, which is the compiler's signal that there is no op to emit.
    finish = np.where(idle, np.maximum(free[:, :, None], released[:, None, :]), finish)
    finish = np.maximum(finish, np.where(idle, np.int16(0), earliest_here[None, None, :]))

    in_time = finish <= day.horizon
    before_latest = start <= latest_here[None, None, :]
    fresh = ~done_by_worker(done, who, m)
    legal = (ready & ~done[:, index] & in_time.any(axis=1) & before_latest.any(axis=1))
    legal = legal[:, None, :] & in_time & before_latest
    # A worker's walk may not begin before the worker exists: a fresh worker's first task at
    # `start` needs its walk in the turns start-hop..start-1, and the earliest of those is the
    # worker's own first hour - so the task cannot sit before own_hour + hop. Without this floor
    # the search committed a first task whose walk crossed the worker's birth and `_room` refused
    # the day (measured: winner[15] d23_harvest, winner[17] d29_collect_fertilizer). `load.hour`
    # IS the workers' own first hours (`_start_hours`, carried by `door_load`); with no load the
    # workers' hours are their raw starts, and `free` holds them (no pickups were priced).
    own_hour = (load.hour[None, :, None] if load is not None
                else free[:, :, None])
    legal = legal & (~fresh[:, :, None]
                     | (hop == 0)
                     | (start >= own_hour + hop))
    # A worker tie: a task and the tasks it must share a worker with - a fetch and the op that
    # consumes it, a drop and the good it banks - are one worker's work. A candidate whose mate is
    # already done by somebody else is not a candidate. Without this a drop could be placed on a
    # worker that never held the good: it hands over nothing, and the day would still be called
    # complete.
    if tasks.ties.size:
        mates = tasks.ties[index]                        # (w, k) row indices, -1 for no mate
        for slot in range(mates.shape[1]):
            mate = mates[:, slot]
            real = mate >= 0
            if not real.any():
                continue
            done_mate = done[:, mate.clip(0)]            # (b, w)
            who_mate = who[:, mate.clip(0)]              # (b, w)
            same = ((~done_mate)[:, None, :]
                    | (who_mate[:, None, :] == np.arange(m)[None, :, None]))
            legal = legal & (same | ~real[None, None, :])
    # The owner's ban, applied once `legal` exists: a take that the worker's own drop has since
    # emptied out of the bag forbids that worker every later use of the same good. It is the
    # anti-pattern the compiler used to answer with a mid-day refetch pickup - never worth
    # writing, so it is a rule now, not a price.
    if LEGAL_BAN[0] is not None:
        legal = legal & ~LEGAL_BAN[0]
    finish = np.where(legal, finish, BIG)

    # On the frontier's own width, not the day's: the selection maps the column it picked back
    # through `index`, so nothing is ever spread to the full list and the step never pays for the
    # columns it did not price.
    earliest = finish.min(axis=1)                            # (b, w): the hour it can be done
    worker = finish.argmin(axis=1)                           # (b, w): and by which worker
    return dict(rows=rows, finish=finish, hop=hop, earliest=earliest, worker=worker,
                done=done, when=when, who=who, free=free, where=where, travel=travel,
                idle=idle, index=index, count=count, has=has)


def _rankings_for(tasks: TaskArray) -> tuple[str, ...]:
    """The rankings a day is worth searching under.

    The extra keys chase an hour. A day with no deadline-bound task - nothing whose `latest` is
    before the horizon - has no hour to chase, and the whole portfolio would be paid for nothing:
    the mixed day is one of those, and a third of its beam is three tasks.
    """
    if (tasks.latest < int(tasks.latest.max())).any():
        return RANKINGS
    return ("finish",)


def _shares(width: int, parts: int) -> list[int]:
    """The slots each ranking keeps: the WHOLE width each, not a slice of it.

    Sharing the width out buys diversity with the one thing the search needs - see `RANKINGS`. The
    cost is the expansion, which is linear in the rows, so a day costs about `parts` times as much.
    """
    return [width] * parts


def _rank_keys(tasks: TaskArray, child_done, child_free, travel, makespan, hands_used, dead):
    """Each ranking's sort keys, in `np.lexsort` order - the LAST key is the primary one.

    No single key is right, which is why there is more than one. The earliest finish keeps the most
    work placed and is blind to an hour that is nearly gone; the least slack sees only the hour and
    gives away work to do it; the deadline count chases the tasks that have to land by an hour at the
    cost of everything else. Measured on the animals' day with the farmer alone: the finish key places
    11 of 12, the slack key 10 of 12, and the day is carryable with 12.
    """
    #: The tasks with an hour of their own - a deadline. `latest` is the horizon for everything else,
    #: and the horizon is not a deadline: nothing has to be done by the last turn of the day.
    bound_rows = np.flatnonzero(tasks.latest < int(tasks.latest.max()))
    # The slack: how many turns are left before the nearest hour passes, over the tasks not yet done.
    # A done task cannot be late, so it is masked out of the minimum.
    latest = getattr(tasks, "latest32", tasks.latest)
    slack = latest[None, :] - child_free.min(axis=1)[:, None]
    slack = np.where(child_done, np.int32(1 << 14), slack).min(axis=1)
    placed_bound = (child_done[:, bound_rows].sum(axis=1) if bound_rows.size
                    else np.zeros(child_done.shape[0], dtype=np.int16))
    return {
        "finish": (travel, makespan, hands_used, dead),
        "slack": (travel, makespan, hands_used, dead, slack),
        "bound": (travel, makespan, hands_used, dead, -placed_bound),
    }


#: Which deterministic selection rule `_select` runs (owner 2026-09-25: no lottery):
#: "hour" - finish hour first, importance among equals; "slack" - urgency (EDF) first.
SELECT_RULE = "hour"
SEED = [0]

BRANCH_IDLE: list = [False]


def _select(expanded, tasks: TaskArray, beam: int, first_hand: int, start_hours,
            pool_of=None, hours_of=None, branch_floor: int = 0,
            band_of: np.ndarray | None = None):
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
    active = _rankings_for(tasks)
    done, when, who = expanded["done"], expanded["when"], expanded["who"]
    free = expanded["free"]
    where, travel, hop = expanded["where"], expanded["travel"], expanded["hop"]
    flat_hour = expanded["earliest"][rows].ravel()           # (live * width)

    legal = np.flatnonzero(flat_hour < BIG)
    if legal.size == 0:
        empty = beam * len(_rankings_for(tasks))
        return (_empty_like(done, empty), _empty_like(when, empty), _empty_like(who, empty),
                _empty_like(free, empty), _empty_like(where, empty),
                np.zeros((empty,), dtype=np.int16), np.zeros((empty,), dtype=bool),
                np.zeros((empty, tasks.n), dtype=np.int16))

    # The shortlist is a few percent of the field - `beam x tasks` candidates - and among candidates
    # whose finish hour ties, which of them it keeps was arbitrary. Too narrow, and the states that
    # would carry the day are cut before the ranking ever sees them: at `beam * 4` a width of 50
    # placed 86 of a real day's 93 tasks, and at `beam * 16` it placed 93.
    budget = min(legal.size, beam * len(active) * 16)
    if pool_of is not None:
        # Multi-pool: the shortlist budget is PER BAND. A global cut starves the
        # small pools' slower candidates while the widest pool fills the budget
        # (measured: the lighter-day bands died of starvation). Rank within each
        # band, take each band's own slice, concatenate - the same deterministic
        # keys, applied per band.
        band = beam * len(active)
        per_band = band * 16
        col_of = legal % width
        task_of = index[col_of]
        importance = -tasks.chain_weight[task_of]
        worker_of = expanded["worker"][rows].ravel()[legal]
        parent_of = np.repeat(rows, width)[legal]
        hop_of = expanded["hop"][parent_of, worker_of, col_of]
        if SELECT_RULE == "hour":
            h_rem = np.maximum(0, 24 - flat_hour[legal]).astype(np.int32)
            cw = tasks.chain_weight[task_of].astype(np.int32)
            alpha = 1 if tasks.n <= 130 else 2
            primary = flat_hour[legal].astype(np.int32) * 48 - cw * h_rem * alpha
        else:
            latest = getattr(tasks, "latest32", tasks.latest)
            primary = latest[task_of] - flat_hour[legal].astype(np.int32)
        needs_item = tasks.items[task_of] >= 0
        has_in_bag = expanded["has"][parent_of, worker_of, col_of]
        self_serve_bonus = -(needs_item & has_in_bag).astype(np.int8)
        band_of_legal = band_of[parent_of]
        order = np.lexsort((band_of_legal, worker_of, self_serve_bonus, hop_of,
                            importance, primary))
        # keep per-band slices: walk the ordered list and cap each band at its budget
        band_counts = np.zeros(int(band_of_legal.max()) + 1, dtype=np.int64)
        cap = np.zeros_like(band_counts)
        cap[band_of_legal] = per_band
        take = np.zeros(order.size, dtype=bool)
        for idx in order:
            b = band_of_legal[idx]
            if band_counts[b] < cap[b]:
                band_counts[b] += 1
                take[idx] = True
        shortlist = legal[take]
    elif legal.size > budget:
        col_of = legal % width
        task_of = index[col_of]
        importance = -tasks.chain_weight[task_of]
        worker_of = expanded["worker"][rows].ravel()[legal]
        parent_of = np.repeat(rows, width)[legal]
        hop_of = expanded["hop"][parent_of, worker_of, col_of]
        if SELECT_RULE == "hour":
            h_rem = np.maximum(0, 24 - flat_hour[legal]).astype(np.int32)
            cw = tasks.chain_weight[task_of].astype(np.int32)
            alpha = 1 if tasks.n <= 130 else 2
            primary = flat_hour[legal].astype(np.int32) * 48 - cw * h_rem * alpha
        else:
            latest = getattr(tasks, "latest32", tasks.latest)
            primary = latest[task_of] - flat_hour[legal].astype(np.int32)
        needs_item = tasks.items[task_of] >= 0
        has_in_bag = expanded["has"][parent_of, worker_of, col_of]
        self_serve_bonus = -(needs_item & has_in_bag).astype(np.int8)
        order = np.lexsort((worker_of, self_serve_bonus, hop_of, importance, primary))
        shortlist = legal[order[:budget]]
    else:
        shortlist = legal

    parent = np.repeat(rows, width)[shortlist]
    column = shortlist % width
    child_group = band_of[parent] if pool_of is not None else None
    # The frontier's column, mapped back to the day's own index - which is what the state and the
    # task's cell are written by.
    task = index[column]
    hour = flat_hour[shortlist]
    worker = expanded["worker"][rows].ravel()[shortlist]

    busy = done_by_worker(done, who, free.shape[1])
    used_now = busy[:, first_hand:].sum(axis=1)
    if pool_of is not None:
        # Multi-pool: the branch fires only in DOUBLED groups (the group's pool >=
        # 2x the hired floor); masked columns are nonexistent workers, not idle ones.
        surplus = pool_of[:, None] - used_now
        has_surplus = ((surplus >= 1).any(axis=1)
                       & (pool_of >= 2 * branch_floor))[parent]
    elif BRANCH_IDLE[0]:
        surplus = free.shape[1] - first_hand - used_now
        has_surplus = (surplus >= 2)[parent]
    else:
        has_surplus = None
    if has_surplus is not None and has_surplus.any():
        finish_all = expanded["finish"]
        busy_finish = np.where(busy[:, :, None], finish_all, BIG)
        worker2 = busy_finish[parent, :, column].argmin(axis=1)
        add = (has_surplus
               & (busy_finish[parent, worker2, column] < BIG)
               & (worker2 != worker))
        parent2 = parent[add]
        column2 = column[add]
        worker2 = worker2[add]
        parent = np.concatenate([parent, parent2])
        column = np.concatenate([column, column2])
        worker = np.concatenate([worker, worker2])
        task = np.concatenate([task, index[column2]])
        hour = np.concatenate([hour, busy_finish[parent2, worker2, column2]])
        if child_group is not None:
            child_group = np.concatenate([child_group, band_of[parent2]])

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
    # Where the worker ends up: the task's tile, or for a DROP the door nearest where it stood.
    target = tasks.cells[task]
    dropping = tasks.is_drop[task]
    if dropping.any():
        stood = where[parent, worker]
        at_door = DOOR_CELL[stood[:, 0].astype(np.int32) * BOARD_SIZE + stood[:, 1].astype(np.int32)]
        target = np.where(dropping[:, None], at_door, target)
    child_where[np.arange(parent.size), worker] = np.where(
        idle_here[:, None], where[parent, worker], target)
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
    if pool_of is not None:
        child_parent = parent[keep]
        child_group = child_group[keep]

    # The tasks this state has already made impossible: unplaced, with a latest hour that has gone
    # by the earliest any worker is free. A route that lost one cannot carry the day, so this leads
    # the ranking - without it the beam keeps the states that look best now and drops the ones that
    # will finish.
    dead = ((tasks.latest[None, :] < child_free.min(axis=1)[:, None]) & ~child_done).sum(axis=1)
    if pool_of is None:
        hands_used = (child_free[:, first_hand:] > start_hours[first_hand:]).sum(axis=1)
    else:
        child_hours = hours_of[child_parent]
        hands_used = (child_free[:, first_hand:] > child_hours[:, first_hand:]).sum(axis=1)
    makespan = child_free.max(axis=1)
    # The portfolio: `_expand` ran once and every candidate is shared, so another ranking costs a sort
    # and a copy rather than an expansion. Each keeps a beam of its own, so a state one key prunes is
    # still examined under another.
    keys = _rank_keys(tasks, child_done, child_free, child_travel, makespan, hands_used, dead)
    if pool_of is None:
        shares = _shares(beam, len(active))
        order = np.concatenate([np.lexsort(keys[rank])[:take]
                                for rank, take in zip(active, shares)])
        out_done = _empty_like(child_done, sum(shares))
        out_when = _empty_like(child_when, sum(shares))
        out_who = _empty_like(child_who, sum(shares))
        out_free = _empty_like(child_free, sum(shares))
        out_where = _empty_like(child_where, sum(shares))
        out_travel = np.zeros((sum(shares),), dtype=np.int16)
        out_live = np.zeros((sum(shares),), dtype=bool)
        out_count = np.zeros((sum(shares), tasks.n), dtype=np.int16)
        k = order.size
        out_done[:k], out_when[:k] = child_done[order], child_when[order]
        out_who[:k] = child_who[order]
        out_free[:k], out_where[:k] = child_free[order], child_where[order]
        out_travel[:k], out_live[:k] = child_travel[order], True
        out_count[:k] = child_count[order]
        return out_done, out_when, out_who, out_free, out_where, out_travel, out_live, out_count

    # Multi-pool: each group's beam lives in a FIXED band of the output (its own
    # slice), so pool_of/hours_of stay static and aligned band-for-band across
    # steps; groups never compete for one another's slots.
    band = beam * len(active)
    n_bands = int(band_of.max()) + 1
    out_done = _empty_like(child_done, band * n_bands)
    out_when = _empty_like(child_when, band * n_bands)
    out_who = _empty_like(child_who, band * n_bands)
    out_free = _empty_like(child_free, band * n_bands)
    out_where = _empty_like(child_where, band * n_bands)
    out_travel = np.zeros((band * n_bands,), dtype=np.int16)
    out_live = np.zeros((band * n_bands,), dtype=bool)
    out_count = np.zeros((band * n_bands, tasks.n), dtype=np.int16)
    for g in range(n_bands):
        rows_g = np.flatnonzero(child_group == g)
        fill = g * band
        for rank in active:
            keys_g = tuple(k[rows_g] for k in keys[rank])
            take = np.lexsort(keys_g)[:beam]
            sel = rows_g[take]
            out_done[fill:fill + take.size] = child_done[sel]
            out_when[fill:fill + take.size] = child_when[sel]
            out_who[fill:fill + take.size] = child_who[sel]
            out_free[fill:fill + take.size] = child_free[sel]
            out_where[fill:fill + take.size] = child_where[sel]
            out_travel[fill:fill + take.size] = child_travel[sel]
            out_live[fill:fill + take.size] = True
            out_count[fill:fill + take.size] = child_count[sel]
            fill += beam
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
    signature = np.ascontiguousarray(np.hstack([
        np.packbits(done, axis=1),
        free.view(np.uint8).reshape(done.shape[0], -1),
        where.view(np.uint8).reshape(done.shape[0], -1),
    ]))
    dt = np.dtype((np.void, signature.shape[1]))
    void_arr = signature.view(dt).ravel()
    _, u_idx = np.unique(void_arr, return_index=True)
    return np.sort(u_idx)


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


def _carried(done, who, tasks: TaskArray, workers: int, when=None, last_drop=None) -> np.ndarray:
    """Whether each worker's bag holds each good, per route - (batch, workers, goods).

    Derived from the route rather than tracked: a worker holds a good exactly when it has done a
    task that consumes it, because the trip that brought it is the same worker's. Two arrays that
    cannot disagree are worth more than one kept in step by hand.

    `last_drop` is each worker's last DROP turn (`_last_drop`). The engine's DROP empties the whole
    bag (`kaggriculture.py:343-356`), so only what a worker did AFTER its last drop is still in it.
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
    if last_drop is not None:
        since = np.take_along_axis(last_drop, np.where(worker >= 0, worker, 0), axis=1)
        live &= when[:, consuming] > since
    if live.any():
        rows = np.arange(batch)[:, None]
        index = ((rows * workers) + np.where(live, worker, 0)) * n_goods + goods[consuming]
        flat[index[live]] = True
    return flat.reshape(batch, workers, n_goods)



def leg_moves(here: Cell, target: Cell, fetch) -> list[tuple]:
    """The ops that carry a worker to its task: the walk, or the walk through a door and a PICKUP."""
    if fetch is None:
        return walk(here, target)
    door = nearest_shed(here)
    fetched = ("PICKUP", ITEM_NAME[int(fetch[0])], int(fetch[1]))
    return walk(here, door) + [fetched] + walk(door, target)

def leg_target(tasks: TaskArray, row: int, here: Cell) -> Cell:
    """Where a worker standing on `here` does task `row`: the task's own tile, or a DROP's door.

    Any of the four shed-access tiles takes a DROP (`kaggriculture.py:138-139`, `:343-356`), so the
    door a bag is handed over at is a property of the route - the one nearest the worker when it
    drops (`nearest_shed`, the refetch's own rule) - and not of the harvest it banks. The search
    (`DROP_DOOR`), the compiler and every reader of where a worker stands go through here.
    """
    if bool(tasks.is_drop[row]):
        return nearest_shed(here)
    return (int(tasks.cells[row][0]), int(tasks.cells[row][1]))

def legs(tasks: TaskArray, entries: list[tuple[int, str]]) -> list[tuple[int, int, tuple | None]]:
    """One worker's day as `(turn, row, fetch)`, where `fetch` is `(good, n)` or None.

    A DROP hands over the whole bag, so a good used after it is no longer carried: the first use of
    each good after a drop walks through the nearest door on the way and picks up `n` - every use
    of that good until the next drop. A good the worker has taken off a tile since the drop is in
    the bag, as before. The search prices this trip and the compiler writes it from here, so the
    two cannot disagree on it.

    A drop with an empty bag (turn -1) is not a leg: it has no turn, no op and no walk, so the
    worker never goes to its door. Reading it as one put the worker on that door from turn 0.
    """
    rows = [(int(turn), tasks.ids.index(task_id)) for turn, task_id in sorted(entries)
            if int(turn) >= 0]
    out: list[tuple[int, int, tuple | None]] = []
    dropped = False
    held: set[int] = set()
    for k, (turn, row) in enumerate(rows):
        fetch = None
        need = int(tasks.items[row])
        if bool(tasks.is_drop[row]):
            dropped, held = True, set()
        elif dropped and need >= 0 and need not in held:
            n = 0
            for _later_turn, later in rows[k:]:
                if bool(tasks.is_drop[later]):
                    break
                n += int(tasks.items[later]) == need
            fetch = (need, n)
        for good in (need, int(tasks.yields[row])):
            if good >= 0:
                held.add(good)
        out.append((turn, row, fetch))
    return out

def bags_of(day: Day, tasks: TaskArray, result: Result) -> list[frozenset[int]]:
    """The goods each worker's own day loads at its door: the distinct goods in its own bag.

    The compiler writes one PICKUP per good, at the worker's door, before its first task that is
    not door work (`loads_before`), each at its own good's hour (`pickup_turns`) - so this is what the search has to charge that worker, and it is a
    property of the route rather than of the day. `_fixed_point` iterates it; `compile_route` writes
    from the same bags.
    """
    per: dict[int, list[tuple[int, str]]] = {}
    for turn, task_id, worker in result.route:
        per.setdefault(int(worker), []).append((int(turn), task_id))
    return [frozenset(_bag(tasks, per.get(worker, [])))
            for worker in range(len(day.units) + result.pool)]





