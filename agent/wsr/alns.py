"""Destroy and repair: the beam's route, taken apart in pieces and re-placed, keeping only what is better.

The beam places every task once and never moves one afterwards: `_expand` adds a task to every route
in the beam, `_select` keeps the best of them, and a choice made in the first generation is carried to
the last. So the route that comes back is the best of the orders the beam happened to build, and no
part of it is ever reconsidered.

This module reconsiders. It takes the route, removes part of it - a hand's whole day, the tasks that
share a deadline, a corner of the board, every task of one kind - and hands the rest back to the beam
as a warm start. `search(..., warm=)` starts from a route, so the removed tasks are simply unplaced
and the search may place them anywhere, including on the workers it had not chosen for them.

Acceptance is strict: a candidate is kept only when it beats the incumbent on the same layered
objective the search ranks by (more work placed, then fewer hands, then the earlier finish, then the
shorter walk). The route this returns is therefore never worse than the one it was handed, which is
what lets a caller wire it in without turning a fit answer into a gamble.

The operators carry weights, and the weight of the one that last produced an improvement grows while
the others decay - so a day that pays for removing a hand's day stops being asked about the corners
of the board. The choice is seeded: the same day and the same seed give the same route.
"""

from __future__ import annotations

import random
import time
from typing import NamedTuple

import numpy as np

from agent.wsr import beam as B
from agent.wsr.emit import check_route, compile_route
from agent.wsr.tasks import TaskArray

#: The destroy operators, by name. Each removes a GROUP of tasks, never a single one: a task on its
#: own is a re-timing the beam already does, and the point here is to free a part of the day.
DESTROYERS: tuple[str, ...] = ("worker", "deadline", "region", "kind")


class Improved(NamedTuple):
    """What the loop did: the route to use, and the facts about how it got there."""

    result: B.Result
    iterations: int = 0
    accepted: int = 0
    #: Candidates the compiler refused: a repaired route whose walk does not fit the turns it chose,
    #: or a task out of its window. Counted rather than swallowed - a rising count is a destroy
    #: operator that keeps proposing days the engine would refuse (F047).
    refused: int = 0
    out_of_time: bool = False

    @property
    def improved(self) -> bool:
        return self.accepted > 0


def keys(day: B.Day, tasks: TaskArray, result: B.Result) -> tuple:
    """The route on the search's own layered objective: work, hands, finish, walk.

    Read back through `_warm_row` - the translation the search itself uses for a route it did not
    build - so this cannot drift from what the beam ranks by. The last key is the walk, which is the
    hop sum between a worker's tasks from the door it lands on.
    """
    n = tasks.n
    first_hand = len(day.units)
    hours = B._start_hours(day, result.pool)
    free = np.asarray(hours, dtype=np.int16)[None, :].copy()
    where = B._start_positions(day, result.pool, result.doors)[None, :, :].astype(np.int16)
    done = np.zeros((1, n), dtype=bool)
    when = np.zeros((1, n), dtype=np.int16)
    who = np.full((1, n), -1, dtype=np.int16)
    travel = np.zeros((1,), dtype=np.int16)
    count = np.zeros((1, n), dtype=np.int16)
    B._warm_row(tasks, result, done, when, who, free, where, travel, count)
    placed = int(done.sum())
    hands_used = int((free[:, first_hand:] > np.asarray(hours)[first_hand:]).sum())
    return (-placed, hands_used, int(free.max()), int(travel[0]))


def writable(day: B.Day, tasks: TaskArray, result: B.Result) -> bool:
    """Whether the engine can be given this route: the compiler's own two refusals, in one place.

    `check_route` names what the day's rules forbid - a task outside its window, a worker the pool
    does not have, a predecessor after its successor, a good consumed before it is in the shed.
    `compile_route` refuses what the day's own arithmetic forbids: a walk that does not fit the turns
    the route chose (`_room` raises `ValueError`), which the rules are silent about - the route is
    legal and still cannot be walked. A route that fails either is not a day, and the engine would run
    it op by op without a word (F047).
    """
    if check_route(day, tasks, result):
        return False
    try:
        compile_route(day, tasks, result, horizon=day.horizon)
    except (ValueError, IndexError):
        # `IndexError` is the compiler's own indexing on a worker the pool does not have: the rules
        # catch that first, so this is what is left if they are ever asked a narrower question.
        return False
    return True


def _rows(tasks: TaskArray) -> dict[str, int]:
    return {task_id: index for index, task_id in enumerate(tasks.ids)}


def _destroy_worker(day: B.Day, tasks: TaskArray, result: B.Result, rng: random.Random) -> list:
    """One hand's whole day: the tasks that keep a hand hired."""
    first_hand = len(day.units)
    hands = sorted({int(worker) for _turn, _t, worker in result.route if int(worker) >= first_hand})
    if not hands:
        return list(result.route)
    victim = rng.choice(hands)
    return [entry for entry in result.route if int(entry[2]) != victim]


def _destroy_deadline(day: B.Day, tasks: TaskArray, result: B.Result, rng: random.Random) -> list:
    """Every task that shares a deadline: the hours that make a day tight."""
    horizon = int(tasks.latest.max())
    hours = sorted({int(tasks.latest[_rows(tasks)[task_id]])
                    for _turn, task_id, _w in result.route
                    if int(tasks.latest[_rows(tasks)[task_id]]) < horizon})
    if not hours:
        return list(result.route)
    hour = rng.choice(hours)
    keep = {task_id for _turn, task_id, _w in result.route
            if int(tasks.latest[_rows(tasks)[task_id]]) != hour}
    return [entry for entry in result.route if entry[1] in keep]


def _destroy_region(day: B.Day, tasks: TaskArray, result: B.Result, rng: random.Random) -> list:
    """A block of the board: the walk a route pays to cross it."""
    row_of = _rows(tasks)
    size = 4
    corners = [(x, y) for x in range(0, 8, size) for y in range(0, 8, size)]
    x0, y0 = rng.choice(corners)
    keep = []
    for entry in result.route:
        cell = tasks.cells[row_of[entry[1]]]
        if x0 <= int(cell[0]) < x0 + size and y0 <= int(cell[1]) < y0 + size:
            continue
        keep.append(entry)
    return keep


def _destroy_kind(day: B.Day, tasks: TaskArray, result: B.Result, rng: random.Random) -> list:
    """Every task of one kind: WATER, HARVEST, PLANT - the chain layer's own vocabulary."""
    row_of = _rows(tasks)
    kinds = sorted({str(tasks.ops[row_of[task_id]][0]) for _turn, task_id, _w in result.route})
    if not kinds:
        return list(result.route)
    kind = rng.choice(kinds)
    return [entry for entry in result.route
            if str(tasks.ops[row_of[entry[1]]][0]) != kind]


_DESTROY = {"worker": _destroy_worker, "deadline": _destroy_deadline,
            "region": _destroy_region, "kind": _destroy_kind}


def _pick(weights: dict[str, float], rng: random.Random) -> str:
    names = list(DESTROYERS)
    total = sum(weights[name] for name in names)
    point = rng.random() * total
    for name in names:
        point -= weights[name]
        if point <= 0.0:
            return name
    return names[-1]


def improve(day: B.Day, tasks: TaskArray, result: B.Result, *,
            budget_s: float | None = None, iterations: int | None = None,
            seed: int = 0, beam: int | None = None) -> Improved:
    """The route, taken apart and re-placed until the budget runs out or nothing is left to try.

    `budget_s` bounds the wall clock and `iterations` bounds the attempts; either alone is enough, and
    the route that comes back is the best found so far - never a partial one, and never worse than the
    one handed in. `beam` is the repair's own width: `None` uses the day's, which is what the search
    would have used anyway.
    """
    best = result
    best_keys = keys(day, tasks, best)
    weights = {name: 1.0 for name in DESTROYERS}
    rng = random.Random(seed)
    deadline = None if budget_s is None else time.perf_counter() + float(budget_s)
    done = accepted = refused = 0
    cut = False

    while iterations is None or done < iterations:
        if deadline is not None and time.perf_counter() >= deadline:
            cut = True
            break
        name = _pick(weights, rng)
        kept = _DESTROY[name](day, tasks, best, rng)
        done += 1
        if len(kept) == len(best.route):
            weights[name] *= 0.9            # this operator had nothing to remove today
            continue
        warm = best._replace(route=kept)
        candidate = B.search(day, tasks, hands=best.pool, max_hands=best.pool,
                             beam=beam, warm=warm)
        if not candidate.route:
            weights[name] *= 0.9
            continue
        if not writable(day, tasks, candidate):
            refused += 1
            weights[name] *= 0.9
            continue
        candidate_keys = keys(day, tasks, candidate)
        if candidate_keys < best_keys:
            best, best_keys = candidate, candidate_keys
            accepted += 1
            weights[name] += 1.0
        else:
            weights[name] *= 0.9

    return Improved(best, iterations=done, accepted=accepted, refused=refused, out_of_time=cut)
