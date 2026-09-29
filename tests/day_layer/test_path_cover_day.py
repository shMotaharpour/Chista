"""The repair pass's own arithmetic, guarded where it can be checked independently of the search.

What is asserted is not that the search finds more work - that is the corpus's job. It is that the
pieces the pass is built from are right on their own: the order it calls optimal IS the optimal one,
the day it re-times IS the day the compiler writes, and a pass handed a route it cannot improve gives
that route back with every task it had.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B  # noqa: E402
from agent.wsr import tasks as T  # noqa: E402
from agent.wsr.emit import check_route, compile_route  # noqa: E402
from agent.world.rules import BOARD_SIZE, earliest_hire_times  # noqa: E402

FIXTURE = pathlib.Path(__file__).parent / "corpus" / "mixed_second_day.json"
_DAY = json.loads(FIXTURE.read_text())
WATERS = [(tuple(cell), tuple(ops), entity)
          for cell, ops, entity in _DAY["chains"] if ops == ["WATER"]]
DOOR = (4, 4)
HORIZON = 24


def _tasks():
    return T.build(WATERS, available={})


def _walked(cells, start, order) -> int:
    here, total = start, 0
    for row in order:
        here, total = cells[row], total + abs(here[0] - cells[row][0]) + abs(here[1] - cells[row][1])
    return total


def test_the_order_it_calls_optimal_is_the_optimal_one() -> None:
    """Held-Karp against every permutation, on small random instances of the real board."""
    rng = np.random.default_rng(11)
    for _ in range(120):
        n = int(rng.integers(2, 8))
        cells = [(int(rng.integers(BOARD_SIZE)), int(rng.integers(BOARD_SIZE))) for _ in range(n)]
        start = (int(rng.integers(BOARD_SIZE)), int(rng.integers(BOARD_SIZE)))
        tasks = T.build([((x, y), ("WATER",), None) for x, y in cells], available={})
        order = B._path_order(tasks, list(range(n)), start)
        assert sorted(order) == list(range(n)), f"the order is not the set: {order}"
        got = _walked(cells, start, order)
        want = min(_walked(cells, start, p) for p in itertools.permutations(range(n)))
        assert got == want, f"order {order} walks {got}, the optimum is {want} ({cells} from {start})"


def test_the_day_it_prices_is_the_walk_the_tiles_cost() -> None:
    """The turns the pass prices a day at, recomputed here by hand from the tiles themselves.

    This is the check that the pass's leg arithmetic cannot drift from the board: a walk counted one
    turn short or long would price days as full that are not, and the search would quietly place less
    work. The recomputation shares no code with the pass - it walks the coordinates.
    """
    tasks = _tasks()
    rows = [int(tasks.row_of[name]) for name in
            ("d0_water", "d5_water", "d10_water", "d14_water", "d19_water", "d21_water")]
    day = B.Day(chains=tuple(WATERS), available={}, hire_times=earliest_hire_times(0))
    landed = B._retime_day(tasks, rows, DOOR, 0, HORIZON)
    assert landed is not None, "the day these six tiles make does not fit"
    here, free, want = DOOR, 0, []
    for _priced, row in landed:
        cell = (int(tasks.cells[row][0]), int(tasks.cells[row][1]))
        turn = free + abs(here[0] - cell[0]) + abs(here[1] - cell[1])
        want.append((turn, row))
        free, here = turn + 1, cell
    assert landed == want, f"the pass priced {landed}, walking the tiles costs {want}"


def test_the_day_it_re_times_is_the_day_the_writer_writes() -> None:
    """The turns the pass prices a worker's day at are the turns the compiled ops land on.

    The compiler writes every walk itself (`leg_moves`), so this is the check that the pass's own leg
    arithmetic and the writer's ops cannot drift apart - the failure mode the whole pass rests on.
    """
    tasks = _tasks()
    rows = [int(tasks.row_of[name]) for name in
            ("d0_water", "d5_water", "d10_water", "d14_water", "d19_water", "d21_water")]
    day = B.Day(chains=tuple(WATERS), available={}, hire_times=earliest_hire_times(0))
    landed = B._retime_day(tasks, rows, DOOR, 0, HORIZON)
    assert landed is not None, "the day these six tiles make does not fit"
    route = [(turn, tasks.ids[row], 0) for turn, row in landed]
    result = B.Result(pool=0, route=sorted(route), complete=False)
    assert not check_route(day, tasks, result), f"the checker refuses its own day: {check_route(day, tasks, result)}"
    ops = compile_route(day, tasks, result)
    # Every tile of this day is a WATER, so the ops are told apart by where they sit, not by name:
    # the k-th WATER op in the unit's list is the k-th task of the day the pass priced.
    written = [turn for turn, op in enumerate(ops.units[0]) if op == ("WATER",)]
    priced = [turn for turn, _row in landed]
    assert len(written) == len(priced), f"the writer ran {len(written)} tasks of the {len(priced)} priced"
    for at, (turn, row) in enumerate(landed):
        assert written[at] == turn, (
            f"{tasks.ids[row]}: the pass priced turn {turn}, the writer wrote it at {written[at]}")


def test_a_route_the_pass_cannot_improve_keeps_every_task_it_had(monkeypatch) -> None:
    """No task is ever lost, and anything the pass hands back compiles and passes the checker."""
    day = B.Day(chains=tuple(WATERS), available={}, hire_times=earliest_hire_times(1))
    tasks = _tasks()
    # The search runs the pass itself, so the route the pass is handed is the search's own answer:
    # this is the pass looking at the day a second time, which is what its callers do.
    searched = B.search(day, tasks, hands=1, max_hands=1)
    before = {task for _turn, task, _worker in searched.route}
    after = B._repair_unplaced(day, tasks, searched)
    kept = {task for _turn, task, _worker in after.route}
    assert before <= kept, f"the pass lost tasks: {sorted(before - kept)}"
    assert not check_route(day, tasks, after), f"the checker refuses the repaired route: {check_route(day, tasks, after)}"
    compile_route(day, tasks, after)                  # raises if the writer cannot run it
    assert after.complete == (len(after.route) == tasks.n), (
        "the pass reports a day that is neither complete nor short")


def test_it_closes_the_day_that_is_one_tile_short(monkeypatch) -> None:
    """The #209 day, as the guard: the farmer and one hand water all twenty-two tiles.

    This is the day the pass was built for - one tile of a five-by-five block left dry, with the walk
    that reaches it eight turns from the door. The search alone comes back with twenty-one of the
    twenty-two; the pass has to find the day that holds all of them, and the compiler and the checker
    have to agree with it. The search's own call to the pass is lifted out first, so what is being
    shown is the pass's work and not the search's.
    """
    day = B.Day(chains=tuple(WATERS), available={}, hire_times=earliest_hire_times(1))
    tasks = _tasks()
    monkeypatch.setattr(B, "_repair_unplaced", lambda day, tasks, result: result)
    alone = B.search(day, tasks, hands=1, max_hands=1)
    assert not alone.complete, "the premise: the search alone leaves a tile behind"
    monkeypatch.undo()
    repaired = B._repair_unplaced(day, tasks, alone)
    assert repaired.complete, (
        f"the pass left the day short: {tasks.n - len(repaired.route)} of {tasks.n} tiles")
    assert not check_route(day, tasks, repaired), check_route(day, tasks, repaired)
    compile_route(day, tasks, repaired)
