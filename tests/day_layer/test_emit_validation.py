"""`check_route`'s rules: every complaint it can make has to be reachable, and no route the search
builds may draw one.

Run:  .venv/bin/python -m pytest tests/day_layer/test_emit_validation.py

`check_route` is what stands between a route and an engine that refuses a bad op in silence (F047), so
a check that never runs is worse than no check at all: it reads like coverage. Two of its checks sat
behind `if row not in placed`, where `placed` held the route's turns rather than its task ids - a set
of ints no task id can be in - so a route that plants before it digs, or draws a good before the good
is in the shed, came back clean.

The corpus days carry no per-task deadline, so these guards aim the rules at a route the search itself
built and move one task where the rule forbids.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route

CORPUS = pathlib.Path(__file__).parent / "corpus" / "winner_days.json"
REAL_DAYS = json.loads(CORPUS.read_text())
BY_SIZE = sorted(REAL_DAYS, key=lambda e: sum(len(ops) for _c, ops, _e in e["chains"]))


def _searched(entry):
    grid = [(tuple(cell), tuple(ops), entity) for cell, ops, entity in entry["chains"]]
    available = {good: int(hour) for good, hour in entry["available"].items()}
    tasks = T.build(grid, available=available)
    hire_times = tuple(entry["hire_times"]) or (1,) * entry["hands"]
    day = B.Day(chains=tuple(grid), available=available, hire_times=hire_times)
    result = B.search(day, tasks, hands=max(entry["hands"] - 1, 0), max_hands=entry["hands"])
    return day, tasks, result


def _moved(route, task_id, turn):
    return [(turn if placed == task_id else was, placed, worker)
            for was, placed, worker in route]


def test_a_task_before_its_predecessor_is_named() -> None:
    """The order the chains impose: the search keeps it, and the compiler has to say when a route does
    not - otherwise the engine is handed a plant whose seed is still in the shed."""
    day, tasks, result = _searched(BY_SIZE[0])
    rows = {task_id: index for index, task_id in enumerate(tasks.ids)}
    late = next(entry for entry in result.route if tasks.pred[rows[entry[1]]].any())
    assert int(late[0]) > 1, late
    complaints = check_route(day, tasks, result._replace(route=_moved(result.route, late[1], 1)))
    assert any("must come before" in complaint for complaint in complaints), complaints


def test_a_good_drawn_before_it_is_in_the_shed_is_named() -> None:
    """A task that needs a good cannot start before the good is in the shed - the engine would run the
    op, bank nothing, and say nothing."""
    day, tasks, result = _searched(BY_SIZE[0])
    row = next(index for index in range(tasks.n) if int(tasks.items[index]) >= 0)
    task_id = tasks.ids[row]
    assert int(tasks.earliest[row]) >= 1
    complaints = check_route(day, tasks, result._replace(route=_moved(result.route, task_id, 0)))
    assert any("in the shed" in complaint for complaint in complaints), complaints


def test_the_searchs_own_routes_draw_no_complaint() -> None:
    """The rules have to stay silent about the days the search builds: a check that starts refusing
    good routes is worse than the one that never ran."""
    for entry in [BY_SIZE[0], BY_SIZE[2], BY_SIZE[10], BY_SIZE[20]]:
        day, tasks, result = _searched(entry)
        assert check_route(day, tasks, result) == [], tasks.n
