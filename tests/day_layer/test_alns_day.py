"""The destroy-and-repair loop: it may not hand back a worse day, and it must place what it can.

Run:  .venv/bin/python -m pytest tests/day_layer/test_alns_day.py

The loop takes the beam's route apart in groups (a hand's whole day, a deadline's tasks, a block of
the board, every task of one kind) and re-places the rest with the search, keeping a candidate only
when it beats the incumbent on the search's own layered objective: more work placed, then fewer
hands, then the earlier finish, then the shorter walk.

Contracts under test:

- the route that comes back is never worse than the one handed in, on that objective;
- every route it returns is one the compiler will write: `check_route` clean and `compile_route`
  silent, which is the only thing standing between a repaired route and a day the engine refuses op
  by op without a word (F047);
- the same seed gives the same route, so a caller can reproduce a day;
- on a day the pool cannot carry, the loop places MORE of the day - measured on the corpus's 146-task
  day: 135 of 146 placed by the beam, 139 after one accepted repair, and the walk down from 142 to
  134. That is the key the search ranks first, so it is the one the loop is worth having for.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.wsr import alns
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route

CORPUS = pathlib.Path(__file__).parent / "corpus" / "real_days.json"
REAL_DAYS = json.loads(CORPUS.read_text())
#: Biggest first: the days whose routes are the most work to build and the most to get wrong.
BY_SIZE = sorted(REAL_DAYS, key=lambda e: -sum(len(ops) for _c, ops, _e in e["chains"]))


def _day(entry):
    grid = [(tuple(cell), tuple(ops), entity) for cell, ops, entity in entry["chains"]]
    available = {good: int(hour) for good, hour in entry["available"].items()}
    tasks = T.build(grid, available=available)
    hire_times = tuple(entry["hire_times"]) or (1,) * entry["hands"]
    return B.Day(chains=tuple(grid), available=available, hire_times=hire_times), tasks


def _searched(entry):
    day, tasks = _day(entry)
    result = B.search(day, tasks, hands=max(entry["hands"] - 1, 0),
                      max_hands=entry["hands"])
    return day, tasks, result


def test_the_loop_never_hands_back_a_worse_day() -> None:
    """Never worse, and always writable: the two promises a caller wires it in on."""
    for entry in BY_SIZE[:3]:
        day, tasks, result = _searched(entry)
        before = alns.keys(day, tasks, result)
        out = alns.improve(day, tasks, result, iterations=2, seed=0)
        after = alns.keys(day, tasks, out.result)
        assert after <= before, (tasks.n, before, after)
        assert not check_route(day, tasks, out.result), check_route(day, tasks, out.result)
        compile_route(day, tasks, out.result, horizon=day.horizon)
        assert out.refused == 0, f"{out.refused} candidates the compiler would not write"


def test_the_same_seed_gives_the_same_route() -> None:
    """A day is reproducible: the operator choice is seeded, not sampled from the clock."""
    entry = BY_SIZE[-1]
    day, tasks, result = _searched(entry)
    first = alns.improve(day, tasks, result, iterations=6, seed=7)
    second = alns.improve(day, tasks, result, iterations=6, seed=7)
    assert first.result.route == second.result.route
    assert (first.iterations, first.accepted) == (second.iterations, second.accepted)


def test_a_day_the_pool_cannot_carry_gets_more_of_it_placed() -> None:
    """The loop's own case: a day too big for its pool, where placing more is the first key."""
    entry = BY_SIZE[2]
    assert entry["hands"] == 11, entry["hands"]        # the corpus day this is about
    day, tasks, result = _searched(entry)
    assert not result.complete, "this day is supposed to be one the pool cannot carry"
    before = alns.keys(day, tasks, result)
    out = alns.improve(day, tasks, result, iterations=3, seed=0)
    after = alns.keys(day, tasks, out.result)
    assert out.accepted >= 1, "nothing was accepted on the day the loop is for"
    assert -after[0] > -before[0], (before, after)


def test_a_route_the_engine_would_refuse_is_counted_not_returned(monkeypatch) -> None:
    """A search that hands back a day the engine would refuse op by op (F047) must be counted and
    dropped, and the incumbent kept."""
    entry = BY_SIZE[-1]
    day, tasks, result = _searched(entry)
    broken = result._replace(route=[(turn, task_id, 999 if index == 0 else worker)
                                    for index, (turn, task_id, worker) in enumerate(result.route)])
    assert not alns.writable(day, tasks, broken), "the broken route is supposed to be refused"
    monkeypatch.setattr(alns.B, "search", lambda *a, **k: broken)
    out = alns.improve(day, tasks, result, iterations=2, seed=0)
    assert out.refused == 2, out
    assert out.accepted == 0 and out.result is result, out


def test_the_validation_catches_both_kinds_of_refusal() -> None:
    """The composite check, on both kinds of refusal.

    A worker the pool does not have is a rule the compiler names. A walk that does not fit the turns
    the route chose is arithmetic the rules have nothing to say about: the route is legal right up to
    the moment the compiler tries to write it down. The second case is found rather than assumed - the
    first task with nothing to wait for and no good to draw, moved to a turn it cannot walk to in
    time - so this guard does not depend on one day's geometry being the shape it was written for.
    """
    entry = BY_SIZE[-1]
    day, tasks, result = _searched(entry)
    assert alns.writable(day, tasks, result), "the beam's own route has to be a day"

    stranger = result._replace(
        route=[(turn, task_id, 999 if index == 0 else worker)
               for index, (turn, task_id, worker) in enumerate(result.route)])
    assert check_route(day, tasks, stranger), "the rules are the half that names this one"
    assert not alns.writable(day, tasks, stranger)

    rows = {task_id: index for index, task_id in enumerate(tasks.ids)}
    plain = [(turn, task_id, worker) for turn, task_id, worker in result.route
             if not tasks.pred[rows[task_id]].any() and int(tasks.items[rows[task_id]]) < 0]
    for turn, task_id, _worker in plain:
        for earlier in range(0, int(turn)):
            moved = result._replace(route=[(earlier if placed == task_id else was, placed, worker)
                                           for was, placed, worker in result.route])
            if not check_route(day, tasks, moved) and not alns.writable(day, tasks, moved):
                return
    raise AssertionError("no route here is silent to the rules and refused by the compiler")
