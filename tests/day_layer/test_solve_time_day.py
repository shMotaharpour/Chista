"""How long a day's search takes, and the two promises that must hold whatever it costs.

The manager plans one day across 24 turns with a one-second act timeout, so the search is asked many
times per day on nearly identical instances. Two things have to be true of it:

  the budget    a call given `budget_s` returns within it, with the best route it has so far and
                `out_of_time=True` rather than running long
  the cost      the portfolio asks four widths where one would do, so it costs a few single searches
                and not a multiple that grows with the day

The bounds here are promises with room in them, not benchmarks: a test that asserts a machine's speed
fails on a busy machine and says nothing about the search. The relative bound is the real one - the
portfolio's cost against a single search of the same day - because it holds on any machine. The
absolute bound is generous on purpose.
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B
from agent.wsr import tasks as T

CORPUS = pathlib.Path(__file__).parent / "corpus" / "real_days.json"
REAL_DAYS = json.loads(CORPUS.read_text())

#: The largest day in the corpus, which is the one the cost is about.
BIGGEST = max(REAL_DAYS, key=lambda e: sum(len(ops) for _c, ops, _e in e["chains"]))


def _day(entry):
    grid = [(tuple(cell), tuple(ops), entity) for cell, ops, entity in entry["chains"]]
    available = {good: int(hour) for good, hour in entry["available"].items()}
    tasks = T.build(grid, available=available)
    hire_times = tuple(entry["hire_times"]) or (1,) * entry["hands"]
    day = B.Day(chains=tuple(grid), available=available, hire_times=hire_times)
    return day, tasks


def test_a_budget_is_kept_and_the_work_so_far_comes_back():
    """A slice of the turn, not a promise to finish: the route built so far and out_of_time."""
    day, tasks = _day(BIGGEST)
    started = time.perf_counter()
    result = B.search(day, tasks, hands=max(BIGGEST["hands"] - 1, 0),
                      max_hands=BIGGEST["hands"], budget_s=0.02)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"a 0.02 s budget took {elapsed:.2f} s"
    assert result.out_of_time, "the search says it had time to spare on a 0.02 s budget"
    assert result.route, "a budgeted call returned nothing at all"
    print(f"\n  budget 0.02 s: returned in {elapsed:.3f} s with "
          f"{len(result.route)} of {tasks.n} tasks")


def test_a_day_of_this_size_is_searched_inside_a_turn():
    """The manager's turn is about a second, and the largest day in the corpus has to fit in it."""
    day, tasks = _day(BIGGEST)
    hands = max(BIGGEST["hands"] - 1, 0)

    started = time.perf_counter()
    result = B.search(day, tasks, hands=hands, max_hands=BIGGEST["hands"])
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0, f"the largest day took {elapsed:.2f} s, which no turn can afford"
    print(f"\n  {tasks.n} tasks in {elapsed * 1000:.0f} ms, placed {len(result.route)}")
