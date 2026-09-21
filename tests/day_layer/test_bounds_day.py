"""The pool a day provably cannot use: refused with a no, not with a search.

`lower_bound` is a proved bound on the workers a day needs - the work against the turns it has, the goods
that need fetching, and the walk a drop forces - so a caller asking about a range whose top sits below it
is asking a question with a known answer. These are the two things that has to mean: the answer arrives
without a route on it, and the bound itself is not mistaken for a refusal, because a bound is the fewest a
day MIGHT need and the day is free to need more.
"""
import json
import pathlib
import sys

sys.path.insert(0, "/chista/pm/world")

from agent.wsr import beam as B
from agent.wsr import tasks as T

CORPUS = pathlib.Path("/chista/pm/world/tests/day_layer/corpus/real_days.json")
#: A day from the archive with a floor worth asking about: 93 tasks, one unit on the field.
DAY_KEY = ("2026-08-24", 98009264, 14)


def _the_day():
    entry = next(e for e in json.loads(CORPUS.read_text())
                 if (e["dump"], e["episode"], e["day"]) == DAY_KEY)
    grid = [(tuple(cell), tuple(ops), entity) for cell, ops, entity in entry["chains"]]
    available = {good: int(hour) for good, hour in entry["available"].items()}
    tasks = T.build(grid, available=available)
    day = B.Day(chains=tuple(grid), available=available,
                hire_times=tuple(entry["hire_times"]) or (1,) * entry["hands"])
    return day, tasks


def test_a_range_below_the_floor_comes_back_as_a_no_with_no_route():
    """The no is an answer about the day, so it carries no route and claims no time was short."""
    day, tasks = _the_day()
    floor = max(0, B.lower_bound(day, tasks) - len(day.units))
    assert floor > 1, f"the day this is built on has to have a floor worth asking about, got {floor}"

    below = B.search(day, tasks, hands=floor - 1, max_hands=floor - 1, budget_s=5.0)

    assert below.infeasible, f"a range under the floor of {floor} was not refused"
    assert below.route == [], "a refused range came back with a route on it"
    assert not below.complete and not below.out_of_time, (
        "the refusal is neither a partial day nor a deadline: it is a bound"
    )


def test_the_floor_itself_is_not_a_refusal():
    """A bound says the fewest a day might need. A range that reaches it is a question for the search.

    The range straddles the floor on purpose - it starts one below and allows one above - which is the
    shape the corpus test asks in, and the shape a gate comparing the starting pool would refuse.
    """
    day, tasks = _the_day()
    floor = max(0, B.lower_bound(day, tasks) - len(day.units))
    assert floor > 1

    straddling = B.search(day, tasks, hands=floor - 1, max_hands=floor + 1, budget_s=5.0)

    assert not straddling.infeasible, (
        f"a range of {floor - 1}..{floor + 1} was refused, but its top reaches the floor of {floor} "
        f"and the bound is not a ceiling"
    )


def test_an_ask_above_the_day_s_own_ceiling_is_refused_rather_than_crashing():
    """The range is empty when the caller's ask starts above the bound, and that is still a no."""
    day, tasks = _the_day()
    bound = B.ceiling_for(day, tasks)
    assert bound >= 1

    above = B.search(day, tasks, hands=bound + 1, max_hands=bound + 1, budget_s=5.0)

    assert above.infeasible, "an ask above the day's own ceiling was not refused"
    assert above.route == []
