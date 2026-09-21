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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B
from agent.wsr import tasks as T

CORPUS = pathlib.Path(__file__).parent / "corpus" / "real_days.json"
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


def test_an_empty_day_is_carried_by_nobody_rather_than_raising():
    """The manager's idle plan is the first thing the layer is handed, and it has no tasks at all."""
    day = B.Day(chains=(), available={})
    result = B.search(day, T.build((), available={}), budget_s=5.0)

    assert result.complete, "a day with no work is not a day that failed"
    assert result.pool == 0 and result.route == []


def _one_tile_day(tasks_on_the_shed, hire_hours):
    """A day whose arithmetic can be written down: every task on the shed's own tile, no walking."""
    cell = (4, 4)
    ops = tuple("WATER" for _ in range(tasks_on_the_shed))
    grid = [(cell, ops, None)]
    available = {}
    day = B.Day(chains=tuple(grid), available=available, hire_times=tuple(hire_hours))
    return day, T.build(grid, available=available)


def test_the_walk_counts_the_shed_once_and_pays_for_the_crossings():
    """Three quadrants, hand-checked: each tile's edge to the shed is 8 from its own nearest door.

    (0,0) is 8 from (4,4), (9,0) is 8 from (5,4) and (0,9) is 8 from (4,5), and 8 beats the 9 a tile
    would pay to reach another tile - so the tree is 8 + 8 + 8 = 24. One representative door reads 26,
    and the four access tiles as four nodes read 27.
    """
    grid = [((0, 0), ("WATER",), None), ((9, 0), ("WATER",), None), ((0, 9), ("WATER",), None)]
    tasks = T.build(grid, available={})

    assert T.spanning_walk(tasks) == 24, (
        f"the tree over three quadrants is 8 + 8 + 8 out of the shed's nearest doors, "
        f"got {T.spanning_walk(tasks)}; 26 is one representative door, 27 is four doors"
    )
    assert T.spanning_walk(tasks) > len(grid) - 1, "the tile count is what the tree exists to improve on"


def test_the_shed_s_edge_costs_the_nearest_door_and_not_a_representative_one():
    """Two corners of the board: 8 out of each nearest door, 16 in all.

    A worker may start on any of the shed's four access tiles, so a tile's edge to the shed is the
    distance to the nearest of them. Charging the representative (4,4) instead reads 8 + 10 = 18, and
    leaving the four as four nodes reads 19.
    """
    grid = [((0, 0), ("WATER",), None), ((9, 9), ("WATER",), None)]
    tasks = T.build(grid, available={})

    assert T.spanning_walk(tasks) == 16, (
        f"(0,0) is 8 from its nearest door and (9,9) is 8 from its own, so the tree is 16, "
        f"got {T.spanning_walk(tasks)}; 18 is one representative door, 19 is four doors"
    )


def test_the_floor_counts_the_ladder_and_not_the_work_divided_by_the_horizon():
    """Forty tasks on the shed's tile, three hands hired at hour 20: 24, 28, 32, 36 turns of capacity.

    Dividing the work by the horizon reads 40/24 and answers two hands. The ladder answers four, which
    is the honest count: a hand hired in turn 20 has four turns in it, not twenty-four.
    """
    day, tasks = _one_tile_day(tasks_on_the_shed=40, hire_hours=(20, 20, 20))
    assert tasks.n == 40

    floor = B.lower_bound(day, tasks)

    assert floor == 4, (
        f"the ladder gives the farmer 24 turns and each hand 4, so 40 tasks need 4 units, got {floor}; "
        f"2 is the work divided by the horizon"
    )
    assert floor > -(-tasks.n // day.horizon), "the floor has to beat the horizon division to matter"


def test_the_ceiling_is_the_hands_the_day_offered_and_not_the_work():
    """The planner's offer caps the pool: a hand nobody pays for is not a hand the day has.

    Seven tasks on one tile with two hands offered is a ceiling of three - the farmer and the two -
    where the work alone would allow eight.
    """
    day, tasks = _one_tile_day(tasks_on_the_shed=7, hire_hours=(1, 1))

    assert B.ceiling_for(day, tasks) == len(day.units) + len(day.hire_times) == 3, (
        f"the offer is two hands and the farmer, so the ceiling is 3, got {B.ceiling_for(day, tasks)}; "
        f"8 is the work plus the units, which is what it used to be"
    )


def test_the_ceiling_still_caps_a_day_whose_work_is_smaller_than_its_offer():
    """Every unit does at least one task, so a pool larger than the work is never the smallest one."""
    day, tasks = _one_tile_day(tasks_on_the_shed=3, hire_hours=(1,) * 10)

    assert B.ceiling_for(day, tasks) == tasks.n + len(day.units) == 4, (
        f"three tasks with ten hands offered is capped by the work at 4, got {B.ceiling_for(day, tasks)}"
    )
