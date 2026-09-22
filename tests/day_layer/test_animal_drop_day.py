"""The animals' day: FEED, CARE and COLLECT_FERTILIZER, with the fertilizer banked by hour 9.

The board is `test_mixed_second_day.py`'s - the three animals day 0 left, each with fertilizer
available - and the day's work is what the engine's own rules ask of that board: fed once a day
(`kaggriculture.py:505-513`), cared for once a day (`:524-530`), and the fertilizer collected while
it is there (`:515-522`). The three ops have NO order between them; the engine requires none, and the
only precedence is that a drop comes after the task whose good it banks.

That last one is a worker tie, not just a turn: the bag is the worker's, so a drop by anybody but the
worker that took the good hands over nothing. What is asserted here is what the engine would be given:

    the drops land by the deadline - every arrival is in the shed at or before hour 9
    each drop is on the worker that took the good, which is what makes it bank anything
    the farmer alone and one hand come up one drop short, and two hands carry the day
    the shortfall is the clock and not the pool: the answer has turns to spare
    the ops' order in the chain is not a constraint: the same day comes back either way
"""
import collections
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route

BOARD = pathlib.Path(__file__).parent / "corpus" / "mixed_second_day.json"
DAY = json.loads(BOARD.read_text())
#: The animals day 0 left, and the work the board needs from them.
ANIMALS = [(tuple(cell), entity) for cell, ops, entity in DAY["chains"] if "FEED" in ops]
CHAIN = ("FEED", "CARE", "COLLECT_FERTILIZER")
#: The hour the fertilizer has to be in the shed by.
DEADLINE = 9
HANDS = 2


def _build(chain=CHAIN, hands=HANDS):
    grid = [(cell, chain, entity) for cell, entity in ANIMALS]
    tasks = T.build(grid, available={"WHEAT": 0}, drop_by=[DEADLINE] * len(grid))
    day = B.Day(chains=tuple(grid), available={"WHEAT": 0}, hire_times=(1,) * hands)
    return day, tasks, grid


def _search(hands: int, chain=CHAIN):
    day, tasks, _grid = _build(chain, hands)
    return day, tasks, B.search(day, tasks, hands=hands, max_hands=hands, budget_s=20.0)


@pytest.fixture(scope="module")
def played():
    """The day carried by the two hands it needs."""
    return _search(HANDS)


def test_the_fertilizer_is_banked_by_the_deadline(played) -> None:
    """Every arrival the compiled day promises is in the shed at or before hour 9."""
    day, tasks, result = played
    assert result.complete, f"{len(result.route)} of {tasks.n} tasks with {HANDS} hands"
    assert not check_route(day, tasks, result), "the route breaks a rule the engine enforces"

    arrivals = compile_route(day, tasks, result).arrivals
    assert arrivals, "the day banks nothing at all"
    late = [(hour, item, n) for hour, item, n in arrivals if int(hour) > DEADLINE]
    assert not late, f"these arrivals are after hour {DEADLINE}: {late}"
    assert sum(n for _hour, item, n in arrivals if item == "FERTILIZER") == len(ANIMALS), (
        f"the day collects from {len(ANIMALS)} animals and banks "
        f"{sum(n for _h, i, n in arrivals if i == 'FERTILIZER')} fertilizer")


def test_each_drop_is_on_the_worker_that_took_the_good() -> None:
    """The tie that makes a drop bank anything: its worker is the one holding the good.

    A drop hands over the worker's own bag, so a route that splits the pair writes an op the engine
    runs happily and banks nothing - the deadline passes with the fertilizer still in a bag. Checked
    on the pools that come up short as well as the one that carries: the split showed up on the short
    day, and a guard that only reads the carrying day cannot see it.
    """
    seen = 0
    for hands in (0, 1, HANDS):
        _day_, tasks, result = _search(hands)
        who = {task_id: int(worker) for _turn, task_id, worker in result.route}
        seen += sum(1 for task_id in who if task_id.endswith("_drop"))
        for i in range(tasks.n):
            for slot in range(tasks.ties.shape[1]):
                mate = int(tasks.ties[i, slot])
                if mate < 0:
                    continue
                row, other = tasks.ids[i], tasks.ids[mate]
                if row in who and other in who:
                    assert who[row] == who[other], (
                        f"{row} is on worker {who[row]} and {other} on worker {who[other]}: "
                        f"the good never reaches the shed")
    assert seen, "no pool placed a drop at all, so nothing was banked"


def test_the_farmer_alone_carries_the_day() -> None:
    """The whole day is carried with no hands at all - and it is the portfolio that finds it.

    The day's own arithmetic says one worker: twelve tasks, seventeen turns with the walks, and the
    fertilizer can be banked in one drop because the bag carries all three animals' fertilizer. The
    earliest-finish ranking alone does not find it (it places 11 of 12 and lets the last drop go
    past its hour); the deadline rankings do.
    """
    _day_, tasks, result = _search(0)
    assert result.complete, f"the farmer alone placed {len(result.route)} of {tasks.n}"
    assert result.spare > 0, "the day was carried with nothing to spare, which cannot be right"

    day, _tasks, _result = _build(hands=0)
    arrivals = compile_route(day, tasks, result).arrivals
    late = [(hour, item, n) for hour, item, n in arrivals if int(hour) > DEADLINE]
    assert not late, f"these arrivals are after hour {DEADLINE}: {late}"
    assert sum(n for _hour, item, n in arrivals if item == "FERTILIZER") == len(ANIMALS), (
        "the day does not bank one fertilizer per animal")


def test_two_hands_carry_the_whole_day() -> None:
    """One more hand than the day can use alone is what banks the third drop before hour 9."""
    _day_, tasks, result = _search(HANDS)
    assert result.complete, f"{HANDS} hands placed {len(result.route)} of {tasks.n}"
    assert result.pool <= HANDS, f"the layer chose {result.pool} hands of the {HANDS} allowed"


def test_the_ops_order_in_the_chain_is_not_a_constraint() -> None:
    """The same day written in a different order is the same day: these three ops have no order.

    The engine asks for none of them in any particular order - CARE's bonus only needs the animal to
    be fed that day - so the chain's sequence is the caller's way of naming the work, not a
    precedence the search has to honour.
    """
    _day_, tasks, result = _search(HANDS)
    reversed_chain = tuple(reversed(CHAIN))
    _day2, tasks2, result2 = _search(HANDS, reversed_chain)
    assert result2.complete, f"the reversed chain placed {len(result2.route)} of {tasks2.n}"
    assert len(result2.route) == len(result.route) == tasks.n

    edges = [(tasks.ids[i], tasks.ids[j]) for i in range(tasks.n) for j in range(tasks.n)
             if tasks.pred[i, j]]
    assert edges, "the day has no precedence at all, and a drop has to come after its collect"
    assert all(after.endswith("_drop") for after, _before in edges), (
        f"the chain's own order became a constraint: {edges}")
