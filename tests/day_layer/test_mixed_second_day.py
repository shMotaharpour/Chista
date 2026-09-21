"""The mixed day's SECOND day: the board day 0 left, and the work that board needs.

A day-0 test cannot reach this shape. The wheat is in the ground and the animals are in their
structures, so the day's work is watering and feeding rather than building and planting - and the
hands, the market and the timetable are all read from a board that already exists. The fixture is
built once by `corpus/build_mixed_second_day.py` (which plays day 0 on the engine and writes the
board at day 1 hour 0), so this test never replays day 0.

What is asserted is the layer's own answer to that board:

    the work is the board's: every planted tile is watered and every animal fed, and nothing else
    the day is carried with the four hands it paid for, and no more than four are chosen
    the same day is carried by a smaller pool - it is a lighter day than day 0, and the layer says so
"""
import collections
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route

FIXTURE = pathlib.Path(__file__).parent / "corpus" / "mixed_second_day.json"
DAY = json.loads(FIXTURE.read_text())
HANDS = DAY["hands"]
#: Every planted tile is watered once and every animal fed once: a tile, its op, in the day's words.
WATERINGS = {tuple(cell) for cell, ops, _entity in DAY["chains"] if ops == ["WATER"]}
ANIMALS = {tuple(cell) for cell, ops, _entity in DAY["chains"] if "FEED" in ops}


def _grid():
    return [(tuple(cell), tuple(ops), entity) for cell, ops, entity in DAY["chains"]]


def _available():
    return {good: int(hour) for good, hour in DAY["available"].items()}


def _day(grid, hands):
    return B.Day(chains=tuple(grid), available=_available(),
                 hire_times=tuple(DAY["hire_times"])[:hands])


def _search(hands: int, max_hands: int):
    grid = _grid()
    day = _day(grid, max_hands)
    tasks = T.build(grid, available=_available())
    result = B.search(day, tasks, hands=hands, max_hands=max_hands, budget_s=20.0)
    return day, tasks, result


@pytest.fixture(scope="module")
def played():
    """The day searched with the hands it paid for."""
    return _search(HANDS, HANDS)


def test_the_fixture_is_the_day_it_claims() -> None:
    """The board and the work agree, so the day is a board's day and not a list of tasks.

    A fixture that drifts from the board it came from stops being a mid-game day without saying so:
    a tile in the ground with no watering, an animal with no feeding, or a work list that asks for
    something the board cannot answer.
    """
    board = {(x, y): tile for (x, y), tile in DAY["board"]}
    assert set(DAY["state"]["unlocked_quadrants"]) == {"NW"}, "the fixture's land is not day 0's"
    assert len(board) == len(WATERINGS) + len(ANIMALS), (
        f"{len(board)} tiles on the board against {len(WATERINGS) + len(ANIMALS)} in the work")

    for cell, tile in board.items():
        if tile.get("crop"):
            assert cell in WATERINGS, f"{cell} holds {tile['crop']} and the day never waters it"
        elif tile.get("animal"):
            assert cell in ANIMALS, f"{cell} holds {tile['animal']} and the day never feeds it"
        else:
            assert False, f"{cell} holds {tile} and the day's work does not answer it"

    fed = sum(1 for _cell, ops, _e in DAY["chains"] if "FEED" in ops)
    assert DAY["state"]["shed"]["WHEAT"] >= fed, (
        f"the day feeds {fed} animals from a shed holding {DAY['state']['shed']['WHEAT']} wheat")


def test_the_day_is_carried_with_the_hands_it_paid_for(played) -> None:
    """The land work is carried out, and the pool is no larger than the four the day paid for."""
    day, tasks, result = played
    assert result.complete, f"{len(result.route)} of {tasks.n} tasks with {HANDS} hands"
    assert result.pool <= HANDS, f"the layer chose {result.pool} hands where the day paid {HANDS}"
    assert not check_route(day, tasks, result), "the route breaks a rule the engine enforces"

    placed = collections.Counter(_pair(task_id) for _turn, task_id, _worker in result.route)
    asked = collections.Counter((index, op.lower())
                                for index, (_cell, ops, _entity) in enumerate(DAY["chains"])
                                for op in ops)
    assert placed == asked, (
        f"placed {sum(placed.values())} of the day's {sum(asked.values())} tile ops; "
        f"missing {(asked - placed).most_common(3)}, extra {(placed - asked).most_common(3)}")


def test_a_lighter_day_is_carried_by_a_smaller_pool() -> None:
    """Day 1 asks less of the farm than day 0 did, and the layer's own ladder says how much less.

    Without this the day could be carried by the full pool and the fixture would not show that a
    mid-game day is a different question - the board is already worked, so the hands are watering and
    feeding rather than clearing land.
    """
    _day_, tasks, result = _search(1, HANDS)
    assert result.complete, f"{len(result.route)} of {tasks.n} tasks with the ladder"
    assert result.pool < HANDS, (
        f"the lighter day chose {result.pool} hands of the {HANDS} allowed")


def _pair(task_id: str) -> tuple[int, str]:
    """A task id back to the tile it works and the op it runs: `d24_water` is (24, 'water')."""
    tile, rest = task_id.split("_", 1)
    return (int(tile[1:]), rest.rstrip("0123456789"))
