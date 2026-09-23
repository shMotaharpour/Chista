"""A good a worker uses only after its own DROP is not charged to its door load.

The compiler loads at the door only the goods a worker uses before its first DROP (`_bag`): the DROP
empties the whole bag (`kaggriculture.py:343-356`), so a good used after it is fetched then, on the
way (`legs`). The search's first pass charges every worker every good of the day at the door
(`preload_turns`). When that pass already carries the day, the worker's first walk still starts a
turn late for a pickup the compiler never writes - a PASS at the head of the day.

The day: the farmer alone, a goose whose fertilizer must be banked by hour 10, and a sheep across the
quadrant that only needs its FEED. Collecting first and feeding after the drop is the route; the
wheat for the feed is picked up after the drop, not at the door.

What is asserted:

    the premise: the farmer's FEED is after its DROP, and it loads nothing at the door
    the farmer's first op is a move at turn 0 - no turn is spent at the door
    the engine banked the fertilizer and fed the sheep
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.dispatch import dispatch_plan
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route, to_plan

FIXTURE = pathlib.Path(__file__).parent / "corpus" / "mixed_second_day.json"
DAY = json.loads(FIXTURE.read_text())
BOARD = {tuple(cell): tile for cell, tile in DAY["board"]}
#: Two of the fixture's animal records, written onto tiles the farmer has to walk to.
GOOSE_TILE, SHEEP_TILE = (2, 2), (0, 4)
CHAINS = [(GOOSE_TILE, ("COLLECT_FERTILIZER",), "GOOSE"),
          (SHEEP_TILE, ("FEED",), "SHEEP")]
DROP_BY = [10, None]
AVAILABLE = {"WHEAT": 0}
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _search():
    tasks = T.build(CHAINS, available=AVAILABLE, drop_by=DROP_BY)
    day = B.Day(chains=tuple(CHAINS), available=AVAILABLE, hire_times=())
    return day, tasks, B.search(day, tasks, hands=0, max_hands=0)


def _play(plan):
    """The compiled day on an engine holding the two animals and the shed's wheat."""
    from offline_lab.fast_sim import FastSim

    sim = FastSim({"episodeSteps": 25})
    live = sim.observations(copy_state=False)[0]
    tiles = live["farms"][0]["tiles"]
    for (x, y), source in ((GOOSE_TILE, (4, 2)), (SHEEP_TILE, (4, 3))):
        tiles[y][x] = json.loads(json.dumps(BOARD[source]))
    live["private"]["shed"]["WHEAT"] = int(DAY["state"]["shed"]["WHEAT"])
    while True:
        obs = sim.observations()[0]
        if int(obs["day"]) != 0 or int(obs["hour"]) == 23:
            return obs
        sim.step([dict(dispatch_plan(plan, obs)), PASS])


@pytest.fixture(scope="module")
def played():
    day, tasks, result = _search()
    assert result.complete, f"{len(result.route)} of {tasks.n} tasks with the farmer alone"
    assert not check_route(day, tasks, result), check_route(day, tasks, result)
    ops = compile_route(day, tasks, result)
    return result, ops, _play(to_plan(ops))


def test_the_feed_is_after_the_drop_and_nothing_is_loaded_at_the_door(played) -> None:
    result, ops, _obs = played
    turn = {task_id: t for t, task_id, _w in result.route}
    assert turn["d1_feed"] > turn["d0_collect_fertilizer_drop"] >= 0, sorted(result.route)
    farmer = ops.units[0]
    first = next(t for t, op in enumerate(farmer) if op[0] != "PASS")
    assert farmer[first][0] != "PICKUP", f"the farmer loads at the door: {farmer[first]}"


def test_no_turn_is_spent_at_the_door_for_a_good_used_after_the_drop(played) -> None:
    _result, ops, _obs = played
    farmer = ops.units[0]
    assert farmer[0][0] != "PASS", (
        f"the farmer waits at its door in turn 0 for a pickup it never makes: "
        f"{[(t, op) for t, op in enumerate(farmer) if op[0] != 'PASS']}")


def test_the_engine_banked_the_fertilizer_and_fed_the_sheep(played) -> None:
    _result, _ops, obs = played
    x, y = SHEEP_TILE
    assert obs["farms"][0]["tiles"][y][x].get("fed_today"), obs["farms"][0]["tiles"][y][x]
    assert int(obs["private"]["shed"].get("FERTILIZER", 0)) >= 1, obs["private"]["shed"]
