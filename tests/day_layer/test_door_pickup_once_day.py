"""A door pickup is paid for once: the goods loaded at the door are in the bag until the first DROP.

`start_hours` delays a worker's first walk by one turn per good its charge loads at the door - the
PICKUPs the compiler writes there before the walk (`pickup_turns`). The first consumer of each of those
goods must then find it in the bag. Charged a second time, it waits a turn the compiler fills with a
PASS, and every task after it on that worker slides a turn later.

The day is the smallest one that shows it: the farmer alone runs a sheep's whole chain two tiles north
of its door, `BUILD_PASTURE+PLACE+FEED+CARE`, with the sheep and its wheat bought in turn 0.

What is asserted:

    the farmer's compiled day has no PASS between its first door pickup and its last task
    the engine placed the sheep, fed it and cared for it
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.dispatch import dispatch_plan
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route, to_plan

SHEEP_TILE = (4, 2)
CHAINS = [(SHEEP_TILE, ("BUILD_PASTURE", "PLACE", "FEED", "CARE"), "SHEEP")]
#: Bought in turn 0, so both are in the shed from hour 1.
AVAILABLE = {"SHEEP": 1, "WHEAT": 1}
ORDERS = [["BUY_ANIMAL", "SHEEP", 1], ["BUY_PRODUCT", "WHEAT", 1]]
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _search():
    tasks = T.build(CHAINS, available=AVAILABLE)
    day = B.Day(chains=tuple(CHAINS), available=AVAILABLE, hire_times=())
    return day, tasks, B.search(day, tasks, hands=0, max_hands=0)


def _play(plan):
    """The compiled day on the engine, to the last hour before the night's refresh."""
    from offline_lab.fast_sim import FastSim

    sim = FastSim({"episodeSteps": 25})
    sim.reset()
    while True:
        obs = sim.observations()[0]
        if int(obs["day"]) != 0 or int(obs["hour"]) == 23:
            return obs
        action = dict(dispatch_plan(plan, obs))
        action["market"] = ORDERS if int(obs["hour"]) == 0 else []
        sim.step([action, PASS])


@pytest.fixture(scope="module")
def played():
    day, tasks, result = _search()
    assert result.complete, f"{len(result.route)} of {tasks.n} tasks with the farmer alone"
    assert not check_route(day, tasks, result), check_route(day, tasks, result)
    ops = compile_route(day, tasks, result)
    return ops, _play(to_plan(ops))


def test_no_turn_is_lost_between_the_door_load_and_the_last_task(played) -> None:
    ops, _obs = played
    farmer = ops.units[0]
    busy = [turn for turn, op in enumerate(farmer) if op[0] != "PASS"]
    assert farmer[busy[0]][0] == "PICKUP", f"the premise: the day loads at the door first: {farmer}"
    idle = [turn for turn in range(busy[0], busy[-1] + 1) if farmer[turn][0] == "PASS"]
    assert not idle, (
        f"the farmer passes at turns {idle} with its goods already in the bag - a door pickup "
        f"charged a second time: {[(t, op) for t, op in enumerate(farmer) if op[0] != 'PASS']}")


def test_the_engine_placed_fed_and_cared_for_the_sheep(played) -> None:
    _ops, obs = played
    x, y = SHEEP_TILE
    tile = obs["farms"][0]["tiles"][y][x]
    assert isinstance(tile, dict) and tile.get("animal") == "SHEEP", tile
    assert tile.get("fed_today") and tile.get("cared_today"), tile
