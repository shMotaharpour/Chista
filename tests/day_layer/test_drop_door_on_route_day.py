"""A drop is handed over at the door nearest the worker, not at the harvest's own door.

Any of the four shed-access tiles takes a DROP (`kaggriculture.py:138-139`, `:343-356`), so where a
bag is handed over is a property of the route, not of the harvest it banks. A drop bound to the door
nearest its harvest makes a worker that harvested on one side of the shed and then worked the other
walk back across the shed to bank it - turns the day does not have.

The day: the farmer alone. A ripe wheat tile east of the shed, (8, 4), whose harvest must be in the
shed by the end of the day, and two plantings south-east of the shed, (9, 6) and (6, 9), each planted
and watered. The route that carries all six tasks harvests, plants and waters both, and drops at the
south-east door (5, 5) on the way back - not at the north-east door (5, 4) nearest the harvest.

The board is written into the engine at hour 0: the east and south quadrants are ours and the wheat
is ripe. What is asserted is what the engine did:

    every task is placed, and the compiled day banks the wheat before the night
    every planting the route placed holds wheat, watered
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.dispatch import dispatch_plan
from agent.world.board import MOVE_DELTA
from agent.world.rules import SHED_ACCESS
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route, to_plan
from agent.wsr.routing import nearest_shed

HARVEST_TILE = (8, 4)
PLANT_TILES = ((9, 6), (6, 9))
CHAINS = ([(HARVEST_TILE, ("HARVEST",), "WHEAT")]
          + [(cell, ("PLANT", "WATER"), "WHEAT") for cell in PLANT_TILES])
#: The harvest is due in the shed by the day's last hour, and the plantings have no drop.
DROP_BY = [23, None, None]
AVAILABLE = {"WHEAT": 0}
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _search():
    tasks = T.build(CHAINS, available=AVAILABLE, drop_by=DROP_BY)
    day = B.Day(chains=tuple(CHAINS), available=AVAILABLE, hire_times=())
    return day, tasks, B.search(day, tasks, hands=0, max_hands=0)


def _board_sim():
    """A fresh engine at hour 0 with the east and south tiles ours and ripe wheat on the harvest."""
    from offline_lab.fast_sim import FastSim

    sim = FastSim({"episodeSteps": 25, "weedSpawnChance": 0.0})
    live = sim.observations(copy_state=False)[0]
    tiles = live["farms"][0]["tiles"]
    for x, y in PLANT_TILES:
        tiles[y][x] = None
    x, y = HARVEST_TILE
    tiles[y][x] = {"kind": "PLANT", "crop": "WHEAT", "planted_day": -10, "watered_today": False,
                   "consecutive_unwatered": 0, "yield_units": 1, "max_lifespan_step": -1,
                   "fertilized_until_day": -1}
    live["private"]["seeds"]["WHEAT"] = len(PLANT_TILES)
    return sim


def _play(plan):
    """The compiled day on the engine, stopped at its last hour - before the night banks every bag."""
    sim = _board_sim()
    while True:
        obs = sim.observations()[0]
        if int(obs["day"]) != 0 or int(obs["hour"]) == 23:
            return obs
        sim.step([dict(dispatch_plan(plan, obs)), PASS])


@pytest.fixture(scope="module")
def played():
    """The searched day, compiled and played - or the compiler's refusal, for the guard to name."""
    day, tasks, result = _search()
    assert not check_route(day, tasks, result), check_route(day, tasks, result)
    try:
        ops = compile_route(day, tasks, result)
    except ValueError as exc:
        return tasks, result, None, str(exc)
    return tasks, result, ops, _play(to_plan(ops))


def _written(played):
    tasks, result, ops, obs = played
    assert ops is not None, (
        f"the compiler refuses the day the search priced - the two disagree on the drop's door: {obs}")
    return tasks, result, ops, obs


def test_the_premise_the_drop_door_differs_from_the_harvest_door(played) -> None:
    """The door nearest the harvest is not the one the route drops at, so the rule is exercised."""
    _tasks, _result, ops, _obs = _written(played)
    farmer = ops.units[0]
    drops = [turn for turn, op in enumerate(farmer) if op[0] == "DROP"]
    assert drops, f"the farmer never drops: {[(t, op) for t, op in enumerate(farmer) if op[0] != 'PASS']}"
    at = B.FARMER_START
    for op in farmer[:drops[0]]:
        if op[0] in MOVE_DELTA:
            at = (at[0] + int(MOVE_DELTA[op[0]][0]), at[1] + int(MOVE_DELTA[op[0]][1]))
    assert at in tuple(tuple(d) for d in SHED_ACCESS), f"the farmer drops off the shed, at {at}"
    assert at != nearest_shed(HARVEST_TILE), (
        f"the route drops at the harvest's own door {at}, so this day does not tell the rules apart")


def test_the_day_is_carried_and_the_wheat_is_banked_before_the_night(played) -> None:
    tasks, result, _ops, obs = _written(played)
    assert result.complete, (
        f"{len(result.route)} of {tasks.n} tasks with the farmer alone - the drop walked back to the "
        f"harvest's own door")
    assert int(obs["private"]["shed"].get("WHEAT", 0)) >= 1, (
        f"the shed holds no wheat at the day's last hour - the harvest was never dropped")


def test_every_planting_is_wheat_and_watered(played) -> None:
    _tasks, _result, _ops, obs = _written(played)
    tiles = obs["farms"][0]["tiles"]
    for x, y in PLANT_TILES:
        tile = tiles[y][x]
        assert isinstance(tile, dict) and tile.get("crop") == "WHEAT", f"{(x, y)} was never planted: {tile}"
        assert tile.get("watered_today"), f"{(x, y)} was planted and never watered: {tile}"
