"""The farmer's turn 0 is not spent waiting at its door for goods bought in turn 0.

Goods bought in turn 0 are in the shed from hour 1, and the farmer acts from hour 0. A worker that has
only worked the four shed-access tiles is still at the shed, where it can pick up (`kaggriculture.py`
`:138-139`, `:358-375`) - so a task on the door the farmer stands on, one that needs no good, can
fill hour 0 and the pickups follow it.

The day: the farmer alone. A wheat plant on its own door (4, 4) to water, a sheep two tiles north to
feed and care for with wheat bought in turn 0, and two wheat plants further out to water. Written into
the engine at hour 0.

What is asserted is what the engine did:

    the farmer's hour 0 is not a PASS
    every plant the day waters is watered, and the sheep is fed and cared for
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

DOOR_PLANT = (4, 4)
FAR_PLANTS = ((1, 1), (2, 1))
SHEEP_TILE = (4, 2)
CHAINS = ([(DOOR_PLANT, ("WATER",), "WHEAT"), (SHEEP_TILE, ("FEED", "CARE"), "SHEEP")]
          + [(cell, ("WATER",), "WHEAT") for cell in FAR_PLANTS])
#: The wheat is bought in turn 0, so it is in the shed from hour 1.
AVAILABLE = {"WHEAT": 1}
ORDERS = [["BUY_PRODUCT", "WHEAT", 1]]
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
PLANT = {"kind": "PLANT", "crop": "WHEAT", "planted_day": -1, "watered_today": False,
         "consecutive_unwatered": 0, "yield_units": 0, "max_lifespan_step": -1,
         "fertilized_until_day": -1}
SHEEP = {"kind": "PASTURE", "animal": "SHEEP", "placed_day": -1, "yield_units": 0,
         "consecutive_unfed": 0, "fed_today": False, "cared_today": False,
         "fertilizer_available": False, "pending_care_bonus": 0}


def _search():
    tasks = T.build(CHAINS, available=AVAILABLE)
    day = B.Day(chains=tuple(CHAINS), available=AVAILABLE, hire_times=())
    return day, tasks, B.search(day, tasks, hands=0, max_hands=0)


def _play(plan):
    """The compiled day on an engine holding the plants and the sheep, to the day's last hour."""
    from offline_lab.fast_sim import FastSim

    sim = FastSim({"episodeSteps": 25, "weedSpawnChance": 0.0})
    live = sim.observations(copy_state=False)[0]
    tiles = live["farms"][0]["tiles"]
    for x, y in (DOOR_PLANT,) + FAR_PLANTS:
        tiles[y][x] = dict(PLANT)
    x, y = SHEEP_TILE
    tiles[y][x] = dict(SHEEP)
    while True:
        obs = sim.observations()[0]
        if int(obs["day"]) != 0 or int(obs["hour"]) == 23:
            return obs
        action = dict(dispatch_plan(plan, obs))
        action["market"] = ORDERS if int(obs["hour"]) == 0 else []
        sim.step([action, PASS])


@pytest.fixture(scope="module")
def played():
    """The searched day, compiled and played - or the compiler's refusal, for the guard to name."""
    day, tasks, result = _search()
    assert result.complete, f"{len(result.route)} of {tasks.n} tasks with the farmer alone"
    assert not check_route(day, tasks, result), check_route(day, tasks, result)
    try:
        ops = compile_route(day, tasks, result)
    except ValueError as exc:
        return None, str(exc)
    return ops, _play(to_plan(ops))


def _written(played):
    ops, obs = played
    assert ops is not None, (
        f"the compiler refuses the day the search priced - the two disagree on when the farmer "
        f"loads: {obs}")
    return ops, obs


def test_the_farmer_does_not_wait_at_its_door_in_hour_zero(played) -> None:
    ops, _obs = _written(played)
    farmer = ops.units[0]
    assert farmer[0][0] != "PASS", (
        f"the farmer waits at its door in hour 0 for wheat that lands at hour 1, with a plant to water "
        f"under its feet: {[(t, op) for t, op in enumerate(farmer) if op[0] != 'PASS']}")


def test_the_engine_watered_every_plant_and_fed_and_cared_for_the_sheep(played) -> None:
    _ops, obs = _written(played)
    tiles = obs["farms"][0]["tiles"]
    for x, y in (DOOR_PLANT,) + FAR_PLANTS:
        assert tiles[y][x].get("watered_today"), f"{(x, y)} was never watered: {tiles[y][x]}"
    x, y = SHEEP_TILE
    assert tiles[y][x].get("fed_today"), f"the sheep was never fed: {tiles[y][x]}"
    assert tiles[y][x].get("cared_today"), f"the sheep was never cared for: {tiles[y][x]}"
