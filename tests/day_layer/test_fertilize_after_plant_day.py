"""A planting day fertilizes the plant, not the ground it goes into.

The engine's FERTILIZE needs a plant on the tile (`kaggriculture.py:475-482`); on bare ground it is
refused in silence (F047) and the fertilizer the day bought for it is spent on nothing. So on a chain
that plants and fertilizes the same tile in one day, the FERTILIZE comes after the PLANT whatever the
crop - the search is free to reorder everything else.

The day is three wheat tiles on the first row, the registry's own `PLANT+FERTILIZE+WATER`, with the
seeds and the fertilizer bought at hour 0 (in the shed from hour 1). Those three tiles with one or
two hands are the smallest days on which the search, left without the edge, fertilized before it
planted.

What is asserted is what the engine did: every tile the route fertilized is fertilized.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.dispatch import dispatch_plan
from agent.tile_dp.chains import chain_id_of, chain_ops
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route, to_plan
from agent.wsr.models import expand_chain

CHAIN = ("PLANT", "FERTILIZE", "WATER")
TILES = [(x, 0) for x in range(3)]
AVAILABLE = {"WHEAT": 1, "FERTILIZER": 1}
POOLS = (1, 2)
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _chains():
    return [(cell, chain_ops(chain_id_of(CHAIN)), "WHEAT") for cell in TILES]


def test_the_chain_is_the_registrys() -> None:
    """The premise: the chain under test is one the DP actually hands the layer."""
    assert tuple(chain_ops(chain_id_of(CHAIN))) == CHAIN


@pytest.mark.parametrize("crop", ["WHEAT", "TOMATO"])
def test_fertilize_waits_for_its_plant(crop) -> None:
    """The edge itself, for a one-shot crop and an ongoing one alike."""
    ex = expand_chain(CHAIN, entity=crop, cell=(0, 0), item=crop)
    assert ("plant", "fertilize") in ex.order, ex.order


@pytest.fixture(scope="module", params=POOLS, ids=[f"hands={h}" for h in POOLS])
def played(request):
    from offline_lab.fast_sim import FastSim

    hands = request.param
    chains = _chains()
    tasks = T.build(chains, available=AVAILABLE)
    day = B.Day(chains=tuple(chains), available=AVAILABLE, hire_times=(1,) * hands)
    result = B.search(day, tasks, hands=hands, max_hands=hands, budget_s=20.0)
    plan = to_plan(compile_route(day, tasks, result))
    orders = ([["BUY_SEED", "WHEAT", len(TILES)], ["BUY_PRODUCT", "FERTILIZER", len(TILES)]]
              + [["HIRE"]] * hands)

    sim = FastSim({"episodeSteps": 25})
    while True:
        obs = sim.observations()[0]
        if int(obs["day"]) != 0 or int(obs["hour"]) == 23:
            break
        action = dict(dispatch_plan(plan, obs))
        action["market"] = orders if int(obs["hour"]) == 0 else []
        sim.step([action, PASS])
    return hands, day, tasks, result, obs


def test_every_fertilizing_lands_on_a_plant(played) -> None:
    hands, day, tasks, result, obs = played
    assert result.complete, f"hands={hands}: {len(result.route)} of {tasks.n}"
    assert not check_route(day, tasks, result), check_route(day, tasks, result)
    tiles = obs["farms"][0]["tiles"]
    placed = {task_id for _turn, task_id, _worker in result.route}
    wasted = [(x, y) for index, (x, y) in enumerate(TILES)
              if f"d{index}_fertilize" in placed
              and int((tiles[y][x] or {}).get("fertilized_until_day", -1)) < 0]
    assert not wasted, (f"hands={hands}: the route fertilized {wasted} and the engine did not - "
                        f"the FERTILIZE ran before the PLANT, on bare ground")
