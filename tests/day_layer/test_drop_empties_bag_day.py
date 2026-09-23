"""A drop hands over the WHOLE bag: the feed a worker still carries goes with it.

The compiler loads every good a worker will consume at its door before the walk, and the engine's
DROP moves every item in the unit's inventory to the shed (`kaggriculture.py:343-356`) - the wheat
for the feedings still ahead included. A route that drops and then feeds on the same worker writes
FEEDs the engine refuses in silence (F047), and two unfed days in a row lose the animal.

The day is `test_animal_drop_day.py`'s: the three animals the mixed day's first day left, FEED,
CARE and COLLECT_FERTILIZER each, with the fertilizer banked by hour 9 - a deadline that makes an
early drop attractive. The board is the fixture's own (`corpus/mixed_second_day.json`), written into
the engine at hour 0 rather than replayed: only the animal tiles and the shed's wheat matter here.

What is asserted is what the engine did:

    every animal whose FEED the route placed is fed at the end of the day
    no worker eats from a bag its own DROP emptied and no PICKUP refilled
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
ANIMALS = [(tuple(cell), entity) for cell, ops, entity in DAY["chains"] if "FEED" in ops]
CHAIN = ("FEED", "CARE", "COLLECT_FERTILIZER")
DEADLINE = 9
AVAILABLE = {"WHEAT": 0}
#: The pools `test_animal_drop_day.py` searches. The farmer alone and two hands both came back
#: complete with an animal the engine never fed.
POOLS = (0, 1, 2)
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
MOVES = {"PASS", "NORTH", "SOUTH", "EAST", "WEST"}


def _search(hands: int):
    grid = [(cell, CHAIN, entity) for cell, entity in ANIMALS]
    tasks = T.build(grid, available=AVAILABLE, drop_by=[DEADLINE] * len(grid))
    day = B.Day(chains=tuple(grid), available=AVAILABLE, hire_times=(1,) * hands)
    return day, tasks, B.search(day, tasks, hands=hands, max_hands=hands, budget_s=20.0)


def _board_sim():
    """A fresh engine at day 0 hour 0 holding the fixture's animals and the shed's wheat."""
    from offline_lab.fast_sim import FastSim

    sim = FastSim({"episodeSteps": 25})
    live = sim.observations(copy_state=False)[0]
    tiles = live["farms"][0]["tiles"]
    for (x, y), _entity in ANIMALS:
        tiles[y][x] = json.loads(json.dumps(BOARD[(x, y)]))
    live["private"]["shed"]["WHEAT"] = int(DAY["state"]["shed"]["WHEAT"])
    return sim


def _play(plan, hands: int):
    """The compiled day on the engine, to the last hour before the night's refresh."""
    sim = _board_sim()
    while True:
        obs = sim.observations()[0]
        if int(obs["day"]) != 0 or int(obs["hour"]) == 23:
            return obs
        action = dict(dispatch_plan(plan, obs))
        action["market"] = [["HIRE"]] * hands if int(obs["hour"]) == 0 else []
        sim.step([action, PASS])


@pytest.fixture(scope="module", params=POOLS, ids=[f"hands={h}" for h in POOLS])
def played(request):
    hands = request.param
    day, tasks, result = _search(hands)
    ops = compile_route(day, tasks, result)
    return hands, day, tasks, result, ops, _play(to_plan(ops), hands)


def test_the_board_is_the_fixture() -> None:
    """The premise: three animals on the board, unfed, and wheat in the shed for all three."""
    sim = _board_sim()
    obs = sim.observations()[0]
    tiles = obs["farms"][0]["tiles"]
    for (x, y), entity in ANIMALS:
        assert tiles[y][x]["animal"] == entity and not tiles[y][x]["fed_today"]
    assert obs["private"]["shed"]["WHEAT"] >= len(ANIMALS)
    assert len(ANIMALS) == 3


def test_every_feeding_the_route_placed_is_eaten(played) -> None:
    """The engine's own flag: an animal the route fed is fed."""
    hands, _day, _tasks, result, _ops, obs = played
    tiles = obs["farms"][0]["tiles"]
    placed = {task_id for _turn, task_id, _worker in result.route}
    starved = [entity for index, ((x, y), entity) in enumerate(ANIMALS)
               if f"d{index}_feed" in placed and not tiles[y][x]["fed_today"]]
    assert not starved, (
        f"hands={hands}: the route fed {starved} and the engine did not - "
        f"a DROP before the FEED took the wheat out of the bag")


def test_no_worker_consumes_from_an_emptied_bag(played) -> None:
    """The compiled ops, read the way the engine reads them: a DROP empties the bag, a PICKUP or a
    take off a tile refills it, and nothing is eaten that is not in it."""
    hands, day, tasks, result, ops, _obs = played
    assert not check_route(day, tasks, result), check_route(day, tasks, result)
    eats = {"FEED": "WHEAT", "FERTILIZE": "FERTILIZER"}
    for worker, unit in enumerate(ops.units):
        bag: dict[str, int] = {}
        for turn, op in enumerate(unit):
            if op[0] == "DROP":
                bag.clear()
            elif op[0] == "PICKUP":
                bag[op[1]] = bag.get(op[1], 0) + int(op[2])
            elif op[0] == "COLLECT_FERTILIZER":
                bag["FERTILIZER"] = bag.get("FERTILIZER", 0) + 1
            elif op[0] in eats:
                good = eats[op[0]]
                assert bag.get(good, 0) > 0, (
                    f"hands={hands}: worker {worker} runs {op[0]} at turn {turn} with no {good} "
                    f"in its bag - a DROP before it emptied the bag")
                bag[good] -= 1
