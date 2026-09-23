"""Each good is picked up at its own hour, and the planner's compile writes the day the search priced.

Two ways a compiled day fails on the engine in silence (F047):

1. A worker loads every good it consumes at its door before the walk. A good is in the shed at its
   own hour (`available`), and a PICKUP before that hour is refused. A cow bought in turn 1 is in
   the shed from hour 2 however early the wheat is, so the cow is picked up at hour 2 and the wheat
   at hour 0, not both counted from the earlier hour.

2. `agent/planner/day.py:compile` is the runtime's call into this layer. It must write the day from
   the doors the search priced the hands on (`Result.doors`). An explicit `settled=` turns those
   doors off, and on a day whose farmer walks in turn 0 the hands are then written from the wrong
   doors and the compiler refuses the day.

What is asserted is what the engine did.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.dispatch import dispatch_plan
from agent.planner import day as D
from agent.tile_dp.chains import chain_id_of, chain_ops
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route, to_plan

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _run_day(plan, orders, board=None, shed=None, until=23):
    """The plan on the engine from day 0 hour 0 to `until`; the last observation.

    `orders` is the market's orders by the hour they are placed in.
    """
    from offline_lab.fast_sim import FastSim

    sim = FastSim({"episodeSteps": 25})
    live = sim.observations(copy_state=False)[0]
    for (x, y), tile in (board or {}).items():
        live["farms"][0]["tiles"][y][x] = json.loads(json.dumps(tile))
    for good, n in (shed or {}).items():
        live["private"]["shed"][good] = int(n)
    while True:
        obs = sim.observations()[0]
        if int(obs["day"]) != 0 or int(obs["hour"]) == until:
            return obs
        action = dict(dispatch_plan(plan, obs))
        action["market"] = list(orders.get(int(obs["hour"]), []))
        sim.step([action, PASS])


# -- 1. each good at its own hour ----------------------------------------------------------------

COW_TILE = (4, 2)
COW_CHAIN = ("BUILD_PASTURE", "PLACE", "FEED")
#: The wheat is on the shelf from hour 0; the cow is bought in turn 1 and lands at hour 2 (a buy
#: settles after the units act, F030). Two hours apart, so loading the goods one turn apart from
#: the earlier one's hour still takes the cow before it is there - whatever order they go in.
COW_AVAILABLE = {"WHEAT": 0, "COW": 2}
COW_ORDERS = {1: [["BUY_ANIMAL", "COW", 1]]}


def test_a_cow_bought_today_is_picked_up_when_it_is_in_the_shed() -> None:
    chains = [(COW_TILE, chain_ops(chain_id_of(COW_CHAIN)), "COW")]
    tasks = T.build(chains, available=COW_AVAILABLE)
    day = B.Day(chains=tuple(chains), available=COW_AVAILABLE, hire_times=())
    result = B.search(day, tasks, hands=0, max_hands=0)
    assert result.complete and not check_route(day, tasks, result)
    ops = compile_route(day, tasks, result)

    early = [(turn, op) for turn, op in enumerate(ops.units[0])
             if op[0] == "PICKUP" and turn < COW_AVAILABLE.get(op[1], 0)]
    assert not early, f"picked up before the good is in the shed: {early}"

    obs = _run_day(to_plan(ops), COW_ORDERS, shed={"WHEAT": 2})
    x, y = COW_TILE
    tile = obs["farms"][0]["tiles"][y][x]
    assert tile.get("animal") == "COW", f"the pasture holds no cow: {tile}"
    assert tile.get("fed_today"), f"the cow was placed and never fed: {tile}"


# -- 2. the planner's own compile ----------------------------------------------------------------

FIXTURE = pathlib.Path(__file__).parent / "corpus" / "mixed_second_day.json"
DAY = json.loads(FIXTURE.read_text())
#: The watering half of the mixed second day: no goods to load, so the farmer walks off its door
#: in turn 0 - which is what moves the door the next hand lands on (F040).
WATERS = [(tuple(cell), tuple(ops), entity) for cell, ops, entity in DAY["chains"]
          if ops == ["WATER"]]
BOARD = {tuple(cell): tile for cell, tile in DAY["board"] if tuple(cell) in {c for c, _o, _e in WATERS}}


def _day_plan(hands: int):
    fit = D.DayFit(chains=tuple(WATERS), placed=len(WATERS), tasks=len(WATERS), pool=hands,
                   complete=True, hours_used=0.0, hours_committed=0.0)
    return D.DayPlan(master=None, choices=[], mixes={}, day=fit, hands=hands)


def _obs():
    from offline_lab.fast_sim import FastSim

    sim = FastSim({"episodeSteps": 25})
    live = sim.observations(copy_state=False)[0]
    for (x, y), tile in BOARD.items():
        live["farms"][0]["tiles"][y][x] = json.loads(json.dumps(tile))
    return sim.observations()[0]


@pytest.mark.parametrize("hands", [1, 2, 3])
def test_the_planners_compile_writes_the_day_the_search_priced(hands) -> None:
    obs = _obs()
    try:
        plan = D.compile(_day_plan(hands), obs, hands=hands)
    except ValueError as exc:
        pytest.fail(f"hands={hands}: the planner's compile refused its own day - the hands were "
                    f"written from doors the search did not price: {exc}")
    assert plan["units"][0][0][0] in ("NORTH", "SOUTH", "EAST", "WEST"), (
        "the premise: the farmer walks off its door in turn 0")

    played = _run_day(plan, {0: [["HIRE"]] * hands}, board=BOARD)
    tiles = played["farms"][0]["tiles"]
    dry = [(x, y) for (x, y) in BOARD if not tiles[y][x]["watered_today"]]
    assert not dry, f"hands={hands}: {len(dry)} of {len(BOARD)} tiles not watered: {dry[:5]}"
