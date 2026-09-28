"""Each good is picked up at its own hour, and the planner's compile writes the day the search priced.

Two ways a compiled day fails on the engine in silence (F047):

1. A worker loads every good it consumes at its door before the walk. A good is in the shed at its
   own hour (`available`), and a PICKUP before that hour is refused. A cow bought in turn 1 is in
   the shed from hour 2 however early the wheat is, so the cow is picked up at hour 2 and the wheat
   at hour 0, not both counted from the earlier hour.

2. `agent/planner/day.py:compile` is the runtime's call into this layer. It must write the day from
   the doors the search priced the hands on (`Result.doors`). Placing the hands from the farmer's
   start cell instead is a different day whenever the farmer walks in turn 0, and the compiler
   refuses it.

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
    """The plan on the engine from day 0 hour 0 THROUGH `until`; the last observation.

    `orders` is the market's orders by the hour they are placed in. `until=23`
    plays the day's LAST turn (0..23) and returns the observation before the day
    rolls over - the old stop-on-hour==23 form dropped turn 23 itself, so any
    day whose last op sat at the horizon read a dry tile the plan had watered.
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
        if int(obs["day"]) != 0:
            return obs
        hour = int(obs["hour"])
        action = dict(dispatch_plan(plan, obs))
        action["market"] = list(orders.get(hour, []))
        sim.step([action, PASS])
        if hour >= until:
            return sim.observations()[0]


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
    # The day rolled: fed_today was cleared by _daily_refresh_animals; the durable
    # record of a fed day is consecutive_unfed == 0 (an unfed day reads 1).
    assert tile.get("consecutive_unfed") == 0, (
        f"the cow was placed and never fed: consecutive_unfed="
        f"{tile.get('consecutive_unfed')}")


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


@pytest.mark.parametrize("hands", [2, 3])
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
    # The day rolled: watered_today was cleared by _daily_refresh_plants; the durable
    # record of a watered day is consecutive_unwatered == 0 (one missed day reads 1).
    dry = [(x, y) for (x, y) in BOARD
           if tiles[y][x].get("consecutive_unwatered", 0) != 0]
    assert not dry, f"hands={hands}: {len(dry)} of {len(BOARD)} tiles not watered: {dry[:5]}"


@pytest.mark.parametrize("hands", [1, 2, 3])
def test_the_compiled_day_hires_exactly_the_hands_it_priced(hands) -> None:
    """One source: the market hires the tuple's own length, not a second count.

    `Day.hire_times` is the day's whole labour and `compile` hires `result.pool` —
    the same number, because the tuple IS that pool's timetable. A compile that
    hired a different count would dispatch hands the day was never costed with,
    which is the mismatch the hours were given one home to prevent.
    """
    plan = D.compile(_day_plan(hands), _obs(), hands=hands)
    hires = sum(1 for hour in plan["market"] for o in hour if o and o[0] == "HIRE")
    assert hires == hands, f"priced {hands} hands, hired {hires}"


def test_one_hand_cannot_water_the_whole_mixed_day() -> None:
    """Why the wiring guard above starts at two hands: the arithmetic says so.

    The fixture's water set is a 5x5 block minus three cells -- 22 WATER ops
    plus at least eight moves between distinct rows and columns is 30 turns,
    and a hand hired in turn 0 first acts at hour 1 (F040), so it has 23. A
    synthetic `DayFit(pool=1, complete=True)` therefore claims a day no
    scheduler could walk, and the guard above may not assert it.

    This test does not restate that number from memory: it re-derives it from
    the fixture, so if the water set ever becomes walkable by one hand this
    fails and says to put `hands=1` back.
    """
    xs = sorted({x for x, _y in BOARD})
    ys = sorted({y for _x, y in BOARD})
    turns = len(BOARD) + (xs[-1] - xs[0]) + (ys[-1] - ys[0])
    available = 24 - 1                    # a hand hired in turn 0 acts from hour 1

    assert turns > available, (
        f"the fixture's water set now fits one hand ({turns} <= {available} "
        f"turns): put hands=1 back into the parametrize above")
