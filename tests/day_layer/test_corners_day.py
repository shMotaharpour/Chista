"""The corners day: two pastures at the board's ends, a goose beside each, an empty barn each.

Two cows and two geese bought on the first turn, four wheat to feed them, one hand hired, and the
second quadrant bought so the far end of the board can be used at all. Each corner runs the whole
chain - build, place, feed, care - and keeps an empty pasture next to it.

The search reproduces the hand-solved route: each worker picks up its own three goods at the door,
walks to its corner, and works outward. The hand's route and the search's agree op for op, which is
the point of the test - a route a person can write out and the search can find are the same route.

What is asserted is what the engine did, not what the compiler intended:

    each corner holds the animal it was built for, and the goose stands beside the cow
    the barns that were meant to stay empty are empty, and are still barns
    every animal that was placed was fed and cared for, which the board records on the animal
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.dispatch import dispatch_plan
from agent.tile_dp.chains import chain_id_of, chain_ops
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route, to_plan

COW_CHAIN = ("BUILD_PASTURE", "PLACE", "FEED", "CARE")
GOOSE_CHAIN = ("BUILD_COOP", "PLACE", "FEED", "CARE")
EMPTY_CHAIN = ("BUILD_PASTURE",)

#: The two corners, each a cow, a goose beside it, and an empty barn beside that.
TILES = [
    ((0, 0), COW_CHAIN, "COW"), ((1, 0), GOOSE_CHAIN, "GOOSE"), ((2, 0), EMPTY_CHAIN, None),
    ((9, 0), COW_CHAIN, "COW"), ((8, 0), GOOSE_CHAIN, "GOOSE"), ((7, 0), EMPTY_CHAIN, None),
]
AVAILABLE = {"COW": 1, "GOOSE": 1, "WHEAT": 1}
HANDS = 1
FED = 4

ORDERS = [["BUY_ANIMAL", "COW", 2], ["BUY_ANIMAL", "GOOSE", 2], ["BUY_PRODUCT", "WHEAT", FED],
          ["HIRE"] * HANDS, ["BUY_LAND"]]


def _day():
    """The day as the search sees it: six tiles, three goods, one hand offered at hour one."""
    chains = [(cell, chain_ops(chain_id_of(ops)), entity) for cell, ops, entity in TILES]
    tasks = T.build(chains, available=AVAILABLE)
    day = B.Day(chains=tuple(chains), available=AVAILABLE,
                hire_times=(1,) * HANDS)
    return day, tasks, chains


def _replay(plan, orders):
    """Run the compiled day against the harness and return the tiles it left behind."""
    from offline_lab.kaggle_env import new_environment

    def agent(obs):
        hour, day_no = int(obs["hour"]), int(obs["day"])
        if day_no != 0:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        action = dispatch_plan(plan, obs)
        action["market"] = orders if hour == 0 else []
        return action

    env = new_environment({"episodeSteps": 25})
    env.run([agent, "random"])
    return env.steps[-1][0]["observation"]["farms"][0]["tiles"]


def _board(tiles):
    """Every tile of the two rows the day touches, as `{(x, y): record}`."""
    out = {}
    for x, y in [(x, 0) for x in range(10)]:
        record = tiles[y][x]
        if isinstance(record, dict) and record:
            out[(x, y)] = record
    return out


@pytest.fixture(scope="module")
def played():
    """The searched day, compiled and replayed once - the harness run is the expensive part."""
    day, tasks, chains = _day()
    result = B.search(day, tasks, beam=64, hands=HANDS, max_hands=HANDS)
    assert result.complete, (
        f"the search could not carry the corners with {HANDS + 1} workers: "
        f"{len(result.route)} of {tasks.n} tasks")
    assert not check_route(day, tasks, result), "the route breaks a rule the engine enforces"

    plan = to_plan(compile_route(day, tasks, result))
    return tasks, result, _board(_replay(plan, ORDERS))


def test_each_corner_holds_the_animal_it_was_built_for(played) -> None:
    """The species on each tile, against the tile the day was planned around.

    A PLACE lands on the tile the worker stands on, so a worker one door from where the search
    thought it was puts its animal on the neighbour - and the board comes back with the corners
    swapped rather than empty.
    """
    _tasks, _result, board = played
    for cell, _ops, entity in TILES:
        if entity is None:
            continue
        record = board.get(cell, {})
        assert record.get("animal") == entity, (
            f"{cell} was built for {entity} and holds {record.get('animal')!r}: {record}")


def test_the_barns_meant_to_stay_empty_are_still_barns(played) -> None:
    """An empty pasture is a built pasture with nothing in it - not a tile the day forgot."""
    _tasks, _result, board = played
    for cell, ops, entity in TILES:
        if entity is not None:
            continue
        record = board.get(cell, {})
        assert record, f"{cell} was meant to hold a barn and is bare"
        assert record.get("kind") == "PASTURE", f"{cell} holds {record.get('kind')!r}: {record}"
        assert not record.get("animal"), f"{cell} was meant to stay empty and holds an animal"


def test_every_animal_placed_was_fed_and_cared_for(played) -> None:
    """The board's own counters, which are what say a feeding and a caring actually landed.

    `consecutive_unfed` stays zero only if a FEED reached the tile, and `pending_care_bonus` is one
    only if a CARE did - both read from the day the engine ran, not from the plan. A FEED removed
    from the same plan leaves `consecutive_unfed` at one and a CARE removed leaves the bonus at
    zero, so these two assert the ops the search placed rather than the ops it intended.
    """
    _tasks, _result, board = played
    for cell, _ops, entity in TILES:
        if entity is None:
            continue
        record = board.get(cell, {})
        assert record.get("consecutive_unfed") == 0, (
            f"the animal at {cell} was never fed: {record}")
        assert record.get("pending_care_bonus", 0) >= 1, (
            f"the animal at {cell} was never cared for: {record}")
