"""The mixed day: three species on the shed's column, wheat across the rest of the quadrant.

Bought on the first turn - a cow, a sheep, a goose, eight wheat to feed them and twenty-five wheat
seeds - with five hands offered, all of them from hour one. Every item is in the shed from hour one
too, so nothing in this day waits on the market: what the search places is the day's own choice.

The three animal tiles are the column beside the shed's north-west door: the cow on the door itself
at (4, 4), the sheep and the goose one and two tiles north of it. The cow runs the shortest pasture
chain the registry has, `BUILD_PASTURE+PLACE`; the sheep and the goose run the whole chain,
`BUILD_PASTURE+PLACE+FEED+CARE` and `BUILD_COOP+PLACE+FEED+CARE`. The rest of the quadrant - the
twenty-two tiles those three leave - is wheat, `PLANT+WATER`.

What is asserted is what the engine did, not what the compiler intended:

    each animal tile holds the species it was built for, in the structure that species lives in
    the sheep and the goose were fed and cared for, and the cow - whose chain stops at PLACE - was
    not, which is how a day that invents ops is told apart from one that ran the chains it was given
    every planting the search planned landed, and every planted tile holds the crop it was planted with
    the quadrant is full: twenty-two plants and three structures, and no tile the day forgot
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.dispatch import dispatch_plan
from agent.tile_dp.chains import chain_id_of, chain_ops
from agent.world.model import CROPS, SHED_ITEMS
from agent.world.rules import ANIMAL_STRUCTURE
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route, to_plan

COW_CHAIN = ("BUILD_PASTURE", "PLACE")
SHEEP_CHAIN = ("BUILD_PASTURE", "PLACE", "FEED", "CARE")
GOOSE_CHAIN = ("BUILD_COOP", "PLACE", "FEED", "CARE")
WHEAT_CHAIN = ("PLANT", "WATER")

#: The three animal tiles, on the column beside the shed's north-west door.
ANIMAL_TILES = [
    ((4, 4), COW_CHAIN, "COW"),
    ((4, 3), SHEEP_CHAIN, "SHEEP"),
    ((4, 2), GOOSE_CHAIN, "GOOSE"),
]
#: The rest of the first quadrant - the land the farm holds on day zero - planted with wheat.
WHEAT_TILES = [((x, y), WHEAT_CHAIN, "WHEAT")
               for y in range(5) for x in range(5)
               if (x, y) not in {cell for cell, _ops, _entity in ANIMAL_TILES}]
TILES = ANIMAL_TILES + WHEAT_TILES

#: Everything the shed can hold, and every crop's seed, is there from hour one.
AVAILABLE = {name: 1 for name in SHED_ITEMS + CROPS}
HANDS = 5
WHEAT_BOUGHT = 8
SEEDS_BOUGHT = 25
WHEAT_SEED = "WHEAT"

#: One turn's market: three animals, their feed, the seeds for the quadrant, and five hands - ten
#: orders, which is the engine's own cap per turn (F031).
ORDERS = ([["BUY_ANIMAL", "COW", 1], ["BUY_ANIMAL", "SHEEP", 1], ["BUY_ANIMAL", "GOOSE", 1],
           ["BUY_PRODUCT", "WHEAT", WHEAT_BOUGHT], ["BUY_SEED", WHEAT_SEED, SEEDS_BOUGHT]]
          + [["HIRE"]] * HANDS)


def _day():
    """The day as the search sees it: twenty-five tiles, five hands offered at hour one."""
    chains = [(cell, chain_ops(chain_id_of(ops)), entity) for cell, ops, entity in TILES]
    tasks = T.build(chains, available=AVAILABLE)
    day = B.Day(chains=tuple(chains), available=AVAILABLE, hire_times=(1,) * HANDS)
    return day, tasks, chains


def _replay(plan, orders):
    """Run the compiled day against the harness and return the board it left behind."""
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
    """The quadrant as `{(x, y): tile record}`, for tiles that hold something."""
    out = {}
    for y in range(5):
        for x in range(5):
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
        f"the search could not carry the day with {HANDS + 1} workers: "
        f"{len(result.route)} of {tasks.n} tasks")
    assert not check_route(day, tasks, result), "the route breaks a rule the engine enforces"

    plan = to_plan(compile_route(day, tasks, result))
    return tasks, result, _board(_replay(plan, ORDERS))


def test_each_animal_tile_holds_the_species_it_was_built_for(played) -> None:
    """The species on the tile the day was planned around, and the structure it lives in.

    A PLACE lands on the tile the worker stands on, so a worker one door from where the search
    thought it was puts its animal on the neighbour - and the board comes back with the column
    shuffled rather than empty. The structure is asserted beside it because a goose on a pasture
    (or a cow on a coop) is a build the day got wrong while the placement still looks right.
    """
    _tasks, _result, board = played
    for cell, _ops, entity in ANIMAL_TILES:
        record = board.get(cell, {})
        assert record.get("animal") == entity, (
            f"{cell} was built for {entity} and holds {record.get('animal')!r}: {record}")
        assert record.get("kind") == ANIMAL_STRUCTURE[entity], (
            f"{cell} holds a {entity} in {record.get('kind')!r}, "
            f"which is not {ANIMAL_STRUCTURE[entity]}: {record}")


def test_the_chains_that_feed_and_care_did(played) -> None:
    """The board's own counters for the tiles whose chains carry FEED and CARE.

    `consecutive_unfed` stays zero only if a FEED reached the tile, and `pending_care_bonus` is one
    only if a CARE did - both read from the day the engine ran, not from the plan.
    """
    _tasks, _result, board = played
    for cell, ops, _entity in ANIMAL_TILES:
        record = board.get(cell, {})
        if "FEED" in ops:
            assert record.get("consecutive_unfed") == 0, (
                f"the animal at {cell} was never fed: {record}")
        if "CARE" in ops:
            assert record.get("pending_care_bonus", 0) >= 1, (
                f"the animal at {cell} was never cared for: {record}")


def test_the_cow_whose_chain_stops_at_place_was_not_fed(played) -> None:
    """The tile the chain leaves alone, asserted as hard as the ones it does not.

    The cow's chain is `BUILD_PASTURE+PLACE`, so a day that fed or cared for it ran ops nobody
    asked for - and a feeding that never happened is invisible unless the untouched animal is read
    back. A newly placed animal starts at `consecutive_unfed = 0`, and the end of the day counts the
    missed feeding once (F017).
    """
    _tasks, _result, board = played
    cow = next(cell for cell, _ops, entity in ANIMAL_TILES if entity == "COW")
    record = board.get(cow, {})
    assert record.get("consecutive_unfed") == 1, (
        f"the cow at {cow} was fed, and its chain never asks for it: {record}")
    assert not record.get("pending_care_bonus", 0), (
        f"the cow at {cow} was cared for, and its chain never asks for it: {record}")


def test_every_wheat_tile_the_search_planned_for_is_planted(played) -> None:
    """The plantings that landed, counted - a lost planting is invisible anywhere else.

    The engine refuses a PLANT on a tile the worker is not standing on, in silence, so a day can
    report success and leave the quadrant bare. This is the count that says whether it did.
    """
    _tasks, result, board = played
    planned = sum(1 for _hour, task_id, _worker in result.route if task_id.endswith("_plant"))
    planted = sum(1 for record in board.values() if record.get("kind") == "PLANT")
    bare = sorted({cell for cell, _ops, _entity in WHEAT_TILES}
                  - {cell for cell, record in board.items() if record.get("kind") == "PLANT"})
    assert planted == planned, (
        f"the search planned {planned} plantings and the board holds {planted}: bare {bare}")


def test_every_planted_tile_holds_the_crop_it_was_planted_with(played) -> None:
    """No crop but wheat, on the quadrant the day was planned over.

    Every planting lands on the tile the worker stands on, so a worker one door from where the
    search thought it was plants its neighbour's tile - and the board comes back with the quadrant
    shuffled rather than short.
    """
    _tasks, _result, board = played
    wrong = {cell: record.get("crop") for cell, record in board.items()
             if record.get("kind") == "PLANT" and record.get("crop") != WHEAT_SEED}
    assert not wrong, f"tiles planted with something other than wheat: {wrong}"


def test_the_quadrant_is_full_and_nothing_is_left_in_a_state_the_day_did_not_plan(played) -> None:
    """Twenty-two plants and three structures over the quadrant the farm holds, and nothing else.

    A watered tile with no crop under it is a planting that was refused (the quadrant test's mark),
    and an empty unlocked tile is where a weed can appear at the day's refresh - so a full quadrant
    is the state that says the day used the land it was given.
    """
    _tasks, _result, board = played
    kinds = {cell: record.get("kind") for cell, record in board.items()}
    plants = [cell for cell, kind in kinds.items() if kind == "PLANT"]
    structures = [cell for cell, kind in kinds.items() if kind in ANIMAL_STRUCTURE.values()]
    assert len(plants) == len(WHEAT_TILES), (
        f"the day planned {len(WHEAT_TILES)} wheat tiles and the board holds {len(plants)}: {kinds}")
    assert len(structures) == len(ANIMAL_TILES), (
        f"the day planned {len(ANIMAL_TILES)} animal tiles and the board holds "
        f"{len(structures)}: {kinds}")
    assert len(board) == len(TILES), (
        f"the quadrant holds {len(TILES)} tiles and the board came back with {len(board)}: {kinds}")
    unplanned = {cell: kind for cell, kind in kinds.items() if kind not in ("PLANT", *ANIMAL_STRUCTURE.values())}
    assert not unplanned, f"tiles left in a state the day did not plan: {unplanned}"
