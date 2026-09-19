"""The quadrant day: twenty-five tiles, five hands, and the engine's own verdict.

A regression test for the whole path - chains in, a searched route, ops out, the harness accepting
them - because the bugs this catches were all silent. A PLANT on a tile the worker never reached is
refused without a word, so a plan that looks right and plants nothing is the failure mode, and only
the board can tell the two apart.

The day is the first quadrant planted in rows: strawberry, melon, tomato, and two of wheat, every
tile running PLANT then WATER, with five hands hired and ten of each seed bought on the first turn.

What is asserted is what the engine did, not what the compiler intended:

    the crops that landed are the crops of their row
    the count is the count the search planned for, or better
    no tile is left watered-but-unplanted, which is how a lost planting shows up
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

#: y -> the crop that row is planted with, for the first quadrant's five rows.
ROWS = {0: "STRAWBERRY", 1: "MELON", 2: "TOMATO", 3: "WHEAT", 4: "WHEAT"}
CHAIN = ("PLANT", "WATER")
HANDS = 5
SEEDS = 10

TILES = [((x, y), CHAIN, crop) for y, crop in ROWS.items() for x in range(5)]
AVAILABLE = {crop: 1 for crop in set(ROWS.values())}


def _day():
    """The day as the search sees it: chains, the timetable, and five hands offered at hour one."""
    ops = chain_ops(chain_id_of(CHAIN))
    chains = [(cell, ops, crop) for cell, _ops, crop in TILES]
    tasks = T.build(chains, available=AVAILABLE)
    day = B.Day(chains=tuple(chains), available=AVAILABLE,
                units=((4, 4),), hire_times=(1,) * HANDS)
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

    env = new_environment({"episodeSteps": 22})
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
    result = B.search(day, tasks, beam=64, hands=HANDS)
    assert result.complete, (
        f"the search could not carry the quadrant with {HANDS} hands: "
        f"{len(result.route)} of {tasks.n} tasks")
    assert not check_route(day, tasks, result), "the route breaks a rule the engine enforces"

    plan = to_plan(compile_route(day, tasks, result))
    orders = ([["BUY_SEED", crop, SEEDS] for crop in sorted(set(ROWS.values()))]
              + [["HIRE"]] * HANDS)
    return tasks, result, _board(_replay(plan, orders))


def test_every_tile_the_search_planned_for_gets_its_crop(played) -> None:
    """The plantings that landed, counted - a lost planting is invisible anywhere else.

    The engine refuses a PLANT on a tile the worker is not standing on, in silence, so a day can
    report success and leave the board empty. This is the count that says whether it did.
    """
    _tasks, result, board = played
    planned = sum(1 for _h, task_id, _w in result.route if task_id.endswith("_plant"))
    planted = sum(1 for record in board.values() if record.get("kind") == "PLANT")
    assert planted == planned, (
        f"the search planned {planned} plantings and the board holds {planted}: "
        f"missing {sorted({(x, y) for (x, y), r in board.items() if r.get('kind') != 'PLANT'})}")


def test_a_row_holds_the_crop_it_was_planted_with(played) -> None:
    """No crop on a tile of another row.

    Every planting lands on the tile the worker stands on, so a worker one door from where the
    search thought it was plants its neighbour's tile - and the board comes back with the rows
    shuffled rather than short.
    """
    _tasks, _result, board = played
    wrong = {cell: record.get("crop") for cell, record in board.items()
             if record.get("kind") == "PLANT" and record.get("crop") != ROWS[cell[1]]}
    assert not wrong, f"crops planted in the wrong row: {wrong}"


def test_no_tile_is_watered_without_being_planted(played) -> None:
    """A watering with no crop under it is a planting that was refused.

    WATER follows PLANT on the same tile, so a tile that is not a PLANT after the day has had
    neither - and the weed that grows there is the visible mark of a lost planting.
    """
    _tasks, _result, board = played
    weeds = {cell: record.get("kind") for cell, record in board.items()
             if record.get("kind") not in ("PLANT", None)}
    assert not weeds, f"tiles left in a state the day did not plan: {weeds}"
