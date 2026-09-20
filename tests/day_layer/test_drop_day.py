"""The drop: a day that grows something puts it in the shed before the night does.

The engine pours every unit's bag into the shed at the end of the day, so a day without a drop
still banks its harvest - it just cannot sell it, and a SELL reads the shed and never a bag. So
"the goods are in the shed" is not the assertion; "they are there BEFORE the night" is, and that is
what a same-day sale needs.

A harvest needs a crop that is ready, and the engine's own table says the earliest is WHEAT's
`first_yield_day: 2` - so this day is three days long: sow and water on day 0, water on day 1, and
take it off the tile on day 2. `yield_units` is not readiness: it reads 1 the moment a seed is in
the ground, which is not a harvest anyone can take.

The control is the same three days compiled with `drop=False` on the harvest day: the same tile is
harvested, the same unit carries the same wheat, and the shed stays empty all day. Without that
reading the assertion would pass on the night's own drop and prove nothing.

What is asserted is what the engine did, not what the compiler intended:

    the compiled day writes the trip, and names what it banks
    the wheat is in the shed at the drop's own hour, not at hour 23
    without the drop it is still in the bag at hour 23
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

TILE = (2, 2)
AVAILABLE = {"WHEAT": 1}
SEEDS = 1
#: day -> the chain that day runs on the tile. WHEAT's first_yield_day is 2.
DAYS = {0: ("PLANT", "WATER"), 1: ("WATER",), 2: ("HARVEST",)}


def _compiled(ops, drop: bool = True):
    """One day's chain, searched and compiled - the day layer's own path."""
    chains = [(TILE, chain_ops(chain_id_of(ops)), "WHEAT")]
    tasks = T.build(chains, available=AVAILABLE)
    day = B.Day(chains=tuple(chains), available=AVAILABLE, units=((4, 4),), hire_times=())
    result = B.search(day, tasks, beam=64, hands=0, max_hands=0)
    assert result.complete, f"the search could not carry {ops}: {len(result.route)}/{tasks.n}"
    assert not check_route(day, tasks, result), f"the route for {ops} breaks an engine rule"
    return tasks, compile_route(day, tasks, result, drop=drop)


def _replay(drop: bool) -> tuple[list[tuple[int, int]], tuple]:
    """Play the three days against the harness; return the shed per (day, hour) and the arrivals."""
    from offline_lab.kaggle_env import new_environment

    plans, arrivals = {}, ()
    for day_no, ops in DAYS.items():
        tasks, day_ops = _compiled(ops, drop=drop)
        if day_no == 2:
            arrivals = day_ops.arrivals
        plans[day_no] = to_plan(day_ops)

    seen: list[tuple[int, int]] = []

    def agent(obs):
        day_no, hour = int(obs["day"]), int(obs["hour"])
        if day_no not in plans:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        shed = obs["private"].get("shed") or {}
        seen.append((day_no * 24 + hour, int(shed.get("WHEAT", 0))))
        action = dispatch_plan(plans[day_no], obs)
        action["market"] = [["BUY_SEED", "WHEAT", SEEDS]] if day_no == 0 and hour == 0 else []
        return action

    env = new_environment({"episodeSteps": 3 * 24 + 1})
    env.run([agent, "random"])
    return seen, arrivals


@pytest.fixture(scope="module")
def played():
    """The three days compiled twice - with the drop and without it - and both replayed."""
    with_drop = _replay(drop=True)
    without = _replay(drop=False)
    return with_drop, without


def test_the_harvest_day_writes_the_trip_and_names_what_it_banks(played) -> None:
    """A day that harvests carries the goods to a door, and says which hour they land."""
    (_banked, arrivals), _without = played
    assert arrivals, "the harvest day banked nothing, so a sell could reach nothing"
    for hour, item, units in arrivals:
        assert item == "WHEAT" and units >= 1, f"the arrival names {item!r} x{units}"
        assert hour < 23, f"the drop landed at hour {hour}, which is the night's own drop"


def test_the_wheat_is_in_the_shed_at_the_drop_hour(played) -> None:
    """The arrival the compiler published is the arrival the engine made, at that hour."""
    (banked, arrivals), _without = played
    drop_hour = 48 + min(hour for hour, _i, _n in arrivals)
    by_hour = dict(banked)
    assert by_hour[drop_hour + 1] >= 1, (
        f"the drop was written for turn {drop_hour} and the shed held {by_hour[drop_hour + 1]} "
        f"the hour after")


def test_without_the_drop_the_wheat_is_still_in_the_bag_at_hour_23(played) -> None:
    """The control. The same harvest, and nothing reaches the shed before the night.

    This is what makes the test above mean something: if the wheat appeared in the shed either way,
    the assertion would be reading the engine's nightly pour rather than the day's own drop.
    """
    _with_drop, (kept, _arrivals) = played
    by_hour = dict(kept)
    assert by_hour[48 + 23] == 0, (
        f"a day compiled with drop=False still had {by_hour[48 + 23]} wheat in the shed at the "
        f"harvest day's hour 23")
