"""The drop: a day that grows something puts it in the shed before the night does.

The engine pours every unit's bag into the shed at the end of the day, so a day without a drop
still banks its harvest - it just cannot sell it, and a SELL reads the shed and never a bag. So
"the goods are in the shed" is not the assertion; "they are there BEFORE the night" is, and that is
what a same-day sale needs.

A harvest needs a crop that is ready, and the engine's own table says the earliest is WHEAT's
`first_yield_day: 2`. So the day under test is the third one, and the two before it are not what
this is about: they are played ONCE, into a state, and both readings start from a clone of it.
`yield_units` is not readiness - it reads 1 the moment a seed is in the ground - so nothing here
asks a tile for a harvest before its day.

The control is the same harvest day with no deadline on the chain at all: the same tile is
harvested, the same unit carries the same wheat, and the shed stays empty all day. Without that
reading the assertion would pass on the engine's nightly pour and prove nothing.

What is asserted is what the engine did, not what the compiler intended:

    the day writes the trip, and names what it banks
    the wheat is in the shed at the drop's own hour, not at hour 23
    with no deadline it is still in the bag at hour 23
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
#: The latest hour the harvest day must have its wheat in the shed - before the night's own pour,
#: so an arrival there is the day's drop and not the engine's end-of-day one.
DROP_BY = 20
#: day -> the chain that day runs on the tile. WHEAT's first_yield_day is 2.
PREFIX = {0: ("PLANT", "WATER"), 1: ("WATER",)}
HARVEST_DAY = 2
TURNS = 24


def _compiled(ops, banked: bool = True):
    """One day's chain, searched and compiled - the day layer's own path.

    `banked` is the whole of the drop's input: a deadline on the harvest, or nothing at all. The
    chain is the same either way, because whether a harvest is dropped is not the chain's business.
    """
    chains = [(TILE, chain_ops(chain_id_of(ops)), "WHEAT")]
    tasks = T.build(chains, available=AVAILABLE, drop_by=[DROP_BY if banked else None])
    day = B.Day(chains=tuple(chains), available=AVAILABLE, hire_times=())
    result = B.search(day, tasks, beam=64, hands=0, max_hands=0)
    assert result.complete, f"the search could not carry {ops}: {len(result.route)}/{tasks.n}"
    assert not check_route(day, tasks, result), f"the route for {ops} breaks an engine rule"
    return tasks, compile_route(day, tasks, result)


def _pass() -> dict:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _step(sim, action, market=()) -> None:
    """One turn of the fast simulator, with the other seat passing.

    FastSim wraps the same interpreter the harness drives (R003), so a board read from it is the
    engine's own verdict - without the harness's schema validation, which is what makes it cheap
    enough to spend two days reaching a state.
    """
    action = dict(action)
    action["market"] = list(market)
    sim.step([action, _pass()])


@pytest.fixture(scope="module")
def at_harvest_day():
    """A simulator standing at the harvest day's first turn.

    The prefix is played once and cloned, not replayed per reading: it is there to leave a planted,
    watered crop behind, and two days of it are the same two days whatever the day under test does
    with the result.
    """
    from offline_lab.fast_sim import FastSim

    sim = FastSim({"episodeSteps": (HARVEST_DAY + 1) * TURNS + 1})
    sim.reset()
    for day_no, ops in PREFIX.items():
        tasks, day_ops = _compiled(ops, banked=False)   # the prefix is not the drop's business
        plan = to_plan(day_ops)
        while not sim.done:
            obs = sim.observations()[0]
            if int(obs["day"]) != day_no:
                break
            hour = int(obs["hour"])
            orders = [["BUY_SEED", "WHEAT", SEEDS]] if day_no == 0 and hour == 0 else []
            _step(sim, dispatch_plan(plan, obs), orders)
    return sim


def _replay(sim, banked: bool) -> tuple[list[tuple[int, int]], tuple]:
    """The harvest day, played from the state the prefix left; the shed per turn, and the arrivals."""
    from offline_lab.fast_sim import FastSim

    tasks, day_ops = _compiled(("HARVEST",), banked=banked)
    plan = to_plan(day_ops)

    sim = sim.clone()
    seen: list[tuple[int, int]] = []
    while not sim.done:
        obs = sim.observations()[0]
        if int(obs["day"]) != HARVEST_DAY:
            break
        shed = obs["private"].get("shed") or {}
        seen.append((int(obs["hour"]), int(shed.get("WHEAT", 0))))
        _step(sim, dispatch_plan(plan, obs))
    return seen, day_ops.arrivals


@pytest.fixture(scope="module")
def played(at_harvest_day):
    """The harvest day compiled twice - with the deadline and without it - and both played."""
    return _replay(at_harvest_day, banked=True), _replay(at_harvest_day, banked=False)


def test_the_harvest_day_writes_the_trip_and_names_what_it_banks(played) -> None:
    """A day that harvests carries the goods to a door, and says which hour they land."""
    (_banked, arrivals), _without = played
    assert arrivals, "the harvest day banked nothing, so a sell could reach nothing"
    for hour, item, units in arrivals:
        assert item == "WHEAT" and units >= 1, f"the arrival names {item!r} x{units}"
        assert hour <= DROP_BY, f"the drop landed at hour {hour}, past its deadline {DROP_BY}"


def test_the_wheat_is_in_the_shed_at_the_drop_hour(played) -> None:
    """The arrival the compiler published is the arrival the engine made, at that hour."""
    (banked, arrivals), _without = played
    drop_hour = min(hour for hour, _i, _n in arrivals)
    by_hour = dict(banked)
    assert by_hour[drop_hour + 1] >= 1, (
        f"the drop was written for hour {drop_hour} and the shed held {by_hour[drop_hour + 1]} "
        f"the hour after: {banked}")


def test_without_a_deadline_the_wheat_is_still_in_the_bag_at_hour_23(played) -> None:
    """The control. The same harvest, and nothing reaches the shed before the night.

    This is what makes the test above mean something: if the wheat appeared in the shed either way,
    the assertion would be reading the engine's nightly pour rather than the day's own drop.
    """
    _with_drop, (kept, _arrivals) = played
    by_hour = dict(kept)
    assert by_hour[TURNS - 1] == 0, (
        f"a day with no deadline still had {by_hour[TURNS - 1]} wheat in the shed at hour "
        f"{TURNS - 1}")
