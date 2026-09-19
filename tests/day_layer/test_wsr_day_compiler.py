"""A hand-built day for the day compiler: chains and an offer in, routes out.

No hires are on offer, so the farmer is the only worker. The market's orders are the caller's -
two wheat seeds and two wheat - and what the compiler is told is when those goods reach the shed:
everything at hour one. A fetch cannot happen before that, and an op whose good never arrives is
reported rather than guessed at.

    (4,4)  BUILD_COOP + PLACE GOOSE + FEED + CARE
    (3,4)  BUILD_PASTURE + PLACE COW
    (2,4)  BUILD_PASTURE + PLACE SHEEP + FEED + CARE
    (3,2)  PLANT WHEAT + WATER
    (2,2)  PLANT WHEAT + WATER

    the caller's market: BUY_SEED WHEAT x2, BUY_PRODUCT WHEAT x2
    the offer:           nothing to hire, COW / SHEEP / GOOSE / WHEAT in the shed from hour 1
"""

from __future__ import annotations

from agent.tile_dp.chains import chain_id_of, chain_ops
from agent.wsr.routing import Offer, compile_day

TILES = [
    ((4, 4), ("BUILD_COOP", "PLACE", "FEED", "CARE"), "GOOSE"),
    ((3, 4), ("BUILD_PASTURE", "PLACE"), "COW"),
    ((2, 4), ("BUILD_PASTURE", "PLACE", "FEED", "CARE"), "SHEEP"),
    ((3, 2), ("PLANT", "WATER"), "WHEAT"),
    ((2, 2), ("PLANT", "WATER"), "WHEAT"),
]
MARKET = [["BUY_SEED", "WHEAT", 1], ["BUY_SEED", "WHEAT", 1],
          ["BUY_PRODUCT", "WHEAT", 1], ["BUY_PRODUCT", "WHEAT", 1]]
OFFER = Offer(hire_times=(), available={"COW": 1, "SHEEP": 1, "GOOSE": 1, "WHEAT": 1})


def _day():
    """The farmer alone, with the caller's offer and the caller's market."""
    tiles = [(cell, chain_ops(chain_id_of(ops)), entity) for cell, ops, entity in TILES]
    return compile_day(tiles, [(4, 4)], offer=OFFER, hands=0, bags=[{}])


def _ops_of(plan, unit: int) -> list[str]:
    return [op[0] for op in plan.units[unit] if op and op[0] != "PASS"]


def test_the_farmer_is_not_idle_while_chains_are_waiting() -> None:
    """One worker and five chains: the day it can do is not thrown away.

    An all-PASS day is the honest answer to work that does not fit, but a chain on one tile does
    fit - the farmer walks to a tile and runs it.
    """
    plan = _day()
    assert plan.worked > 0, "the farmer did nothing while five chains were waiting"


def test_a_worked_tile_runs_its_own_chain() -> None:
    """The ops a unit runs belong to the tile the plan assigns it.

    A coop's chain builds a coop and places a goose; a wheat chain plants and waters. If the
    assignment and the ops disagree, the plan describes a day nobody planned.
    """
    plan = _day()
    chains = {cell: set(ops) for cell, ops, _ in TILES}
    for unit, cell in enumerate(plan.assignments):
        if cell is None:
            continue
        assert cell in chains, f"the plan works {cell}, which was never given a chain"
        ops = set(_ops_of(plan, unit))
        assert ops <= chains[cell], (
            f"the unit on {cell} runs {sorted(ops)} but that tile's chain is {sorted(chains[cell])}")


def test_a_chain_is_not_started_and_abandoned() -> None:
    """Every op of a started chain is there.

    A chain begun and abandoned leaves the tile half-worked: a coop built with no goose in it, or
    a pasture built and never used. Those cost money and return nothing, so a plan that cannot
    finish a chain must not start it.
    """
    plan = _day()
    chains = {cell: list(ops) for cell, ops, _ in TILES}
    for unit, cell in enumerate(plan.assignments):
        if cell is None:
            continue
        ops = _ops_of(plan, unit)
        missing = [op for op in chains[cell] if op not in ops]
        assert not missing, f"the unit on {cell} started its chain but never ran {missing}"


def test_a_fetch_never_precedes_the_good_reaching_the_shed() -> None:
    """The offer's timetable is the floor: nothing is picked up before it is there.

    The engine refuses a PICKUP of an empty shelf in silence, so a fetch written before its good
    arrives is a plan that lies about its day.
    """
    plan = _day()
    for route in plan.routes:
        for hour, op in enumerate(route.ops):
            if op and op[0] == "PICKUP":
                item = op[1]
                assert hour >= OFFER.available[item], (
                    f"unit {route.unit} picks up {item} at hour {hour}, "
                    f"before the offer says it is in the shed")


def test_nothing_is_fetched_that_the_offer_never_promised() -> None:
    """A good the planner never said arrives is reported, not assumed."""
    plan = _day()
    fetched = {op[1] for route in plan.routes for op in route.ops if op and op[0] == "PICKUP"}
    assert fetched <= set(OFFER.available), f"the day fetched {sorted(fetched - set(OFFER.available))}"
