"""A hand-built day for the day compiler: one coop, two pastures, two wheat tiles.

Only the farmer works - no hires - and the market's orders are the caller's, given in full. So
what is under test is the compiler alone: five chains in, and a day out where every op lands on
the tile it was chosen for, and no chain is left half-run for want of a turn the plan never had.

    (4,4)  BUILD_COOP + PLACE GOOSE + FEED + CARE
    (3,4)  BUILD_PASTURE + PLACE COW
    (2,4)  BUILD_PASTURE + PLACE SHEEP
    (3,2)  PLANT WHEAT + WATER
    (2,2)  PLANT WHEAT + WATER

    market hour 0: BUY_ANIMAL COW, BUY_ANIMAL GOOSE, BUY_ANIMAL SHEEP, BUY_SEED WHEAT x2
"""

from __future__ import annotations

from agent.tile_dp.chains import chain_id_of, chain_ops
from agent.wsr.routing import plan_day

TILES = [
    ((4, 4), ("BUILD_COOP", "PLACE", "FEED", "CARE"), "GOOSE"),
    ((3, 4), ("BUILD_PASTURE", "PLACE"), "COW"),
    ((2, 4), ("BUILD_PASTURE", "PLACE"), "SHEEP"),
    ((3, 2), ("PLANT", "WATER"), "WHEAT"),
    ((2, 2), ("PLANT", "WATER"), "WHEAT"),
]
MARKET = [["BUY_ANIMAL", "COW", 1], ["BUY_ANIMAL", "GOOSE", 1],
          ["BUY_ANIMAL", "SHEEP", 1], ["BUY_SEED", "WHEAT", 1], ["BUY_SEED", "WHEAT", 1]]


def _day():
    """The farmer alone, with the caller's own market orders."""
    tiles = [(cell, chain_ops(chain_id_of(ops)), entity) for cell, ops, entity in TILES]
    return plan_day(tiles, [(4, 4)], new_hands=0, sells=[MARKET],
                    bags=[{}], shed={}, money=3000.0, prices={})


def _worked(plan) -> dict[tuple[int, int], list[str]]:
    """cell -> the ops the unit that works it runs."""
    out: dict[tuple[int, int], list[str]] = {}
    for unit, cell in enumerate(plan.assignments):
        if cell is None:
            continue
        out[cell] = [op[0] for op in plan.units[unit] if op and op[0] != "PASS"]
    return out


def test_the_farmer_is_not_idle_while_chains_are_waiting() -> None:
    """One worker and five chains: the day it can do is not thrown away.

    An all-PASS day is the honest answer to work that does not fit, but five chains on one tile
    each do fit - the farmer walks to a tile and runs its chain.
    """
    plan = _day()
    assert any(op for ops in _worked(plan).values() for op in ops), (
        "the farmer did nothing while five chains were waiting")


def test_a_worked_tile_runs_its_own_chain() -> None:
    """The ops a unit runs belong to the tile the plan says it works.

    A coop's chain builds a coop and places a goose; a wheat chain plants and waters. If the
    assignment and the ops disagree, the plan describes a day nobody planned.
    """
    plan = _day()
    chains = {cell: ops for cell, ops, _ in TILES}
    for cell, ops in _worked(plan).items():
        assert cell in chains, f"the plan works {cell}, which was never given a chain"
        assert set(ops) <= set(chains[cell]), (
            f"the unit on {cell} runs {sorted(ops)} but that tile's chain is {chains[cell]}")


def test_a_chain_is_not_started_and_abandoned() -> None:
    """Every op of a started chain is there, in the order the chain's own rules allow.

    A chain begun and abandoned leaves the tile half-worked: a coop built with no goose in it, or
    a seed planted that is never watered. Those cost money and return nothing, so a plan that
    cannot finish a chain must not start it.
    """
    plan = _day()
    chains = {cell: ops for cell, ops, _ in TILES}
    for cell, ops in _worked(plan).items():
        missing = [op for op in chains[cell] if op not in ops]
        assert not missing, (
            f"the unit on {cell} started its chain but never ran {missing}")


def test_the_plan_does_not_touch_the_market_it_was_given() -> None:
    """The market is the caller's: the day compiler schedules units, not orders.

    Orders already queued for a turn are not re-ordered and not added to - the engine would buy
    twice, and the second buy is refused in silence.
    """
    plan = _day()
    given = [list(order) for order in MARKET]
    for row in plan.market:
        for order in row:
            assert list(order) in given, f"the plan queued {list(order)}, which the caller did not"
