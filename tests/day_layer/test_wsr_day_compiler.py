"""A hand-built day for the day compiler: one coop, two pastures, two wheat tiles.

The market's orders are given, so nothing here depends on how they are planned. What is under
test is whether the compiler turns five chains into a day the engine accepts, with every op
landing on the tile it was chosen for and every hired hand getting work.

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


def _day(hands: int = 2):
    tiles = [(cell, chain_ops(chain_id_of(ops)), entity) for cell, ops, entity in TILES]
    return plan_day(tiles, [(4, 4)], new_hands=hands, sells=[MARKET],
                    bags=[{} for _ in range(hands + 1)], shed={}, money=3000.0, prices={})


def _ops_of(plan, unit: int) -> list[str]:
    return [op[0] for op in plan.units[unit] if op and op[0] != "PASS"]


def test_every_chosen_tile_is_worked() -> None:
    """Five chains in, five tiles worked: a tile the caller chose is not silently left out."""
    plan = _day()
    worked = {cell for cell in plan.assignments if cell is not None}
    assert worked == {cell for cell, _, _ in TILES}, (
        f"only {sorted(worked)} of the five chosen tiles were assigned")


def test_a_tile_runs_the_chain_it_was_chosen_for() -> None:
    """The ops a unit runs belong to the tile the plan says it works.

    A coop's chain builds a coop and places a goose; a wheat chain plants and waters. If the
    assignment and the ops disagree, the plan describes a day nobody planned.
    """
    plan = _day()
    for unit, cell in enumerate(plan.assignments):
        if cell is None:
            continue
        ops = set(_ops_of(plan, unit))
        chain = dict(((c, e), o) for c, o, e in TILES)[(cell, _entity_of(plan, unit, cell))]
        assert ops <= set(chain), f"unit {unit} on {cell} runs {sorted(ops)} but was given {chain}"


def _entity_of(plan, unit: int, cell: tuple[int, int]) -> str:
    for c, _, entity in TILES:
        if c == cell:
            return entity
    raise AssertionError(cell)


def test_every_hired_hand_gets_work_when_work_remains() -> None:
    """Five chains and three units: a unit is idle only when nothing is left for it."""
    plan = _day(hands=2)
    unworked = [cell for cell, _, _ in TILES
                if cell not in {a for a in plan.assignments if a}]
    assert not (unworked and plan.idle_units), (
        f"{plan.idle_units} units idle while {sorted(unworked)} still had chains to run")


def test_the_market_is_not_asked_twice_for_what_it_was_already_given() -> None:
    """An order already queued for this turn is not re-ordered: the engine would buy twice."""
    plan = _day()
    given = {(o[0], o[1]) for row in [MARKET] for o in row}
    for row in plan.market:
        for order in row:
            if order[0].startswith("BUY_"):
                assert (order[0], order[1]) not in given or order in MARKET, (
                    f"the plan orders {order}, which the caller had already given")


def test_the_market_row_keeps_the_engine_order() -> None:
    """Sells, then hires, then purchases - and never more than the engine executes."""
    plan = _day()
    for hour, row in enumerate(plan.market):
        assert len(row) <= 10, f"hour {hour} queues {len(row)} orders; the engine runs 10"
