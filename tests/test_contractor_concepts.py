"""Conceptual tests: does the DP put its work where the value is?

Both settings keep the cost side constant in shape and vary only where the value is, so what
the recovered plan does is the DP's own decision, not an input.

  1. Nothing is worth anything except CARROT on days 10 and 25, where it is worth 30. The
     plan must harvest carrots on exactly those two days, and each harvest must be the
     carrot's maximum (4 - the fertilised one), because a unit left in the field on a day
     with no price is worth nothing.
  2. The same, with MELON worth 30 on day 20, and a labour hour costing 1 on even days and
     10 on odd ones. A melon needs ten days, so the planting has to happen on day 8 or
     earlier, and every hour the plan spends must be spent on a cheap day.

To watch either fail: give CARROT a price on a third day (test 1 stops harvesting exactly
there), or make the odd days cheap instead of dear (test 2 starts working on odd days).
"""

import numpy as np
import pytest

from agent.artifact import artifact_path
from agent.tile_dp.chains import load_chains, entity_of_code
from agent.tile_dp.contractor import HORIZON_DAYS, TileContractor
from agent.tile_dp.graph import TileGraph
from agent.world.model import N_RESOURCE, RESOURCE_ID

DAYS = HORIZON_DAYS
LABOR = RESOURCE_ID["LABOR"]


@pytest.fixture(scope="module")
def contractor() -> TileContractor:
    return TileContractor(TileGraph.load(artifact_path("tile_graph", ".npz")))


@pytest.fixture(scope="module")
def chains():
    return load_chains()


def duals(price_days, wage_of_day):
    """`p` (what a unit of a product is worth) and `w` (what a labour hour costs) per day."""
    p = np.zeros((DAYS, N_RESOURCE), dtype=np.float64)
    w = np.zeros((DAYS, N_RESOURCE), dtype=np.float64)
    for day in range(DAYS):
        for item, value in price_days.get(day, {}).items():
            p[day, RESOURCE_ID[item]] = value
        w[day, LABOR] = wage_of_day(day)
    return p, w


def one_bare_tile(graph: TileGraph) -> list[int]:
    """The graph's root: the bare tile every plan starts from."""
    return [0]


def harvests(plan, chains, product: str) -> dict[int, int]:
    """day -> units of `product` the plan takes off the tile that day."""
    out: dict[int, int] = {}
    for day, _state, chain_id in plan:
        ops = chains[chain_id]
        if "HARVEST" in ops:
            out[day] = out.get(day, 0)
    return out


def produced_per_day(board, chains, tile: int, product: str) -> dict[int, int]:
    column = RESOURCE_ID[product]
    return {day: int(board.per_day_produce[tile, day, column])
            for day in range(board.days)
            if board.per_day_produce[tile, day, column]}


def worked_days(board, tile: int) -> list[int]:
    return [day for day in range(board.days) if board.per_day_cost[tile, day, LABOR]]


def test_carrot_is_harvested_on_the_two_days_it_is_worth_something(contractor, chains):
    p, w = duals({10: {"CARROT": 30.0}, 25: {"CARROT": 30.0}}, lambda day: 1.0)
    board = contractor.price(p, w, one_bare_tile(contractor.graph))
    taken = produced_per_day(board, chains, 0, "CARROT")

    assert set(taken) == {10, 25}, (
        f"carrots came off the tile on {sorted(taken)}, not on the two days they are worth "
        f"30: a unit harvested on any other day earns nothing")
    assert taken == {10: 4, 25: 4}, (
        f"harvests were {taken}: the carrot's maximum is 4, and a smaller one wastes a day "
        f"that had a price")


def test_nothing_else_is_produced_when_nothing_else_has_a_price(contractor, chains):
    p, w = duals({10: {"CARROT": 30.0}, 25: {"CARROT": 30.0}}, lambda day: 1.0)
    board = contractor.price(p, w, one_bare_tile(contractor.graph))
    produced = {RESOURCE_ID["CARROT"]}
    stray = [(day, item) for day in range(board.days)
             for item in range(N_RESOURCE)
             if board.per_day_produce[0, day, item] and item not in produced]
    assert not stray, f"the plan produced something with no price: {stray}"


def test_melon_is_harvested_on_the_day_it_is_worth_something(contractor, chains):
    p, w = duals({20: {"MELON": 30.0}}, lambda day: 1.0 if day % 2 == 0 else 10.0)
    board = contractor.price(p, w, one_bare_tile(contractor.graph))
    taken = produced_per_day(board, chains, 0, "MELON")

    assert set(taken) == {20}, f"melons came off the tile on {sorted(taken)}, not on day 20"
    assert taken[20] == 6, f"the melon harvest was {taken[20]}, its maximum is 6"


def test_every_worked_day_is_a_cheap_day(contractor, chains):
    p, w = duals({20: {"MELON": 30.0}}, lambda day: 1.0 if day % 2 == 0 else 10.0)
    board = contractor.price(p, w, one_bare_tile(contractor.graph))
    worked = worked_days(board, 0)
    dear = [day for day in worked if day % 2]

    assert not dear, (
        f"the plan worked on {dear}: an hour costs 10 there and 1 on the even days, and the "
        f"only deadline is day 20")
