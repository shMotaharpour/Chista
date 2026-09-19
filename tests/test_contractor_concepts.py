"""Conceptual tests: does the DP put its work where the value is?

The cost side is held constant in shape (day one's prices) and only the value moves, so what
the plan does is the DP's own decision rather than an input:

  1. nothing is worth anything except CARROT on days 10 and 25, where it is worth 30;
  2. the same with MELON worth 30 on day 20, and a labour hour costing 1 on even days and 10
     on odd ones.

Two of them come in pairs, because the same setting answers differently depending on whether
the dose is free: a carrot harvest reaches the crop's maximum of 4 with a free fertilizer, and
stops at 3 when a dose costs 100 and adds one unit worth 30.

To watch them fail: give CARROT a price on a third day (the harvest follows it), make the odd
days cheap instead of dear (the plan starts working on odd days), or make the dose free again
(the harvest grows back to 4).
"""

import numpy as np
import pytest

from agent.artifact import artifact_path
from agent.tile_dp.chains import load_chains
from agent.tile_dp.contractor import HORIZON_DAYS, TileContractor
from agent.tile_dp.graph import TileGraph
from agent.world.model import N_RESOURCE, RESOURCE_ID

DAYS = HORIZON_DAYS
LABOR = RESOURCE_ID["LABOR"]

#: What day one charges, constant across the season: a labour hour (the DP's own wage), the
#: seed shop, and the market's buy quote for a fertilizer at its starting inventory.
DAY_ONE_COSTS = {"LABOR": 1.0, "SEED_CARROT": 20.0, "SEED_MELON": 80.0, "FERTILIZER": 100.0}
FREE_DOSE = {**DAY_ONE_COSTS, "FERTILIZER": 0.0}


@pytest.fixture(scope="module")
def contractor() -> TileContractor:
    return TileContractor(TileGraph.load(artifact_path("tile_graph", ".npz")))


@pytest.fixture(scope="module")
def chains():
    return load_chains()


def duals(price_days, wage_of_day, costs=DAY_ONE_COSTS):
    """`p` (what a unit of a product is worth) and `w` (what a unit costs) per day."""
    p = np.zeros((DAYS, N_RESOURCE), dtype=np.float64)
    w = np.zeros((DAYS, N_RESOURCE), dtype=np.float64)
    for day in range(DAYS):
        for item, value in price_days.get(day, {}).items():
            p[day, RESOURCE_ID[item]] = value
        for item, value in costs.items():
            w[day, RESOURCE_ID[item]] = value
        w[day, LABOR] = wage_of_day(day)
    return p, w


def one_bare_tile() -> list[int]:
    """The graph's root: the bare tile every plan starts from."""
    return [0]


def produced_per_day(board, product: str, tile: int = 0) -> dict[int, int]:
    column = RESOURCE_ID[product]
    return {day: int(board.per_day_produce[tile, day, column])
            for day in range(board.days) if board.per_day_produce[tile, day, column]}


def worked_days(board, tile: int = 0) -> list[int]:
    return [day for day in range(board.days) if board.per_day_cost[tile, day, LABOR]]


CARROT_DAYS = {10: {"CARROT": 30.0}, 25: {"CARROT": 30.0}}
MELON_DAY = {20: {"MELON": 30.0}}


def test_carrot_is_harvested_on_the_two_days_it_is_worth_something(contractor):
    p, w = duals(CARROT_DAYS, lambda day: 1.0)
    board = contractor.price(p, w, one_bare_tile())
    taken = produced_per_day(board, "CARROT")

    assert set(taken) == {10, 25}, (
        f"carrots came off the tile on {sorted(taken)}, not on the two days they are worth "
        f"30: a unit harvested on any other day earns nothing")


def test_a_carrot_harvest_is_the_maximum_when_the_dose_is_free(contractor):
    p, w = duals(CARROT_DAYS, lambda day: 1.0, costs=FREE_DOSE)
    board = contractor.price(p, w, one_bare_tile())

    assert produced_per_day(board, "CARROT") == {10: 4, 25: 4}, (
        "with a free dose the carrot reaches 4; a smaller harvest wastes a day that had a "
        "price")


def test_a_carrot_harvest_stops_at_three_when_the_dose_costs_more_than_it_adds(contractor):
    p, w = duals(CARROT_DAYS, lambda day: 1.0)
    board = contractor.price(p, w, one_bare_tile())

    assert produced_per_day(board, "CARROT") == {10: 3, 25: 3}, (
        "a dose costs 100 and adds one unit worth 30, so the plan must not pay for one")


def test_nothing_else_is_produced_when_nothing_else_has_a_price(contractor):
    p, w = duals(CARROT_DAYS, lambda day: 1.0)
    board = contractor.price(p, w, one_bare_tile())
    stray = [(day, item) for day in range(board.days) for item in range(N_RESOURCE)
             if board.per_day_produce[0, day, item] and item != RESOURCE_ID["CARROT"]]

    assert not stray, f"the plan produced something with no price: {stray}"


def test_melon_is_harvested_on_the_day_it_is_worth_something(contractor):
    p, w = duals(MELON_DAY, lambda day: 1.0 if day % 2 == 0 else 10.0)
    board = contractor.price(p, w, one_bare_tile())

    assert produced_per_day(board, "MELON") == {20: 6}, (
        "the melon's day is 20 and its maximum is 6, and it reaches 6 without a dose")


def test_a_dear_hour_is_worked_only_when_it_pays(contractor):
    p, w = duals(MELON_DAY, lambda day: 1.0 if day % 2 == 0 else 10.0)
    board = contractor.price(p, w, one_bare_tile())
    dear = [day for day in worked_days(board) if day % 2]

    # An hour on a dear day costs 10 and can only be paying for the unit it adds to the day-20
    # harvest (worth 30), so it may happen on the eve of the harvest and nowhere else.
    assert all(day == 19 for day in dear), (
        f"the plan worked on dear days {dear}: an hour costs 10 there and only the unit it "
        f"adds to the day-20 harvest pays for it")
