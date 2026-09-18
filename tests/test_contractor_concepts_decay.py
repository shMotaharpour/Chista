"""Scenario 3: a decaying milk price, a constant carrot price, and a dose that gets cheap.

  * MILK starts at 200 and loses 10 a day until it reaches 1, then stays there;
  * CARROT is worth the same on every day;
  * every other product is worth nothing, and nothing else costs anything to buy;
  * a dose costs 100 for the first 20 days and 1 after that;
  * a labour hour costs 1 throughout.

What the plan does with that is the test: milk while it is dear (it can only start on the
cow's first yield day, and every day it waits costs 10 a unit), carrots with no timing
pressure at all, and no dose at all until one is cheap.

To watch these fail: flip the milk price to rise instead of fall (the plan waits), or hold the
dose at 100 for the whole season (the late doses disappear).
"""

import numpy as np
import pytest

from agent.artifact import artifact_path
from agent.tile_dp.chains import load_chains
from agent.tile_dp.contractor import HORIZON_DAYS, TileContractor
from agent.tile_dp.graph import TileGraph
from agent.world.model import N_RESOURCE, RESOURCE_ID
from agent.world.prices import price
from offline_lab.build.graph import _new_sim

DAYS = HORIZON_DAYS
LABOR, MILK, CARROT, FERTILIZER = (RESOURCE_ID[k] for k in
                                   ("LABOR", "MILK", "CARROT", "FERTILIZER"))
#: The market starts with ten thousand of every product, so a sale (price at inventory, F033)
#: is the abundant end of the curve: the carrot's constant price.
CARROT_PRICE = float(price("CARROT", _new_sim().observations()[0]["market"]["inventory"]["CARROT"]))
CHEAP_DOSE_FROM = 20


def milk_price(day: int) -> float:
    return max(1.0, 200.0 - 10.0 * day)


def dose_cost(day: int) -> float:
    return 100.0 if day < CHEAP_DOSE_FROM else 1.0


@pytest.fixture(scope="module")
def board():
    graph = TileGraph.load(artifact_path("tile_graph", ".npz"))
    contractor = TileContractor(graph)
    p = np.zeros((DAYS, N_RESOURCE))
    w = np.zeros((DAYS, N_RESOURCE))
    for day in range(DAYS):
        p[day, MILK] = milk_price(day)
        p[day, CARROT] = CARROT_PRICE
        w[day, LABOR] = 1.0
        w[day, RESOURCE_ID["SEED_CARROT"]] = 20.0
        w[day, RESOURCE_ID["SEED_MELON"]] = 80.0
        w[day, FERTILIZER] = dose_cost(day)
    return contractor.price(p, w, [0])


@pytest.fixture(scope="module")
def chains():
    return load_chains()


def days_producing(board, column: int) -> dict[int, int]:
    return {day: int(board.per_day_produce[0, day, column])
            for day in range(board.days) if board.per_day_produce[0, day, column]}


def test_milk_is_taken_while_it_is_dear(board):
    milked = days_producing(board, MILK)

    assert milked, "the plan never milked a cow, and milk is the only thing worth money"
    assert min(milked) == 8, (
        f"the first milk is on day {min(milked)}: a cow's first yield day is 8, and every day "
        f"of waiting costs 10 a unit")
    assert max(milked) < CHEAP_DOSE_FROM, (
        f"the plan milked on day {max(milked)}: the price is down to "
        f"{milk_price(max(milked))} there")


def test_no_dose_is_bought_while_it_costs_a_hundred(board, chains):
    early = [(day, int(v)) for day in range(CHEAP_DOSE_FROM)
             for v in [board.per_day_cost[0, day, FERTILIZER]] if v]

    assert not early, f"a dose was bought at 100 on {early}: it cannot pay for itself there"


def test_the_dose_is_used_once_it_is_cheap(board, chains):
    late = [day for day in range(CHEAP_DOSE_FROM, board.days)
            if board.per_day_cost[0, day, FERTILIZER]]

    assert late, "a dose costs 1 from day 20 on and the plan never used one"


def test_carrot_is_harvested_steadily_when_its_price_does_not_move(board):
    carrots = days_producing(board, CARROT)

    assert len(carrots) >= 4, (
        f"the plan harvested carrots on {sorted(carrots)}: with a price that does not move "
        f"there is no reason to stop")
    assert set(carrots.values()) == {3}, (
        f"carrot harvests were {carrots}: three is what the crop gives without a dose")
