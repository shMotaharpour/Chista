"""The carrot cycle, as tests: two days, three carrots, and the dose goes to the next plant.

Measured on the engine by the owner, and produced by the plan: plant and water on day 19, an
idle day 20, then from day 21 on every other day - water, harvest 3, plant again, dose, water.
The dose of a harvest day belongs to the plant that is put in the ground right after the
harvest, which is what makes the next harvest worth 3 as well.

Three things are pinned here: the cadence, the count, and that the plan's own states chain up
(each day's state must be the successor its chain reaches, or the plan describes a walk the
graph does not contain).

To watch these fail: price a dose at 100 for the whole season (the cadence breaks), or make
the carrot's price decay too (the plan stops harvesting on the cheap days).
"""

import numpy as np
import pytest

from agent.artifact import artifact_path
from agent.tile_dp.chains import load_chains, chain_name
from agent.tile_dp.contractor import HORIZON_DAYS, TileContractor
from agent.tile_dp.graph import TileGraph
from agent.world.model import N_RESOURCE, RESOURCE_ID
from agent.world.prices import price
from offline_lab.build.graph import _new_sim

DAYS = HORIZON_DAYS
LABOR, MILK, CARROT, FERTILIZER = (RESOURCE_ID[k] for k in
                                   ("LABOR", "MILK", "CARROT", "FERTILIZER"))
CARROT_PRICE = float(price("CARROT",
                           _new_sim().observations()[0]["market"]["inventory"]["CARROT"]))


@pytest.fixture(scope="module")
def scenario():
    """Scenario 3: decaying milk, a constant carrot, and a dose that gets cheap at day 20."""
    graph = TileGraph.load(artifact_path("tile_graph", ".npz"))
    p = np.zeros((DAYS, N_RESOURCE))
    w = np.zeros((DAYS, N_RESOURCE))
    for day in range(DAYS):
        p[day, MILK] = max(1.0, 200.0 - 10.0 * day)
        p[day, CARROT] = CARROT_PRICE
        w[day, LABOR] = 1.0
        w[day, RESOURCE_ID["SEED_CARROT"]] = 20.0
        w[day, FERTILIZER] = 100.0 if day < 20 else 1.0
    board = TileContractor(graph).price(p, w, [0])
    return graph, board, load_chains()


def carrot_days(board) -> dict[int, int]:
    return {day: int(board.per_day_produce[0, day, CARROT])
            for day in range(board.days) if board.per_day_produce[0, day, CARROT]}


def test_the_carrot_cycle_is_two_days_and_three_carrots(scenario):
    _graph, board, _chains = scenario
    days = sorted(carrot_days(board))

    assert days, "the plan never harvested a carrot, and carrots are worth 35 all season"
    assert all(b - a == 2 for a, b in zip(days, days[1:])), (
        f"carrot harvests fell on {days}: a two-day cycle is what the plan settles on")
    assert set(carrot_days(board).values()) == {3}, (
        f"harvests were {carrot_days(board)}: the cycle gives three, not four, and three is "
        f"what it gives by watering the plant on its harvest day")


def test_the_dose_of_a_harvest_day_belongs_to_the_next_plant(scenario):
    _graph, board, chains = scenario
    doses = [day for day in range(board.days) if board.per_day_cost[0, day, FERTILIZER]]
    carrot_harvests = set(carrot_days(board))

    assert doses, "the plan never used a dose, though one costs 1 from day 20 on"
    replanted = 0
    for day in sorted(carrot_harvests):
        _d, _state, chain_id = board.plans[0][day]
        ops = chains[chain_id]
        assert "HARVEST" in ops, f"day {day} was counted as a harvest but its chain has none"
        after = ops[ops.index("HARVEST") + 1:]
        if "PLANT" not in after:
            continue          # the season's last day: nothing to dose for
        replanted += 1
        assert "FERTILIZE" in after, (
            f"day {day}: {chain_name(ops)} doses before the harvest, so the dose would be "
            f"spent on the crop being taken off the tile")

    assert replanted >= 4, (
        f"only {replanted} harvests were followed by a replanting: the cycle is what keeps the "
        f"tile earning")


def test_the_plan_chains_up(scenario):
    graph, board, chains = scenario
    bad = []
    plan = board.plans[0]
    for day in range(len(plan) - 1):
        _d, state_id, chain_id = plan[day]
        nxt = plan[day + 1][1]
        reachable = any(int(graph.edge_chain[e]) == chain_id and int(graph.edge_next[e]) == nxt
                        for e in range(int(graph.edge_offsets[state_id]),
                                       int(graph.edge_offsets[state_id + 1])))
        if not reachable:
            bad.append((day, state_id, chain_name(chains[chain_id]), nxt))

    assert not bad, (
        f"these days describe a walk the graph does not contain (day, state, chain, next "
        f"state): {bad}")
