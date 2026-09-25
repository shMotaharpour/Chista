"""The manager: the turn's clock, and the memory between days.

It decides nothing — `planner/` does — so what is asserted here is the two
things it owns. A solve that respects the budget it was given, and a column
pool carried forward that means the same thing on the day it is reused.

R007: every guard was broken and seen red before it was trusted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.manager.core import Manager
from agent.world.rules import TURNS_PER_DAY


# --- config: numbers, and the ones that are not legal ---------------------

def test_a_turn_budget_inside_its_own_reserve_is_rejected():
    with pytest.raises(ValueError, match="leaves nothing after"):
        Config(turn_budget_ms=100.0, reserve_ms=100.0)


def test_an_unknown_config_key_is_an_error_not_a_shrug(tmp_path):
    """A number that is not read is a number that is not tuned."""
    bad = tmp_path / "config.json"
    bad.write_text('{"damping": 0.4, "dampign": 0.9}')
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.load(bad)


def test_a_missing_config_file_is_the_ordinary_case(tmp_path):
    assert Config.load(tmp_path / "nothing.json") == Config()


def test_a_round_trip_through_the_artifact_keeps_every_number(tmp_path):
    out = tmp_path / "config.json"
    Config(damping=0.35, max_hands=6).dump(out)
    assert Config.load(out) == Config(damping=0.35, max_hands=6)


# --- the clock ------------------------------------------------------------

def test_the_solve_stops_inside_the_budget_it_was_given():
    """A round costs what the last one cost, so one that cannot finish is not
    started — and the answer degrades instead of overrunning.

    The deadline used to be assigned over inside `equilibrate` by the rung's
    own (absent) one, so a caller that passed a budget got none: 25 turns of a
    season went past 965 ms, the worst at 2.7 s.
    """
    from agent.planner import columns as C
    from agent.planner import day as D
    from agent.planner import master as M
    from agent.planner.colgen import classes_of
    from agent.planner.inputs import GRAPH_PATH, load_contractor
    from agent.obs import decode_world
    from agent.tile_dp.graph import TileGraph
    from offline_lab.kaggle_env import new_environment

    env = new_environment({"seed": 0})
    env.reset(2)
    obs = env.state[0].observation
    graph = TileGraph.load(GRAPH_PATH)
    contractor = load_contractor(days=20)
    _r, _c, of_tile = classes_of(M._owned_states(object(), obs),
                                 M._owned_distances(obs, C.shed_distance()))
    view = decode_world(obs, at_day_start=True,
                        graph_keys=frozenset(graph.key_index))
    walker = iter(of_tile)
    class_of_tile = [next(walker, None) if int(k) >= 0 else None
                     for k in np.asarray(view.me.keys).reshape(-1)]

    seen = []
    for budget_ms in (150.0, 400.0):
        started = time.perf_counter()
        result = D.plan(obs, contractor, M.supply_from_obs(obs),
                        class_of_tile=class_of_tile, iter_cap=200, hands=4,
                        budget_s=0.25, rounds=2, pool=[],
                        deadline=started + budget_ms / 1000.0)
        elapsed = (time.perf_counter() - started) * 1000.0
        assert elapsed < budget_ms * 1.5 + 60, (
            f"a {budget_ms:.0f} ms budget took {elapsed:.0f} ms")
        assert result.master.stopped == "budget", (
            f"the solve was not stopped by the clock: "
            f"{result.master.stopped!r}")
        seen.append(result.master.objective)

    assert seen[1] >= seen[0], (
        f"more time bought a worse answer: {seen[0]:.0f} at 150 ms, "
        f"{seen[1]:.0f} at 400 ms")


# --- the memory -----------------------------------------------------------

def test_a_warm_pool_reaches_the_same_answer_in_a_fraction_of_the_rounds():
    """A column is a plan, and a plan is still a plan when the prices move."""
    from agent.planner import master as M
    from agent.planner.inputs import load_contractor
    from offline_lab.kaggle_env import new_environment

    env = new_environment({"seed": 0})
    env.reset(2)
    obs = env.state[0].observation
    contractor = load_contractor(days=20)
    supply = M.supply_from_obs(obs)

    cold = M.equilibrate(object(), obs, contractor, supply, iter_cap=200)
    warm = M.equilibrate(object(), obs, contractor, supply, iter_cap=200,
                         pool=cold.pool)

    assert cold.certified and warm.certified
    assert warm.objective == pytest.approx(cold.objective, rel=1e-9)
    assert warm.rounds < cold.rounds / 4, (
        f"the warm pool saved nothing: {warm.rounds} rounds against "
        f"{cold.rounds}")


def test_a_carried_column_is_matched_by_class_KEY_not_by_index():
    """An index is positional: tomorrow's class 3 is not today's.

    A pool matched by index hands a plan for an empty field over as a plan for
    a grown crop, and the master prices a fiction it then commits tiles to.
    """
    from agent.planner.colgen import Column, generate, solve_master

    def column(cls, cls_key, revenue):
        return Column(cls=cls, cost=np.zeros((2, 1)), spend=np.zeros(2),
                      earn=np.zeros(2), revenue=revenue, cls_key=cls_key,
                      key=("plan", revenue))

    idle = [column(0, (7, 0), 0.0), column(1, (9, 3), 0.0)]
    idle = [Column(cls=c.cls, cost=c.cost, spend=c.spend, earn=c.earn,
                   revenue=0.0, cls_key=c.cls_key, key=("idle",))
            for c in idle]
    # A pool from a board where the SAME classes sat at the other indices.
    stale = [column(1, (7, 0), 5.0), column(0, (9, 3), 7.0),
             column(0, (4, 4), 9.0)]          # a class this board does not have

    def price(duals):                      # the one-argument contract
        y, cash = duals.y, duals.cash
        return np.zeros(2), [column(0, (7, 0), 0.0), column(1, (9, 3), 0.0)]

    out = generate(price, np.full(2, 10.0), 100.0, np.array([1, 1]), 2, 1,
                   idle, rounds=1, warm=stale)
    carried = {(c.cls, c.cls_key) for c in out.pool if c.key != ("idle",)}
    assert carried == {(0, (7, 0)), (1, (9, 3))}, (
        f"the warm columns landed on the wrong classes: {carried}")


def test_the_plan_is_legal_before_any_day_is_observed():
    """`best()` is never None: idle is a legal answer and the honest one."""
    plan = Manager().best()
    assert plan["units"] and len(plan["units"][0]) == TURNS_PER_DAY
    assert all(op == ["PASS"] for op in plan["units"][0])


def test_unmodelled_tile_does_not_shift_classes_of_subsequent_tiles():
    """An unmodelled tile key must get None and NOT shift subsequent tiles (#152)."""
    import copy
    from offline_lab.kaggle_env import new_environment
    from agent.planner.colgen import classes_of
    from agent.planner import master as M
    from agent.planner import columns as C

    env = new_environment({"seed": 0})
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    manager = Manager()

    # Base line: all 25 tiles modelled
    owned0 = M._owned_states(object(), obs)
    dists0 = M._owned_distances(obs, manager.steps)
    _, _, of_tile0 = classes_of(owned0, dists0)
    normal = manager._class_of_tile(obs, of_tile0)

    # Inject an unmodelled state at row 0, col 2
    obs["farms"][0]["tiles"][0][2] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "yield_units": 2
    }
    owned1 = M._owned_states(object(), obs)
    dists1 = M._owned_distances(obs, manager.steps)
    reps1, _, of_tile1 = classes_of(owned1, dists1)
    shifted = manager._class_of_tile(obs, of_tile1)

    # Tile (0, 2) must be None because its state is not modelled
    assert shifted[2] is None, f"unmodelled tile should be None, got {shifted[2]}"
    # Last owned tile (row 4, col 4 -> index 44) must NOT be None
    assert shifted[44] is not None, "last owned tile was shifted out to None"
    # Tile (0, 3) must match its own state and distance, not be shifted from tile 2
    assert reps1[shifted[3]] == (0, int(manager.steps[3])), (
        f"tile 3 class shifted: got {reps1[shifted[3]]}, expected (0, {manager.steps[3]})"
    )
    # Tile (4, 4) must match its own state and distance
    assert reps1[shifted[44]] == (0, int(manager.steps[44])), (
        f"tile 44 class shifted: got {reps1[shifted[44]]}, expected (0, {manager.steps[44]})"
    )

