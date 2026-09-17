"""Replanner rung tests (issue #11 §6): duals, projection, cadence, bail.

Run:  .venv/bin/python -m tests.test_agent_replan

Contracts under test:
- The dual stand-in is non-negative (R006), flat over the horizon, and every
  component traces to the engine or to the observation (R005).
- A unit works the tile it stands on: LOCKED and unmodelled tiles are dropped
  rather than guessed at, and the plan's columns line up with the units.
- Every recovered plan dispatches through `agent/dispatch.py` into shape-valid
  actions for all 30 days, and `world.actions.validate_action` accepts each.
- The rung polls the deadline mid-work and bails (the between-rung gate alone
  would only ever see a replan that had already spent the turn).
- It runs once per day through `Runtime.act`, and dispatches for the rest.
- It is OFF by default: the rung cannot carry the inputs its chains assume
  until the secretary (#14) exists, so greedy stays the honest brain.
"""

from __future__ import annotations

import os
import time

import numpy as np

from world.fast_sim import FastSim
from world.actions import validate_action

from agent.dispatch import dispatch_plan
from agent.replan import (dual_stand_in, load_contractor, replan_day,
                          unit_state_ids)
from agent.runtime import Deadline, Runtime
from tile_dp.chains import CHAIN_NAMES, ENTITY_NAMES, N_RESOURCE, RESOURCE_ID
from tile_dp.contractor import HORIZON_DAYS

_RESOURCES = None


def _resources():
    """The shipped graph and one contractor, shared by the tests."""
    global _RESOURCES
    if _RESOURCES is None:
        contractor = load_contractor()
        _RESOURCES = (contractor.graph, contractor)
    return _RESOURCES


def _obs(day: int = 0, hour: int = 0, tile=None, hands=None) -> dict:
    """A real day-start observation from the engine, with the tile we want."""
    sim = FastSim({"episodeSteps": 24 * 3, "seed": 1})
    obs = dict(sim.observations()[0])
    obs["day"], obs["hour"] = day, hour
    obs["farms"] = [dict(obs["farms"][0])]
    tiles = [[None] * 10 for _ in range(10)]
    if tile is not None:
        tiles[4][4] = tile
    obs["farms"][0]["tiles"] = tiles
    obs["farms"][0]["hands"] = hands or []
    return obs


# ----------------------------------------------------------- the dual stand-in

def test_dual_stand_in_is_non_negative_and_sourced() -> None:
    """R006: the stand-in may never hand the contractor a negative dual."""
    p, w = dual_stand_in(_obs())
    assert p.shape == (HORIZON_DAYS, N_RESOURCE)
    assert w.shape == (HORIZON_DAYS, N_RESOURCE)
    assert float(p.min()) >= 0.0 and float(w.min()) >= 0.0
    # every product the engine quotes is priced, and no other component is
    assert float(p[:, RESOURCE_ID["WHEAT"]].min()) > 0.0
    assert float(p[:, RESOURCE_ID["LABOR_HOURS"]].max()) == 0.0
    # the wage side carries the engine's seed and animal costs (R002)
    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    assert w[0, RESOURCE_ID["SEED_WHEAT"]] == float(K.CROPS["WHEAT"]["seed"])
    assert w[0, RESOURCE_ID["ANIMAL_COW"]] == float(K.ANIMALS["COW"]["cost"])
    # and the hour's price is the engine's own hire cost over 23 hours (F040)
    assert w[0, RESOURCE_ID["LABOR_HOURS"]] == \
        float(K._hire_cost(0)) / 23.0
    # flat over the horizon: it is a quote, not a forecast
    assert np.all(p == p[0]) and np.all(w == w[0])


# --------------------------------------------------------------- the projection

def test_units_work_the_tile_they_stand_on() -> None:
    """LOCKED tiles are dropped (F042), not guessed at; units map to columns."""
    graph, _ = _resources()
    obs = _obs()                                     # bare tile under the farmer
    from agent.obs import decode_world
    view = decode_world(obs, at_day_start=True,
                        graph_keys=frozenset(graph.key_index))
    ids = unit_state_ids(view, graph)
    assert ids == [graph.key_index[view.me.keys[4][4]]]

    locked = _obs(tile="LOCKED")
    view = decode_world(locked, at_day_start=True,
                        graph_keys=frozenset(graph.key_index))
    assert unit_state_ids(view, graph) == [None]


def _board_for(view, graph):
    """A real board for the tiles `view` owns."""
    from agent.replan import dual_stand_in
    p, w = dual_stand_in(_obs())
    owned = [s for s in unit_state_ids(view, graph) if s is not None]
    _graph, contractor = _resources()
    return contractor.price(p, w, owned)


# ------------------------------------------------------------------ the rung

def test_replanned_plan_dispatches_and_validates() -> None:
    """Every recovered plan is a legal action, hour by hour, for 30 days."""
    graph, contractor = _resources()
    runtime = Runtime()
    runtime._replan_resources = (graph, contractor)
    obs = _obs()
    plan = replan_day(runtime, obs, graph=graph, contractor=contractor)
    assert isinstance(plan, dict) and set(plan) == {"units", "market"}
    assert plan["units"], "the farmer stands on a priceable tile"
    for day in range(30):
        for hour in range(24):
            turn_obs = dict(obs)
            turn_obs["day"], turn_obs["hour"] = day, hour
            action = dispatch_plan(plan, turn_obs)
            validate_action(0, action)
            assert set(action) == {"farmer", "hands", "market"}


def test_rung_polls_the_deadline_mid_work() -> None:
    """A slow step inside the rung bails with TimeoutError, after the work."""
    graph, contractor = _resources()

    class SlowContractor:
        def __init__(self, real):
            self.real = real
            self.days = real.days          # the master reads the horizon

        def price(self, *args, **kwargs):
            time.sleep(0.05)                      # one slow pricing step
            return self.real.price(*args, **kwargs)

    import agent.runtime as R
    runtime = Runtime()
    saved = R.WORKING_BUDGET_S
    R.WORKING_BUDGET_S = 0.02                     # shrink the wall
    runtime._deadline = Deadline(60.0)
    try:
        try:
            replan_day(runtime, _obs(), graph=graph,
                       contractor=SlowContractor(contractor))
        except TimeoutError as exc:
            assert "over budget" in str(exc), exc
        else:
            raise AssertionError("the rung ran past a shrunk deadline")
    finally:
        R.WORKING_BUDGET_S = saved


def test_replanner_runs_once_per_day() -> None:
    """Hour 0 replans, hours 1-23 dispatch the stored plan (never replan)."""
    runtime = Runtime()
    calls = []

    def counting(runtime_, obs):
        calls.append((obs["day"], obs["hour"]))
        return {"units": [[["WATER"], ["NORTH"]]], "market": []}

    runtime.replanner = counting
    seen_ops = []
    for day in range(2):
        for hour in range(24):
            action = runtime.act(_obs(day=day, hour=hour))
            seen_ops.append(action["farmer"][0])
    assert calls == [(0, 0), (1, 0)], calls
    # hour 1 of each day dispatched the stored plan's second op
    assert seen_ops[1] == "NORTH" and seen_ops[25] == "NORTH", seen_ops[:3]


def test_replanner_is_off_by_default() -> None:
    """The rung is opt-in until the secretary (#14) can carry the inputs."""
    assert "CHISTA_REPLAN" not in os.environ or \
        os.environ["CHISTA_REPLAN"] != "1"
    assert Runtime().replanner is None
    runtime = Runtime()
    action = runtime.act(_obs())
    assert action["farmer"][0] in ("PASS", "PLANT", "WATER", "HARVEST")


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - test runner
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {name}")
    if failures:
        print(f"{failures} test(s) failed")
        return 1
    print("all replanner tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
