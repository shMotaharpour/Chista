"""The integer decision path must run, and say so.

Every defect found on this path was found by a human reading a traceback the
fallback had swallowed, not by the suite: a duplicate keyword made the decision
solve raise on every integral run, the land rows failed to broadcast for more
than two quadrants, and the quadrants the solve bought never crossed the
boundary to the caller. Each one left the same quiet signature -- an empty pool
and an objective of zero -- and each one passed every test in the repository.

The guard below is the one that would have gone red: it drives a day-0 board
with the integer solve ON and land buyable, and asserts the three things those
defects emptied. `fallback_reason` is asserted FIRST and by name, because a
fallback is how the path reports a failure it survived.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from agent.config import Config
from agent.main import AGENT
from agent.world.rules import LAND_PRICES
from offline_lab.fast_sim import FastSim


def _day_zero(config: Config):
    sim = FastSim({"episodeSteps": 24 * 3 + 6, "seed": 33, "farmHandCostMult": 1})
    obs = sim.observations(copy_state=False)[0]
    AGENT.cfg = config
    AGENT(obs, sim.configuration)
    return AGENT.manager.day.master


def test_the_integer_decision_path_runs_and_publishes_its_land() -> None:
    quadrants = 4
    master = _day_zero(replace(Config(), master_rounds=4, day_integral=True,
                               land_quadrants=quadrants))
    assert getattr(master, "fallback_reason", "") == "", (
        "the decision path fell back instead of solving: "
        f"{getattr(master, 'fallback_reason', None)!r}")
    assert bool(master.integral), (
        "the day asked for the integer solve and the result is not the integral one")
    bought = getattr(master, "land_bought", None)
    assert bought is not None, (
        "the integer solve bought quadrants but the caller cannot see them")
    days = int(np.asarray(AGENT.manager.contractor.days))
    nq = min(quadrants, len(LAND_PRICES))
    assert np.asarray(bought).shape == (nq, days), (
        f"land_bought is {np.asarray(bought).shape}, expected {(nq, days)}")
