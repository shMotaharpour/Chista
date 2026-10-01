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


def _fresh_day_zero(config: Config):
    """A NEW agent, so `config` is the one its manager is CONSTRUCTED with."""
    from agent.main import Agent

    sim = FastSim({"episodeSteps": 24 * 3 + 6, "seed": 33, "farmHandCostMult": 1})
    obs = sim.observations(copy_state=False)[0]
    agent = Agent(config)
    agent(obs, sim.configuration)
    return agent


def test_the_runs_land_count_reaches_the_loop_as_an_argument(monkeypatch) -> None:
    """The count must reach `generate` as an ARGUMENT, not be re-read inside it.

    The loop's last solve and the decision solve beside it are two halves of ONE
    matrix, so the quadrant count the run injected has to arrive at the pricing
    call whole. A `generate` that reads a default of its own is a second source
    for the run's own number -- a defect that is invisible here, where the two
    agree, and lethal the day they do not.
    """
    from agent.planner import colgen

    seen: list = []
    real = colgen.generate

    def spy(*args, **kwargs):
        seen.append(kwargs.get("land", "<absent>"))
        return real(*args, **kwargs)

    monkeypatch.setattr(colgen, "generate", spy)
    quadrants = 4
    agent = _fresh_day_zero(replace(Config(), master_rounds=2, day_integral=True,
                                    land_quadrants=quadrants))
    master = agent.manager.day.master
    assert seen and set(seen) == {quadrants}, (
        f"generate was handed land={sorted(set(seen), key=str)}: the run's "
        "quadrant count did not reach the pricing call")
    days = int(np.asarray(agent.manager.contractor.days))
    nq = min(quadrants, len(LAND_PRICES))
    # The land block is `nq` binaries per day; its duals are the model's own
    # receipt that the rows were in the matrix the pricing priced on.
    assert master.land_dual is not None and len(master.land_dual) == nq, (
        f"land rows are {getattr(master, 'land_dual', None)!r}, expected {nq}")
    assert master.rent is not None and len(master.rent) == days, (
        f"rent is {getattr(master, 'rent', None)!r}, expected {days} days")
    assert np.asarray(master.land_bought).shape == (nq, days)


def test_zero_quadrants_keep_every_land_row_off(monkeypatch) -> None:
    """`land_quadrants = 0` is the shipped LP path, and it stays untouched."""
    from agent.planner import colgen

    seen: list = []
    real = colgen.generate

    def spy(*args, **kwargs):
        seen.append(kwargs.get("land", "<absent>"))
        return real(*args, **kwargs)

    monkeypatch.setattr(colgen, "generate", spy)
    agent = _fresh_day_zero(replace(Config(), master_rounds=2, day_integral=True,
                                    land_quadrants=0))
    master = agent.manager.day.master
    assert seen and set(seen) == {None}, (
        f"generate was handed land={sorted(set(seen), key=str)} at zero quadrants")
    assert getattr(master, "land_bought", None) is None
    assert getattr(master, "land_dual", None) is None


def test_a_config_swapped_after_a_turn_reaches_the_next_manager() -> None:
    """A NEW config object is a new run: its numbers must rebuild the manager.

    The shared `AGENT` is the shape every probe and every regime-pinning test
    uses, and a manager left over from the previous config silently plays the
    OLD numbers -- measured: `AGENT.cfg = ...(land_quadrants=4)` after one turn
    still built no land rows, and `land_bought` stayed None.
    """
    sim = FastSim({"episodeSteps": 24 * 3 + 6, "seed": 33, "farmHandCostMult": 1})
    obs = sim.observations(copy_state=False)[0]
    AGENT.cfg = replace(Config(), master_rounds=2, day_integral=True,
                        land_quadrants=0)
    AGENT(obs, sim.configuration)
    assert getattr(AGENT.manager.day.master, "land_bought", None) is None, (
        "the zero-quadrant run bought land")
    quadrants = 2
    AGENT.cfg = replace(Config(), master_rounds=2, day_integral=True,
                        land_quadrants=quadrants)
    AGENT(obs, sim.configuration)
    assert getattr(AGENT.manager.cfg, "land_quadrants", None) == quadrants, (
        "the swapped config never reached the manager: "
        f"{getattr(AGENT.manager.cfg, 'land_quadrants', None)!r}")
    bought = getattr(AGENT.manager.day.master, "land_bought", None)
    assert bought is not None and np.asarray(bought).shape[0] == quadrants, (
        f"land_bought is {None if bought is None else np.asarray(bought).shape} "
        f"after the config was swapped to {quadrants} quadrants")
