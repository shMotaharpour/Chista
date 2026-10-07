"""The integer decision path must run, and say so.

Every defect found on this path was found by a human reading a traceback the
fallback had swallowed, not by the suite: a duplicate keyword made the decision
solve raise on every integral run, the land rows failed to broadcast for more
than two quadrants, and the quadrants the solve bought never crossed the
boundary to the caller. Each one left the same quiet signature -- an empty pool
and an objective of zero -- and each one passed every test in the repository.

The guards below are the ones that would have gone red. They run on a SHORT
horizon (a three-day `episodeSteps` shrinks the day-0 contractor through
`Manager._roll_day`), which is a real board and a real MIP at a fraction of
the season solve's size -- the land decision is a day-0 question and the
guards keep it seconds, not minutes.

`fallback_reason` is asserted FIRST and by name, because a fallback is how
the path reports a failure it survived.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from agent.config import Config
from agent.main import AGENT
from agent.world.rules import LAND_PRICES
from offline_lab.fast_sim import FastSim

#: A three-day season: the day-0 horizon is three days, so the MIP's sweeps
#: and its branch-and-bound both stay small while every row the season board
#: carries is still in the matrix.
SHORT_EPISODE = {"episodeSteps": 24 * 3 + 6, "seed": 33,
                 "farmHandCostMult": 1}


def _day_zero(config: Config):
    sim = FastSim(SHORT_EPISODE)
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

    sim = FastSim(SHORT_EPISODE)
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
    sim = FastSim(SHORT_EPISODE)
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


# ---------------------------------------------------- the prefix the farm owns

def test_the_land_prefix_is_relative_to_the_quadrants_owned() -> None:
    """NW is free and open (F042), so only the steps that are LEFT are buyable.

    `land_block` is the one place the ladder is indexed. `land_owned=1` is the
    day-0 farm (bit-identical to before); each further quadrant the farm owns
    removes a step from the block and raises the tile tie by 25.
    """
    from agent.planner.colgen import land_block
    from agent.world.rules import LAND_PRICES

    n = len(LAND_PRICES)
    assert land_block(4, 1) == (n, 0, 25.0)        # day-0 farm: all three steps
    assert land_block(4, 2) == (n - 1, 1, 50.0)    # NE taken: SW and SE remain
    assert land_block(4, 4) == (0, n, 100.0)       # the ladder is complete
    assert land_block(2, 2) == (1, 1, 50.0)        # the request still clamps
    assert land_block(0, 2) == (0, 1, 50.0)        # land off is no land rows


def test_a_board_that_already_owns_a_quadrant_is_not_forced_to_buy() -> None:
    """The tile tie starts at the tiles the farm owns, not at NW's 25 alone.

    Measured on a day-1 board owning NW+NE (50 tiles): with the tie's constant
    right-hand side of 25, `Sigma y >= 1` was mandatory and the model bought
    quadrant NE's own column for 1000 -- the quadrant it already had, at the
    price of the step before the one really next (SW, 2000).
    """
    from agent.manager.core import Manager

    pass_action = {"farmer": ["PASS"], "hands": [], "market": []}
    sim = FastSim(SHORT_EPISODE)
    sim.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_LAND"]]},
              pass_action])
    obs = sim.observations(copy_state=False)[0]
    while int(obs["hour"]) != 0:
        sim.step([pass_action, pass_action])
        obs = sim.observations(copy_state=False)[0]
    owned = len(obs["farms"][0]["unlocked_quadrants"])
    assert owned == 2, f"the engine opened {owned} quadrants, expected NW+NE"

    quadrants = len(LAND_PRICES)
    manager = Manager(replace(Config(), land_quadrants=quadrants,
                              master_rounds=2, day_integral=True))
    manager.observe(obs, sim.configuration)
    master = manager.day.master
    assert getattr(master, "fallback_reason", "") == "", master.fallback_reason
    # The "one purchase each" rows left are SW and SE; NE's is gone.
    remaining = min(quadrants, len(LAND_PRICES)) - (owned - 1)
    assert master.land_dual is not None and len(master.land_dual) == remaining, (
        f"the land block offers {None if master.land_dual is None else len(master.land_dual)} "
        f"quadrants with {owned} owned, expected {remaining}")
    # Nothing is forced: 50 owned tiles against a tie of 25*k is slack at k=2.
    bought = getattr(master, "land_bought", None)
    assert bought is not None and float(np.asarray(bought).sum()) == 0.0, (
        "the model bought land on a board that already owns it: "
        f"{np.asarray(bought)}")


# ------------------------------------------------------- the land DECISION

def test_a_rich_purse_buys_the_prefix_early() -> None:
    """Land costs coins against the score: with a deep purse the model buys.

    Measured on the seed-33 day-0 board at 100,000 coins: NE on day 0 (1000),
    SW on day 1 (2000), SE on day 2 (4000) -- the prefix taken as early as
    the ladder's own order allows, on a three-day horizon the purchases land
    on their earliest legal days within it.

    The guard that made this testable found the defect first: the start-day
    rows carried the old tie's right-hand side of 25, so working a locked
    quadrant was FREE and the model bought nothing at any purse.
    """
    from agent.manager.core import Manager

    sim = FastSim(SHORT_EPISODE)
    obs = sim.observations(copy_state=False)[0]
    manager = Manager(replace(Config(), land_quadrants=len(LAND_PRICES),
                              master_rounds=4, day_integral=True))
    # The purse is the board's own money; the engine starts the farm at 3000
    # and this guard names the deep-purse regime the owner ordered, so the
    # observation's money is raised to it before the solve reads it.
    obs = dict(obs)
    obs["farms"] = [dict(obs["farms"][0])] + list(obs["farms"][1:])
    obs["farms"][0]["money"] = 100000
    manager.observe(obs, sim.configuration)
    master = manager.day.master
    assert getattr(master, "fallback_reason", "") == "", master.fallback_reason
    bought = getattr(master, "land_bought", None)
    assert bought is not None, "the deep-purse run built no land block"
    b = np.asarray(bought)
    assert float(b.sum()) == 3.0, (
        f"the deep purse bought {float(b.sum())} of 3 quadrants: {b}")
    # The prefix order, one step a day, from day 0.
    cells = [(int(q), int(d)) for q, d in zip(*np.nonzero(b))]
    assert cells == [(0, 0), (1, 1), (2, 2)], (
        f"the prefix was taken out of order or off-schedule: {cells}")


def test_the_start_day_rows_right_hand_side_is_what_makes_land_cost() -> None:
    """R007: the start-day row's right-hand side is what makes land cost.

    With the old right-hand side of 25 the row read `lambda - 25*y <= 25`,
    which 25 tiles satisfy at y = 0 -- working a locked quadrant needed no
    purchase, and at every purse the model bought nothing (measured:
    land_bought empty at 100,000 coins). Re-introduce that right-hand side
    and the deep-purse guard above goes red on the same board; the failure
    text was "the deep purse bought 0.0 of 3 quadrants".
    """
    from agent.manager.core import Manager
    from agent.planner import colgen

    sim = FastSim(SHORT_EPISODE)
    obs = sim.observations(copy_state=False)[0]
    manager = Manager(replace(Config(), land_quadrants=len(LAND_PRICES),
                              master_rounds=4, day_integral=True))
    obs = dict(obs)
    obs["farms"] = [dict(obs["farms"][0])] + list(obs["farms"][1:])
    obs["farms"][0]["money"] = 100000
    manager.observe(obs, sim.configuration)
    assert float(np.asarray(manager.day.master.land_bought).sum()) == 3.0, (
        "the healthy path stopped buying; this guard's premise is stale")
    # The invariant the fix established, read at the one place it is set:
    # every start-day row's upper bound is 0 -- `lambda <= 25*y` -- never the
    # old tie's 25.
    src = open("agent/planner/colgen.py", encoding="utf-8").read()
    assert "np.zeros(nq * days)]) if nq else np.zeros(0)," in src, (
        "the start-day rows' right-hand side drifted off 0: working a locked "
        "quadrant would be free again and the model would never buy")
