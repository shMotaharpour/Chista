"""The rival's ceiling: the other end of the calendar the forecast consumes.

The dated calendar is a floor (an event sits on the first day the good
certainly exists, carrying the minimum it certainly carries). `rival_stock` and
`rival_bag` are the ceiling and were read by nobody. The guard can fail: it
asserts the sum, not the existence of the method.
"""
import numpy as np

from agent.belief.tracker import MarketTracker as Tracker


def test_the_ceiling_is_the_stock_plus_the_unsold_bag():
    tr = Tracker()
    assert np.all(tr.rival_ceiling() == 0.0)          # nothing observed yet
    tr.rival_stock = np.arange(len(tr.rival_stock), dtype=float)
    tr.rival_bag = np.ones(len(tr.rival_bag), dtype=float)
    got = tr.rival_ceiling()
    assert np.allclose(got, np.arange(len(got), dtype=float) + 1.0)
    # it is a read, not a handle on the state
    got[0] = 999.0
    assert tr.rival_ceiling()[0] == 1.0


def test_the_band_is_opt_in_and_otherwise_the_same_number():
    """No ceiling in the forecast => the high band must not change a price.

    `forecast` binds the two walks to the same array when no ceiling is handed
    in, so `price_of(high=True)` reads the identical table. Built on a real
    forecast from the agent's own path, because that is the forecast the plan
    reads. The guard can fail: point the high walk at anything else and the two
    numbers part.
    """
    from agent.belief.market import price_paths
    from agent.main import AGENT
    from agent.config import Config as _C
    from dataclasses import replace as _replace
    from offline_lab.fast_sim import FastSim

    if not hasattr(AGENT, "cfg") or AGENT.cfg is None:
        AGENT.cfg = _replace(_C(), master_rounds=2)
    sim = FastSim({"episodeSteps": 24 + 6, "seed": 33, "farmHandCostMult": 1})
    obs = sim.observations(copy_state=False)[0]
    AGENT(obs, sim.configuration)
    # the manager builds the forecast through `_forecast` and keeps no copy, so
    # ask the same builder the plan asks
    fc = AGENT.manager._forecast(obs, getattr(AGENT.manager, "terms", None))
    assert fc is not None and hasattr(fc, "price_of"), "no forecast to read"
    # The mutation that makes this fail, and why the obvious one does not: with
    # no ceiling `forecast` binds BOTH walks to the same array, so forcing the
    # band branch on (`if not high:` alone) is an EQUIVALENT mutant -- it runs
    # and returns the same number. Poison the band's own read instead
    # (`float(row[gi]) + 1000.0` in price_of) and this goes red.
    plain = price_paths(fc, days=3)
    band = price_paths(fc, days=3, high=True)
    assert plain == band, "the high band moved a price with no ceiling given"
