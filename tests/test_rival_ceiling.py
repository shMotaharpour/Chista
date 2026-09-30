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
