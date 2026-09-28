"""The sale ladder's risk shave: the drain's own spread, priced as a price.

`belief.opponent.quantile_price_floor` names risk as one more number on the same
curve — the drain falling `z·sd` short is the ladder walked `z·sd` units higher.
`sell_blocks` reads that shave as a `pad`, so the model prices a fuller market
without a second price existing anywhere.
"""

from __future__ import annotations

import numpy as np

from agent.belief.depth import sell_blocks
from agent.belief.market import forecast
from offline_lab.fast_sim import FastSim

GOODS = ["MILK", "WHEAT", "FERTILIZER", "EGG", "WOOL"]


def _forecast():
    """One board's forecast — the same seed every run, so the ladder is a fixture."""
    sim = FastSim({"episodeSteps": 25, "seed": 33, "farmHandCostMult": 0})
    return forecast(sim.observations(copy_state=False)[0], days=5)


def test_a_zero_pad_is_the_mean_ladder_bit_for_bit():
    """R007: add the pad to the inventory unconditionally and this goes red."""
    fc = _forecast()
    units, mean = sell_blocks(fc, GOODS, 0, 5, 100, blocks=3)
    zero_units, zero = sell_blocks(fc, GOODS, 0, 5, 100, blocks=3,
                                   pad=np.zeros(len(GOODS)))
    assert (units == zero_units).all(), "a zero pad changed the blocks themselves"
    assert np.array_equal(mean, zero), "a zero pad changed the mean ladder"


def test_a_pad_prices_every_block_from_a_fuller_market():
    """A fuller market pays less, good by good, and only where the pad is set."""
    fc = _forecast()
    _, mean = sell_blocks(fc, GOODS, 0, 5, 100, blocks=3)
    pad = np.zeros(len(GOODS))
    pad[0] = 20.0                                   # MILK only
    _, priced = sell_blocks(fc, GOODS, 0, 5, 100, blocks=3, pad=pad)

    assert (priced[0] <= mean[0] + 1e-9).all(), "a padded block fetched MORE"
    assert (priced[0] < mean[0]).any(), "a 20-unit pad changed nothing at all"
    assert np.array_equal(priced[1:], mean[1:]), (
        "a good whose pad is zero must be the mean ladder, untouched")
