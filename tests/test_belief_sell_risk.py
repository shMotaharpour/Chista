"""The sale ladder's risk shave: the drain's own spread, priced as a price.

`belief.opponent.quantile_price_floor` names risk as one more number on the same
curve — the drain falling `z·sd` short is the ladder walked `z·sd` units higher.
`sell_blocks` reads that shave as a `pad`, so the model prices a fuller market
without a second price existing anywhere.
"""

from __future__ import annotations

import numpy as np

from agent.belief.depth import depth_coins, sell_blocks
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


def _block_revenue(units, prices, lot: int) -> float:
    """What the model credits a `lot` of one good, filling the rich blocks first."""
    left, rev = int(lot), 0.0
    for i in range(units.shape[2]):
        take = min(left, int(units[0, 0, i]))
        rev += take * float(prices[0, 0, i])
        left -= take
        if left == 0:
            break
    return rev


def _worst_error(fc, good, blocks, cap=100):
    """The largest relative gap between the block model and the ladder, all lots."""
    units, prices = sell_blocks(fc, [good], 0, 1, cap, blocks=blocks)
    worst = 0.0
    for lot in range(1, cap + 1):
        exact = float(depth_coins(fc, good, 0, lot))
        worst = max(worst, abs(_block_revenue(units, prices, lot) - exact) / max(1.0, exact))
    return worst


def test_the_named_lot_is_priced_by_the_ladder_not_by_a_hidden_block():
    """The pathology this split was built for: a big lot inside ONE block.

    With geometric edges (1, 2, 4, 8, 100) a 42-unit MILK lot landed inside the
    92-unit last block and was credited 3063 against the ladder's 4914 -- 38% low,
    which is the 7-cow day priced wrong. Equal-REVENUE bands put the edges where
    the curve bends (9, 18, 29, 43, 100) and the same lot is within 1%. R007: put
    the geometric split back and this goes red.
    """
    fc = _forecast()
    units, prices = sell_blocks(fc, ["MILK"], 0, 1, 100, blocks=5)
    lot = 42
    exact = float(depth_coins(fc, "MILK", 0, lot))
    got = _block_revenue(units, prices, lot)
    assert got / exact >= 0.99, (
        f"{lot} MILK: the blocks credit {got:.1f} against the ladder's {exact:.1f}")


def test_more_blocks_never_price_worse():
    """The real cost/benefit knob: the error falls with the block count.

    Measured worst case over lots 1..100 on MILK day 0: 9.606% at 5 blocks,
    6.432% at 8, 5.237% at 10, 4.134% at 12 -- the block model cannot be exact at
    a small K, so K is the fidelity/cost dial and it must be monotone. R007:
    reverse the edge builder's order and this goes red.
    """
    fc = _forecast()
    errors = [_worst_error(fc, "MILK", k) for k in (5, 8, 10, 12)]
    assert all(b <= a + 1e-9 for a, b in zip(errors, errors[1:])), (
        "a bigger block count priced worse: " + ", ".join(f"{e:.3%}" for e in errors))


def test_the_block_edges_rise_and_end_at_the_cap():
    """The shape contract: strictly rising edges whose last one is the cap."""
    from agent.belief.depth import equal_revenue_edges
    fc = _forecast()
    edges = equal_revenue_edges(fc, "MILK", 0, 100, 5)
    assert edges[-1] == 100, edges
    assert all(b > a for a, b in zip(edges, edges[1:])), edges
    units, _ = sell_blocks(fc, ["MILK"], 0, 1, 100, blocks=5)
    assert int(units.sum()) == 100, int(units.sum())
