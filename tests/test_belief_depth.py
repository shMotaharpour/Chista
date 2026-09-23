"""Guards for the market's DEPTH surface (`belief/depth.py`).

The claim under test: belief publishes what a LOT fetches, not only what one
unit fetches — the walk's inventory at the day (or the hour) a sale lands, fed
through the ladder belief already owns. The identities that must hold:

  * one definition — the curve IS `ladder.sell_coins` at the forecast's own
    walked inventory (patch the ladder and the curve must move);
  * the curve's first unit is the forecast's own quote (the depth surface
    continues the price table, it is not a second price);
  * the marginal never rises in volume and stalls at the engine's $1 floor;
  * the blocks are exact at their boundaries and never credit a PARTIAL fill
    above the ladder (the conservative direction an LP needs);
  * a flat fraction of the quote is neither a bound nor an estimate — the same
    0.5 is too rich for one good and too poor for another (measured here);
  * the day split runs on the WALK's own drains, not on zeros;
  * the hourly inventory and the hourly prices read the SAME walk row.

Run:  .venv/bin/python -m tests.test_belief_depth   (also under pytest)
"""

from __future__ import annotations

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K
from agent.belief import depth as D
from agent.belief.ladder import FLOOR, sell_coins, split_days
from agent.belief.market import (PRODUCTS, forecast, hourly_inventory,
                                 hourly_prices)
from agent.world.model import PRODUCTS as WORLD_PRODUCTS


def _sim(seed: int = 0):
    from offline_lab.fast_sim import FastSim
    sim = FastSim(configuration={"seed": seed, "episodeSteps": 720},
                  validate="dev")
    PASS = {"farmer": ["PASS"], "hands": [], "market": []}
    return sim, PASS


def _fc(seed: int = 0, days: int = 30):
    sim, _ = _sim(seed)
    return forecast(sim.observations()[0], days=days)


def test_the_depth_curve_is_the_ladder_at_the_forecasts_own_inventory() -> None:
    """Every point of the curve is `sell_coins` at the walked inventory."""
    fc = _fc()
    for item in ("CARROT", "MILK", "MELON", "WHEAT"):
        for day in (0, 7, 29):
            inv = int(fc.inventory_of(item, day))
            for n in (1, 5, 40, 200):
                assert D.depth_coins(fc, item, day, n) == sell_coins(item, inv, n), \
                    f"{item} day {day} n={n}: the curve is not the ladder"
    # and the module CALLS the ladder rather than re-deriving it
    real = D.sell_coins
    try:
        D.sell_coins = lambda good, inv, units: 777
        assert D.depth_coins(fc, "MILK", 0, 3) == 777, \
            "patching the ladder must move the curve (no second ladder)"
    finally:
        D.sell_coins = real


def test_the_curves_first_unit_is_the_forecasts_own_quote() -> None:
    """A single unit fetches the day's quote: the depth surface continues it."""
    fc = _fc()
    for item in ("CARROT", "TOMATO", "MILK", "MELON", "FERTILIZER"):
        for day in (0, 12, 29):
            assert D.marginal_price(fc, item, day, 1) == fc.price_of(item, day), \
                f"{item} day {day}: the first unit is not the forecast's quote"


def test_the_marginal_never_rises_and_stalls_at_the_floor() -> None:
    """Volume only cheapens: the marginal declines, and the floor is $1."""
    fc = _fc()
    for item in ("CARROT", "TOMATO", "STRAWBERRY", "MELON", "MILK", "WOOL"):
        prev = float("inf")
        for n in range(1, 201):
            m = D.marginal_price(fc, item, 0, n)
            assert m <= prev, f"{item}: the marginal rose at unit {n}"
            assert m >= 1.0, f"{item}: a unit fetched below the $1 floor"
            prev = m
    # the stall: past the floor inventory every unit fetches exactly 1
    inv = fc.inventory_of("MILK", 0)
    floor_at = FLOOR[WORLD_PRODUCTS.index("MILK")]
    assert floor_at is not None, "MILK is expected to reach the floor"
    stall = int(floor_at) - int(inv)
    assert stall > 0, "the day-0 MILK inventory should be above the floor"
    assert D.marginal_price(fc, "MILK", 0, stall + 1) == 1.0
    assert D.marginal_price(fc, "MILK", 0, stall + 50) == 1.0


def test_the_blocks_sum_to_the_exact_coins_and_decline() -> None:
    """The blocks are a representation of the curve, not a second price."""
    fc = _fc()
    for item in ("CARROT", "MILK", "MELON"):
        for units in (7, 40, 150):
            blocks = D.depth_blocks(fc, item, 0, units, blocks=3)
            assert sum(k for k, _ in blocks) == units
            total = sum(k * price for k, price in blocks)
            assert abs(total - D.depth_coins(fc, item, 0, units)) < 1e-9, \
                f"{item}: the blocks do not sum to the ladder's coins"
            prices = [price for _, price in blocks]
            assert all(a >= b for a, b in zip(prices, prices[1:])), \
                f"{item}: the block prices must not rise"
    # On a lot big enough for the ladder to step, the blocks must DECLINE
    # strictly: a flat block price is the tier this curve retires, and on a
    # 7-unit lot the quote does not move at all, so equality is legitimate
    # there and the strict form belongs on the larger lots.
    for item, units in (("CARROT", 40), ("MILK", 150), ("MELON", 60)):
        prices = [price for _, price in D.depth_blocks(fc, item, 0, units, blocks=3)]
        assert all(a > b for a, b in zip(prices, prices[1:])), (
            f"{item} {units}u: the block prices must DECLINE, got {prices}")


def test_the_blocks_never_credit_a_partial_fill_above_the_ladder() -> None:
    """Exact at the boundaries, conservative inside a block — the LP's shape."""
    fc = _fc()
    for item in ("CARROT", "TOMATO", "MILK", "MELON"):
        units, blocks_n = 60, 3
        blocks = D.depth_blocks(fc, item, 0, units, blocks=blocks_n)
        for x in range(0, units + 1):
            credited = D.block_revenue(blocks, x)
            exact = D.depth_coins(fc, item, 0, x)
            assert credited <= exact + 1e-9, (
                f"{item}: a partial fill of {x} units was credited "
                f"{credited:.1f} against the ladder's {exact}")
            if x % (units // blocks_n) == 0:       # a block boundary
                assert abs(credited - exact) < 1e-9, \
                    f"{item}: boundary {x} must be exact"


def test_a_flat_fraction_of_the_quote_is_neither_a_bound_nor_an_estimate() -> None:
    """The measured shape that retires the flat deep tier (#151).

    On the day-0 board the first 100 MILK units average BELOW half the quote
    and the first 200 MELON units ABOVE it, so the same 0.5 factor is too rich
    for one good and too poor for the other. Both directions are asserted: a
    model with one flat fraction is wrong in sign for one of them whichever
    fraction it picks.
    """
    fc = _fc()
    milk_avg = D.depth_coins(fc, "MILK", 0, 100) / 100
    melon_avg = D.depth_coins(fc, "MELON", 0, 200) / 200
    milk_half = 0.5 * fc.price_of("MILK", 0)
    melon_half = 0.5 * fc.price_of("MELON", 0)
    assert milk_avg < milk_half, (
        f"milk 100u average {milk_avg:.1f} is not below the flat tier "
        f"{milk_half:.1f}; the flat model would be conservative here")
    assert melon_avg > melon_half, (
        f"melon 200u average {melon_avg:.1f} is not above the flat tier "
        f"{melon_half:.1f}; the flat model would over-credit here")
    # the block model tracks the same two cases in the right direction
    for item, units, half in (("MILK", 100, milk_half), ("MELON", 200, melon_half)):
        blocks = D.depth_blocks(fc, item, 0, units, blocks=3)
        avg = D.block_revenue(blocks, units) / units
        assert abs(avg - D.depth_coins(fc, item, 0, units) / units) < 1e-9
        assert (avg < half) == (item == "MILK")


def test_the_best_day_split_uses_the_walks_own_drains() -> None:
    """The split's drains are the forecast's day-to-day drops, not zeros."""
    fc = _fc()
    seen: dict = {}
    real = D.split_days

    def spy(good, inventory, lot, drains):
        seen["drains"] = np.asarray(drains, dtype=np.int64).copy()
        seen["inventory"] = int(inventory)
        seen["lot"] = int(lot)
        return real(good, inventory, lot, drains)

    try:
        D.split_days = spy
        plan, coins = D.best_day_split(fc, "CARROT", 0, 60, days=8)
    finally:
        D.split_days = real

    walk_drains = np.array(
        [max(0, int(fc.inventory_of("CARROT", d))
             - int(fc.inventory_of("CARROT", d + 1)))
         for d in range(8)], dtype=np.int64)
    assert seen["drains"].tolist() == walk_drains.tolist(), \
        "the split must run on the walk's own drains"
    assert seen["inventory"] == fc.inventory_of("CARROT", 0)
    assert seen["lot"] == 60
    assert int(walk_drains.sum()) > 0, "this board's walk must drain something"
    assert int(plan.sum()) == 60, "every unit of the lot is sold"
    # the plan is priced by the evaluator at the SAME drains
    from agent.belief.ladder import plan_coins
    assert plan_coins("CARROT", int(fc.inventory_of("CARROT", 0)), plan,
                      walk_drains) == coins
    # and the walk's drains are what makes it non-trivial: with zeros the DP
    # may defer, with the walk's drains it must not (the time direction)
    zero_plan, zero_coins = real("CARROT", int(fc.inventory_of("CARROT", 0)), 60,
                                 np.zeros(8, dtype=np.int64))
    assert (plan.tolist() != zero_plan.tolist()) or (coins != zero_coins), \
        "the walk's drains must change the split (they are not zeros)"


def test_the_hourly_inventory_and_prices_read_the_same_row() -> None:
    """One walk, two readings: the quote must be the engine's at that inventory."""
    sim, _ = _sim(3)
    obs = sim.observations()[0]
    fc = forecast(obs, days=2)
    INV = hourly_inventory(fc)
    H = hourly_prices(fc)
    assert INV.shape == H.shape == (2 * 24, len(PRODUCTS))
    for row in range(INV.shape[0]):
        for gi, item in enumerate(PRODUCTS):
            assert int(H[row, gi]) == int(K.market_price(item, float(INV[row, gi]))), \
                f"row {row} {item}: the price is not the engine's at this inventory"
    # the depth curve at an hour reads that same row
    for hour in (0, 6, 12, 23):
        inv = D.inventory_at(fc, "MILK", 0, hour)
        assert inv == int(INV[hour, PRODUCTS.index("MILK")]), \
            "the hourly curve must read the hourly inventory row"
        assert D.depth_coins(fc, "MILK", 0, 25, hour=hour) == sell_coins("MILK", inv, 25)
    # and the rows are the WALK's, not one snapshot repeated: a day boundary in
    # the hourly table must land on the day-start table's own inventory
    for d in (1,):
        for gi, item in enumerate(PRODUCTS):
            assert int(INV[d * 24, gi]) == int(fc.inventory_of(item, d)), (
                f"day {d} hour 0 {item}: {int(INV[d * 24, gi])} != the "
                f"day-start table's {int(fc.inventory_of(item, d))}")
    assert (INV != INV[0]).any(), \
        "the hourly inventory table must move (it is not one snapshot repeated)"


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} belief depth checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
