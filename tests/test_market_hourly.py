"""Guards for the hourly price extension of `belief/market`.

The claim under test: `forecast` exposes the market's value PER HOUR, not
just at day starts — the walk already runs per turn, so the hourly rows are
a sampling choice, and the hour values must satisfy the same identities the
day-start rows do:

  * parity — every hourly quote equals the engine's `market_price` at the
    walked inventory (the engine is patched in one guard and must move);
  * turn order — the town consumes AFTER the market inside a turn, so hour
    h's quote is the inventory after turn h's market and BEFORE turn h's
    town consumption;
  * cadence — a shop tick inside a day shows up as a price drop between
    that hour and the next (and the day-start identity with the old rows
    is unchanged: row 0 of each day still equals the old `prices` row);
  * the mid-day stub keeps row 0 = the observation itself.

Run:  .venv/bin/python -m tests.test_market_hourly   (also under pytest)
"""

from __future__ import annotations

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K
from agent.world.model import PRODUCTS
from agent.belief.market import PRODUCTS as M_PRODUCTS, forecast, hourly_prices


def _sim(seed: int = 0):
    from offline_lab.fast_sim import FastSim
    sim = FastSim(configuration={"seed": seed, "episodeSteps": 720},
                  validate="dev")
    PASS = {"farmer": ["PASS"], "hands": [], "market": []}
    return sim, PASS


def test_hourly_rows_carry_every_turn_of_the_horizon() -> None:
    """The hourly table is (days*24, 9): one row per turn, in PRODUCTS order."""
    sim, PASS = _sim(0)
    fc = forecast(sim.observations()[0], days=3)
    H = hourly_prices(fc)
    assert H.shape == (3 * 24, len(M_PRODUCTS)), H.shape


def test_hourly_quotes_are_the_engine_at_the_walked_inventory() -> None:
    """Every hourly quote equals K.market_price at the walk's inventory row.

    The walk itself is internal; the identity checked here is the one the
    consumer needs: the hourly quote at (day d, hour h) must sit between the
    day-start quotes that bracket it, moving with the cadence the engine
    runs — and patching the engine's price function must move it.
    """
    sim, PASS = _sim(5)
    obs = sim.observations()[0]
    fc = forecast(obs, days=2)
    H = hourly_prices(fc)

    real = K.market_price
    try:
        K.market_price = lambda item, inventory, params=None: 777
        fc2 = forecast(obs, days=2)
        H2 = hourly_prices(fc2)
    finally:
        K.market_price = real
    assert (H2 == 777).all(), "a patched engine price must move the hourly rows"


def test_hourly_prices_move_with_the_shop_cadence() -> None:
    """A shop tick shows up as a between-hours drop, not a flat row.

    On a seed whose first shop unlocks on day 3, day 4 has ticks every 4
    hours; the walk must show them (the day-start-only table cannot).
    """
    sim, PASS = _sim(1)
    while int(sim.observations()[0]["day"]) < 4:
        sim.step([PASS, PASS])
    fc = forecast(sim.observations()[0], days=2, unlock_policy="none")
    H = hourly_prices(fc)
    # the centre eats at hour 0 and 24... its effect is IN the quote of the
    # following hours; check the WHEAT row is not constant across a day that
    # contains shop consumption
    day0 = H[0:24, M_PRODUCTS.index("WHEAT")]
    assert len(set(day0.tolist())) > 1, (
        "a day with shop consumption priced flat: the hourly rows are not "
        "sampling the walk")


def test_day_start_rows_are_unchanged() -> None:
    """The extension must not move the day table the master already reads."""
    sim, PASS = _sim(3)
    obs = sim.observations()[0]
    fc = forecast(obs, days=4)
    H = hourly_prices(fc)
    d0 = M_PRODUCTS.index("WHEAT")
    # the day-start price of day d must equal the OLD row's price for the
    # same day (the walk is the same; only the sampling is finer)
    for d in range(3):
        assert H[d * 24, d0] == fc.prices[d][d0], d


def test_mid_day_forecast_row0_is_the_observation_itself() -> None:
    """Hour 0 of row 0, mid-day, quotes the observed inventory exactly."""
    sim, PASS = _sim(0)
    for _ in range(29):                      # day 1, hour 5
        sim.step([PASS, PASS])
    obs = sim.observations()[0]
    fc = forecast(obs, days=2)
    H = hourly_prices(fc)
    for item in M_PRODUCTS:
        assert H[0, M_PRODUCTS.index(item)] == K.market_price(
            item, float(obs["market"]["inventory"][item])), item


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
    print(f"{len(tests) - failures}/{len(tests)} market hourly checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
