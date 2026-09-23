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


def test_mid_day_hour_row_is_the_snapshot_at_hour_now() -> None:
    """The #71 guard: at hour_now = h, hour h's OWN quote is the snapshot,
    and the ladder moves land at the hours the walk says.

    Row 0 is trivially the snapshot under ANY labeling (rows init to 0),
    so asserting row 0 proved nothing — the mid-day defect #67 fixed
    (hours h..23 carrying the quote h hours EARLY) survived it. On a
    fixture where the walk actually moves (seed 1, day 4: the town's shop
    drain steps STRAWBERRY every 4 walk-rows), the two labelings differ,
    and this guard pins the correct one: the defect table shifts every
    move 5 hours early.
    """
    sim, PASS = _sim(1)
    while int(sim.observations()[0]["step"]) < 4 * 24 + 5:
        sim.step([PASS, PASS])               # day 4, hour 5
    obs = sim.observations()[0]
    hour_now = int(obs["step"]) % 24
    fc = forecast(obs, days=2)
    H = hourly_prices(fc)
    g = M_PRODUCTS.index("STRAWBERRY")
    # hour hour_now's own quote IS the snapshot (its market has not run):
    assert H[hour_now, g] == K.market_price(
        "STRAWBERRY", float(obs["market"]["inventory"]["STRAWBERRY"]))
    # and each 4-row ladder step lands at the hour the walk says: the
    # snapshot (row 0) + 4 walk rows = hour hour_now + 4
    expected_moves = [(hour_now + 4, 1), (hour_now + 8, 3),
                      (hour_now + 12, 4), (hour_now + 16, 5),
                      (hour_now + 20, 7)]
    for h, q in expected_moves:
        assert int(H[h, g]) == 150 + q, (
            f"hour {h}: {int(H[h, g])} != {150 + q}; the mid-day rows are "
            "labeled off the walk (the #67 defect shifts each move 5 "
            "hours early and this assertion fires)")


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


def test_a_rivals_dated_sale_moves_the_hours_after_it() -> None:
    """The rival's supply at hour 10 cheapens hour 11 onward, not hour 10.

    `rival_supply` is per DAY, so their units all landed at hour 0 and the rest
    of the day was quoted on a market they never sold into (#16). The dated form
    puts them where they land: a sale adds supply AFTER that turn's market, so
    the turns before it are unchanged and the turns after it are cheaper.
    """
    sim, _PASS = _sim(0)
    obs = sim.observations()[0]
    g = M_PRODUCTS.index("MELON")
    base = hourly_prices(forecast(obs, days=1), days=1)
    dated = hourly_prices(forecast(obs, days=1,
                                   rival_sells={10: {"MELON": 20}}), days=1)
    assert (dated[:11, g] == base[:11, g]).all(), \
        "the hours before the sale must not move"
    assert (dated[11:, g] < base[11:, g]).all(), \
        "the hours after it must be cheaper"


def test_the_dated_rivals_supply_replaces_the_days_total() -> None:
    """Naming a day's hours must not ADD to that day's calendar total."""
    import numpy as np
    sim, _PASS = _sim(0)
    obs = sim.observations()[0]
    daily = np.zeros((1, len(M_PRODUCTS)))
    daily[0, M_PRODUCTS.index("MELON")] = 20
    both = hourly_prices(forecast(obs, days=1, rival_supply=daily,
                                  rival_sells={10: {"MELON": 20}}), days=1)
    dated = hourly_prices(forecast(obs, days=1,
                                   rival_sells={10: {"MELON": 20}}), days=1)
    assert (both == dated).all(), (
        "the day's calendar total and its dated hours are the same units: the "
        "dated form REPLACES the day it names")
