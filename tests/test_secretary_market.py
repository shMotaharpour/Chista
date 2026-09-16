"""Secretary B (#15): the forecast, the shed guard, the queue, the layer.

Run:  .venv/bin/python -m tests.test_secretary_market

Every guard here is shown to FAIL before it is trusted (R007). The two
engine-level legs of the shed guard are the pair that matters: the same
scenario destroys product with the market layer off and destroys none
with it on, on the real interpreter (`world.fast_sim`).

Numbers this module asserts without re-measuring are named where they
appear and reproducible by `bench/bench_market_forecast.py`, which prints
the 20-seed table these bounds come from (R005).
"""

from __future__ import annotations

import time

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from agent.dispatch import dispatch_plan, market_at
from agent.market_layer import MarketLayer, from_env, _market_row
from secretary.inventory import (MAX_ORDERS_PER_TURN, Sale, ShedState,
                                 _assert_within_cap, market_queue,
                                 orders_by_hour, plan_sales, shed_state)
from secretary.market import (PRODUCTS, MarketForecast, forecast, shop_demand,
                              town_deltas)
from world.fast_sim import FastSim

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
I0 = {item: float(K.MARKET_PARAMS[item]["I0"]) for item in PRODUCTS}


def _sim(seed: int = 0, weeds: float = 0.0) -> FastSim:
    return FastSim({"episodeSteps": 720, "seed": seed,
                    "weedSpawnChance": weeds})


def _realised(seed: int, until_day: int) -> list[dict]:
    """Day-start market inventories of a PASS-vs-PASS episode."""
    sim = _sim(seed)
    rows = []
    while not sim.done and len(rows) <= until_day:
        obs = sim.observations()[0]
        if int(obs["hour"]) == 0:
            rows.append(dict(obs["market"]["inventory"]))
        sim.step([PASS, PASS])
    return rows


# --- the cadence model (class B: the forecast must match the engine) -----

def test_cadence_model_reproduces_the_engine_exactly_to_day_3():
    """Days 0..3 are exact: the first unlock lands at the start of day 3.

    Nothing is random before it: shops unlock at the END of day 2
    (`next_day % townShopUnlockInterval == 0`), so the day-3 day-start row
    precedes the first shop consumption. This is the leg that proves the
    cadence (shops every 4 turns, centre every 24, single-product shops at
    2x) was not guessed.
    """
    sim = _sim(7)
    fc = forecast(sim.observations()[0], days=30, unlock_policy="none")
    realised = _realised(7, until_day=3)
    for day in range(4):
        for item in PRODUCTS:
            assert fc.inventory_of(item, day) == realised[day][item], (
                day, item, fc.inventory_of(item, day), realised[day][item])
        real_prices = {i: K.market_price(i, realised[day][i]) for i in PRODUCTS}
        assert fc.prices[day] == tuple(real_prices[i] for i in PRODUCTS), day


def test_day_10_error_stays_inside_the_acceptance_bound():
    """Worst inventory error over the bench's 20 seeds, per policy.

    Measured (`bench/bench_market_forecast.py --error --seeds 20`,
    PASS-vs-PASS, no weeds, seeds 0..19): worst |pred - realised| / I0 at
    day 10 is 1.32 % for `none` and 1.14 % for `mean`; at day 3 both are
    0.000 %. The issue's bound is 5 % at 3 days and 15 % at 10. Asserted at
    2 % here over seeds 0..2 with the same probe, so this is the same
    quantity with a stated margin, not a second, looser claim.
    """
    for seed in (0, 1, 2):
        sim = _sim(seed)
        fcs = {p: forecast(sim.observations()[0], days=30, unlock_policy=p)
               for p in ("none", "mean")}
        realised = _realised(seed, until_day=10)
        for policy, fc in fcs.items():
            for day, horizon in ((3, 3), (10, 10)):
                worst = max(abs(fc.inventory_of(i, horizon)
                                - realised[horizon][i]) / I0[i]
                            for i in PRODUCTS)
                assert worst <= 0.02, (seed, policy, horizon, worst)


def test_unlock_policy_none_is_biased_toward_higher_prices():
    """The named bias, asserted in its direction.

    Missing unlocks under-counts town consumption, so the forecast holds
    MORE inventory than reality and prices it lower: the sign is part of
    the contract, not an accident (the docstring says so).
    """
    seed = 1
    sim = _sim(seed)
    fc = forecast(sim.observations()[0], days=30, unlock_policy="none")
    realised = _realised(seed, until_day=20)
    worse = 0
    for day in range(10, 21):
        for item in PRODUCTS:
            if fc.inventory_of(item, day) < realised[day][item]:
                worse += 1
    assert worse == 0, f"{worse} item-days dipped below the realised inventory"


def test_price_is_the_engine_function_not_a_copy():
    """Patch `K.market_price` and watch every forecast price move with it."""
    sim = _sim(0)
    obs = sim.observations()[0]
    real = K.market_price
    try:
        K.market_price = lambda item, inventory, params=None: 777  # noqa: ARG005
        fc = forecast(obs, days=2)
    finally:
        K.market_price = real
    assert fc.prices == ((777,) * len(PRODUCTS), (777,) * len(PRODUCTS))


def test_our_own_sells_are_modelled():
    """`our_sells` is the farm's own price impact (F033/F036)."""
    sim = _sim(0)
    obs = sim.observations()[0]
    base = forecast(obs, days=4)
    selling = forecast(obs, days=4, our_sells={48: {"WHEAT": 200}})
    assert selling.inventory_of("WHEAT", 3) > base.inventory_of("WHEAT", 3)
    assert selling.price_of("WHEAT", 3) <= base.price_of("WHEAT", 3)


def test_town_deltas_counts_a_single_product_shop_twice():
    """PET_CAFE demands carrots only, so its instance consumes 2 (F037)."""
    assert town_deltas(["PET_CAFE"], step=4, shop_interval=4,
                       center_interval=24) == {"CARROT": 2}
    assert shop_demand(["PET_CAFE", "PET_CAFE"]) == {"CARROT": 4}
    assert town_deltas([], 0, 4, 24)["WHEAT"] == 1     # town centre only


# --- the shed guard, on the real interpreter (R007 pair) -----------------

def _run_a_full_day(destroyed_scenario: bool) -> dict:
    """Fill the shed to 95 with 20 wheat in a bag, then play out the day.

    `destroyed_scenario=True` runs the day with NO market layer (the control
    leg): the nightly drop then destroys what does not fit.
    """
    sim = _sim(0)
    while int(sim.observations()[0]["day"]) < 1:
        sim.step([PASS, PASS])
    priv = sim.state[0].observation.private
    priv["shed"]["WHEAT"] = 95
    priv["inventories"] = [{"WHEAT": 20}]
    start_total = 95 + 20                       # everything that wants to fit
    money_before = float(sim.observations()[0]["farms"][0]["money"])
    layer = None if destroyed_scenario else MarketLayer("spread")
    sold = 0
    while int(sim.observations()[0]["hour"]) != 23:
        obs = sim.observations()[0]
        action = dict(PASS) if layer is None else layer.attach(dict(PASS), obs)
        sold += sum(int(o[2]) for o in action.get("market", [])
                    if o and o[0] == "SELL")
        sim.step([action, PASS])
    sim.step([PASS, PASS])                      # the drop happens here
    after = sum(sim.observations()[0]["private"]["shed"].values())
    return {"before": start_total, "after": after, "sold": sold,
            "destroyed": start_total - after - sold,
            "money": float(sim.observations()[0]["farms"][0]["money"])
                     - money_before}


def test_the_shed_guard_prevents_a_real_destruction_event():
    """With the market layer on: nothing is destroyed, and the stock is sold."""
    out = _run_a_full_day(destroyed_scenario=False)
    assert out["sold"] >= 15, out
    assert out["destroyed"] == 0, out
    assert out["money"] > 0, out


def test_without_the_market_layer_the_same_day_destroys_product():
    """R007: the control leg, and the reason the guard exists at all."""
    out = _run_a_full_day(destroyed_scenario=True)
    assert out["destroyed"] == 15, out


def test_shed_state_reads_the_bags_and_blocks_buys_at_the_cap():
    obs = {"private": {"shed": {"WHEAT": 99, "MELON": 1},
                       "inventories": [{"CARROT": 3}, {}]}}
    state = shed_state(obs)
    assert (state.held, state.carried, state.room) == (100, 3, 0)
    assert state.buys_blocked                        # F043's second half
    assert state.night_overflow() == 3               # nothing fits
    assert state.sellable() == {"WHEAT": 99, "MELON": 1}


# --- the schedule and the queue ------------------------------------------

def _forecast_stub(prices: dict, days: int = 30):
    class _F:
        def price_of(self, item, day):
            return prices.get(item, 25)

        @property
        def days(self):
            return days
    return _F()


def test_shed_guard_sells_the_overflow_plus_its_margin():
    sales = plan_sales({"WHEAT": 120}, _forecast_stub({}), day=5, hour=0,
                       capacity_room=0, harvest_expected=0)
    assert sum(s.units for s in sales) >= 25, sales       # 20 + margin 5
    assert all(s.reason == "shed-guard" for s in sales)


def test_season_end_liquidates_everything():
    sales = plan_sales({"WHEAT": 40, "MELON": 7}, _forecast_stub({}), day=29,
                       hour=12, capacity_room=53, end_day=29)
    assert sum(s.units for s in sales) == 47
    assert {s.reason for s in sales} == {"season-end"}
    assert max(s.hour for s in sales) <= 23


def test_a_basket_is_spread_not_dumped():
    """F036: one big basket walks the ladder down; split it across turns."""
    sales = plan_sales({"WHEAT": 50}, _forecast_stub({}), day=3, hour=0,
                       capacity_room=50)
    assert len(sales) > 1, sales
    assert max(s.units for s in sales) < 50, sales
    assert sum(s.units for s in sales) == 50
    assert sorted({s.hour for s in sales}) == list(range(len(sales)))


def test_a_full_shed_with_empty_forecast_never_raises():
    assert plan_sales({}, _forecast_stub({}), day=0, hour=0,
                      capacity_room=100) == ()


def test_sell_orders_are_capped_per_turn_and_the_guard_fires():
    """F031: one order per item per turn cannot reach 10 - and an 11th raises."""
    queue = orders_by_hour([Sale(hour=0, item=item, units=1, reason="x")
                            for item in PRODUCTS], hours=24)
    assert len(queue[0]) == len(PRODUCTS) <= MAX_ORDERS_PER_TURN
    _assert_within_cap(queue)                        # legal: no raise
    try:
        _assert_within_cap([[["SELL", "WHEAT", 1]] * 11])
    except ValueError as exc:
        assert "F031" in str(exc)
    else:                                            # pragma: no cover
        raise AssertionError("the F031 cap guard did not fire on 11 orders")


def test_dispatch_slices_a_per_hour_queue_and_still_takes_the_flat_list():
    queue = [[] for _ in range(24)]
    queue[3] = [["SELL", "WHEAT", 2]]
    assert market_at(queue, 3) == [["SELL", "WHEAT", 2]]
    assert market_at(queue, 4) == []
    flat = [["SELL", "WHEAT", 2]]
    assert market_at(flat, 0) == flat and market_at(flat, 5) == []
    plan = {"units": [["PASS"]], "market": queue}
    for hour in (0, 3, 23):
        action = dispatch_plan(plan, {"hour": hour, "player": 0,
                                      "farms": [{"hands": []}, {}]})
        assert action["market"] == (queue[hour] if hour == 3 else [])


def test_market_queue_reads_the_observation_and_plans_a_day():
    sim = _sim(0)
    obs = sim.observations()[0]
    queue = market_queue(obs)
    assert len(queue) == 24
    assert all(len(row) <= MAX_ORDERS_PER_TURN for row in queue)


# --- the layer, and its contract with the ladder -------------------------

def test_the_layer_returns_the_rung_action_untouched_when_it_fails():
    """The ladder's contract outranks the market layer (R007: seen to fail)."""
    sim = _sim(0)
    obs = sim.observations()[0]
    rung_action = {"farmer": ["PASS"], "hands": [],
                   "market": [["SELL", "WHEAT", 4]]}
    layer = MarketLayer("spread")

    def _boom(_obs):
        raise RuntimeError("planned failure")
    layer.plan_day = _boom
    out = layer.attach(rung_action, obs)
    assert out == rung_action, out
    assert "planned failure" in layer.last.get("error", "")


def test_the_layer_replaces_the_rungs_sells_and_keeps_its_buys():
    sim = _sim(0)
    obs = sim.observations()[0]
    layer = MarketLayer("dump")
    action = {"farmer": ["PASS"], "hands": [],
              "market": [["SELL", "WHEAT", 4], ["BUY_SEED", "WHEAT", 1]]}
    out = layer.attach(action, obs)
    kinds = [order[0] for order in out["market"]]
    assert kinds == ["BUY_SEED"], out      # the rung's SELL is superseded


def test_dump_mode_sells_the_whole_shed_at_hour_zero():
    sim = _sim(0)
    while int(sim.observations()[0]["day"]) < 1:
        sim.step([PASS, PASS])
    sim.state[0].observation.private["shed"]["WHEAT"] = 30
    sim.state[0].observation.private["shed"]["MELON"] = 4
    obs = sim.observations()[0]
    queue = MarketLayer("dump").plan_day(obs)
    assert queue[0] == [["SELL", "MELON", 4], ["SELL", "WHEAT", 30]]
    assert all(row == [] for row in queue[1:])


def test_mode_switch_reads_the_environment():
    import os
    for value, want in (("spread", "spread"), ("dump", "dump"), ("", None),
                        ("0", None), ("nonsense", None)):
        os.environ["CHISTA_MARKET"] = value
        layer = from_env()
        assert (layer.mode if layer else None) == want, value
    del os.environ["CHISTA_MARKET"]


def test_the_layer_costs_under_ten_milliseconds_at_p99():
    """The issue's budget, measured on this box (warm, one day plan each)."""
    sim = _sim(0)
    obs = sim.observations()[0]
    layer = MarketLayer("spread")
    for _ in range(5):                       # warm the imports and the caches
        layer.plan_day(obs)
    samples = []
    for _ in range(200):
        t0 = time.perf_counter()
        layer.plan_day(obs)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    p99 = samples[int(0.99 * (len(samples) - 1))]
    assert p99 <= 10.0, f"p99 {p99:.2f} ms"


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:          # noqa: BLE001 - test runner
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {name}")
    if failures:
        print(f"{failures} test(s) failed")
        return 1
    print("all secretary-market tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
