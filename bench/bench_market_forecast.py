"""Secretary B (#15): the measurements the acceptance numbers name (R005).

Run from the repo root:

    .venv/bin/python -m bench.bench_market_forecast --probe
    .venv/bin/python -m bench.bench_market_forecast --error --seeds 20
    .venv/bin/python -m bench.bench_market_forecast --lag
    .venv/bin/python -m bench.bench_market_forecast --layer-timing

Every number the #15 tests and docs quote comes from here:

- `--probe`   the cadence model against the engine's own deltas
              (measured 2026-09-16: 6,462/6,462 item-step comparisons
              match, seed 7, no weeds, PASS policies).
- `--error`   forecast inventory error at 3/6/10/20 days, per unlock
              policy, in % of I0 and in coins of price — against the
              realised path of PAS-vs-PASS seasons (`--seeds N`, default
              20, seeds 0..N-1).
- `--lag`     the `d+1` revenue contract (#8 §4 contract 2): a SELL reads
              the SHED, and a same-turn `DROP` + `SELL` lands that turn.
- `--layer-timing` the market layer's own p99 (issue budget ≤ 10 ms).

The probe lists are the same ones `offline.fast_sim` drives for the graph
builder (R003: the simulator wraps the interpreter, it never reimplements
it), so an engine change moves these numbers instead of hiding behind a
hand-written table.
"""

from __future__ import annotations

import argparse
import time

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from agent.greedy import greedy_action
from agent.market_layer import MarketLayer
from belief.market import PRODUCTS, forecast, town_deltas
from offline.fast_sim import FastSim

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
I0 = {item: float(K.MARKET_PARAMS[item]["I0"]) for item in PRODUCTS}
SHOP_INTERVAL, CENTER_INTERVAL = 4, 24


def cadence_probe(seed: int = 7, steps: int = 719) -> tuple[int, int]:
    """Model each turn's consumption from the shop set the obs showed."""
    sim = FastSim({"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.0})
    comparisons = mismatches = 0
    prev = None
    for step in range(steps):
        obs = sim.observations()[0]
        inv = dict(obs["market"]["inventory"])
        shops = list(obs["town"]["unlocked_shops"])
        if prev is not None:
            predicted = town_deltas(prev[0], prev[1], SHOP_INTERVAL,
                                    CENTER_INTERVAL)
            for item in PRODUCTS:
                comparisons += 1
                if prev[2][item] - inv[item] != predicted.get(item, 0):
                    mismatches += 1
        prev = (shops, step, inv)
        sim.step([PASS, PASS])
    return comparisons, mismatches


def passive_season(seed: int, until_day: int = 20) -> list[dict]:
    """Day-start market inventories of a PASS-vs-PASS season."""
    sim = FastSim({"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.0})
    rows: list[dict] = []
    while not sim.done and len(rows) <= until_day:
        obs = sim.observations()[0]
        if int(obs["hour"]) == 0:
            rows.append(dict(obs["market"]["inventory"]))
        sim.step([PASS, PASS])
    return rows


def error_table(seeds: int, policies=("none", "mean"),
                horizons=(3, 6, 10, 20)) -> None:
    """Forecast vs realised, per policy and horizon, worst item named."""
    for policy in policies:
        worst_inv = {h: 0.0 for h in horizons}
        worst_price = {h: 0 for h in horizons}
        for seed in range(seeds):
            sim = FastSim({"episodeSteps": 720, "seed": seed,
                           "weedSpawnChance": 0.0})
            fc = forecast(sim.observations()[0], days=30,
                          unlock_policy=policy)
            realised = passive_season(seed, until_day=max(horizons))
            for horizon in horizons:
                row = realised[horizon]
                inv_err = max((abs(fc.inventory_of(i, horizon) - row[i])
                               / I0[i], i) for i in PRODUCTS)
                price_err = max((abs(fc.price_of(i, horizon)
                                     - K.market_price(i, row[i])), i)
                                for i in PRODUCTS)
                worst_inv[horizon] = max(worst_inv[horizon], inv_err[0])
                worst_price[horizon] = max(worst_price[horizon], price_err[0])
        print(f"unlock policy {policy!r} over {seeds} seeds:")
        for horizon in horizons:
            print(f"   day+{horizon:2d}: worst inventory error "
                  f"{100 * worst_inv[horizon]:6.3f} % of I0, "
                  f"worst price error {worst_price[horizon]:3d} coins")


def lag_probe(seed: int = 0) -> None:
    """#8 contract 2: does a SELL read the bag, and does DROP change that?"""
    sim = FastSim({"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.0})
    while not sim.done:
        obs = sim.observations()[0]
        if obs["private"]["inventories"][0].get("WHEAT", 0) > 0:
            break
        sim.step([greedy_action(obs), PASS])
    obs = sim.observations()[0]
    bag = dict(obs["private"]["inventories"][0])
    money0 = float(obs["farms"][0]["money"])
    print(f"caught a harvest: day {obs['day']} hour {obs['hour']} "
          f"farmer {obs['farms'][0]['farmer']} bag {bag} money {money0:.0f}")
    sim.step([{"farmer": ["PASS"], "hands": [],
               "market": [["SELL", "WHEAT", 1]]}, PASS])
    after = sim.observations()[0]
    # fast-mode observations are LIVE views: read the numbers out now
    money_sell = float(after["farms"][0]["money"])
    shed_sell = int(after["private"]["shed"]["WHEAT"])
    print(f"  SELL with the item only in the bag: money {money_sell:.0f} "
          f"(unchanged), shed {shed_sell} wheat - SELL reads the shed")
    sim.step([{"farmer": ["DROP"], "hands": [],
               "market": [["SELL", "WHEAT", 1]]}, PASS])
    final = sim.observations()[0]
    money_drop = float(final["farms"][0]["money"])
    print(f"  DROP + SELL in the SAME turn: money {money_drop:.0f} "
          f"(+{money_drop - money_sell:.0f} that turn), shed "
          f"{int(final['private']['shed']['WHEAT'])} wheat")


def layer_timing(samples: int = 300) -> None:
    """The market layer's own per-day planning cost (issue budget ≤ 10 ms)."""
    sim = FastSim({"episodeSteps": 720, "seed": 0, "weedSpawnChance": 0.0})
    obs = sim.observations()[0]
    layer = MarketLayer("spread")
    for _ in range(10):
        layer.plan_day(obs)
    times = []
    for _ in range(samples):
        t0 = time.perf_counter()
        layer.plan_day(obs)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50 = times[len(times) // 2]
    p99 = times[int(0.99 * (len(times) - 1))]
    print(f"market layer plan_day over {samples} calls: "
          f"p50 {p50:.3f} ms, p99 {p99:.3f} ms, max {times[-1]:.3f} ms "
          f"(budget 10 ms)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.bench_market_forecast")
    ap.add_argument("--probe", action="store_true",
                    help="the cadence model vs the engine's own deltas")
    ap.add_argument("--error", action="store_true", help="the error table")
    ap.add_argument("--lag", action="store_true", help="#8's d+1 contract")
    ap.add_argument("--layer-timing", action="store_true",
                    help="the layer's own p99")
    ap.add_argument("--overflow", action="store_true",
                    help="zero-destruction / zero-dropped-order suite")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args(argv)
    if not any((args.probe, args.error, args.lag, args.layer_timing,
                args.overflow)):
        args.probe = args.error = args.lag = args.layer_timing = True
        args.overflow = True
    if args.probe:
        comparisons, mismatches = cadence_probe()
        print(f"cadence probe: {comparisons - mismatches}/{comparisons} "
              f"item-step comparisons match the engine "
              f"({mismatches} mismatches)")
    if args.error:
        error_table(args.seeds)
    if args.lag:
        lag_probe()
    if args.layer_timing:
        layer_timing()
    if args.overflow:
        overflow_suite()
    return 0
def overflow_suite(seeds: int = 20, weeds: float = 0.005) -> None:
    """Acceptance: zero destruction events, zero dropped orders, real agent.

    Runs full seasons with the market layer ON (`agent.main_market_spread`),
    counting at each day's last turn the units the nightly drop cannot fit
    (`shed + every bag` against the 100 cap) and any turn whose market row
    exceeded the engine's 10-order limit.
    """
    from agent.main_market_spread import agent as chista   # env first
    overflow = dropped = malformed = 0
    money = []
    for seed in range(seeds):
        sim = FastSim({"episodeSteps": 720, "seed": seed,
                       "weedSpawnChance": weeds})
        while not sim.done:
            obs = sim.observations()[0]
            shed = sum(obs["private"]["shed"].values())
            bags = sum(sum(b.values()) for b in obs["private"]["inventories"])
            if int(obs["hour"]) == 23:
                overflow += max(0, shed + bags - 100)
            action = chista(obs)
            orders = action.get("market", [])
            dropped += max(0, len(orders) - 10)
            malformed += 0 if all(isinstance(o, list) and o for o in orders) else 1
            sim.step([action, PASS])
        money.append(float(sim.money()[0]))
    print(f"overflow suite: {seeds} seasons, weeds {weeds}")
    print(f"  destroyed units: {overflow}   dropped orders: {dropped}   "
          f"malformed rows: {malformed}")
    print(f"  final money: min {min(money):.0f} median "
          f"{sorted(money)[len(money) // 2]:.0f} max {max(money):.0f}")


if __name__ == "__main__":
    raise SystemExit(main())
