"""The own-supply half of #110: the master's chosen plan shifts the price
path its own objective is priced on — one fixed-point iteration.

`_product_price_path` today builds the forecast from the observation
(town drain + rival calendar) and prices every plan at that path. The
plan's OWN projected sells (`Column.produce`, already `(days,
N_RESOURCE)`) never enter, so a 20-melon plan is priced at the
no-melon-sold ladder and the objective overstates it by the ladder's
move (measured: 7 melons/day on days 10-12 moves MELON 272 → 249 by
day 13).

`plan_supply_sells(produce, lam, days)` converts the chosen mix into the
`our_sells` shape `forecast` already consumes, and
`_product_price_path` re-prices once when a caller passes it.

Run:  .venv/bin/python -m tests.test_plan_supply_pricing   (also pytest)
"""

from __future__ import annotations

import numpy as np

from agent.belief.market import PRODUCTS, forecast
from offline_lab.kaggle_env import new_environment


from agent.planner.plan_supply import plan_supply_sells


def test_plan_supply_sells_shapes_the_forecast_input() -> None:
    """The converter's output is exactly `forecast(our_sells=)`'s shape.

    `produce` arrives in RESOURCE space (n, days, N_RESOURCE=18); the
    melon column is RESOURCE_ID['MELON']=13."""
    from agent.world.model import RESOURCE_ID
    melon = RESOURCE_ID["MELON"]
    produce = np.zeros((2, 5, 18))
    produce[0, :5, melon] = 10.0        # 10 melons/day, column 0
    produce[1, :5, melon] = 4.0         # column 1
    sells = plan_supply_sells(produce, np.array([0.5, 0.5]), 5, 0)
    steps = [d * 24 for d in range(5)]
    assert set(sells) == set(steps)
    assert all(sells[s] == {"MELON": 7} for s in steps), sells


def test_plan_aware_forecast_prices_the_ladder_move() -> None:
    """Selling 7 melons/day on days 10-12 must depress day-11+ prices
    against the untouched path — the ladder walks down with supply."""
    env = new_environment()
    obs = env.state[0].observation
    aware = forecast(obs, days=20,
                     our_sells=_plan_sells((10, 11, 12), "MELON", 7),
                     rival_supply=np.zeros((20, 9)))
    naive = forecast(obs, days=20, rival_supply=np.zeros((20, 9)))
    moved = [(d, aware.price_of("MELON", d), naive.price_of("MELON", d))
             for d in (11, 12, 13)]
    assert any(a < b for _d, a, b in moved), (
        f"the plan's own sells left the ladder unmoved: {moved}")


def test_before_the_first_sell_the_paths_agree() -> None:
    """Conditioning must not move days BEFORE the plan sells anything."""
    env = new_environment()
    obs = env.state[0].observation
    aware = forecast(obs, days=20,
                     our_sells=_plan_sells((10, 11, 12), "MELON", 7),
                     rival_supply=np.zeros((20, 9)))
    naive = forecast(obs, days=20, rival_supply=np.zeros((20, 9)))
    for d in (0, 3, 9):
        assert aware.price_of("MELON", d) == naive.price_of("MELON", d), (
            f"day {d} moved without any sell before it")


def _plan_sells(days: tuple[int, ...], item: str, per_day: int) -> dict:
    return {f"{d * 24}": {item: per_day} for d in days}


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
    print(f"{len(tests) - failures}/{len(tests)} plan-supply checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
