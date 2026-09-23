"""The #110 own-supply wiring: one fixed-point pass.

`equilibrate` solves, the chosen mix's `produce` is the plan's own
projected supply, and that supply moves the price path the NEXT
objective is priced on. The pass lives here (in the manager's loop
between days) rather than inside `equilibrate`, so the master stays a
pure function of the forecast it is handed — and the second pass is
counted and compared, not silently mixed into the first.

Run:  .venv/bin/python -m tests.test_plan_supply_iteration   (also pytest)
"""

from __future__ import annotations

import numpy as np

from agent.belief.market import forecast
from agent.planner.colgen import Column
from agent.planner.plan_supply import plan_supply_sells, plan_supply_sells_from_pool
from offline_lab.kaggle_env import new_environment


def test_plan_supply_sells_from_pool_matches_columns() -> None:
    """Two warm columns at 50/50: the projected sells are the average of
    their per-day produce, indexed at absolute steps."""
    days = 5
    c0 = Column(cls=0, cost=np.zeros((days, 11)), spend=np.zeros(days),
                earn=np.zeros(days), revenue=0.0,
                produce=_produce(days, {"MELON": 8}),
                cls_key=(1, 0), key=("a",))
    c1 = Column(cls=0, cost=np.zeros((days, 11)), spend=np.zeros(days),
                earn=np.zeros(days), revenue=0.0,
                produce=_produce(days, {"MELON": 4}),
                cls_key=(1, 0), key=("b",))
    sells = plan_supply_sells_from_pool([c0, c1], np.array([0.5, 0.5]),
                                        days, first_day=3)
    assert all(sells[(3 + d) * 24]["MELON"] == 6 for d in range(days)), sells


def _produce(days: int, per_day: dict[str, int]) -> np.ndarray:
    from agent.world.model import RESOURCE_ID
    out = np.zeros((days, len(RESOURCE_ID)))
    for name, units in per_day.items():
        out[:, RESOURCE_ID[name]] = units
    return out


def test_the_second_pass_moves_the_price_path_down() -> None:
    """The property the iteration is for: with the plan's own sells in the
    forecast, the NEXT objective is priced on a lower path — measured on
    a real forecast, MELON sold from day 10."""
    env = new_environment()
    obs = env.state[0].observation
    sells = plan_supply_sells_from_pool(
        [_warm_column()], np.array([1.0]), days=20, first_day=0)
    assert sells, "the warm column produced nothing to sell"
    fc_first = forecast(obs, days=20, rival_supply=np.zeros((20, 9)))
    fc_second = forecast(obs, days=20, our_sells=sells,
                         rival_supply=np.zeros((20, 9)))
    good = next(iter(next(iter(sells.values()))))
    moved = [(d, fc_second.price_of(good, d), fc_first.price_of(good, d))
             for d in sorted({d for d in sells})]
    assert any(a < b for _d, a, b in moved), moved


def _warm_column() -> Column:
    from agent.world.model import RESOURCE_ID
    days = 20
    produce = np.zeros((days, len(RESOURCE_ID)))
    produce[10:13, RESOURCE_ID["MELON"]] = 7.0
    return Column(cls=0, cost=np.zeros((days, 11)), spend=np.zeros(days),
                  earn=np.zeros(days), revenue=0.0, produce=produce,
                  cls_key=(1, 0), key=("warm",))


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
    print(f"{len(tests) - failures}/{len(tests)} plan-supply iteration "
          "checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_the_projected_sells_land_on_the_plans_own_hour() -> None:
    """A sell at hour 14 supplies the market at hour 14, not at hour 0.

    The walk accepts per-turn sells (`our_sells` is `{absolute_step: {item:
    units}}`), so pricing every day's projection at hour 0 made it blind to our
    own intra-day supply: the hours after a sale are quoted on a market we never
    sold into (#110).
    """
    import numpy as np
    from agent.planner.plan_supply import plan_supply_sells
    from agent.world.model import N_RESOURCE, RESOURCE_ID
    produce = np.zeros((1, 2, N_RESOURCE), dtype=np.float64)
    produce[0, 1, RESOURCE_ID["CARROT"]] = 3.0
    lam = np.array([1.0])
    plain = plan_supply_sells(produce, lam, 2, 5)
    assert plain == {6 * 24: {"CARROT": 3}}, plain
    dated = plan_supply_sells(produce, lam, 2, 5, hours={"CARROT": 14})
    assert dated == {6 * 24 + 14: {"CARROT": 3}}, dated
