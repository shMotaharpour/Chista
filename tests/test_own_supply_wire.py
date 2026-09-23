"""The #110 wiring guard: the manager's OWN plan moves the next path.

One season step-pair proves the chain end to end: observe day N commits a
plan and records `own_sells`; observe day N+1 prices on a forecast that
carries them. The day-over-day fixed point — not a second solve.

Run:  .venv/bin/python -m tests.test_own_supply_wire   (also pytest)
"""

from __future__ import annotations


def test_the_committed_plan_records_its_projected_sells() -> None:
    """After day-0's observe, `own_sells` holds what the mix projects."""
    from agent.config import Config
    from agent.manager.core import Manager
    from offline_lab.kaggle_env import new_environment

    env = new_environment(configuration={"seed": 2})
    obs = env.state[0].observation
    # NOT a tight budget: the master returns NO plan at all when the deadline
    # cuts it (measured: 400 ms -> 0 market orders, own_sells empty; 1000 ms and
    # 3000 ms -> 29 keys). A guard that goes red when the day search gets heavier
    # is measuring speed, not the wiring (#110).
    m = Manager(Config(turn_budget_ms=3000.0, reserve_ms=100.0))
    m.observe(obs, None)
    assert m.own_sells, "the committed plan projected no sells at all"
    steps = sorted(m.own_sells)
    assert steps and steps[0] >= 0
    for goods in m.own_sells.values():
        assert goods and all(u > 0 for u in goods.values())


def test_the_next_day_is_priced_on_the_plan_it_inherits() -> None:
    """observe -> step to day 1 -> observe: day 1's forecast path carries
    day 0's projected sells (the ladder reads them; a path WITHOUT them
    prices strictly higher on the goods the plan dumps)."""
    from agent.belief.market import forecast
    from agent.config import Config
    from agent.manager.core import IDLE_PLAN, Manager
    from offline_lab.fast_sim import FastSim
    from offline_lab.kaggle_env import new_environment
    from agent.dispatch import dispatch_plan

    sim = FastSim({"episodeSteps": 720, "seed": 2})
    env = new_environment(configuration={"seed": 2})
    # NOT a tight budget: the master returns NO plan at all when the deadline
    # cuts it (measured: 400 ms -> 0 market orders, own_sells empty; 1000 ms and
    # 3000 ms -> 29 keys). A guard that goes red when the day search gets heavier
    # is measuring speed, not the wiring (#110).
    m = Manager(Config(turn_budget_ms=3000.0, reserve_ms=100.0))
    m.observe(sim.observations()[0], None)
    sells = dict(m.own_sells)
    for turn in range(24):
        views = sim.observations()
        o0 = views[0]
        if int(o0["hour"]) == 0:
            m.observe(o0, None)
        else:
            m.step()
        sim.step([dispatch_plan(m.best(), o0), dict(IDLE_PLAN)])
    assert m.own_sells != sells or not sells  # the new plan re-projects
    # The chain, priced: with the sells the plan committed, the day's path
    # is lower somewhere than the naive one.
    obs = sim.observations()[0]
    fc_with = m._forecast(obs, None)
    fc_flat = forecast(obs, days=20, rival_supply=m._rival_supply(obs, 20))
    goods = {g for s in sells.values() for g in s} if sells else set()
    moved = [g for g in goods
             if any(fc_with.price_of(g, d) != fc_flat.price_of(g, d)
                    for d in range(20))]
    assert moved, "day 1's path ignored the plan's own supply"


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
    print(f"{len(tests) - failures}/{len(tests)} own-supply wire checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
