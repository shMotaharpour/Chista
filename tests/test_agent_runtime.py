"""agent spine tests: entry point, dispatch, deadline, fallback ladder.

Run:  .venv/bin/python -m tests.test_agent_runtime

Contracts under test:
- The entry point returns a shape-valid action dict for any input — a
  real observation, garbage, None. It never raises.
- The dispatcher slices a day plan by hour, honours the F031 market cap
  (10, per turn), and hands PASS to units past their list.
- The fallback ladder: a plan error falls to the previous plan, then to
  the greedy policy, then to PASS — every rung shape-valid.
- The greedy stub never breaks F002 (watering a planted day) and never
  sends more than 10 market orders (F031).
"""

from __future__ import annotations

from agent.main import agent
from agent.dispatch import dispatch_plan, PASS_ACTION, MAX_MARKET_ORDERS
from agent.runtime import Runtime, RUNTIME, WORKING_BUDGET_S


def _obs(day=0, hour=0, seeds=2, shed=None, money=100):
    return {"player": 0, "step": day * 24 + hour, "day": day, "hour": hour,
            "farms": [{"money": money,
                       "tiles": [[None] * 10 for _ in range(10)],
                       "farmer": [4, 4], "hands": [],
                       "unlocked_quadrants": ["NW"], "hires_today": 0}],
            "private": {"shed": shed or {}, "seeds": {"WHEAT": seeds},
                        "inventories": [[]]},
            "market": {"inventory": {}, "prices": {}},
            "town": {"unlocked_shops": []},
            "remainingOverageTime": 60.0}


def _plant_tile(day, watered=False, yield_units=0):
    return {"kind": "PLANT", "crop": "WHEAT", "planted_day": day,
            "watered_today": watered, "consecutive_unwatered": 0,
            "yield_units": yield_units, "max_lifespan_step": 96,
            "fertilized_until_day": -1}


def _fresh_runtime() -> Runtime:
    r = Runtime()
    return r


def test_entry_point_never_raises() -> None:
    """Any input — real-shaped, empty, garbage, None — gets a legal dict."""
    for bad_input in (None, 42, {}, {"farms": "no"}, _obs()):
        action = agent(bad_input)
        assert set(action) == {"farmer", "hands", "market"}, action
        assert isinstance(action["farmer"], list)
        assert isinstance(action["hands"], list)
        assert isinstance(action["market"], list)


def test_entry_point_is_deterministic_per_state() -> None:
    """The same observation twice gives the same action (stateless stub)."""
    fresh = _fresh_runtime()
    RUNTIME.__dict__.update(fresh.__dict__)
    a1 = agent(_obs())
    RUNTIME.__dict__.update(_fresh_runtime().__dict__)
    a2 = agent(_obs())
    assert a1 == a2


def test_dispatch_slices_by_hour() -> None:
    """Hour h gives every unit its h-th op; exhausted units PASS."""
    plan = {"units": [[["PLANT", "WHEAT"], ["WATER"], ["HARVEST"]],
                      [["PASS"], ["NORTH"], ["PASS"]]],
            "market": [["BUY_SEED", "WHEAT", 1]]}
    obs = _obs(hour=0)
    obs["farms"][0]["hands"] = [[1, 1]]       # one real hand (F031 truth)
    a0 = dispatch_plan(plan, obs)
    assert a0["farmer"] == ["PLANT", "WHEAT"]
    assert a0["hands"] == [["PASS"]]
    assert a0["market"] == [["BUY_SEED", "WHEAT", 1]]      # hour 0 carries it
    obs1 = _obs(hour=1); obs1["farms"][0]["hands"] = [[1, 1]]
    a1 = dispatch_plan(plan, obs1)
    assert a1["farmer"] == ["WATER"]
    assert a1["hands"] == [["NORTH"]]
    assert a1["market"] == []                              # only hour 0
    obs2 = _obs(hour=2); obs2["farms"][0]["hands"] = [[1, 1]]
    a2 = dispatch_plan(plan, obs2)
    assert a2["farmer"] == ["HARVEST"]
    obs5 = _obs(hour=5); obs5["farms"][0]["hands"] = [[1, 1]]
    a5 = dispatch_plan(plan, obs5)
    assert a5["farmer"] == ["PASS"] and a5["hands"] == [["PASS"]]


def test_dispatch_market_cap_f031() -> None:
    """More than 10 market orders never leave the agent (F031 cap)."""
    plan = {"units": [[["PASS"]]],
            "market": [["BUY_SEED", "WHEAT", 1]] * 15}
    a = dispatch_plan(plan, _obs(hour=0))
    assert len(a["market"]) == MAX_MARKET_ORDERS == 10
    later = dispatch_plan(plan, _obs(hour=1))
    assert later["market"] == []            # per turn, not per day

def test_fallback_ladder_rungs() -> None:
    """A plan error falls through: plan -> prev plan -> greedy -> PASS."""
    r = _fresh_runtime()
    broken = {"units": "not-a-list"}          # dispatch_plan raises on it
    r.plan = broken
    r.prev_plan = {"units": [[["WATER"]]], "market": []}
    obs = _obs(hour=0)
    action = (r._rung_plan(obs) or r._rung_prev_plan(obs)
              or r._rung_greedy(obs) or r._rung_pass(obs))
    assert action["farmer"] == ["WATER"]      # prev plan caught the fall
    # both plans broken -> greedy
    r.prev_plan = broken
    action = (r._rung_plan(obs) or r._rung_prev_plan(obs)
              or r._rung_greedy(obs) or r._rung_pass(obs))
    assert action["farmer"][0] in ("PLANT", "WATER", "PASS", "HARVEST")
    # no plans at all -> greedy; greedy itself broken -> PASS
    r.plan = r.prev_plan = None
    assert r._rung_pass(obs) == {"farmer": ["PASS"], "hands": [], "market": []}


def test_act_survives_a_raising_plan() -> None:
    """act() catches everything inside and still returns a legal dict."""
    r = _fresh_runtime()
    r.plan = {"units": "garbage"}
    action = r.act(_obs(hour=0))
    assert set(action) == {"farmer", "hands", "market"}


def test_act_never_raises_on_hostile_obs() -> None:
    r = _fresh_runtime()
    for bad in (None, 42, {}, {"hour": "x"}, {"farms": []}):
        action = r.act(bad)
        assert set(action) == {"farmer", "hands", "market"}


def test_deadline_reads_overage() -> None:
    """The deadline reads remainingOverageTime from the observation."""
    from agent.runtime import Deadline
    d = Deadline(60.0)
    assert not d.expired()
    assert d.remaining_ms() <= int(WORKING_BUDGET_S * 1000)
    d2 = Deadline(0.0)
    assert d2.remaining_overage_s == 0.0      # exhausted bank: F046 forfeit


def test_deadline_gates_the_ladder() -> None:
    """A rung that burns the budget hands the turn to PASS, fast.

    The drill must be able to FAIL: a sleeping rung makes act() return in
    well under the sleep, because the deadline is consulted between rungs
    (review 2, finding 1 - the old drill passed whatever the budget was).
    """
    import time

    def slow_dispatch(runtime, obs):
        # a well-behaved heavy rung polls the deadline and bails
        for _ in range(120):                  # 1.2 s in 10 ms slices
            if runtime._deadline.expired():
                raise TimeoutError("plan rung over budget")
            time.sleep(0.01)
        return {"farmer": ["WATER"], "hands": [], "market": []}

    r = _fresh_runtime()
    r.plan = {"units": [[["WATER"]]], "market": []}
    saved = Runtime._rung_plan                # act() calls the BOUND method
    saved_budget = WORKING_BUDGET_S
    import agent.runtime as R
    R.WORKING_BUDGET_S = 0.05                 # shrink the wall for the drill
    Runtime._rung_plan = lambda self, obs: slow_dispatch(self, obs)
    try:
        t0 = time.perf_counter()
        action = r.act(_obs(hour=1))          # hour 1: plan rung runs first
        elapsed = time.perf_counter() - t0
    finally:
        Runtime._rung_plan = saved
        R.WORKING_BUDGET_S = saved_budget
    # the plan rung saw the expired budget and bailed; the ladder handed
    # the turn to PASS instead of running more rungs past the wall
    assert action["farmer"] == ["PASS"], action
    assert elapsed < 0.5, f"act() took {elapsed:.3f}s - the gate did not fire"


def test_empty_units_plan_is_legal() -> None:
    """{"units": []} is a legitimate plan: everyone passes, no raise."""
    plan = {"units": [], "market": [["SELL", "WHEAT", 5]]}
    a = dispatch_plan(plan, _obs(hour=0))
    assert a["farmer"] == ["PASS"]
    assert a["hands"] == []
    assert a["market"] == [["SELL", "WHEAT", 5]]


def test_dispatch_reconciles_hands_with_obs() -> None:
    """F031: hires can fail silently, so the OBS's hand count is
    authoritative - plan ops for non-existent hands are dropped."""
    plan = {"units": [[["PASS"]],            # farmer
                      [["NORTH"]],           # planned hand 1
                      [["SOUTH"]]],          # planned hand 2
            "market": []}
    obs = _obs(hour=0)
    a = dispatch_plan(plan, obs)              # obs has no hands hired
    assert a["hands"] == []                   # both dropped: no real hands
    obs["farms"][0]["hands"] = [[1, 1]]       # one real hand
    a = dispatch_plan(plan, obs)
    assert a["hands"] == [["NORTH"]]          # hand 1 kept, hand 2 gone
    obs["farms"][0]["hands"] = [[1, 1], [2, 2], [3, 3]]   # 3 real hands
    a = dispatch_plan(plan, obs)
    assert a["hands"] == [["NORTH"], ["SOUTH"], ["PASS"]]


def test_greedy_f002_water_first() -> None:
    """A planted tile is watered before anything else (F002)."""
    obs = _obs(day=1)
    obs["farms"][0]["tiles"][4][4] = _plant_tile(day=0, watered=False)
    action = agent(obs)
    assert action["farmer"] == ["WATER"], action


def test_greedy_harvests_from_first_yield() -> None:
    """WHEAT first_yield_day = 2: harvest at age >= 2 with yield."""
    obs = _obs(day=3)
    obs["farms"][0]["tiles"][4][4] = _plant_tile(day=0, watered=True,
                                                 yield_units=3)
    action = agent(obs)
    assert action["farmer"] == ["HARVEST"], action


def test_greedy_buys_seed_when_out() -> None:
    obs = _obs(day=0, seeds=0, money=100)
    action = agent(obs)
    assert ["BUY_SEED", "WHEAT", 1] in action["market"], action


def test_greedy_market_cap_f031() -> None:
    """Greedy never sends more than 10 orders, even with many shed items."""
    shed = {f"ITEM{i}": 99 for i in range(20)}
    obs = _obs(day=0, shed=shed)
    action = agent(obs)
    assert len(action["market"]) <= 10


def test_plan_day_roll_over() -> None:
    """At hour 0 the committed plan archives to prev_plan (rung 2 feed).

    With no replanner enabled (the default - the rung is opt-in until the
    day #14 can carry its chains' inputs) plan stays None and greedy
    drives the day; the archived plan is dispatched on the next hour.
    """
    r = _fresh_runtime()
    r.plan = {"units": [[["WATER"], ["NORTH"]]], "market": []}
    a0 = r.act(_obs(day=0, hour=0))
    assert r.plan is None                     # archived at hour 0
    assert r.prev_plan == {"units": [[["WATER"], ["NORTH"]]], "market": []}
    a1 = r.act(_obs(day=0, hour=1))
    # rung 2 dispatched the archived plan: farmer does hour-1 op = NORTH
    assert a1["farmer"] == ["NORTH"], a1


def test_full_episode_smoke() -> None:
    """720 turns of greedy + rolling hour, no exception, all shape-valid."""
    r = _fresh_runtime()
    day = 0
    for step in range(720):
        hour = step % 24
        if step % 24 == 0 and step > 0:
            day += 1
        obs = _obs(day=day, hour=hour)
        if hour == 12:
            obs["farms"][0]["tiles"][4][4] = _plant_tile(
                day=day, watered=(hour > 12), yield_units=2 if day > 2 else 0)
        action = r.act(obs)
        assert set(action) == {"farmer", "hands", "market"}


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - test runner
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {name}")
    if failures:
        print(f"{failures} test(s) failed")
        return 1
    print("all agent spine tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
