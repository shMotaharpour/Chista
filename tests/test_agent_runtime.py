"""The spine's guards: the entry point, the dispatcher, the manager's clock, and visible failure.

Run:  .venv/bin/python -m tests.test_agent_runtime

Contracts under test:
- The entry point returns a shape-valid action dict for any input - a real
  observation, garbage, None. It never raises.
- Hour 0 calls `Manager.observe` once and `Manager.step` never; every later hour
  calls `step` with a budget inside the turn's working second (F046/F058).
- What the manager returns is what the engine gets: the plan is sliced by hour
  under the F031 market cap, and ops for hands that do not exist are dropped.
- A manager that raises is RECORDED (`Runtime.failures`, `failed_days`) and the
  turn PASSes. No second policy answers in its place - that is the failure mode
  `docs/ARCHITECTURE.md` section 5 exists to prevent.
- The spine imports no fallback-ladder module: the ladder is gone, not dormant.
"""

from __future__ import annotations

import ast
import contextlib
import io
import pathlib
from typing import Any

from agent.config import Config
from agent.dispatch import MAX_MARKET_ORDERS, PASS_ACTION, dispatch_plan
from agent.main import agent
from agent.runtime import RUNTIME, Runtime

SPINE = pathlib.Path(__file__).resolve().parents[1] / "agent" / "runtime.py"


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


class _FakeManager:
    """The manager's interface, with the calls recorded instead of priced."""

    def __init__(self, plan=None, raises: str | None = None) -> None:
        self.observed: list[tuple] = []
        self.stepped: list[Any] = []
        self.plan = plan if plan is not None else {"units": [[["PASS"]]], "market": []}
        self.raises = raises
        self.certified = True
        self.pool: list = []

    def observe(self, obs, config=None) -> None:
        self.observed.append((obs, config))
        if self.raises == "observe":
            raise RuntimeError("observe blew up")

    def step(self, budget_ms: float | None = None) -> bool:
        self.stepped.append(budget_ms)
        if self.raises == "step":
            raise RuntimeError("step blew up")
        return self.certified

    def best(self) -> dict:
        return self.plan


def _runtime(fake: _FakeManager, **cfg) -> Runtime:
    """A Runtime whose manager is the fake (the spine builds one only if None)."""
    runtime = Runtime(Config(**cfg))
    runtime.manager = fake                     # type: ignore[assignment]
    return runtime


def _quiet(fn, *args, **kwargs) -> tuple[str, Any]:
    """Run `fn`, returning (stdout, result) - the log is part of the contract."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return buf.getvalue(), result


# --------------------------------------------------------------- the entry point

def test_entry_point_never_raises() -> None:
    """Any input - real-shaped, empty, garbage, None - gets a legal dict."""
    fake = _FakeManager(raises="observe")
    runtime = _runtime(fake)
    for bad_input in (None, 42, {}, {"farms": "no"}, _obs()):
        action = runtime.act(bad_input)
        assert set(action) == {"farmer", "hands", "market"}, action
        assert isinstance(action["farmer"], list)
        assert isinstance(action["hands"], list)
        assert isinstance(action["market"], list)


def test_the_module_level_entry_point_is_legal_on_garbage() -> None:
    """`agent.main:agent` - the harness's own path - survives a junk observation.

    This is the one guard that builds the REAL manager, so it restores the
    singleton afterwards and leaves no state for the tests that follow.
    """
    saved = dict(RUNTIME.__dict__)
    try:
        RUNTIME.__dict__.clear()
        RUNTIME.__dict__.update(Runtime().__dict__)
        stdout, action = _quiet(agent, {"farms": []})
        assert set(action) == {"farmer", "hands", "market"}, action
        assert RUNTIME.failures, "a junk observation must be recorded as a failure"
        assert "A day=" in stdout and "error=" in stdout, stdout
    finally:
        RUNTIME.__dict__.clear()
        RUNTIME.__dict__.update(saved)


# --------------------------------------------------------------- the manager's clock

def test_hour_zero_observes_and_every_later_hour_steps() -> None:
    """One observe per day, one step per remaining turn - never the other way."""
    fake = _FakeManager()
    runtime = _runtime(fake)
    runtime.act(_obs(day=0, hour=0))
    assert len(fake.observed) == 1 and fake.stepped == []
    for hour in (1, 7, 23):
        runtime.act(_obs(day=0, hour=hour))
    assert len(fake.observed) == 1, "observe ran more than once in a day"
    assert len(fake.stepped) == 3, fake.stepped
    runtime.act(_obs(day=1, hour=0))
    assert len(fake.observed) == 2, "the next day did not observe"


def test_the_step_budget_is_inside_the_turn() -> None:
    """Every step gets a positive budget no larger than the working second."""
    fake = _FakeManager()
    cfg = Config()
    runtime = _runtime(fake)
    runtime.act(_obs(hour=0))
    runtime.act(_obs(hour=1))
    assert fake.stepped, "no step was made"
    for budget in fake.stepped:
        assert 0.0 < budget <= cfg.solve_budget_ms, budget


def test_the_managers_plan_is_what_gets_dispatched() -> None:
    """The plan the manager holds is sliced by hour, not re-decided."""
    plan = {"units": [[["PLANT", "WHEAT"], ["WATER"], ["HARVEST"]]],
            "market": [[["BUY_SEED", "WHEAT", 1]], []]}
    fake = _FakeManager(plan=plan)
    runtime = _runtime(fake)
    assert runtime.act(_obs(hour=0))["farmer"] == ["PLANT", "WHEAT"]
    assert runtime.act(_obs(hour=1))["farmer"] == ["WATER"]
    assert runtime.act(_obs(hour=2))["farmer"] == ["HARVEST"]
    assert runtime.act(_obs(hour=3))["farmer"] == ["PASS"]
    # the market row rides on its own hour (the queue is per hour, F031)
    assert runtime.act(_obs(hour=0))["market"] == [["BUY_SEED", "WHEAT", 1]]
    assert runtime.act(_obs(hour=1))["market"] == []


# --------------------------------------------------------------- visible failure

def test_a_failing_observe_passes_and_is_recorded() -> None:
    """A raising manager: PASS, a recorded failure, its day marked, a log line."""
    fake = _FakeManager(raises="observe")
    runtime = _runtime(fake)
    stdout, action = _quiet(runtime.act, _obs(day=4, hour=0))
    assert action == PASS_ACTION, action
    assert len(runtime.failures) == 1, runtime.failures
    assert "day 4 hour 0" in runtime.failures[0]
    assert "RuntimeError" in runtime.failures[0]
    assert runtime.failed_days == {4}
    assert "A day=4 hour=0" in stdout and "error=" in stdout, stdout
    # no second policy answered in its place
    assert action["farmer"] == ["PASS"]


def test_a_failing_step_passes_and_is_recorded() -> None:
    """A failure on any turn of the day is recorded, not swallowed."""
    fake = _FakeManager(raises="step")
    runtime = _runtime(fake)
    action = runtime.act(_obs(day=2, hour=5))
    assert action == PASS_ACTION
    assert runtime.failed_days == {2}
    assert "day 2 hour 5" in runtime.failures[0]


def test_a_failure_does_not_stop_the_next_turn() -> None:
    """One bad turn is one bad turn: the next turn still calls the manager."""
    fake = _FakeManager(raises="step")
    runtime = _runtime(fake)
    runtime.act(_obs(hour=1))
    fake.raises = None
    action = runtime.act(_obs(hour=2))
    assert action != PASS_ACTION or fake.plan["units"][0][0] == ["PASS"]
    assert runtime.turns == 2 and len(runtime.failures) == 1


def test_the_day_line_is_written_once_per_day() -> None:
    """The `D` line: one per day, at hour 0, carrying the manager's own numbers."""
    fake = _FakeManager()
    runtime = _runtime(fake)
    stdout, _ = _quiet(runtime.act, _obs(day=0, hour=0))
    assert stdout.count("D 0 ") == 1, stdout
    assert "certified=" in stdout and "pool=" in stdout
    stdout, _ = _quiet(runtime.act, _obs(day=0, hour=1))
    assert "D 0 " not in stdout, "the day line repeated inside the day"


def test_an_over_budget_turn_is_logged_as_an_anomaly() -> None:
    """A turn over the working budget gets an `A` line even when nothing raised."""
    class _SlowManager(_FakeManager):
        def step(self, budget_ms: float | None = None) -> bool:
            import time
            time.sleep(0.05)
            return super().step(budget_ms)

    runtime = _runtime(_SlowManager(), turn_budget_ms=5.0, reserve_ms=1.0)
    stdout, action = _quiet(runtime.act, _obs(day=0, hour=1))
    assert set(action) == {"farmer", "hands", "market"}
    assert "A day=0 hour=1" in stdout and "over_budget" in stdout, stdout
    assert runtime.worst_overrun_s > 0.0


# --------------------------------------------------------------- the dispatcher

def test_dispatch_slices_by_hour() -> None:
    """Hour h gives every unit its h-th op; exhausted units PASS."""
    plan = {"units": [[["PLANT", "WHEAT"], ["WATER"], ["HARVEST"]],
                      [["PASS"], ["NORTH"], ["PASS"]]],
            "market": [[["BUY_SEED", "WHEAT", 1]]]}
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
    assert dispatch_plan(plan, obs2)["farmer"] == ["HARVEST"]
    obs5 = _obs(hour=5); obs5["farms"][0]["hands"] = [[1, 1]]
    a5 = dispatch_plan(plan, obs5)
    assert a5["farmer"] == ["PASS"] and a5["hands"] == [["PASS"]]


def test_dispatch_market_cap_f031() -> None:
    """More than 10 market orders never leave the agent (F031 cap)."""
    plan = {"units": [[["PASS"]]],
            "market": [[["BUY_SEED", "WHEAT", 1]] * 15]}
    a = dispatch_plan(plan, _obs(hour=0))
    assert len(a["market"]) == MAX_MARKET_ORDERS == 10
    assert dispatch_plan(plan, _obs(hour=1))["market"] == []   # per turn


def test_dispatch_reconciles_hands_with_obs() -> None:
    """F031: hires can fail silently, so the OBS's hand count is authoritative."""
    plan = {"units": [[["PASS"]], [["NORTH"]], [["SOUTH"]]], "market": []}
    obs = _obs(hour=0)
    assert dispatch_plan(plan, obs)["hands"] == []      # no real hands: dropped
    obs["farms"][0]["hands"] = [[1, 1]]
    assert dispatch_plan(plan, obs)["hands"] == [["NORTH"]]
    obs["farms"][0]["hands"] = [[1, 1], [2, 2], [3, 3]]
    assert dispatch_plan(plan, obs)["hands"] == [["NORTH"], ["SOUTH"], ["PASS"]]


def test_empty_units_plan_is_legal() -> None:
    """{"units": []} is a legitimate plan: everyone passes, no raise."""
    plan = {"units": [], "market": [[["SELL", "WHEAT", 5]]]}
    a = dispatch_plan(plan, _obs(hour=0))
    assert a["farmer"] == ["PASS"]
    assert a["hands"] == []
    assert a["market"] == [["SELL", "WHEAT", 5]]


# --------------------------------------------------------------- the whole season

def test_full_episode_smoke_with_a_stub_manager() -> None:
    """719 turns of the spine's own loop: 30 observes, 689 steps, all legal."""
    fake = _FakeManager(plan={"units": [[["PASS"]]], "market": []})
    runtime = _runtime(fake)
    for step in range(719):                    # F048: 719 decisions, not 720
        action = runtime.act(_obs(day=step // 24, hour=step % 24))
        assert set(action) == {"farmer", "hands", "market"}
    assert len(fake.observed) == 30, len(fake.observed)
    assert len(fake.stepped) == 689, len(fake.stepped)
    assert runtime.failures == [] and runtime.turns == 719


def test_the_spine_imports_no_fallback_ladder() -> None:
    """Structural: the spine's imports are the manager, the dispatcher, config.

    The ladder is gone rather than dormant - `greedy`, `replan` and
    `market_layer` are not reachable from the entry point any more, and this
    guard fails the day one of them is imported back in.
    """
    tree = ast.parse(SPINE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "agent.manager.core" in imported, imported
    assert "agent.dispatch" in imported, imported
    assert "agent.config" in imported, imported
    for doomed in ("agent.greedy", "agent.replan", "agent.market_layer",
                   "agent.belief.ladder", "agent.planner.master"):
        assert doomed not in imported, f"{doomed} is back in the spine"


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
