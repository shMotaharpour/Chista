"""The entry's guards: the harness contract, the dispatcher, and visible failure.

Run:  .venv/bin/python -m tests.test_agent_main

Contracts under test:
- The harness's own pick rule lands on the LAST callable in `agent/main.py`, and
  that callable takes ONE argument: an instance is callable but has no
  `__code__`, so it would be handed `(obs, config)` and raise. `get_last_callable`
  is called here, on this file's real source, rather than re-implemented.
- The entry point returns a shape-valid action dict for any input - a real
  observation, garbage, None. It never raises with `Config.never_raise` ON.
- Hour 0 calls `Manager.observe` once and `Manager.step` never; every later hour
  calls `step`. No budget is derived from any clock: there is no clock here.
- What the manager returns is what the engine gets: the plan is sliced by hour
  under the F031 market cap, and ops for hands that do not exist are dropped.
- A manager that raises is RECORDED (`Agent.failures`, `failed_days`) and the
  turn PASSes. No second policy answers in its place - that is the failure mode
  `docs/ARCHITECTURE.md` section 5 exists to prevent.

What is NOT here any more: the wall-clock budget, the over-budget anomaly, the
gap statistics and the retired ladder. Those guards went with `agent/runtime.py`
- a turn's length is the harness's business, and a spine that measured its own
clock made the plan follow the machine.
"""

from __future__ import annotations

import ast
import contextlib
import io
import pathlib
from typing import Any

import pytest

from agent.config import Config
from agent.dispatch import MAX_MARKET_ORDERS, PASS_ACTION, dispatch_plan
from agent.main import AGENT, Agent, agent

ENTRY = pathlib.Path(__file__).resolve().parents[1] / "agent" / "main.py"
SPINE = ENTRY


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

    def step(self, obs=None, budget_ms: float | None = None) -> bool:
        self.stepped.append(budget_ms)
        if self.raises == "step":
            raise RuntimeError("step blew up")
        return self.certified

    def best(self) -> dict:
        return self.plan


def _agent(fake: _FakeManager, **cfg) -> Agent:
    """An Agent whose manager is the fake (the entry builds one only if None)."""
    built = Agent(Config(**cfg))
    built.manager = fake                     # type: ignore[assignment]
    return built


def _quiet(fn, *args, **kwargs) -> tuple[str, Any]:
    """Run `fn`, returning (stdout, result) - the log is part of the contract."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return buf.getvalue(), result


# ------------------------------------------------------- the harness's own rule

def _top_level_callables(src: str) -> list[tuple[str, int | None]]:
    """`(name, argcount)` for every callable a TOP-LEVEL statement binds, in order.

    A `def` binds a function (its positional count), a `class` binds a class, and
    an assignment whose value is a call binds whatever the call returns — the
    instance. Imported names are left out: the harness's `get_last_callable`
    reads the executed module's globals, and an import lands in them too, but the
    definition ORDER is what this guard is about.
    """
    out: list[tuple[str, int | None]] = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, len(node.args.args)))
        elif isinstance(node, ast.ClassDef):
            out.append((node.name, None))
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.append((target.id, None))
    return out


def test_the_harness_picks_the_last_callable_and_it_takes_one_argument() -> None:
    """The reason this module's shape is what it is, checked against the harness.

    `kaggle_environments.agent.get_last_callable` returns the LAST callable value
    in the module's globals and `Agent.act` truncates the call to that callable's
    `co_argcount` — so a one-argument function is handed the observation alone.
    Anything callable defined after the entry breaks the turn, and so does an
    instance as the last value: an object has no `__code__`, cannot be truncated,
    and would be handed `(obs, config)`.
    """
    import inspect

    from kaggle_environments.agent import get_last_callable

    src = ENTRY.read_text(encoding="utf-8")
    picked = get_last_callable(src, path=str(ENTRY))
    assert inspect.isfunction(picked), type(picked)
    assert picked.__name__ == agent.__name__ == "agent", picked.__name__
    assert picked.__code__.co_argcount == 1, picked.__code__.co_argcount

    bound = _top_level_callables(src)
    assert bound[-1] == ("agent", 1), (
        f"the entry is not the LAST callable this module binds: {bound}")


# --------------------------------------------------------------- the entry point

def test_entry_point_never_raises() -> None:
    """Any input - real-shaped, empty, garbage, None - gets a legal dict.

    `never_raise=True` names the contract under test: the submission's arm. The
    OFF arm is `test_the_never_raise_switch_decides_whether_a_failure_escapes`.
    """
    fake = _FakeManager(raises="observe")
    entry = _agent(fake, never_raise=True)
    for bad_input in (None, 42, {}, {"farms": "no"}, _obs()):
        action = entry(bad_input)
        assert set(action) == {"farmer", "hands", "market"}, action
        assert isinstance(action["farmer"], list)
        assert isinstance(action["hands"], list)
        assert isinstance(action["market"], list)


def test_the_never_raise_switch_decides_whether_a_failure_escapes() -> None:
    """Both arms of `Config.never_raise`, on the same failing turn.

    ON: all-PASS, the failure on `Agent.failures`, the day marked, the `A` line
    written. OFF: the same three records, and then the exception comes out with
    its traceback — a hunt wants to know what reached the boundary instead of
    reading a season of PASSed days.

    The switch decides whether the run CONTINUES, never whether the failure is
    written down: both arms are asserted to leave the same record.
    """
    caught = _agent(_FakeManager(raises="observe"), never_raise=True)
    stdout, action = _quiet(caught, _obs(day=4, hour=0))
    assert action == PASS_ACTION
    assert len(caught.failures) == 1 and caught.failed_days == {4}
    assert "A day=4 hour=0" in stdout and "error=" in stdout, stdout

    loud = _agent(_FakeManager(raises="observe"), never_raise=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pytest.raises(RuntimeError, match="observe blew up"):
            loud(_obs(day=4, hour=0))
    assert len(loud.failures) == 1, loud.failures
    assert "day 4 hour 0" in loud.failures[0]
    assert loud.failed_days == {4}
    assert "A day=4 hour=0" in buf.getvalue(), (
        "the OFF arm must write the same log line before it raises")


def test_the_never_raise_switch_default_is_the_owners_current_setting() -> None:
    """The DEFAULT is a temporary owner setting, and the guard says which.

    It is OFF right now (the owner's order: the hunt wants the traceback) and
    MUST be ON in the submission. This guard exists so a flip is a deliberate
    edit that names itself, not a silent drift — it is not a claim that OFF is
    correct.
    """
    assert Config().never_raise is False, (
        "the default flipped: if this is the submission, the harness can now be "
        "handed an exception — update this guard's docstring with the reason")


def test_the_module_level_entry_point_is_legal_on_garbage() -> None:
    """`agent.main:agent` - the harness's own path - survives a junk observation.

    This is the one guard that builds the REAL manager, so it restores the
    singleton afterwards and leaves no state for the tests that follow.
    """
    saved = dict(AGENT.__dict__)
    try:
        AGENT.__dict__.clear()
        AGENT.__dict__.update(Agent(Config(never_raise=True)).__dict__)
        stdout, action = _quiet(agent, {"farms": []})
        assert set(action) == {"farmer", "hands", "market"}, action
        assert AGENT.failures, "a junk observation must be recorded as a failure"
        assert "A day=" in stdout and "error=" in stdout, stdout
    finally:
        AGENT.__dict__.clear()
        AGENT.__dict__.update(saved)


# ------------------------------------------------------------------ the day

def test_hour_zero_observes_and_every_later_hour_steps() -> None:
    """One observe per day, one step per remaining turn - never the other way."""
    fake = _FakeManager()
    entry = _agent(fake)
    entry(_obs(day=0, hour=0))
    assert len(fake.observed) == 1 and fake.stepped == []
    for hour in (1, 7, 23):
        entry(_obs(day=0, hour=hour))
    assert len(fake.observed) == 1, "observe ran more than once in a day"
    assert len(fake.stepped) == 3, fake.stepped
    entry(_obs(day=1, hour=0))
    assert len(fake.observed) == 2, "the next day did not observe"


def test_no_budget_is_handed_to_the_manager() -> None:
    """`step` is called with the observation and nothing else.

    The retired spine derived a per-turn millisecond budget from the clock and
    passed it down; the manager's own numbers are `Config`'s, and a turn no
    longer decides how much work happens.
    """
    fake = _FakeManager()
    entry = _agent(fake)
    entry(_obs(hour=0))
    entry(_obs(hour=1))
    assert fake.stepped == [None], fake.stepped


def test_the_managers_plan_is_what_gets_dispatched() -> None:
    """The plan the manager holds is sliced by hour, not re-decided."""
    plan = {"units": [[["PLANT", "WHEAT"], ["WATER"], ["HARVEST"]]],
            "market": [[["BUY_SEED", "WHEAT", 1]], []]}
    fake = _FakeManager(plan=plan)
    entry = _agent(fake)
    assert entry(_obs(hour=0))["farmer"] == ["PLANT", "WHEAT"]
    assert entry(_obs(hour=1))["farmer"] == ["WATER"]
    assert entry(_obs(hour=2))["farmer"] == ["HARVEST"]
    assert entry(_obs(hour=3))["farmer"] == ["PASS"]
    # the market row rides on its own hour (the queue is per hour, F031)
    assert entry(_obs(hour=0))["market"] == [["BUY_SEED", "WHEAT", 1]]
    assert entry(_obs(hour=1))["market"] == []


# --------------------------------------------------------------- visible failure

def test_a_failing_observe_passes_and_is_recorded() -> None:
    """A raising manager: PASS, a recorded failure, its day marked, a log line.

    `never_raise=True` is the arm that answers; the OFF arm is asserted beside
    it in `test_the_never_raise_switch_decides_whether_a_failure_escapes`.
    """
    fake = _FakeManager(raises="observe")
    entry = _agent(fake, never_raise=True)
    stdout, action = _quiet(entry, _obs(day=4, hour=0))
    assert action == PASS_ACTION, action
    assert len(entry.failures) == 1, entry.failures
    assert "day 4 hour 0" in entry.failures[0]
    assert "RuntimeError" in entry.failures[0]
    assert entry.failed_days == {4}
    assert "A day=4 hour=0" in stdout and "error=" in stdout, stdout
    # no second policy answered in its place
    assert action["farmer"] == ["PASS"]


def test_a_failing_step_passes_and_is_recorded() -> None:
    """A failure on any turn of the day is recorded, not swallowed."""
    fake = _FakeManager(raises="step")
    entry = _agent(fake, never_raise=True)
    action = entry(_obs(day=2, hour=5))
    assert action == PASS_ACTION
    assert entry.failed_days == {2}
    assert "day 2 hour 5" in entry.failures[0]


def test_a_failure_does_not_stop_the_next_turn() -> None:
    """One bad turn is one bad turn: the next turn still calls the manager."""
    fake = _FakeManager(raises="step")
    entry = _agent(fake, never_raise=True)
    entry(_obs(hour=1))
    fake.raises = None
    action = entry(_obs(hour=2))
    assert action != PASS_ACTION or fake.plan["units"][0][0] == ["PASS"]
    assert len(entry.failures) == 1, entry.failures


def test_the_day_line_is_written_once_per_day() -> None:
    """The `D` line: one per day, at hour 0, carrying the manager's own numbers."""
    fake = _FakeManager()
    entry = _agent(fake)
    stdout, _ = _quiet(entry, _obs(day=0, hour=0))
    assert stdout.count("D 0 ") == 1, stdout
    assert "certified=" in stdout and "pool=" in stdout
    stdout, _ = _quiet(entry, _obs(day=0, hour=1))
    assert "D 0 " not in stdout, "the day line repeated inside the day"


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
    """719 turns of the entry's own loop: 30 observes, 689 steps, all legal."""
    fake = _FakeManager(plan={"units": [[["PASS"]]], "market": []})
    entry = _agent(fake)
    for step in range(719):                    # F048: 719 decisions, not 720
        action = entry(_obs(day=step // 24, hour=step % 24))
        assert set(action) == {"farmer", "hands", "market"}
    assert len(fake.observed) == 30, len(fake.observed)
    assert len(fake.stepped) == 689, len(fake.stepped)
    assert entry.failures == []


def test_the_spine_imports_no_fallback_ladder_and_no_runtime() -> None:
    """Structural: the entry's imports are the manager, the dispatcher, config.

    The ladder is gone rather than dormant - `greedy`, `replan` and
    `market_layer` are not reachable from the entry point any more - and so is
    the clock: `agent/runtime.py` is retired, and this guard is what fails the
    day it (or another spine that measures the wall clock) is imported back in.
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
                   "agent.belief.ladder", "agent.planner.master",
                   "agent.runtime"):
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
    print("all agent entry tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
