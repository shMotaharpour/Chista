"""F048: the season has 720 states and 719 decisions; the last one counts.

Run:  .venv/bin/python -m tests.test_episode_boundary

"720 turns" (F029) counts STATES. The agent is asked for an action 719 times,
at steps 0..718, because the terminal state is a result and not a prompt. The
distinction is invisible until it bites: a plan that schedules anything for
"the last turn" by writing 719, or that treats step 718 as too late to matter,
is wrong in opposite directions.

Both halves are pinned here because both were guessed wrong once. The first
guess was that the agent is called 720 times; the second, after the record
count said otherwise, was that step 718's action is therefore not processed and
a final sale would be lost. It is processed — that is what
`test_last_decision_is_processed` measures, with money rather than an argument.

Runs on the official harness path (`world.kaggle_env`, per AGENTS.md), because
this is a fact about how the harness invokes an agent, not about the rules.
"""
from __future__ import annotations

from world.kaggle_env import run_episode

TURNS_PER_DAY = 24          # competition rule, fixed (F029)
EPISODE_STATES = 720        # F029: "720 turns" - states, as this file shows
LAST_DECISION_STEP = 718    # EPISODE_STATES - 2 (0-based, terminal needs none)


def _passer(obs, config=None) -> dict:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _recording_agent(market_at: dict[int, list] | None = None):
    """A PASS agent that records every step it is called with.

    `market_at` maps a step to the market orders to emit there, so a test can
    ask for an order at a step the agent may never be given.
    """
    seen: list[tuple[int, int, int]] = []

    def agent(obs, config=None) -> dict:
        step = int(obs.get("step", -1))
        seen.append((step, int(obs.get("day", -1)), int(obs.get("hour", -1))))
        orders = list((market_at or {}).get(step, []))
        return {"farmer": ["PASS"], "hands": [], "market": orders}

    agent.seen = seen                                    # type: ignore[attr-defined]
    return agent


def test_states_and_decisions_differ_by_one() -> None:
    agent = _recording_agent()
    env = run_episode([agent, _passer])
    steps = [s for s, _, _ in agent.seen]               # type: ignore[attr-defined]

    assert len(env.steps) == EPISODE_STATES, (
        f"{len(env.steps)} states, expected {EPISODE_STATES} (F029)")
    assert len(steps) == EPISODE_STATES - 1, (
        f"agent called {len(steps)} times, expected {EPISODE_STATES - 1}: "
        "one decision per transition, none for the terminal state")
    assert steps == list(range(EPISODE_STATES - 1)), (
        "the agent must be called once per step, 0..718, in order")
    assert max(steps) == LAST_DECISION_STEP


def test_day_and_hour_follow_the_step() -> None:
    agent = _recording_agent()
    run_episode([agent, _passer])
    bad = [(s, d, h) for s, d, h in agent.seen            # type: ignore[attr-defined]
           if d != s // TURNS_PER_DAY or h != s % TURNS_PER_DAY]
    assert not bad, f"day/hour diverged from the step for {len(bad)} calls: {bad[:3]}"


def test_final_day_is_one_decision_short() -> None:
    """Days 0..28 get 24 decisions; day 29 gets 23, hours 0..22.

    Not a lost opportunity - see `test_last_decision_is_processed` - just one
    fewer action slot, which is what an end-of-season plan has to budget for.
    """
    agent = _recording_agent()
    run_episode([agent, _passer])
    per_day: dict[int, list[int]] = {}
    for _, day, hour in agent.seen:                       # type: ignore[attr-defined]
        per_day.setdefault(day, []).append(hour)

    last_day = max(per_day)
    for day in sorted(per_day):
        hours = per_day[day]
        want = TURNS_PER_DAY - 1 if day == last_day else TURNS_PER_DAY
        assert len(hours) == want, (
            f"day {day} got {len(hours)} decisions, expected {want}")
        assert hours == list(range(want)), f"day {day} hours out of order"
    assert per_day[last_day][-1] == TURNS_PER_DAY - 2      # hour 22


def test_last_decision_is_processed() -> None:
    """The action at step 718 reaches the engine - measured in money.

    A PASS agent never spends, so money sits at startingMoney all season and
    any change is exactly the order that was processed. Buying one wheat seed
    at step 718 must show up in the terminal state.
    """
    buy = [["BUY_SEED", "WHEAT", 1]]
    agent = _recording_agent({LAST_DECISION_STEP: buy})
    env = run_episode([agent, _passer])

    before = env.steps[LAST_DECISION_STEP][0].observation["farms"][0]["money"]
    final = env.steps[-1][0].observation["farms"][0]["money"]
    seeds = (env.steps[-1][0].observation.get("private") or {}).get("seeds") or {}

    assert final < before, (
        f"money {before} -> {final}: the step-{LAST_DECISION_STEP} order never "
        "reached the engine, so the last decision does NOT count")
    assert seeds.get("WHEAT", 0) >= 1, "the seed bought at the last decision is missing"
    assert env.steps[-1][0].reward == final, (
        "the reward is the money in the terminal state, which the agent never sees")


def test_agent_is_never_asked_past_the_last_decision() -> None:
    """Orders written for steps past 718 are never emitted, because the agent
    is never called there - the plan is silently one turn short, not refused."""
    beyond = {LAST_DECISION_STEP + 1: [["BUY_SEED", "WHEAT", 1]],
              LAST_DECISION_STEP + 2: [["BUY_SEED", "WHEAT", 1]]}
    agent = _recording_agent(beyond)
    env = run_episode([agent, _passer])

    steps = {s for s, _, _ in agent.seen}                  # type: ignore[attr-defined]
    assert not (steps & set(beyond)), (
        f"agent was called at {steps & set(beyond)}, which this finding says "
        "cannot happen")
    final = env.steps[-1][0].observation["farms"][0]["money"]
    start = env.steps[0][0].observation["farms"][0]["money"]
    assert final == start, (
        f"money moved {start} -> {final} although every order was written for a "
        "step the agent is never given")


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:                       # noqa: BLE001 - test runner
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {name}")
    if failures:
        print(f"{failures} test(s) failed")
        return 1
    print(f"episode boundary pinned: {EPISODE_STATES} states, "
          f"{EPISODE_STATES - 1} decisions, last at step {LAST_DECISION_STEP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
