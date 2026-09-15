"""Bench the turn budget: a full 720-turn episode, per-turn timing.

Run from the repo root:

    .venv/bin/python -m bench.bench_turn_budget [seeds]

Plays the agent spine against `random` on the official path
(world.kaggle_env.run_episode), records every turn's self_ms and prints
the p50/p95/p99/max table plus the bank draw. Acceptance (#9): bank
drawn = 0 s, max <= 400 ms; this table is the baseline later milestones
compare against, and SAFETY / FLOOR_S for the bank policy derive from it
(not yet applied in M1 - the bank policy lands with the replanner).
"""

from __future__ import annotations

import statistics
import sys
import time

from agent.main import agent


def bench(seeds: int = 3) -> int:
    from world.kaggle_env import run_episode

    all_ms: list[float] = []
    first_turns: list[float] = []
    for seed in range(seeds):
        turn_times: list[float] = []

        def timed_agent(obs, config=None):
            t0 = time.perf_counter()
            action = agent(obs, config)
            turn_times.append((time.perf_counter() - t0) * 1000.0)
            return action

        env = run_episode([timed_agent, "random"],
                          configuration={"episodeSteps": 720, "seed": seed})
        rewards = [s.reward for s in env.steps[-1]]
        print(f"seed {seed}: rewards {rewards}, turns {len(turn_times)}, "
              f"max {max(turn_times):.1f} ms")
        all_ms.extend(turn_times)
        first_turns.append(turn_times[0])

    all_ms.sort()
    p50 = statistics.median(all_ms)
    p95 = all_ms[int(0.95 * len(all_ms)) - 1]
    p99 = all_ms[int(0.99 * len(all_ms)) - 1]
    mx = all_ms[-1]
    print(f"\nturns {len(all_ms)}  p50 {p50:.2f} ms  p95 {p95:.2f} ms  "
          f"p99 {p99:.2f} ms  max {mx:.2f} ms")
    print(f"first-turn (cold) max: {max(first_turns):.2f} ms")
    bank_drawn = max(0.0, (mx - 1000.0)) / 1000.0
    print(f"bank drawn: {bank_drawn:.3f} s (from the max turn; "
          f"0.000 = untouched)")
    return 0


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    raise SystemExit(bench(seeds))
