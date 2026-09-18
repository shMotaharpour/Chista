"""Reproduce the R003 speed claim on the machine you are running on.

Usage:
    .venv/bin/python -m bench.bench_paths                # 720 steps, 3 reps
    .venv/bin/python -m bench.bench_paths --steps 240 --reps 5

Three timings per season, identical seed/config and PASS policies:
  harness_with_agents : make() + env.run() with the shipped "pass" agent (what a
                        submission-style evaluation costs end to end)
  harness_no_agents   : the same harness driven turn-by-turn without agents (its
                        own bookkeeping, agent indirection removed)
  fast_sim            : FastSim.run() -- the number R003 should quote

These are wall-clock numbers for one machine; they are not portable, which is
why the script prints the platform, cpu_count and installed versions.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import platform
import statistics
import time
from typing import Any

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as K

from offline.fast_sim import FastSim

PASS_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}


def pass_policy(obs: Any) -> dict[str, Any]:
    """Same behaviour as the shipped "pass" agent, without the indirection."""
    return dict(PASS_ACTION)


def harness_with_agents(config: dict[str, Any]) -> float:
    start = time.perf_counter()
    env = make("kaggriculture", configuration=dict(config))
    env.run([K.agents["pass"], K.agents["pass"]])
    return time.perf_counter() - start


def harness_no_agents(config: dict[str, Any]) -> float:
    start = time.perf_counter()
    env = make("kaggriculture", configuration=dict(config))
    env.reset()
    while not env.done:
        env.step([dict(PASS_ACTION), dict(PASS_ACTION)])
    return time.perf_counter() - start


def fast_sim(config: dict[str, Any]) -> float:
    start = time.perf_counter()
    sim = FastSim(dict(config), validate="fast")
    sim.run([pass_policy, pass_policy])
    return time.perf_counter() - start


def median_seconds(fn, config, reps: int) -> float:
    return statistics.median(fn(dict(config)) for _ in range(reps))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=720, help="episode length in turns")
    ap.add_argument("--reps", type=int, default=3, help="repetitions (median reported)")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    config = {"seed": args.seed, "episodeSteps": args.steps}

    print(f"platform={platform.platform()} cpu_count={__import__('os').cpu_count()}")
    print(f"python={platform.python_version()} "
          f"kaggle-environments={metadata.version('kaggle-environments')}")
    print(f"config={config} reps={args.reps}")

    harness_agents = median_seconds(harness_with_agents, config, args.reps)
    harness_bare = median_seconds(harness_no_agents, config, args.reps)
    fast = median_seconds(fast_sim, config, args.reps)

    print(f"harness_with_agents : {harness_agents:8.3f} s/season "
          f"({args.steps / harness_agents:8.1f} turns/s)")
    print(f"harness_no_agents   : {harness_bare:8.3f} s/season")
    print(f"fast_sim            : {fast:8.4f} s/season "
          f"({args.steps / fast:8.1f} turns/s)")
    print(f"speedup vs harness_with_agents : {harness_agents / fast:5.1f}x")
    print(f"speedup vs harness_no_agents   : {harness_bare / fast:5.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
