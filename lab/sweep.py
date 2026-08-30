"""Parameter sweep: prove agent constants empirically with the real environment.

Sweeps one parameter at a time (others at baseline), runs the real env,
and reports the best value per parameter. No opponent modeling — our farm vs 'pass'.

Usage:
    python -m lab.sweep [--seeds 1..5] [--all] [--only land_day,hire_day0]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

from kaggle_environments import make

AGENT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent", "main.py")


def load_agent_module():
    spec = importlib.util.spec_from_file_location("chista_v1", AGENT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_one(agent_fn, seed: int) -> float:
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run([agent_fn, "pass"])  # no opponent dynamics — pure farm economics
    return float(env.steps[-1][0].observation.farms[0]["money"])


def evaluate(agent_fn, seeds) -> dict:
    scores = [run_one(agent_fn, s) for s in seeds]
    return {"mean": sum(scores) / len(scores), "min": min(scores), "max": max(scores), "scores": scores}


# ---------------------------------------------------------------- sweeps
SWEEPS = {
    "DAY0_TILES":     [4, 8, 12, 16, 20, 25],
    "HIRE_FRACTION":  [0.02, 0.05, 0.1, 0.2, 0.4],
    "LAND_DAY":       [1, 2, 3, 5, 8],
    "CREW_RATE":      [3, 4, 5, 6, 8],
    "MELON_START_DAY": [3, 5, 7, 9, 12],
}


def sweep(param: str, mod, seeds, values=None) -> list[dict]:
    values = values or SWEEPS[param]
    old = getattr(mod, param)
    results = []
    for v in values:
        setattr(mod, param, v)
        res = evaluate(mod.agent, seeds)
        results.append({"param": param, "value": v, **res})
        print(f"  {param}={v}: mean=${res['mean']:.0f}  min=${res['min']:.0f}  max=${res['max']:.0f}")
    setattr(mod, param, old)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1..5")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="lab/results/sweep.json")
    args = ap.parse_args()

    a, b = args.seeds.split("..")
    seeds = list(range(int(a), int(b) + 1))
    mod = load_agent_module()

    which = args.only.split(",") if args.only else list(SWEEPS)
    all_results = {}
    t0 = time.time()
    for param in which:
        print(f"\n=== sweeping {param} ===")
        all_results[param] = sweep(param, mod, seeds)
    print(f"\nTotal sweep time: {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"wrote {args.out}")

    print("\n=== BEST PER PARAMETER ===")
    for param, results in all_results.items():
        best = max(results, key=lambda r: r["mean"])
        print(f"{param}: {best['value']} (mean ${best['mean']:.0f})")


if __name__ == "__main__":
    main()
