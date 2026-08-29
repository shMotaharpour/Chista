"""Paired-seed game runner.

Usage:
    python -m lab.runner --a starter --b random --seeds 1..18 --out lab/results
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

from kaggle_environments import make

from lab.metrics import extract_result
from lab import registry


def _parse_seeds(spec: str) -> list[int]:
    m = re.fullmatch(r"(\d+)\.\.(\d+)", spec)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def run_game(agent_a: str, agent_b: str, seed: int | None = None, episode_steps: int = 720):
    env = make("kaggriculture", configuration={"episodeSteps": episode_steps, "seed": seed}, debug=False)
    env.run([registry.resolve(agent_a), registry.resolve(agent_b)])
    return extract_result(env, agent_a, agent_b, seed, episode_steps)


def run_paired(agent_a: str, agent_b: str, seeds: list[int], episode_steps: int = 720):
    """Each seed twice, sides swapped, to cancel first-mover advantage."""
    for seed in seeds:
        yield run_game(agent_a, agent_b, seed, episode_steps)
        yield run_game(agent_b, agent_a, seed, episode_steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--seeds", default="1..18")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--out", default="lab/results")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, f"{run_id}_{args.a}_vs_{args.b}")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i, res in enumerate(run_paired(args.a, args.b, seeds, args.steps)):
        path = os.path.join(out_dir, f"game_{i:03d}.json")
        res.save_json(path)
        results.append(path)
        print(f"[{i + 1}/{2 * len(seeds)}] seed={res.seed} sides=({res.agent_a},{res.agent_b}) "
              f"rewards={res.rewards} residue_value=({res.residue_value_a:.0f},{res.residue_value_b:.0f})")

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"a": args.a, "b": args.b, "seeds": seeds, "steps": args.steps, "files": results}, f, indent=2)
    print(f"\nWrote {len(results)} results to {out_dir}")


if __name__ == "__main__":
    main()
