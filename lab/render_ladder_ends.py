"""Render a game: our v1 agent vs the best (rank 1) and worst (rank 19) ladder opponents.

Usage: python -m lab.render_ladder_ends [--seed 77]
"""
import argparse
import importlib.util
import json
import os

from kaggle_environments import make


def load_agent(path):
    spec = importlib.util.spec_from_file_location("chista_v1", "agent/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=77)
    args = ap.parse_args()

    with open("lab/results/opponent_ladder.json") as f:
        ladder = json.load(f)
    ranked = ladder["ranked"]
    best = ranked[0]
    worst = ranked[-1]

    our_agent = load_agent("agent/main.py")

    for label, opp in (("best", best), ("worst", worst)):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": args.seed}, debug=False)
        env.run([our_agent, f"lab/opponents/{opp}/agent.py"])
        html = env.render(mode="html")
        out = f"lab/results/v1_vs_{label}_{opp[:30]}.html"
        with open(out, "w") as f:
            f.write(html)
        print(f"vs {label} ({opp}): rewards={[s.reward for s in env.steps[-1]]} -> {out}")


if __name__ == "__main__":
    main()
