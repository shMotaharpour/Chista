"""Render a full game of our v1 agent vs a ladder opponent via env.render(mode='html').

Usage: python -m lab.render_game [--a agent/main.py] [--b <opponent>] [--seed N] [--out path]
"""
import argparse
import importlib.util
import os

from kaggle_environments import make


def load_agent(path):
    spec = importlib.util.spec_from_file_location("chista_v1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="agent/main.py")
    ap.add_argument("--b", default="random")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out", default="lab/results/v1_game.html")
    args = ap.parse_args()

    agent = load_agent(args.a)
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": args.seed}, debug=False)
    env.run([agent, args.b])
    html = env.render(mode="html")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"rewards: {[s.reward for s in env.steps[-1]]}")
    print(f"wrote {args.out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
