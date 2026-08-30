"""Melon fertilizer A/B replay via the official env render(mode='html').

Runs two games (fertilized vs plain melon) and saves the Kaggle-built
HTML visualizer output of each.

Usage: python -m lab.kaggle_replay [--outdir lab/results]
"""
from __future__ import annotations

import argparse
import os

from kaggle_environments import make


def run_melon(fert: bool, harvest_at: int, seed: int = 70):
    env = make("kaggriculture", configuration={"episodeSteps": 14 * 24, "seed": seed}, debug=False)

    def ag(obs):
        me = obs["farms"][0]; priv = obs["private"]
        fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
        day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]
        if step == 0:
            o = [["BUY_SEED", "MELON", 1]]
            if fert:
                o.append(["BUY_PRODUCT", "FERTILIZER", 3])
            return {"farmer": ["PASS"], "hands": [], "market": o}
        if step == 1 and fert:
            return {"farmer": ["PICKUP", "FERTILIZER", 3], "hands": [], "market": []}
        if tile is None and priv["seeds"].get("MELON", 0):
            return {"farmer": ["PLANT", "MELON"], "hands": [], "market": []}
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            age = day - tile["planted_day"]
            if not tile["watered_today"]:
                return {"farmer": ["WATER"], "hands": [], "market": []}
            if fert and age == 6 and hour == 1 and priv["shed"].get("FERTILIZER", 0):
                return {"farmer": ["PICKUP", "FERTILIZER", 3], "hands": [], "market": []}
            if fert and age == 6 and hour == 2 and priv["inventories"][0].get("FERTILIZER", 0):
                return {"farmer": ["FERTILIZE"], "hands": [], "market": []}
            if age >= harvest_at and hour == 1 and tile["yield_units"] > 0:
                return {"farmer": ["HARVEST"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env.run([ag, "pass"])
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="lab/results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    env_f = run_melon(True, 8)
    html_f = env_f.render(mode="html")
    path_f = os.path.join(args.outdir, "melon_fert.html")
    with open(path_f, "w") as f:
        f.write(html_f)

    env_p = run_melon(False, 10)
    html_p = env_p.render(mode="html")
    path_p = os.path.join(args.outdir, "melon_plain.html")
    with open(path_p, "w") as f:
        f.write(html_p)

    print(f"wrote {path_f} ({len(html_f)} bytes)")
    print(f"wrote {path_p} ({len(html_p)} bytes)")


if __name__ == "__main__":
    main()
