"""Season measurement on FastSim, one subprocess per (rival, seed).

Why subprocesses: the master's column generation is bounded by the clock, so two
seasons sharing cores can take different plans. A separate process per game keeps
the wall-clock contention off the plan's path, and `--jobs` decides how many run at
once. Every game is FastSim — the kaggle harness caps a turn at one second
(`kaggriculture.json`: actTimeout 1) and that cap is what made the plan's round count
follow the machine's speed.

Each seat gets ITS OWN observation: the pool agents read `obs["player"]` to know
which seat they are, and handing the rival player 0's view makes it play our farm.

usage:
  bench/season_runner.py --seeds 33,34,35 --rivals v3-agent,pass --jobs 3
  bench/season_runner.py --one v3-agent 33            # the child form (internal)
  bench/season_runner.py --agents /tmp/fs_main ...    # measure another checkout
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _one(repo: str, slug: str, seed: int, hands: int, rounds: int | None,
         budget_ms: float | None = None, search_s: float | None = None) -> dict:
    """Play one season in THIS process and return the result as a dict."""
    sys.path.insert(0, repo)
    from agent.config import Config
    import agent.runtime as R
    from offline_lab.fast_sim import FastSim

    cfg = Config()
    object.__setattr__(cfg, "max_hands", int(hands))
    if rounds:
        object.__setattr__(cfg, "master_rounds", int(rounds))
    if budget_ms:
        object.__setattr__(cfg, "turn_budget_ms", float(budget_ms))
    if search_s:
        object.__setattr__(cfg, "search_budget_s", float(search_s))
    R.RUNTIME.cfg = cfg

    loaded = None
    call = None                      # bound only when a rival is loaded
    if slug != "pass":
        from offline_lab.pool.loader import call as _call, load
        loaded, call = load(slug), _call

    def rival(obs):
        if loaded is None:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        return call(loaded, obs, None)

    sim = FastSim({"episodeSteps": 720, "seed": int(seed)})
    ours = theirs = 0.0
    crashes: list[str] = []
    for _ in range(719):
        views = sim.observations(copy_state=False)
        o0 = views[0]
        try:
            action = R.RUNTIME.act(o0, None)
        except Exception as exc:                       # a crash is a RESULT
            crashes.append(f"{type(exc).__name__}: {exc}")
            action = {"farmer": ["PASS"], "hands": [], "market": []}
        sim.step([action, rival(views[1])])
        last = sim.observations(copy_state=False)
        ours = float(last[0]["farms"][0]["money"])
        theirs = float(last[1]["farms"][1]["money"])
    return {"rival": slug, "seed": int(seed), "ours": ours, "theirs": theirs,
            "crashes": len(crashes), "first_crash": crashes[0] if crashes else ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default=str(REPO), help="checkout to measure")
    ap.add_argument("--seeds", default="33,34,35")
    ap.add_argument("--rivals", default="v3-agent,pass")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--hands", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=None)
    # The clock knobs are OURS, not FastSim's: FastSim has no per-turn cap, but the
    # manager still reads Config.turn_budget_ms and the day search its
    # search_budget_s, so a measurement that leaves them at the shipped 965 ms is
    # still measuring a machine's speed rather than a plan.
    ap.add_argument("--budget-ms", type=float, default=None)
    ap.add_argument("--search-s", type=float, default=None)
    ap.add_argument("--one", nargs=2, metavar=("RIVAL", "SEED"), help="child form")
    args = ap.parse_args()

    if args.one:
        slug, seed = args.one
        print(json.dumps(_one(args.agents, slug, int(seed), args.hands,
                              args.rounds, args.budget_ms, args.search_s)),
              flush=True)
        return 0

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rivals = [r for r in args.rivals.split(",") if r.strip()]
    games = [(r, s) for r in rivals for s in seeds]
    print(f"=== {len(games)} games on FastSim, {args.jobs} at a time, "
          f"agents={args.agents} ===", flush=True)

    def run(job):
        slug, seed = job
        cmd = [sys.executable, str(Path(__file__).resolve()), "--one", slug,
               str(seed), "--agents", args.agents, "--hands", str(args.hands)]
        if args.rounds:
            cmd += ["--rounds", str(args.rounds)]
        if args.budget_ms:
            cmd += ["--budget-ms", str(args.budget_ms)]
        if args.search_s:
            cmd += ["--search-s", str(args.search_s)]
        started = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        out = (proc.stdout or "").strip().splitlines()
        row = json.loads(out[-1]) if out else {"rival": slug, "seed": seed,
                                              "ours": 0.0, "theirs": 0.0,
                                              "crashes": -1,
                                              "first_crash": proc.stderr[-160:]}
        row["seconds"] = round(time.time() - started, 1)
        return row

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = list(pool.map(run, games))

    for row in sorted(rows, key=lambda r: (r["rival"], r["seed"])):
        print(f"{row['rival']:>9} seed {row['seed']}: OURS={row['ours']:9.0f} "
              f"THEIRS={row['theirs']:9.0f} crashes={row['crashes']:2d} "
              f"({row['seconds']:5.1f}s)"
              + (f"  {row['first_crash'][:80]}" if row.get("first_crash") else ""),
              flush=True)
    by_rival: dict[str, list[float]] = {}
    for row in rows:
        by_rival.setdefault(row["rival"], []).append(row["ours"])
    for slug, scores in sorted(by_rival.items()):
        mean = sum(scores) / len(scores)
        print(f"mean vs {slug}: {mean:9.0f}  over {len(scores)} seeds "
              f"({min(scores):.0f}-{max(scores):.0f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
