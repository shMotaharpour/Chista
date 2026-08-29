"""Aggregate report from a results directory.

Usage:
    python -m lab.report <run-dir> [...]
"""
from __future__ import annotations

import glob
import json
import math
import sys


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_run(run_dir: str) -> list[dict]:
    files = sorted(glob.glob(f"{run_dir}/game_*.json"))
    if not files:
        raise SystemExit(f"No game_*.json in {run_dir}")
    return [json.load(open(f)) for f in files]


def summarize(results: list[dict], side: str = "a") -> dict:
    """Delta from the perspective of the designated agent across both side-swaps."""
    deltas = []
    wins = 0
    for r in results:
        if r["agent_a"] == side or True:  # agent names tracked per game
            pass
    # Perspective: 'first-named agent of the run' — we use manifest instead.
    return {}


def summarize_manifest(run_dir: str) -> dict:
    manifest = json.load(open(f"{run_dir}/manifest.json"))
    a, b = manifest["a"], manifest["b"]
    deltas, wins, n = [], 0, 0
    for r in load_run(run_dir):
        # Always report from A's perspective: if sides were swapped in this
        # game (agent_a == b), flip rewards back.
        ra, rb = r["rewards"]
        if r["agent_a"] == b and r["agent_b"] == a:
            ra, rb = rb, ra
        deltas.append(ra - rb)
        n += 1
        if ra > rb:
            wins += 1
    deltas.sort()
    mean = sum(deltas) / n
    median = deltas[n // 2] if n % 2 else (deltas[n // 2 - 1] + deltas[n // 2]) / 2
    var = sum((d - mean) ** 2 for d in deltas) / max(1, n - 1)
    se = math.sqrt(var / n) if n > 1 else 0.0
    lo, hi = wilson_ci(wins, n)
    return {
        "a": a, "b": b, "games": n,
        "mean_delta": mean, "median_delta": median, "se": se,
        "win_rate": wins / n, "win_ci95": (lo, hi),
    }


def print_table(s: dict) -> None:
    lo, hi = s["win_ci95"]
    print(f"\n=== {s['a']} vs {s['b']} ({s['games']} paired games, sides swapped) ===")
    print(f"  mean delta  (A-B): {s['mean_delta']:+.1f}")
    print(f"  median delta     : {s['median_delta']:+.1f}")
    print(f"  std error        : {s['se']:.1f}")
    print(f"  win rate         : {s['win_rate']:.1%}  [95% CI {lo:.1%} – {hi:.1%}]")


def main() -> None:
    for run_dir in sys.argv[1:]:
        print_table(summarize_manifest(run_dir))


if __name__ == "__main__":
    main()
