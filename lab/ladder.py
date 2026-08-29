"""Opponent ladder — strength ranking of vendored opponents.

Method (per user spec):
  1. While unranked agents remain: pick two at random, play them; winner is
     inserted above the loser.
  2. Each subsequent candidate (winner or loser of the first round) is placed
     by binary insertion: play the middle of the current ranked list; win ->
     recurse into upper half, lose -> lower half, until final position.
  3. Ties: replay once with a fresh seed; if still tied, insert at that spot.
  4. 1 seed per game.

Usage:
    python -m lab.ladder [--seeds-start 1000] [--out docs/research/005-opponent-ladder.md]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from kaggle_environments import make

OPPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opponents")


def list_opponents() -> dict[str, str]:
    found = {}
    for d in sorted(os.listdir(OPPONENT_DIR)):
        p = os.path.join(OPPONENT_DIR, d, "agent.py")
        if os.path.exists(p):
            found[d] = p
    return found


def play(a_path: str, b_path: str, seed: int) -> tuple[float, float]:
    """One full game. Returns (reward_a, reward_b)."""
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run([a_path, b_path])
    rewards = [float(s.reward) if s.reward is not None else 0.0 for s in env.steps[-1]]
    return rewards[0], rewards[1]


def decide(a_path: str, b_path: str, seed: int) -> tuple[str, list[tuple[int, float, float]]]:
    """Returns 'a' if a wins, 'b' if b wins, 'tie' after one rematch."""
    games = []
    ra, rb = play(a_path, b_path, seed)
    games.append((seed, ra, rb))
    if ra > rb:
        return "a", games
    if rb > ra:
        return "b", games
    # tie -> one rematch with fresh seed
    seed2 = seed + 7919
    ra2, rb2 = play(a_path, b_path, seed2)
    games.append((seed2, ra2, rb2))
    if ra2 > rb2:
        return "a", games
    if rb2 > ra2:
        return "b", games
    return "tie", games


class Ladder:
    def __init__(self):
        self.ranked: list[str] = []      # strongest first
        self.games: list[dict] = []
        self.seed_counter = 1000

    def next_seed(self) -> int:
        self.seed_counter += 1
        return self.seed_counter

    def _record(self, a: str, b: str, outcome: str, game_logs: list[tuple[int, float, float]]):
        for seed, ra, rb in game_logs:
            self.games.append({"a": a, "b": b, "seed": seed, "ra": ra, "rb": rb, "outcome": outcome})

    def _insert(self, name: str, lo: int, hi: int) -> None:
        """Binary insertion of `name` into ranked[lo:hi]; plays the middle."""
        if lo >= hi:
            self.ranked.insert(lo, name)
            return
        mid = (lo + hi) // 2
        opp = self.ranked[mid]
        outcome, logs = decide(self._path(name), self._path(opp), self.next_seed())
        self._record(name, opp, outcome, logs)
        if outcome == "a":
            self._insert(name, lo, mid)      # won -> upper half (stronger side)
        elif outcome == "b":
            self._insert(name, mid + 1, hi)  # lost -> lower half
        else:
            self.ranked.insert(mid, name)    # still tied -> insert here

    @staticmethod
    def _path(name: str) -> str:
        return os.path.join(OPPONENT_DIR, name, "agent.py")

    def place_pair(self, a: str, b: str) -> None:
        """Play two unranked agents; winner placed above loser via binary insertion."""
        outcome, logs = decide(self._path(a), self._path(b), self.next_seed())
        self._record(a, b, outcome, logs)
        if outcome == "tie":
            # Insert both at the same midpoint position.
            mid = len(self.ranked) // 2
            self.ranked.insert(mid, a)
            self.ranked.insert(mid, b)
            return
        winner, loser = (a, b) if outcome == "a" else (b, a)
        self._insert(winner, 0, len(self.ranked))
        self._insert(loser, 0, len(self.ranked))

    def run(self, opponents: dict[str, str]) -> list[str]:
        names = list(opponents)
        rng = random.Random(42)
        while names:
            a = rng.choice(names)
            names.remove(a)
            if not names:
                self._insert(a, 0, len(self.ranked))
                break
            b = rng.choice(names)
            names.remove(b)
            self.place_pair(a, b)
            print(f"  ranked {len(self.ranked)}/{len(opponents)}: {a} vs {b} -> {outcome_str(self.ranked, a, b)}")
        return self.ranked


def outcome_str(ranked, a, b):
    return f"A@{ranked.index(a) + 1} B@{ranked.index(b) + 1}" if a in ranked and b in ranked else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/research/005-opponent-ladder.md")
    ap.add_argument("--json-out", default="lab/results/opponent_ladder.json")
    args = ap.parse_args()

    opponents = list_opponents()
    print(f"Ranking {len(opponents)} opponents...")
    ladder = Ladder()
    t0 = time.time()
    ranked = ladder.run(opponents)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min, {len(ladder.games)} games\n")

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump({"ranked": ranked, "games": ladder.games}, f, indent=2)

    lines = [
        "# Opponent Ladder — 19 vendored opponents",
        "",
        f"Sorted tournament: random pair → winner above loser, binary insertion vs middle,",
        f"tie → 1 rematch → insert at spot. 1 seed/game. {len(ladder.games)} games total.",
        "",
        "| Rank | Opponent |",
        "|---|---|",
    ]
    lines += [f"| {i + 1} | `{n}` |" for i, n in enumerate(ranked)]
    lines += ["", "## Game log", "", "| A | B | Seed | RA | RB | Winner |", "|---|---|---|---|---|---|"]
    for g in ladder.games:
        w = g["a"] if g["ra"] > g["rb"] else (g["b"] if g["rb"] > g["ra"] else "TIE")
        lines.append(f"| `{g['a']}` | `{g['b']}` | {g['seed']} | {g['ra']:.0f} | {g['rb']:.0f} | `{w[:30]}` |")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Ladder:")
    for i, n in enumerate(ranked):
        print(f"  {i + 1:2d}. {n}")
    print(f"\nWrote {args.out} and {args.json_out}")


if __name__ == "__main__":
    main()
