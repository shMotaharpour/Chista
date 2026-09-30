"""One numpy archive of the planning worlds, built from the extracted scenario axes.

Sources (produced by offline_lab/scenario_axes/extract.py, written outside the repo):
  rival_realised.parquet  what their orders actually sold, from the city inventory identity
                          (built by offline_lab/build/realised_supply.py)
  demand_{p}.parquet   the town's demand vector per (game, day)
  harvest_v2.parquet   every cut of a positive counter, with its delay past the gate

Writes agent/artifact/scenario_worlds.npz (AGENTS.md: the agent loads what lives inside agent/):
  rival_supply (W, days, 9) float32  their harvesting, put on the market by day
  demand       (W, days, 9) float32  the town's appetite, by day
  weights      (W,)         float32  how much of the archive each world stands for
  crop         (9,)         <U10    the good order every array uses
  phase0/phase1 (W,)        int8     the day window the world was cut from (0-9 / 10-14 / 15+)
  support      (W,)         int32    game-days behind the world: its evidence (R005)
Prices are deliberately NOT stored: they are the agent's own `_priced_paths` output, so a
world becomes prices in exactly one place.
"""
from __future__ import annotations

import glob
import os
import sys

import duckdb
import numpy as np
import pandas as pd

GOODS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
PHASES = ((0, 9), (10, 14), (15, 30))     # the three regimes the archive shows
DAYS = 31
SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scenario_axes"
OUT = sys.argv[2] if len(sys.argv) > 2 else "agent/artifact/scenario_worlds.npz"
W = int(sys.argv[3]) if len(sys.argv) > 3 else 96
SEED = 20260930


def tensor(frame: pd.DataFrame, value_cols) -> tuple[np.ndarray, list]:
    """(games, DAYS, 9) — one vector per game-day; the archive's silence stays zero."""
    games = pd.unique(frame["episode_id"]).tolist()
    gi = {g: i for i, g in enumerate(games)}
    cols = {g: j for j, g in enumerate(GOODS)}
    T = np.zeros((len(games), DAYS, 9), np.float32)
    for g in value_cols:
        if g not in frame.columns:
            continue
        e = frame["episode_id"].map(gi).to_numpy()
        d = frame["day"].to_numpy().astype(np.int32)
        T[e, d, cols[g]] = frame[g].fillna(0.0).to_numpy()
    return T, games


def read(pattern: str) -> pd.DataFrame:
    """DuckDB reads the parquet (pandas has no engine installed here, and this box is small)."""
    con = duckdb.connect()
    for st in ("SET memory_limit='1500MB'", "SET threads=2", "SET temp_directory='/tmp/duckdb_swap'"):
        con.execute(st)
    files = sorted(glob.glob(pattern.replace("*", "*")))
    df = con.execute("SELECT * FROM read_parquet([" + ", ".join(f"'{f}'" for f in files) + "])").fetchdf()
    con.close()
    return df


r = read(f"{SRC}/rival_realised.parquet")   # REALISED flow (inventory identity), not the logged orders
R, games = tensor(r, GOODS)
d = read(f"{SRC}/demand_*.parquet")
D, dgames = tensor(d, GOODS)

shared = np.array([g for g in games if g in set(dgames)])
rng = np.random.default_rng(SEED)
idx: list[tuple[object, int, int]] = []
for a, b in PHASES:                                  # stratified: every phase is represented
    pick = rng.choice(shared, size=min(len(shared), max(1, W // len(PHASES))), replace=False)
    idx += [(g, a, b) for g in pick]
idx = idx[:W]
# every world carries the FULL season span so the arrays stack; the phase is a label
# (`phase0`/`phase1`). A world's weight is the phase's share of the archive, split evenly
# among the worlds cut from it -- so the weights sum to 1 by construction, not by luck.
rival_supply = np.stack([R[games.index(g)] for g, _, _ in idx]).astype(np.float32)
demand = np.stack([D[dgames.index(g)] for g, _, _ in idx]).astype(np.float32)
phase_len = np.array([b - a + 1 for a, b in PHASES], dtype=np.float64)      # days in each phase
phase_days = phase_len * np.array([sum(1 for _, a2, b2 in idx if (a2, b2) == (a, b)) for a, b in PHASES],
                                  dtype=np.float64)                          # game-days in each phase
phase_share = phase_days / phase_days.sum()
support = np.array([phase_days[PHASES.index((a, b))] / max(1, sum(1 for _, x, y in idx if (x, y) == (a, b)))
                    for _, a, b in idx], np.float32)                          # game-days behind THIS world
weights = np.array([phase_share[PHASES.index((a, b))] / max(1, sum(1 for _, x, y in idx if (x, y) == (a, b)))
                    for _, a, b in idx], np.float32)
assert abs(weights.sum() - 1.0) < 1e-5, f"weights must sum to 1, got {weights.sum()}"
assert (support > 0).all(), "every world must name the evidence behind it"

np.savez_compressed(OUT, rival_supply=rival_supply, demand=demand, weights=weights,
                    crop=np.array(GOODS), phase0=np.array([a for _, a, _ in idx], np.int8),
                    phase1=np.array([b for _, _, b in idx], np.int8), support=support)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)")
print(f"  worlds={rival_supply.shape[0]}  days={rival_supply.shape[1]}  goods={rival_supply.shape[2]}")
print(f"  weights: sum={weights.sum():.4f} min={weights.min():.5f} max={weights.max():.5f}")
print(f"  rival_supply: mean={rival_supply.mean():.2f} max={rival_supply.max():.0f} | demand: mean={demand.mean():.2f} max={demand.max():.0f}")
print(f"  support (game-days per world): min={support.min()} max={support.max()}")
