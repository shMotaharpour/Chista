"""Kaggle replay -> imitation dataset (format 1, see FORMAT.md).

Contract (verified against episode 111017932, not assumed):

- the action recorded at `steps[t][seat]` is what CAUSED the transition
  `obs[t-1] -> obs[t]` (sell revenue lands in obs[t].money; the day's first
  action is `steps[24D+1]`, answering the hour-0 observation `steps[24D]`);
- `action[t].hands` aligns with `obs[t-1].farms[s].hands` (the pre-turn crew),
  and HIRE orders settle atomically before the per-unit loop (F031/F040);
- a sample is ONE (episode, seat, day): state = hour-0 observation, labels =
  the day's 24 actions with each unit's cell taken from the observation one
  turn before it acts.

RAM contract: one replay per process (~32 MB json, a few hundred MB of
objects), one npz shard out, then exit — the 1 GB box reclaims it all.

Usage:
    .venv/bin/python offline_lab/build/kaggle_dataset.py <replay.json>... \
        --out offline_lab/build/kaggle_shards
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from agent.obs import LOCKED_KEY
from agent.planner.inputs import GRAPH_PATH as _GRAPH_PATH
from agent.tile_dp.graph import TileGraph
from agent.tile_dp.tile_state import decode_tile
from agent.world.model import ANIMALS, PRODUCTS
from agent.world.rules import BOARD_SIZE, SHOPS, TURNS_PER_DAY

#: The engine's own orders (world/model.py + master.py's shed_stock order).
PRODUCTS_ORDER = list(PRODUCTS)
SHED_ORDER = PRODUCTS_ORDER + list(ANIMALS)
SEED_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
SHOPS_ORDER = sorted(SHOPS)


def _cell_index(x: int, y: int) -> int:
    """Row-major cell id — the board's own walk order."""
    return int(y) * BOARD_SIZE + int(x)


class Vocab:
    """String -> small int, data-driven and shipped as vocab.json."""

    def __init__(self) -> None:
        self.strings: list[str] = []
        self.index: dict[str, int] = {}

    def code(self, s: str | None) -> int:
        if s is None:
            return -1
        if s not in self.index:
            self.index[s] = len(self.strings)
            self.strings.append(s)
        return self.index[s]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"codes": self.index, "list": self.strings},
                                   indent=1))


def _tiles_to_keys(tiles: list, day: int, key_index: dict) -> tuple[np.ndarray, int]:
    """One farm's 100 tiles -> graph state keys, row-major."""
    out = np.empty(BOARD_SIZE * BOARD_SIZE, dtype=np.int32)
    unknown = 0
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            tile = tiles[y][x]
            if isinstance(tile, str) and tile == "LOCKED":
                out[_cell_index(x, y)] = LOCKED_KEY
                continue
            key = int(decode_tile(tile, day).pack())
            if key not in key_index:
                unknown += 1
            out[_cell_index(x, y)] = key
    return out, unknown


def _state_from_obs(obs: dict, seat: int, key_index: dict) -> tuple[dict, int]:
    """The hour-0 sample state, straight from the seat's own observation."""
    farm = obs["farms"][seat]
    other = obs["farms"][1 - seat]
    day = int(obs["day"])
    tiles_own, unk_own = _tiles_to_keys(farm["tiles"], day, key_index)
    tiles_opp, unk_opp = _tiles_to_keys(other["tiles"], day, key_index)

    own_priv = obs.get("private", {}) or {}
    # The rival's private block: each seat's observation carries its OWN
    # private only, so the rival shed is read from the RIVAL SEAT's row of
    # the same step (the convertor keeps both seats per step).
    shed_own = own_priv.get("shed", {}) or {}
    seeds_own = own_priv.get("seeds", {}) or {}
    market = obs.get("market", {}) or {}
    prices = market.get("prices", {}) or {}
    inventory = market.get("inventory", {}) or {}
    shops = obs.get("town", {}).get("unlocked_shops", []) or []

    return {
        "tiles_own": tiles_own, "tiles_opp": tiles_opp,
        "money_own": np.float32(np.log1p(max(0.0, float(farm["money"])))),
        "money_opp": np.float32(np.log1p(max(0.0, float(other["money"])))),
        "shed_own": np.array([int(shed_own.get(g, 0)) for g in SHED_ORDER],
                             dtype=np.int32),
        "shed_opp": np.zeros(len(SHED_ORDER), dtype=np.int32),  # filled by caller
        "seeds": np.array([int(seeds_own.get(g, 0)) for g in SEED_ORDER],
                          dtype=np.int32),
        "prices": np.array([float(prices.get(g, 0.0)) for g in PRODUCTS_ORDER],
                           dtype=np.float32),
        "mkt_inv": np.array([float(inventory.get(g, 0.0)) for g in PRODUCTS_ORDER],
                            dtype=np.float32),
        "shops": np.array([1 if s in shops else 0 for s in SHOPS_ORDER],
                          dtype=np.uint8),
        "day": np.uint8(day),
    }, unk_own + unk_opp


def _labels_for_day(steps: list, seat: int, day: int, n_steps: int,
                    v: Vocab) -> dict[str, list]:
    """The day's unit ops and market orders, with pre-turn cells.

    The day window is `24D+1 .. 24D+24` (the contract FORMAT.md records) and
    the LAST of those actions belongs to hour 0 of day D+1 by `t % 24`. It is
    still THIS day's plan — the planner wrote it at hour 0 of D — so the
    label keeps the day it was PLANNED on (`lab` hours run 1..24), while
    state side of the next sample carries the tile's rolled-over state.
    """
    ops: list[tuple] = []
    orders: list[tuple] = []
    lo = day * TURNS_PER_DAY + 1
    hi = min(lo + TURNS_PER_DAY, n_steps)
    for t in range(lo, hi):
        act = steps[t][seat].get("action") or {}
        prev = steps[t - 1][seat]["observation"]      # where units stood
        farm = prev["farms"][seat]
        hour = 1 + (t - lo)                           # 1..24, planned-day hours

        positions = [tuple(farm["farmer"])]
        positions += [tuple(h) for h in (farm.get("hands") or [])]
        unit_ops = [act.get("farmer") or []]
        unit_ops += [list(o or []) for o in (act.get("hands") or [])]
        for actor, op in enumerate(unit_ops):
            if not op:
                continue
            x, y = positions[actor] if actor < len(positions) else (-1, -1)
            item = str(op[1]) if len(op) > 1 else None
            n = int(op[2]) if len(op) > 2 else 0
            ops.append((hour, actor, _cell_index(x, y) if x >= 0 else -1,
                        v.code(str(op[0])), v.code(item), n))
        for idx, mo in enumerate(act.get("market") or []):
            if not mo:
                continue
            item = str(mo[1]) if len(mo) > 1 else None
            n = int(mo[2]) if len(mo) > 2 else 0
            orders.append((hour, idx, v.code(str(mo[0])), v.code(item), n))
    return {"ops": ops, "orders": orders}


def convert_replay(path: Path, key_index: dict, v: Vocab) -> dict | None:
    """One replay file -> one shard dict, or None when it must be skipped."""
    with open(path, "rb") as fh:
        raw = json.load(fh)
    steps = raw["steps"]
    n_steps = len(steps)
    info = raw.get("info", {})
    agents = [a["Name"] for a in info.get("Agents", [])]
    rewards = raw.get("rewards", [0, 0])
    episode_id = int(info.get("EpisodeId") or 0)
    seed = int(info.get("seed") or 0)

    samples: list[dict] = []
    ops_rows: list[tuple] = []
    ord_rows: list[tuple] = []
    op_ptr, ord_ptr = [0], [0]
    unknown_total = 0

    for seat in (0, 1):
        opp_sheds = {}
        # The rival shed per day: the rival seat's own private at the same
        # step. Read once per day, outside the sample dict.
        for day in range(30):
            t0 = day * TURNS_PER_DAY
            if t0 >= n_steps:
                break
            opp_priv = steps[t0][1 - seat].get("observation", {}).get("private", {}) or {}
            opp_sheds[day] = opp_priv.get("shed", {}) or {}

        for day in range(30):
            t0 = day * TURNS_PER_DAY
            if t0 >= n_steps:
                break
            obs = steps[t0][seat]["observation"]
            state, unk = _state_from_obs(obs, seat, key_index)
            unknown_total += unk
            opp_shed = opp_sheds[day]
            state["shed_opp"] = np.array(
                [int(opp_shed.get(g, 0)) for g in SHED_ORDER], dtype=np.int32)
            labels = _labels_for_day(steps, seat, day, n_steps, v)
            ops_rows.extend(labels["ops"])
            ord_rows.extend(labels["orders"])
            samples.append({
                "episode_id": episode_id, "seed": seed, "seat": seat,
                "agent": agents[seat] if seat < len(agents) else "",
                "reward": int(rewards[seat] if seat < len(rewards) else 0),
                **state,
            })
            op_ptr.append(len(ops_rows))
            ord_ptr.append(len(ord_rows))

    n = len(samples)
    if n == 0:
        return None
    return {
        "n_samples": n,
        "episode_id": np.full(n, episode_id, dtype=np.int64),
        "seed": np.full(n, seed, dtype=np.int64),
        "seat": np.array([s["seat"] for s in samples], dtype=np.int8),
        "agent": np.array([s["agent"] for s in samples], dtype=np.str_),
        "reward": np.array([s["reward"] for s in samples], dtype=np.int64),
        "day": np.array([int(s["day"]) for s in samples], dtype=np.uint8),
        "tiles_own": np.stack([s["tiles_own"] for s in samples]),
        "tiles_opp": np.stack([s["tiles_opp"] for s in samples]),
        "money_own": np.array([s["money_own"] for s in samples], dtype=np.float32),
        "money_opp": np.array([s["money_opp"] for s in samples], dtype=np.float32),
        "shed_own": np.stack([s["shed_own"] for s in samples]),
        "shed_opp": np.stack([s["shed_opp"] for s in samples]),
        "seeds": np.stack([s["seeds"] for s in samples]),
        "prices": np.stack([s["prices"] for s in samples]),
        "mkt_inv": np.stack([s["mkt_inv"] for s in samples]),
        "shops": np.stack([s["shops"] for s in samples]),
        "op_ptr": np.array(op_ptr, dtype=np.int64),
        "op_hour": np.array([r[0] for r in ops_rows], dtype=np.int8),
        "op_actor": np.array([r[1] for r in ops_rows], dtype=np.int8),
        "op_cell": np.array([r[2] for r in ops_rows], dtype=np.int16),
        "op_code": np.array([r[3] for r in ops_rows], dtype=np.int16),
        "op_item": np.array([r[4] for r in ops_rows], dtype=np.int16),
        "op_qty": np.array([r[5] for r in ops_rows], dtype=np.int32),
        "ord_ptr": np.array(ord_ptr, dtype=np.int64),
        "ord_hour": np.array([r[0] for r in ord_rows], dtype=np.int8),
        "ord_idx": np.array([r[1] for r in ord_rows], dtype=np.int8),
        "ord_code": np.array([r[2] for r in ord_rows], dtype=np.int16),
        "ord_item": np.array([r[3] for r in ord_rows], dtype=np.int16),
        "ord_qty": np.array([r[4] for r in ord_rows], dtype=np.int32),
        "unknown_keys": np.int64(unknown_total),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="+")
    ap.add_argument("--out", default="offline_lab/build/kaggle_shards")
    args = ap.parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = TileGraph.load(_GRAPH_PATH)
    key_index = graph.key_index
    v = Vocab()
    print(f"graph: {len(key_index)} states")

    total_samples = 0
    for path in args.replays:
        path = Path(path)
        shard = convert_replay(path, key_index, v)
        if shard is None:
            print(f"{path.name}: nothing to write")
            continue
        np.savez_compressed(out_dir / f"{int(shard['episode_id'][0])}.npz", **shard)
        total_samples += int(shard["n_samples"])
        print(f"{path.name}: {shard['n_samples']} samples, "
              f"{len(shard['op_hour'])} ops, {len(shard['ord_hour'])} orders, "
              f"{int(shard['unknown_keys'])} unknown keys")
    v.save(out_dir / "vocab.json")
    print(f"total samples: {total_samples}, vocab: {len(v.strings)}")


if __name__ == "__main__":
    main()