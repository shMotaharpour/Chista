"""ONE-tile truth probe: workers -> tile -> per-action effect -> edge match.

The owner's method, applied to a single tile end-to-end:

  1. worker-tile match: every action of the day whose actor STOOD on the
     cell when acting (pre-turn position, the planner's own knowledge);
  2. per-action effect: the tile's engine fields are diffed across that
     turn (`steps[t-1]` vs `steps[t]`) — an action that changes nothing on
     the tile is PHANTOM (a second WATER on a wet tile, a PLACE the engine
     refused);
  3. the day's actions on the tile collect into a MULTISET (counts matter);
  4. that multiset is matched against the ALLOWED edges of the tile's
     hour-0 state — edges are unordered chains, so the match is by op
     counts; the to-state must agree as well;
  5. report: the chosen edge, the phantom ops with their actor/hour, and
     whether the transition is a graph gap.

Usage:
    .venv/bin/python offline_lab/build/probe_one_tile.py \
        <replay.json> <shard.npz> <sample> <x> <y>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from agent.planner.inputs import GRAPH_PATH as _GRAPH_PATH
from agent.tile_dp.chains import chain_ops
from agent.tile_dp.graph import TileGraph
from agent.tile_dp.tile_state import decode_tile
from agent.world.rules import TURNS_PER_DAY

NON_TILE = {"PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP",
            "INVALID", "HIRE", "BUY_LAND"}


def tile_fields(tile) -> dict:
    if tile is None:
        return {"kind": "EMPTY"}
    if isinstance(tile, str):
        return {"kind": tile}
    return {k: tile.get(k) for k in tile}


def main() -> None:
    replay, shard_path, sample, x, y = (
        sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    cell = y * 10 + x
    sh = np.load(shard_path, allow_pickle=False)
    seat, day = int(sh["seat"][sample]), int(sh["day"][sample])
    voc = json.loads((Path(shard_path).parent / "vocab.json").read_text())["list"]
    raw = json.load(open(replay))
    steps = raw["steps"]

    graph = TileGraph.load(_GRAPH_PATH)
    from_key = int(sh["tiles_own"][sample][cell])
    to_key = int(sh["tiles_own"][sample + 1][cell])   # seat/day +1 = same seat next day
    sid_f = graph.key_index.get(from_key)
    st_f, st_t = decode_tile(steps[day * 24][seat]["observation"]["farms"][seat]["tiles"][y][x], day), \
        decode_tile(steps[(day + 1) * 24][seat]["observation"]["farms"][seat]["tiles"][y][x], day + 1)
    print(f"== tile ({x},{y})  sample {sample} (seat {seat}, day {day}) ==")
    print(f"from: {st_f}")
    print(f"to:   {st_t}")

    # 1+2: the day's actions on the cell, each with its own before/after diff
    acts = []
    for h in range(1, TURNS_PER_DAY + 1):
        t = day * TURNS_PER_DAY + h
        if t >= len(steps):
            break
        prev = steps[t - 1][seat]["observation"]["farms"][seat]
        act = steps[t][seat].get("action") or {}
        units = [("farmer", act.get("farmer") or [])]
        units += [(f"hand{i}", list(o or [])) for i, o in enumerate(act.get("hands") or [])]
        positions = [tuple(prev["farmer"])] + [tuple(p) for p in (prev.get("hands") or [])]
        before = steps[t - 1][seat]["observation"]["farms"][seat]["tiles"][y][x]
        after = steps[t][seat]["observation"]["farms"][seat]["tiles"][y][x]
        changed = tile_fields(before) != tile_fields(after)
        for actor, (name, op) in enumerate(units):
            if not op or str(op[0]) in NON_TILE:
                continue
            stood = tuple(positions[actor]) == (x, y) if actor < len(positions) else False
            acts.append((h, name, str(op[0]), stood, changed))

    print("\nactions on this cell (hour, actor, op, stood-here, tile-changed):")
    for a in acts:
        print("  ", a)

    observed = Counter(a[2] for a in acts if a[3])
    phantom_ops = [a for a in acts if a[3] and not a[4] and a[2] != "COLLECT_FERTILIZER"]
    effective = Counter(a[2] for a in acts if a[3] and (a[4] or a[2] == "COLLECT_FERTILIZER"))
    print(f"\nobserved multiset: {dict(observed)}")
    print(f"effective (changed the tile): {dict(effective)}")
    print(f"phantom (no tile change): {[(a[0], a[1], a[2]) for a in phantom_ops]}")

    # 4: allowed edges of the from-state, matched by op COUNTS
    print("\nallowed edges out of from-state (op-count bag -> chain id/name),")
    print("matched against the EFFECTIVE multiset:")
    cands = []
    for e in graph.edges_from(sid_f):
        bag = Counter(str(o) for o in chain_ops(int(e.chain_id)))
        to_key_e = int(graph.state_keys[int(e.to_id)])
        cands.append((bag, int(e.chain_id), chain_ops(int(e.chain_id)), to_key_e == to_key))
    exact = [c for c in cands if c[0] == effective and c[3]]
    subset = [c for c in cands if c[0] <= effective and c[0] and c[3]]
    for bag, cid, ops, to_ok in sorted(cands, key=lambda c: -sum(c[0].values()))[:8]:
        mark = " <== EXACT" if (bag == effective and to_ok) else \
               (" <= subset" if (bag <= effective and bag and to_ok) else "")
        print(f"   {dict(bag)} chain={cid} to_ok={to_ok}{mark}")
    if exact:
        print(f"\nVERDICT: chain {exact[0][1]} {exact[0][2]} (exact multiset match)")
    elif subset:
        best = max(subset, key=lambda c: (sum(c[0].values()), -c[1]))
        extra = effective - best[0]
        print(f"\nVERDICT: chain {best[1]} {best[2]}; excess ops (phantom beyond chain): {dict(extra)}")
    else:
        print("\nVERDICT: NO edge matches (graph gap or transition mismatch)")


if __name__ == "__main__":
    main()