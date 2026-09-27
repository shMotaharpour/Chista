"""Edge labels from worker positions, per-action tile effects, and transitions.

The owner's method, confirmed on one tile end-to-end (probe_one_tile.py):

  1. WORKER-TILE MATCH: `op_cell` is where the actor STOOD one turn before
     acting — the planner's own knowledge at decision time. Tile ops apply to
     the standing tile, so the cell IS the tile acted on.
  2. PER-ACTION EFFECT: the tile's engine dict is diffed across each turn
     (`steps[t-1]` vs `steps[t]`). An op-hour on a tile that changes nothing
     is PHANTOM (a second WATER on a wet tile, a PLACE the engine refused).
     Hour 24's turn spans the nightly reset and cannot be judged from the
     raw diff, so its ops stay in the multiset unverified and the transition
     match arbitrates them.
  3. DAY MULTISET: the day's EFFECTIVE ops on the tile collect WITH COUNTS
     (edges are unordered chains; matching is by op counts, never order).
  4. EDGE MATCH: candidate edges of the tile's hour-0 state whose TO state
     equals the tile's next-day state; among them, the largest op-count bag
     contained in the effective multiset wins, ties to the lowest chain id —
     the minimal-work principle: the state delta is the truth.
  5. A transition no edge explains is LAB_OUT_OF_GRAPH (counted by kind);
     day 29 has no next state and labels LAB_LAST_DAY.

Adds per shard (format 1 extension, see FORMAT.md):
    lab_ptr (n+1,) int64 · lab_cell (ΣL,) int16 · lab_chain (ΣL,) int32
    lab_phantom (ΣL,) int16 · lab_from_kind (ΣL,) int8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from agent.planner.inputs import GRAPH_PATH as _GRAPH_PATH
from agent.tile_dp.chains import chain_ops
from agent.tile_dp.graph import TileGraph
from agent.tile_dp.tile_state import KIND_CODES

kind_code_of = {k: i for i, k in enumerate(KIND_CODES)}

LAB_OUT_OF_GRAPH = -2
LAB_LAST_DAY = -3

#: Ops that are NOT tile work and belong to no chain: movement (the routing
#: layer's decision), PASS, unit-bag logistics, the engine's rejects.
NON_TILE_OPS = frozenset({
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "INVALID",
    "HIRE", "BUY_LAND",
})


def _tile_signature(tile) -> tuple:
    """The tile dict as a comparable tuple (None/LOCKED included)."""
    if tile is None:
        return (None,)
    if isinstance(tile, str):
        return (tile,)
    return tuple(sorted((k, str(v)) for k, v in tile.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", nargs="+", required=True,
                    help="the raw replay JSONs, for the per-turn tile diffs")
    ap.add_argument("--shards", nargs="+", required=True)
    args = ap.parse_args()

    graph = TileGraph.load(_GRAPH_PATH)
    key_index = graph.key_index
    state_keys = np.asarray(graph.state_keys, dtype=np.int64)

    # Edge index: from_state_id -> to_state_id -> [(op bag WITH COUNTS, chain_id, edge row)]
    by_transition: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for sid in range(graph.n_states):
        lo, hi = graph.edges_of(sid)
        for row in range(lo, hi):
            to_id = int(graph.edge_next[row])
            chain_id = int(graph.edge_chain[row])
            bag = Counter(str(op) for op in chain_ops(chain_id))
            by_transition[sid][int(state_keys[to_id])].append((bag, chain_id, row))

    #: The graph's own entity names -> the shard's item-vocabulary codes. The
    #: edge entity is the crop a PLANT plants or the species a PLACE places;
    #: the training loss consumes the vocabulary code directly.
    kind_code = {sid: kind_code_of[graph.state_of(sid).kind]
                 for sid in range(graph.n_states)}

    replay_of = {}
    for path in args.replays:
        path = Path(path)
        replay_of[int(path.name.split("-")[1])] = path

    #: The entity-name -> vocabulary-code table. The vocabulary is per build
    #: (`vocab.json`), so the codes are read from the FIRST shard's table and
    #: reused for every shard of the same build.
    from agent.tile_dp.chains import ENTITY_NAMES, ENTITY_OF_CODE
    first_voc = json.loads((Path(args.shards[0]).parent / "vocab.json").read_text())["list"]
    _ENTITY_OF = {name: first_voc.index(name) for name in ENTITY_NAMES}

    totals: Counter = Counter()
    for shard_path in args.shards:
        shard_path = Path(shard_path)
        sh = np.load(shard_path, allow_pickle=False)
        n = int(sh["n_samples"])
        episode_id = int(sh["episode_id"][0])
        raw = json.load(open(replay_of[episode_id]))
        steps = raw["steps"]
        n_steps = len(steps)
        voc = json.loads((shard_path.parent / "vocab.json").read_text())["list"]

        index_of = {(int(sh["seat"][i]), int(sh["day"][i])): i for i in range(n)}
        op_cell, op_code = sh["op_cell"], sh["op_code"]
        op_ptr = sh["op_ptr"]

        lab_cell: list[int] = []
        lab_chain: list[int] = []
        lab_item: list[int] = []
        lab_phantom: list[int] = []
        lab_from_kind: list[int] = []
        lab_ptr = np.zeros(n + 1, dtype=np.int64)
        shard_stats: Counter = Counter()

        for i in range(n):
            seat, day = int(sh["seat"][i]), int(sh["day"][i])
            tiles = sh["tiles_own"][i]

            # 1. the day's tile ops per cell, with the hours they ran
            cell_hours: dict[int, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
            for j in range(int(op_ptr[i]), int(op_ptr[i + 1])):
                name = str(voc[op_code[j]])
                if name in NON_TILE_OPS:
                    continue
                cell = int(op_cell[j])
                if cell >= 0:
                    cell_hours[cell][int(sh["op_hour"][j])][name] += 1

            nxt = index_of.get((seat, day + 1))

            for cell in range(100):
                from_key = int(tiles[cell])
                if from_key < 0:            # LOCKED sentinel
                    continue
                sid_f = key_index.get(from_key)
                if sid_f is None:           # not modelled even after remap
                    shard_stats["unmodelled_from"] += 1
                    continue

                hours = cell_hours.get(cell, {})
                y, x = divmod(cell, 10)

                # 2. per-action effect: which op-hours changed the tile
                effective: Counter = Counter()
                phantom = 0
                for h, counts in hours.items():
                    t = day * 24 + h        # hour 1..24 -> absolute step
                    if t >= n_steps:
                        continue
                    if h == 24:             # crosses the nightly reset
                        effective.update(counts)
                        continue
                    changed = _tile_signature(
                        steps[t - 1][seat]["observation"]["farms"][seat]["tiles"][y][x]) \
                        != _tile_signature(
                            steps[t][seat]["observation"]["farms"][seat]["tiles"][y][x])
                    if changed:
                        effective.update(counts)
                    else:
                        phantom += int(sum(counts.values()))
                        shard_stats["phantom"] += int(sum(counts.values()))

                if nxt is None:             # day 29: no next-day state
                    lab_cell.append(cell)
                    lab_chain.append(LAB_LAST_DAY)
                    lab_item.append(-1)
                    lab_phantom.append(phantom)
                    lab_from_kind.append(kind_code[sid_f])
                    shard_stats["last_day"] += 1
                    continue

                # 4. the transition's candidates, matched by op COUNTS
                to_key = int(sh["tiles_own"][nxt][cell])
                cands = by_transition.get(sid_f, {}).get(to_key, [])
                if not cands:
                    lab_cell.append(cell)
                    lab_chain.append(LAB_OUT_OF_GRAPH)
                    lab_item.append(-1)
                    lab_phantom.append(phantom + int(sum(effective.values())))
                    lab_from_kind.append(kind_code[sid_f])
                    shard_stats["out_of_graph"] += 1
                    continue

                best = None
                for bag, chain_id, row in cands:
                    if not all(effective.get(op, 0) >= cnt
                               for op, cnt in bag.items()):
                        continue
                    key = (-sum(bag.values()), chain_id)   # max coverage, then lowest id
                    if best is None or key < best[0]:
                        best = (key, chain_id, row)
                if best is None:
                    lab_cell.append(cell)
                    lab_chain.append(LAB_OUT_OF_GRAPH)
                    lab_item.append(-1)
                    lab_phantom.append(phantom + int(sum(effective.values())))
                    lab_from_kind.append(kind_code[sid_f])
                    shard_stats["opset_gap"] += 1
                    continue

                _key, chain_id, row = best
                # The edge's own entity, from the graph: a PLANT chain's crop
                # and a PLACE chain's species — the label the training loss
                # consumes directly. The entity is the transition's own fact
                # (candidates are filtered to it), so this is a read, not an
                # inference.
                entity_code = int(graph.edge_entity[row])
                entity_name = ENTITY_OF_CODE[entity_code] if entity_code else None
                item_code = _ENTITY_OF[entity_name] if entity_name else -1
                lab_cell.append(cell)
                lab_chain.append(chain_id)
                lab_item.append(item_code)
                lab_phantom.append(phantom)
                lab_from_kind.append(kind_code[sid_f])
                shard_stats["matched" if bag else "no_action"] += 1

            lab_ptr[i + 1] = len(lab_cell)

        out = {k: sh[k] for k in sh.files if not k.startswith("lab_")}
        out.update(
            lab_ptr=lab_ptr,
            lab_cell=np.array(lab_cell, dtype=np.int16),
            lab_chain=np.array(lab_chain, dtype=np.int32),
            lab_item=np.array(lab_item, dtype=np.int16),
            lab_phantom=np.array(lab_phantom, dtype=np.int16),
            lab_from_kind=np.array(lab_from_kind, dtype=np.int8),
        )
        np.savez_compressed(shard_path, **out)
        totals.update(shard_stats)
        print(f"{shard_path.name}: matched={shard_stats['matched']} "
              f"no_action={shard_stats['no_action']} "
              f"out_of_graph={shard_stats['out_of_graph'] + shard_stats['opset_gap']} "
              f"phantom={shard_stats['phantom']}")

    print(f"\nTOTAL: {dict(totals)}")


if __name__ == "__main__":
    main()