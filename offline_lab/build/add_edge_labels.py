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

Two entry points:
    label_shard(shard_path, replay_path) — the CLI/one-file path
    label_replay(shard_dict, replay_path, graph, voc) — the notebook path:
      labels an in-memory shard dict and adds the summary counters
      (lab_matched / lab_no_action / lab_out_of_graph / lab_phantom_total)
      the build report reads.

Adds per shard (format 1 extension, see FORMAT.md):
    lab_ptr (n+1,) int64 · lab_cell (ΣL,) int16 · lab_chain (ΣL,) int32
    lab_item (ΣL,) int16 · lab_phantom (ΣL,) int16 · lab_from_kind (ΣL,) int8
    lab_matched / lab_no_action / lab_out_of_graph / lab_phantom_total (scalars)
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
from agent.tile_dp.chains import ENTITY_OF_CODE, chain_ops
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


def _label_shard_arrays(sh: dict, steps: list, graph: TileGraph,
                        voc: list[str], entity_codes: dict[str, int]
                        ) -> tuple[dict, Counter]:
    """The label pass over one in-memory shard dict + its raw replay.

    `sh` maps field name -> np.ndarray (the shard's own arrays); returns the
    label arrays to merge in, plus the shard's summary counters.
    """
    key_index = graph.key_index
    state_keys = np.asarray(graph.state_keys, dtype=np.int64)

    # Edge index: from_state_id -> to_state_key -> [(op bag WITH COUNTS, chain_id, row)]
    by_transition: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for sid in range(graph.n_states):
        lo, hi = graph.edges_of(sid)
        for row in range(lo, hi):
            to_id = int(graph.edge_next[row])
            chain_id = int(graph.edge_chain[row])
            bag = Counter(str(op) for op in chain_ops(chain_id))
            by_transition[sid][int(state_keys[to_id])].append((bag, chain_id, row))

    kind_code = {sid: kind_code_of[graph.state_of(sid).kind]
                 for sid in range(graph.n_states)}

    n = len(sh["seat"])
    index_of = {(int(sh["seat"][i]), int(sh["day"][i])): i for i in range(n)}
    op_cell, op_code, op_ptr = sh["op_cell"], sh["op_code"], sh["op_ptr"]
    n_steps = len(steps)

    lab_cell: list[int] = []
    lab_chain: list[int] = []
    lab_item: list[int] = []
    lab_phantom: list[int] = []
    lab_from_kind: list[int] = []
    lab_ptr = np.zeros(n + 1, dtype=np.int64)
    stats: Counter = Counter()

    for i in range(n):
        seat, day = int(sh["seat"][i]), int(sh["day"][i])
        tiles = sh["tiles_own"][i]

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
            if from_key < 0:                 # LOCKED sentinel
                continue
            sid_f = key_index.get(from_key)
            if sid_f is None:
                stats["unmodelled_from"] += 1
                continue

            hours = cell_hours.get(cell, {})
            y, x = divmod(cell, 10)

            effective: Counter = Counter()
            phantom = 0
            for h, counts in hours.items():
                t = day * 24 + h
                if t >= n_steps:
                    continue
                if h == 24:                  # crosses the nightly reset
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
                    stats["phantom"] += int(sum(counts.values()))

            if nxt is None:                  # day 29: no next-day state
                lab_cell.append(cell)
                lab_chain.append(LAB_LAST_DAY)
                lab_item.append(-1)
                lab_phantom.append(phantom)
                lab_from_kind.append(kind_code[sid_f])
                stats["last_day"] += 1
                continue

            to_key = int(sh["tiles_own"][nxt][cell])
            cands = by_transition.get(sid_f, {}).get(to_key, [])
            if not cands:
                lab_cell.append(cell)
                lab_chain.append(LAB_OUT_OF_GRAPH)
                lab_item.append(-1)
                lab_phantom.append(phantom + int(sum(effective.values())))
                lab_from_kind.append(kind_code[sid_f])
                stats["out_of_graph"] += 1
                continue

            best = None
            for bag, chain_id, row in cands:
                if not all(effective.get(op, 0) >= cnt for op, cnt in bag.items()):
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
                stats["opset_gap"] += 1
                continue

            _key, chain_id, row = best
            # The edge's own entity (a PLANT's crop, a PLACE's species) — the
            # training loss consumes it directly. The transition already fixes
            # it; this is a read, not an inference.
            entity_code = int(graph.edge_entity[row])
            entity_name = ENTITY_OF_CODE[entity_code] if entity_code else None
            item_code = entity_codes.get(entity_name, -1) if entity_name else -1
            lab_cell.append(cell)
            lab_chain.append(chain_id)
            lab_item.append(item_code)
            lab_phantom.append(phantom)
            lab_from_kind.append(kind_code[sid_f])
            bag_empty = not _chain_has_ops(chain_id)
            stats["matched" if not bag_empty else "no_action"] += 1

        lab_ptr[i + 1] = len(lab_cell)

    arrays = {
        "lab_ptr": lab_ptr,
        "lab_cell": np.array(lab_cell, dtype=np.int16),
        "lab_chain": np.array(lab_chain, dtype=np.int32),
        "lab_item": np.array(lab_item, dtype=np.int16),
        "lab_phantom": np.array(lab_phantom, dtype=np.int16),
        "lab_from_kind": np.array(lab_from_kind, dtype=np.int8),
    }
    summary = Counter({k: v for k, v in stats.items()})
    return arrays, summary


def _chain_has_ops(chain_id: int) -> bool:
    return any(True for _ in chain_ops(chain_id))


def label_replay(shard: dict, replay_path, graph: TileGraph) -> dict:
    """The notebook path: label an in-memory shard dict IN PLACE (adds `lab_*`
    arrays + summary scalars) and return it."""
    from offline_lab.build.kaggle_dataset import Vocab  # noqa: F401 (par with builder)
    raw = json.load(open(replay_path))
    steps = raw["steps"]
    voc_list = shard["vocab_list"]           # the builder stashes its table
    entity_codes = shard["vocab_entity_codes"]
    arrays, summary = _label_shard_arrays(shard, steps, graph, voc_list, entity_codes)
    shard.update(arrays)
    for k, v in summary.items():
        shard[f"lab_{k}"] = np.int64(v)
    return shard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", nargs="+", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    args = ap.parse_args()

    graph = TileGraph.load(_GRAPH_PATH)

    replay_of = {}
    for path in args.replays:
        path = Path(path)
        replay_of[int(path.name.split("-")[1])] = path

    totals: Counter = Counter()
    for shard_path in args.shards:
        shard_path = Path(shard_path)
        sh = np.load(shard_path, allow_pickle=False)
        n = int(sh["n_samples"])
        episode_id = int(sh["episode_id"][0])
        voc = json.loads((shard_path.parent / "vocab.json").read_text())["list"]
        entity_codes = {name: i for i, name in enumerate(voc)}

        shard_dict = {k: sh[k] for k in sh.files if not k.startswith("lab_")}
        arrays, summary = _label_shard_arrays(
            shard_dict, json.load(open(replay_of[episode_id]))["steps"],
            graph, voc, entity_codes)
        shard_dict.update(arrays)
        for k, v in summary.items():
            shard_dict[f"lab_{k}"] = np.int64(v)
        np.savez_compressed(shard_path, **shard_dict)
        totals.update(summary)
        print(f"{shard_path.name}: matched={summary['matched']} "
              f"no_action={summary['no_action']} "
              f"out_of_graph={summary['out_of_graph']} "
              f"phantom={summary['phantom_total']}")

    print(f"\nTOTAL: {dict(totals)}")


if __name__ == "__main__":
    main()