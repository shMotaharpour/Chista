"""Hidden-state probe (issue #24, class C): is a modelled state hiding a
difference that changes the future?

`TileState` is an abstraction: several raw engine tiles decode to one state
(the known deliberate merges - care_bank capped at max_held, consec/unfed
saturating - were probed and are decision-equivalent, but nothing checks the
NEXT merge). This probe hunts for them mechanically:

1. Enumerate short legal op histories on a fresh engine tile (single-op daily
   chains through the real executor) and record every day-start raw tile with
   its decoded state.
2. Group them by decoded state; a group with **two different raw signatures
   over the fields the decoder does not read** is a candidate hidden state.
3. Run the graph's own candidate chains on BOTH members of a candidate pair
   and compare the decoded successors - and their 1- and 2-day continuations.
   A difference means the abstraction dropped a dimension that decides the
   future, not merely one that looks different.

Usage:
    .venv/bin/python -m bench.probe_hidden_state [--depth 4] [--entity CARROT]

Exit code 1 when a candidate pair's futures differ (a lost dimension), 0 when
every candidate pair is decision-equivalent. The modelled/unmodelled field
split is read off `decode_tile`'s own source, never hand-listed.
"""

from __future__ import annotations

import argparse
import inspect
import re


from world.fast_sim import FastSim

from tile_dp import graph as G
from tile_dp.tile_state import decode_tile


def _decoder_keys() -> set[str]:
    """The raw tile fields `decode_tile` actually reads (from its source).

    Both forms count: `tile["kind"]` (subscript) and `tile.get("yield_units")`.
    """
    src = inspect.getsource(decode_tile)
    return set(re.findall(r'tile(?:\.get\(|\[)\s*["\']([a-z_]+)["\']', src))


def _raw_signature(tile) -> tuple:
    """The tile's fields the decoder ignores, as a comparable signature."""
    if not isinstance(tile, dict):
        return (("<raw>", repr(tile)),)
    unmodelled = _decoder_keys()
    return tuple(sorted((k, repr(v)) for k, v in tile.items()
                        if k not in unmodelled))


def _print_field_split(tile) -> None:
    if not isinstance(tile, dict):
        print(f"  (a bare tile is {tile!r}, nothing to split)")
        return
    read = _decoder_keys()
    modelled = sorted(k for k in tile if k in read)
    unmodelled = sorted(k for k in tile if k not in read)
    print(f"  decoder reads: {modelled}")
    print(f"  decoder IGNORES: {unmodelled}")


def _one_day(sim: FastSim, state, ops: tuple[str, ...], entity):
    """Execute one single-op chain; None when the engine refuses/mismatches."""
    try:
        return G._exec_chain(sim, state, ops, entity), None
    except Exception as exc:                      # noqa: BLE001 - probe
        return None, f"{type(exc).__name__}: {str(exc)[:60]}"


def _state_sims(entity: str, keep: int = 4) -> dict[int, list]:
    """The graph's states, each with up to `keep` sims that reached it.

    One sim per state is enough to EXPAND the graph (that is what the builder
    does), but a hidden dimension lives precisely in the extra sims: the same
    decoded state reached by different histories, whose raw tiles differ in a
    field the decoder ignores. So: expand every state once, keep every sim
    that arrives.

    Single ops are not enough to walk: `PLANT` alone plants a plant the engine
    already counts as one unwatered day (`consecutive_unwatered` is 1 right
    after `PLANT`, measured), so the plan's chains - `(PLANT, WATER)` first -
    are the realisable ones. This walks those.
    """
    root_sim = G._new_sim()
    root = decode_tile(None, 0)
    buckets: dict[int, list] = {root.pack(): [(root_sim, root)]}
    expanded: set[int] = set()
    queue = [root]
    while queue:
        state = queue.pop(0)
        key = state.pack()
        if key in expanded:
            continue
        expanded.add(key)
        source = buckets[key][0][0]
        for run_entity, ops, _code in G._plan(state, entity):
            branch = source.clone()
            outcome, _err = _one_day(branch, state, ops, run_entity)
            if outcome is None:
                continue
            child = outcome.next_state
            bucket = buckets.setdefault(child.pack(), [])
            if len(bucket) < keep:
                bucket.append((branch, child))
            if child.pack() not in expanded:
                queue.append(child)
    return buckets


def _continuation_diffs(a, b, child_a, child_b, entity, days: int = 2):
    """Follow the successors with idle days; a difference that shows up later
    is still a lost dimension (the merge hid a field that decides the future).
    """
    diffs = []
    sa, sb = a.clone(), b.clone()
    ops = ("NO_ACT",)
    for step in range(days):
        oa = _one_day(sa, child_a, ops, entity)[0]
        ob = _one_day(sb, child_b, ops, entity)[0]
        if oa is None or ob is None:
            break
        if oa.next_state.pack() != ob.next_state.pack():
            diffs.append((f"NO_ACT x{step + 1}",
                          f"{oa.next_state.describe()} vs "
                          f"{ob.next_state.describe()}"))
            break
        child_a, child_b = oa.next_state, ob.next_state
    return diffs


def _futures_differ(a: FastSim, b: FastSim, state, entity: str):
    """Run every planned chain on both sims; compare decoded successors."""
    diffs = []
    for run_entity, ops, _code in G._plan(state, entity):
        oa, ea = _one_day(a.clone(), state, ops, run_entity)
        ob, eb = _one_day(b.clone(), state, ops, run_entity)
        if (oa is None) != (ob is None):
            diffs.append((ops, f"one side refused: {ea} vs {eb}"))
            continue
        if oa is None or ob is None:
            continue
        if oa.next_state.pack() != ob.next_state.pack():
            diffs.append((ops, f"{oa.next_state.describe()} vs "
                               f"{ob.next_state.describe()}"))
        else:
            for ops2, why in _continuation_diffs(a, b, oa.next_state,
                                                 ob.next_state, entity):
                diffs.append((ops, f"after {ops2}: {why}"))
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=4,
                    help="sims kept per decoded state (the extra paths)")
    ap.add_argument("--entity", default=None,
                    help="one entity's graph; omit for the merged graph")
    args = ap.parse_args()

    print(f"hidden-state probe: entity {args.entity}, sims kept per state "
          f"{args.keep}", flush=True)
    buckets = _state_sims(args.entity, keep=args.keep)
    tiles = [G._tile_and_day(sim)[0] for _key, members in buckets.items()
             for sim, _st in members]
    print(f"  day-start sims collected: {len(tiles)} over "
          f"{len(buckets)} decoded states", flush=True)
    _print_field_split(max((t for t in tiles if isinstance(t, dict)),
                           key=len, default=None))

    groups = {key: [(sim, state, G._tile_and_day(sim)[0])
                    for sim, state in members]
              for key, members in buckets.items()}
    candidates = []
    for key, members in groups.items():
        sigs = {}
        for sim, state, tile in members:
            sigs.setdefault(_raw_signature(tile), (sim, state, tile))
        if len(sigs) > 1:
            candidates.append((members[0][1], list(sigs.values())))
    print(f"  decoded states seen: {len(groups)} | states carrying MORE THAN "
          f"ONE unmodelled raw signature: {len(candidates)}", flush=True)

    lost = 0
    checked = 0
    for state, members in candidates:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sim_a, _st_a, tile_a = members[i]
                sim_b, _st_b, tile_b = members[j]
                if _raw_signature(tile_a) == _raw_signature(tile_b):
                    continue
                checked += 1
                diffs = _futures_differ(sim_a, sim_b, state, args.entity)
                if diffs:
                    lost += 1
                    print(f"  LOST DIMENSION {state.describe()}: "
                          f"{_raw_signature(tile_a)} vs "
                          f"{_raw_signature(tile_b)}", flush=True)
                    for ops, why in diffs[:3]:
                        print(f"    chain {ops}: {why}", flush=True)
    print(f"candidate pairs checked: {checked} | with differing futures: "
          f"{lost} (0 = the abstraction is decision-equivalent on this sweep)",
          flush=True)
    return 1 if lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
