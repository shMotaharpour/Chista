"""Brute-force transition oracle (issue #24, class B): which ENGINE
transitions does the graph not carry?

The graph's edges come from a canon: one chain per subset of ops in the fixed
order PLANT -> FERTILIZE -> WATER -> HARVEST (plus DIG and the animal ops).
That canon is an owner decision, not an engine fact, so "the graph agrees with
itself" cannot tell us whether a legal engine sequence produces a transition
nobody registered. This enumerates ordered worker-op sequences directly on a
real engine tile (through the same executor the builder uses, one sequence =
one game day) and diffs the outcomes against the graph:

  * COVERED      - the graph has this (from-state -> to-state) transition with
                   a better-or-equal (cost, produce) edge: canon or dominance
                   already accounts for it;
  * ALIAS        - same to-state and same economics, different op order: the
                   canon picks one representative, nothing is lost;
  * UNCOVERED    - the engine reached a state the graph has NO edge to from
                   there, or only to it with strictly worse economics: a
                   registry / applicability hole (the finding this check
                   exists for);
  * REFUSED      - the executor's own model assertion rejected the sequence
                   (e.g. HARVEST before WATER on a dry plant): not a hole,
                   reported separately so its count is on record.

Usage:
    .venv/bin/python -m bench.oracle_transitions [--entity CARROT]
                                               [--max-ops 3] [--starts 2]

Exit code 1 when any UNCOVERED transition is found.
"""

from __future__ import annotations

import argparse
from itertools import permutations

from tile_dp import graph as G
from tile_dp.chains import RESOURCE_NAMES

# Ops the oracle permutes on a crop tile; the market-buying preconditions are
# the executor's business (`_exec_chain` supplies them, exactly as the builder
# relies on).
OP_VOCAB = ("FERTILIZE", "WATER", "HARVEST", "DIG", "NO_ACT")


def _start_sims(entity: str):
    """Real day-start (sim, state) pairs: one sim per graph state.

    Single ops cannot walk the graph - `PLANT` alone already leaves the plant
    one unwatered day old (`consecutive_unwatered` = 1 immediately, measured),
    so the engine turns it into a weed and the plan's `(PLANT, WATER)` is what
    is realisable. The oracle therefore starts from every state the graph
    itself reaches, then permutes ops from there.
    """
    root_sim = G._new_sim()
    root = G.decode_tile(None, 0)
    sims = {root.pack(): (root_sim, root)}
    queue = [root]
    while queue:
        state = queue.pop(0)
        source = sims[state.pack()][0]
        for run_entity, ops, _code in G._plan(state, entity):
            branch = source.clone()
            try:
                outcome = G._exec_chain(branch, state, ops, run_entity)
            except Exception:                     # noqa: BLE001 - probe
                continue
            child = outcome.next_state
            if child.pack() not in sims:
                sims[child.pack()] = (branch, child)
                queue.append(child)
    return list(sims.values())


def _edges_from(graph, sid: int):
    """(to_id, cost, produce, chain_id) for every edge leaving `sid`."""
    start = int(graph.edge_offsets[sid])
    stop = int(graph.edge_offsets[sid + 1])
    for e in range(start, stop):
        yield (int(graph.edge_next[e]),
               tuple(int(x) for x in graph.edge_cost[e]),
               tuple(int(x) for x in graph.edge_produce[e]),
               int(graph.edge_chain[e]))


def _yield_of(graph, sid: int) -> int:
    """The decoded `yield_units` of a graph node (0 when not a crop)."""
    from tile_dp.tile_state import TileState

    state = TileState.unpack(int(graph.state_keys[sid]))
    return state.yield_units


def _covers(edge, to_state: int, cost, produce) -> str:
    """'same-state' if the graph reaches `to_state` at least as well, else ''."""
    if edge[0] != to_state:
        return ""
    if all(a <= b for a, b in zip(edge[1], cost)) and \
       all(a >= b for a, b in zip(edge[2], produce)):
        return "same-state"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="CARROT")
    ap.add_argument("--max-ops", type=int, default=3)
    args = ap.parse_args()

    graph = G.build_graph(args.entity)
    starts = _start_sims(args.entity)
    print(f"oracle: entity {args.entity}, graph {graph.n_states} states / "
          f"{graph.n_edges} edges, start nodes {len(starts)}, sequences up to "
          f"{args.max_ops} ops", flush=True)

    counts = {"COVERED": 0, "ALIAS": 0, "NOOP": 0, "UNCOVERED": 0,
              "REFUSED": 0, "REFUSED_BUT_BILLED": 0, "PERM_BETTER": 0,
              "CANON_WORSE_ORDER": 0}
    uncovered = []
    billed = []
    better = []
    seen_states = set()
    for sim, state in starts:
        sid = graph.key_index.get(state.pack())
        if sid is None:
            continue
        seen_states.add(sid)
        before_tile = G._tile_and_day(sim)[0]
        for n in range(1, args.max_ops + 1):
            for ops in permutations(OP_VOCAB, n):
                if "NO_ACT" in ops and len(ops) > 1:
                    continue          # NO_ACT is a whole-chain op (contract)
                branch = sim.clone()
                try:
                    outcome = G._exec_chain(branch, state, ops,
                                            args.entity)
                except Exception:                 # noqa: BLE001 - probe
                    counts["REFUSED"] += 1
                    continue
                after_tile = G._tile_and_day(branch)[0]
                # What the engine actually did: a chain whose ops it refused
                # leaves the tile untouched and collects nothing.
                did_something = (before_tile != after_tile
                                 or outcome.harvest
                                 or outcome.fert_collect)
                if not did_something and any(outcome.cost):
                    counts["REFUSED_BUT_BILLED"] += 1
                    billed.append((state, ops, outcome.cost))
                cost = tuple(outcome.cost)
                produce = tuple(outcome.produce)
                to_sid = graph.key_index.get(outcome.next_state.pack())
                same_set = [e for e in _edges_from(graph, sid)
                            if sorted(G.chain_ops(e[3])) == sorted(ops)]
                if to_sid is None:
                    # An off-canon ORDER of a registered SET: not a hole when
                    # the canon's own order is at least as good (measured:
                    # (WATER, FERTILIZE) reaches y=2 while (FERTILIZE, WATER)
                    # reaches y=3 from the same state).
                    canon_yield = (max(_yield_of(graph, e[0])
                                       for e in same_set)
                                   if same_set else None)
                    if canon_yield is not None \
                            and outcome.next_state.yield_units <= canon_yield:
                        counts["CANON_WORSE_ORDER"] += 1
                        continue
                    counts["UNCOVERED"] += 1
                    uncovered.append((state, ops, "state not in the graph",
                                      outcome.next_state.describe()))
                    continue
                if outcome.next_state.pack() == state.pack() \
                        and not did_something:
                    counts["NOOP"] += 1        # F047: the engine no-op'd it
                    continue
                best = ""
                for edge in _edges_from(graph, sid):
                    cover = _covers(edge, to_sid, cost, produce)
                    if cover:
                        best = cover
                        break
                if best:
                    counts["COVERED"] += 1
                    continue
                if same_set:
                    # A non-canon ORDER of a registered SET: the canon's own
                    # edge must be at least as good, or the canon discards
                    # value (an op order the engine can exploit).
                    canon_yield = max(_yield_of(graph, e[0]) for e in same_set)
                    if outcome.next_state.yield_units > canon_yield:
                        counts["PERM_BETTER"] += 1
                        better.append((state, ops, canon_yield,
                                       outcome.next_state.yield_units))
                    counts["ALIAS"] += 1
                    continue
                if any(edge[0] == to_sid for edge in _edges_from(graph, sid)):
                    counts["ALIAS"] += 1
                else:
                    counts["UNCOVERED"] += 1
                    uncovered.append((state, ops, "no edge to that state",
                                      outcome.next_state.describe()))
    for name, n in counts.items():
        print(f"  {name}: {n}", flush=True)
    print(f"  distinct start states exercised: {len(seen_states)} of "
          f"{graph.n_states}", flush=True)
    for state, ops, why, desc in uncovered[:8]:
        print(f"  UNCOVERED {state.describe()} -{ops}-> {desc} ({why})",
              flush=True)
    for state, ops, cost in billed[:4]:
        spent = [f"{RESOURCE_NAMES[i]}={c}" for i, c in enumerate(cost) if c]
        print(f"  REFUSED-BUT-BILLED {state.describe()} -{ops}-> "
              f"the model bills {spent} while the engine did nothing",
              flush=True)
    for state, ops, canon_y, perm_y in better[:4]:
        print(f"  PERMUTATION BEATS THE CANON {state.describe()} -{ops}-> "
              f"yield {perm_y} against the canon's {canon_y} (an op order the "
              f"engine exploits and the graph does not carry)", flush=True)
    return 1 if counts["UNCOVERED"] or counts["PERM_BETTER"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
