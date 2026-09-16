"""Prune-invariance check (issue #24, class B): does dominance ever discard an
edge that matters?

Dominance pruning claims to be optimality-preserving while every price and
wage is >= 0 componentwise (R006). That claim had never been tested on the
real graph - the suite checks the RULE (`_dominates` unit cases) and the
precondition's guard, never the consequence. This rebuilds the same graph with
dominance disabled, runs the backward DP on BOTH graphs over a battery of
random non-negative price/wage vectors from a fixed seed, and compares V per
(day, state): one differing value means dominance dropped an edge the DP
wanted.

Usage:
    .venv/bin/python -m bench.check_prune_invariance [--entity CARROT]
                                                     [--vectors 50] [--days 30]

The negative side of R006 is not tested here - the contractor refuses a
negative component on entry (`_check_non_negative`), which IS the guard; this
script prints that refusal once so the boundary is on record rather than
implied. The comparison carries its own positive control: a copy of the pruned
graph with one edge's produce raised must produce different values, otherwise
"identical" would prove nothing.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np

from tile_dp import graph as G
from tile_dp.contractor import TileContractor
from tile_dp.graph import build_graph


def _unpruned_build(entity: str | None):
    """Build the graph with the dominance pass disabled (no-op sweep kept)."""
    real = G._prune

    def no_dominance(state_id: int, edges: list):
        kept = [e for e in edges if not G._is_noop_edge(state_id, e)]
        return kept, len(edges) - len(kept)

    G._prune = no_dominance
    try:
        return G.build_graph(entity)
    finally:
        G._prune = real


def _edges(graph, sid: int):
    """(to_id, cost, produce, chain_id) for every edge leaving `sid`."""
    for e in range(int(graph.edge_offsets[sid]), int(graph.edge_offsets[sid + 1])):
        yield (int(graph.edge_next[e]),
               tuple(int(x) for x in graph.edge_cost[e]),
               tuple(int(x) for x in graph.edge_produce[e]),
               int(graph.edge_chain[e]))


def _vectors(rng: np.random.Generator, days: int, hi: int = 40):
    """One random NON-NEGATIVE (days, N_RESOURCE) price vector and wage vector."""
    p = rng.integers(0, hi, size=(days, G.N_RESOURCE)).astype(np.float32)
    w = rng.integers(0, hi, size=(days, G.N_RESOURCE)).astype(np.float32)
    return p, w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=None,
                    help="one entity's graph; omit for the merged graph")
    ap.add_argument("--vectors", type=int, default=50)
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    t0 = time.time()
    pruned = build_graph(args.entity)
    full = _unpruned_build(args.entity)
    print(f"{args.entity or 'TILE (merged)'}: pruned {pruned.n_states} states / "
          f"{pruned.n_edges} edges vs unpruned {full.n_states} states / "
          f"{full.n_edges} edges (dominance dropped "
          f"{full.n_edges - pruned.n_edges} edges)", flush=True)

    a = TileContractor(pruned, days=args.days)
    b = TileContractor(full, days=args.days)
    rng = np.random.default_rng(20260916)
    mismatches = 0
    worst = (0.0, None)
    for i in range(args.vectors):
        p, w = _vectors(rng, args.days)
        Va, _ = a.sweep(p, w)
        Vb, _ = b.sweep(p, w)
        diff = np.abs(Va - Vb)
        m = float(diff.max())
        if m > 0.0:
            mismatches += 1
            d, s = np.unravel_index(int(diff.argmax()), diff.shape)
            if m > worst[0]:
                worst = (m, (i, int(d), int(s), float(Va[d, s]), float(Vb[d, s])))
    print(f"vectors tested: {args.vectors} (non-negative, seeded) | "
          f"mismatching sweeps: {mismatches} | worst |V_pruned - V_full| = "
          f"{worst[0]:.4f}", flush=True)
    if worst[1]:
        i, d, s, va, vb = worst[1]
        print(f"  worst case: vector {i}, day {d}, state {s}: pruned {va} vs "
              f"unpruned {vb} - dominance discarded an edge the DP wanted",
              flush=True)
    else:
        print("  no mismatch: dominance changed no optimal value in this "
              "regime (the soundness claim holds for p, w >= 0)", flush=True)

    # Item D (exact twins): does the DP care yet? Same (from, to, cost,
    # produce), different entity - a COW/SHEEP pair on one structure. The
    # issue's threshold is a few percent; this measures the current share and
    # the share of the sweep's edge work they occupy, so the decision to drop
    # them (prune-level group, plan-level skip) rests on a number.

    key_of = {}
    for sid in range(pruned.n_states):
        for to_id, cost, produce, chain in _edges(pruned, sid):
            key_of.setdefault((sid, to_id, cost, produce), []).append(chain)
    twins = {k: v for k, v in key_of.items() if len(v) > 1}
    n_twin = sum(len(v) - 1 for v in twins.values())
    print(f"exact twins (same from/to/cost/produce, different chain): {n_twin} "
          f"extra edges of {pruned.n_edges} = "
          f"{100.0 * n_twin / max(1, pruned.n_edges):.1f} % - the DP reads one "
          f"candidate per edge, so this is the sweep's redundant share",
          flush=True)

    # Positive control: the comparison must be able to see a value change.
    ep = pruned.edge_produce.copy()
    ep[0, 0] = ep[0, 0] + 1
    nudged = replace(pruned, edge_produce=ep)
    seeded = np.random.default_rng(20260916)
    p, w = _vectors(seeded, args.days)
    Vn, _ = TileContractor(nudged, days=args.days).sweep(p, w)
    Vp, _ = a.sweep(p, w)
    ok = bool(np.abs(Vn - Vp).max() > 0.0)
    print(f"positive control (one edge's produce raised): difference detected "
          f"= {ok}", flush=True)

    # R006's other side: the guard itself.
    neg = np.zeros((args.days, G.N_RESOURCE), dtype=np.float32)
    neg[0, 0] = -1.0
    try:
        a.sweep(neg, np.zeros_like(neg))
        print("R006 boundary: a NEGATIVE price was accepted - the guard is "
              "not firing", flush=True)
    except ValueError as exc:
        print(f"R006 boundary: a negative price is refused on entry as "
              f"designed ({str(exc)[:60]}...)", flush=True)

    print(f"elapsed {time.time() - t0:.1f} s", flush=True)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
