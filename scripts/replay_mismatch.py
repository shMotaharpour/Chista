"""Report how often a graph node's canonical replay reproduces that node.

build_graph() labels a node's outgoing edges by replaying the node with a
canonical chain (water / feed / care every day). If that replay lands on a
different day-start state, every edge of the node is computed from the wrong
state. This script measures that mismatch per entity.

Usage (from the repo root):
    PYTHONPATH=. .venv/bin/python scripts/replay_mismatch.py [ENTITY ...]

With no argument it runs all eight entities and prints a summary table.
"""
from __future__ import annotations

import sys

from tile_dp.graph import (LIFE_DAYS, _new_sim, _replay_node, build_graph,
                           decode_tile)

ANIMALS = ("GOOSE", "COW", "SHEEP")
ENTITIES = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON") + ANIMALS


def check(entity: str, show: int = 3) -> tuple[int, int]:
    """Return (mismatches, states) for one entity's freshly built graph."""
    kind = "animal" if entity in ANIMALS else "crop"
    graph = build_graph(entity)
    bad = []
    for state_id in range(graph.n_states):
        state = graph.state_of(state_id)
        sim = _new_sim(LIFE_DAYS[entity])
        _replay_node(sim, state, entity, kind)
        obs = sim.observations()
        me = obs[0]["farms"][0]
        fx, fy = me["farmer"]
        tile = me["tiles"][fy][fx]
        got = decode_tile(tile, obs[0]["day"])
        if got.pack() != state.pack():
            bad.append((state, got))
    print(f"{entity}: mismatch {len(bad)}/{graph.n_states} "
          f"(edges {int(graph.edge_offsets[-1])})", flush=True)
    for want, got in bad[:show]:
        print(f"    want {want.describe()}  got {got.describe()}", flush=True)
    return len(bad), graph.n_states


def main(argv: list[str]) -> int:
    targets = [a.upper() for a in argv] or list(ENTITIES)
    total_bad = total = 0
    for entity in targets:
        bad, states = check(entity)
        total_bad += bad
        total += states
    print(f"\nTOTAL mismatch {total_bad}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
