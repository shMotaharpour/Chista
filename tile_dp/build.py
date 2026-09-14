"""Build the shipped tile graph model.

Run from the repo root:

    .venv/bin/python -m tile_dp.build

Writes `tile_dp/models/graph_tile_lifecycle.npz` (the merged graph) and
`tile_dp/models/build_report.json` (node / edge / kind counts for the merged
graph and the 8 restricted per-entity views), then reloads the written file to
prove the contract- and registry-guarded round-trip works.

The artifact carries no version number: `contract` is computed from everything
the graph depends on (chain registry, engine source, turns per day, packed-key
layout), so a changed input makes an old artifact unusable by construction
instead of relying on a hand-bumped `vNN` label (owner, 2026-09-14).

The report keeps two numbers apart (owner's item 12): `restricted_*` is a build
whose candidate entities are limited to one entity, so the search itself shrinks;
`merged_edges_for_entity` counts the merged graph's edges whose chain runs for
that entity. They are not the same decomposition - in the merged graph animal
chains also sit on PLANT nodes, so the merged count is far larger.
"""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

from tile_dp.chains import ENTITY_NAMES, entity_of_code, registry_fingerprint
from tile_dp.graph import TileGraph, build_graph

MODEL_DIR = Path(__file__).resolve().parent / "models"
GRAPH_PATH = MODEL_DIR / "graph_tile_lifecycle.npz"
REPORT_PATH = MODEL_DIR / "build_report.json"


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    g = build_graph()
    g.save(GRAPH_PATH)
    print("MERGED", g.report.describe(), flush=True)
    print("MERGED kinds", g.report.kinds, flush=True)
    print("MERGED bytes", os.path.getsize(GRAPH_PATH),
          "build_s", round(time.time() - t0, 1), "peak_rss_mb",
          round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
          flush=True)

    t1 = time.time()
    by_entity: dict[str, int] = {}
    for code in g.edge_entity:
        name = entity_of_code(int(code)) or "TILE"
        by_entity[name] = by_entity.get(name, 0) + 1
    report = {"contract": g.engine_tag,
              "registry": registry_fingerprint(),
              "entities": {"TILE": {"nodes": g.n_states, "edges": g.n_edges,
                                    "kinds": g.report.kinds,
                                    "merged_edges_for_entity": by_entity}}}
    for entity in ENTITY_NAMES:
        view = build_graph(entity)
        report["entities"][entity] = {
            "restricted_nodes": view.n_states,
            "restricted_edges": int(view.edge_offsets[-1]),
            "merged_edges_for_entity": by_entity.get(entity, 0),
            "kinds": view.report.kinds}
        print(f"RESTRICTED {entity} n_states={view.n_states} "
              f"n_edges={int(view.edge_offsets[-1])} "
              f"merged={by_entity.get(entity, 0)} kinds={view.report.kinds}",
              flush=True)
    print("restricted_s", round(time.time() - t1, 1), flush=True)

    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    back = TileGraph.load(GRAPH_PATH)
    print("RELOAD", back.n_states, back.n_edges, back.entity, back.engine_tag)
    try:
        shown = GRAPH_PATH.relative_to(Path.cwd()).as_posix()
    except ValueError:      # run from another cwd: print the absolute path
        shown = GRAPH_PATH.as_posix()
    print("model:", shown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
