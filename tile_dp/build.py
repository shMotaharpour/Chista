"""Build the shipped tile graph model.

Run from the repo root:

    .venv/bin/python -m tile_dp.build

Writes `tile_dp/models/graph_tile_lifecycle.npz` (the merged graph, engine tag
`tile-dp-v15`) and `tile_dp/models/build_report.json` (node / edge / kind counts
for the merged graph and the 8 restricted per-entity views), then reloads the
written file to prove the tag-guarded round-trip works.
"""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

from tile_dp.chains import ENTITY_NAMES
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
    report = {"engine_tag": g.engine_tag,
              "entities": {"TILE": {"nodes": g.n_states, "edges": g.n_edges,
                                    "kinds": g.report.kinds}}}
    for entity in ENTITY_NAMES:
        view = build_graph(entity)
        report["entities"][entity] = {"nodes": view.n_states,
                                      "edges": int(view.edge_offsets[-1]),
                                      "kinds": view.report.kinds}
        print(f"RESTRICTED {entity} n_states={view.n_states} "
              f"n_edges={int(view.edge_offsets[-1])} kinds={view.report.kinds}",
              flush=True)
    print("restricted_s", round(time.time() - t1, 1), flush=True)

    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    back = TileGraph.load(GRAPH_PATH)
    print("RELOAD", back.n_states, back.n_edges, back.entity, back.engine_tag)
    print("model:", GRAPH_PATH.relative_to(Path.cwd()).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
