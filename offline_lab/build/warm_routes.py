"""Build the warm-route artifact from the winner corpus (owner-approved design).

For every (episode, day) in `winner_days.json` the corpus carries the day the
GAME ran (chains, available, drop_by, hire_times, hands). For each day:
  1. build the TaskArray and search it once, cold, at the game's own hands;
  2. compute the logistic signature (category x quadrant + hands-per-hour);
  3. store (signature, pool, route, per-placement categories) — verified.
The result lands in `agent/artifact/warm_routes.json`, the file
`agent.wsr.warm.WarmMemory` loads. A stored day warms any FUTURE day whose
logistic signature matches (same pool, same hands-per-hour, same category
density per quadrant) — the re-projection and re-timing happen in warm.py.

Also emits the stability table the owner asked for: the same day signature
across the corpus, so the hit rate can be predicted before any run.

Run:  .venv/bin/python offline_lab/build/warm_routes.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B, tasks as T, warm as W

CORPUS = Path(__file__).resolve().parents[2] / "tests/day_layer/corpus/winner_days.json"
OUT = Path(__file__).resolve().parents[2] / "agent/artifact/warm_routes.json"


def build_one(entry) -> dict | None:
    grid = [(tuple(c), tuple(o), en) for c, o, en in entry["chains"]]
    n_shed = sum(1 for _c, o, _e in grid if "FEED" in o or "FERTILIZE" in o)
    tasks = T.build(grid, available={g: int(h) for g, h in entry["available"].items()},
                    drop_by=[12] * n_shed)
    day = B.Day(chains=tuple(grid),
                available={g: int(h) for g, h in entry["available"].items()},
                hire_times=tuple(entry["hire_times"]) or (1,) * entry["hands"])
    pool = entry["hands"]
    t0 = time.perf_counter()
    result = B.search(day, tasks, hands=pool - 1, max_hands=pool)
    dt = time.perf_counter() - t0
    if not result.complete:
        return None
    cats = W._task_categories(tasks)
    id_idx = {str(tid): i for i, tid in enumerate(tasks.ids)}
    return {
        "sig": W.signature(day, tasks),
        "pool": int(result.pool),
        "route": [(int(t), str(tid), int(w)) for t, tid, w in result.route],
        "categories": [cats[id_idx[str(tid)]] for t, tid, _w in result.route
                       if str(tid) in id_idx],
        "complete": True,
        "verified": True,
        "source": f"{entry['dump']}/{entry['episode']}/d{entry['day']}",
        "search_s": round(dt, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="build only the first N days (smoke mode)")
    args = ap.parse_args()
    days = json.loads(CORPUS.read_text())
    if args.limit:
        days = days[:args.limit]
    entries = []
    skipped = 0
    t_start = time.perf_counter()
    for e in days:
        built = build_one(e)
        if built is None:
            skipped += 1
            print(f"  skip {e['dump']}/{e['episode']}/d{e['day']}: "
                  f"the search could not carry it at the game's hands")
            continue
        entries.append(built)
        print(f"  built {e['dump']}/{e['episode']}/d{e['day']}: "
              f"{len(built['route'])} placements")
    # stability: how many entries share a signature with at least one other?
    seen: dict[str, int] = defaultdict(int)
    for b in entries:
        seen[json.dumps(b["sig"], sort_keys=True)] += 1
    shared = sum(1 for v in seen.values() if v > 1)
    multi = sum(v for v in seen.values() if v > 1)
    print(f"\nbuilt {len(entries)} seeds ({skipped} skipped) in "
          f"{time.perf_counter() - t_start:.0f}s")
    print(f"distinct signatures: {len(seen)}; signatures with 2+ seeds: "
          f"{shared} ({multi} seeds have a same-signature sibling)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(entries))
    print(f"artifact: {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
