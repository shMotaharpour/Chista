"""Build the self-warm artifact from the winner corpus.

For every corpus day (the game's own days, won by elite players), this builds
the day's TaskArray, searches it once at the game's own hands to get a
COMPLETE route, and stores that route + its logistic signature + the
per-worker first hours + the settled doors into
`agent/artifact/warm_routes.npz` (compact numpy, no JSON).

The seeds are stored under the pool they were solved with, so a future query
at the same pool + same hands-per-hour vector + similar logistic signature
receives them as beam-start seeds (selfwarm.SelfWarm.lookup).

Run:  .venv/bin/python offline_lab/build/warm_routes.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B, tasks as T
from agent.wsr.selfwarm import SOFT_WIDTH, WARM_K, soft_vector, task_categories

CORPUS = Path(__file__).resolve().parents[2] / "tests/day_layer/corpus/winner_days.json"
OUT = Path(__file__).resolve().parents[2] / "agent/artifact/warm_routes.npz"
BOARD = 10


def quad(cell) -> int:
    x, y = int(cell[0]), int(cell[1])
    return (y // (BOARD // 2)) * 2 + (x // (BOARD // 2))


def build_entry(entry) -> dict | None:
    grid = [(tuple(c), tuple(o), en) for c, o, en in entry["chains"]]
    n_shed = sum(1 for _c, o, _e in grid if "FEED" in o or "FERTILIZE" in o)
    tasks = T.build(grid, available={g: int(h) for g, h in entry["available"].items()},
                    drop_by=[12] * n_shed)
    day = B.Day(chains=tuple(grid),
                available={g: int(h) for g, h in entry["available"].items()},
                hire_times=tuple(entry["hire_times"]) or (1,) * entry["hands"])
    t0 = time.perf_counter()
    result = B.search(day, tasks, hands=entry["hands"] - 1,
                      max_hands=entry["hands"])
    dt = time.perf_counter() - t0
    if not result.complete:
        return None
    return {
        "tasks": tasks, "day": day, "result": result,
        "soft": soft_vector(day, tasks),
        "source": f"{entry['dump']}/{entry['episode']}/d{entry['day']}",
        "search_s": dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    days = json.loads(CORPUS.read_text())
    if args.limit:
        days = days[:args.limit]

    built = []
    skipped = 0
    t_start = time.perf_counter()
    for e in days:
        built_one = build_entry(e)
        if built_one is None:
            skipped += 1
            print(f"  skip {e['dump']}/{e['episode']}/d{e['day']}")
            continue
        built.append(built_one)
        print(f"  built {e['dump']}/{e['episode']}/d{e['day']}: "
              f"{len(built_one['result'].route)} placements "
              f"({built_one['search_s']:.1f}s)")

    # shared vocabulary: every task id that appears in any stored route
    vocab: list[str] = []
    vidx: dict[str, int] = {}
    for b in built:
        for _t, tid, _w in b["result"].route:
            tid = str(tid)
            if tid not in vidx:
                vidx[tid] = len(vocab)
                vocab.append(tid)
    V = len(vocab)
    max_len = max((len(b["result"].route) for b in built), default=0)
    max_hands = max((b["result"].pool for b in built), default=0)
    max_hire = max((len(b["day"].hire_times) for b in built), default=0)

    n = len(built)
    pool = np.zeros(n, dtype=np.int8)
    hire_hours = np.zeros((n, max_hire), dtype=np.int8)
    soft = np.zeros((n, SOFT_WIDTH), dtype=np.int32)
    route_hour = np.full((n, max_len), -1, dtype=np.int16)
    route_task = np.full((n, max_len), -1, dtype=np.int16)
    route_worker = np.full((n, max_len), -1, dtype=np.int8)
    route_cat = np.full((n, max_len), -1, dtype=np.int8)
    route_len = np.zeros(n, dtype=np.int16)
    doors = np.zeros((n, max_hands, 2), dtype=np.int8)

    for i, b in enumerate(built):
        r = b["result"]
        pool[i] = r.pool
        ht = tuple(int(h) for h in b["day"].hire_times)
        hire_hours[i, :len(ht)] = ht
        soft[i] = b["soft"]
        route_len[i] = len(r.route)
        for k, (t, tid, w) in enumerate(r.route):
            route_hour[i, k] = t
            route_task[i, k] = vidx[str(tid)]
            route_worker[i, k] = w
        for k, d in enumerate(r.doors):
            doors[i, k] = d

    np.savez_compressed(
        OUT,
        pool=pool, hire_hours=hire_hours, soft=soft,
        route_hour=route_hour, route_task=route_task,
        route_worker=route_worker, route_cat=route_cat,
        route_len=route_len, doors=doors,
        vocab=np.array(vocab),
    )
    print(f"\nbuilt {n} seeds ({skipped} skipped) in {time.perf_counter() - t_start:.0f}s")
    print(f"artifact: {OUT} ({OUT.stat().st_size / 1024:.1f} KB), "
          f"vocab {V} task ids, max route {max_len}, k={WARM_K}")


if __name__ == "__main__":
    main()
