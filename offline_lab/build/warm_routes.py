"""Build the self-warm artifact from the winner's OWN recorded routes.

For every corpus day this reads the winning unit's real op stream from the
September parquet store (`hands_steps` — real hours, real cells), maps each
action op onto the day's wsr tasks by (op, item) + nearest cell, and stores
the route INCREMENTALLY COMPILE-VERIFIED: a placement joins the seed only if
the partial route still compiles, so every stored seed is legal by
construction (probes showed the verbatim recorded hours can refuse — the
compiler charges door walks the recorded stream does not spell out).

Each seed carries everything the self-warm re-projection needs: the logistic
signature (category x quadrant + tiles), the pool it was solved at, the
per-worker first hours, the settled doors, and the route as
(hour, task_id, worker) with the task id in the shared vocabulary.

Run:  .venv/bin/python offline_lab/build/warm_routes.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.wsr import beam as B, tasks as T
from agent.wsr.selfwarm import SOFT_WIDTH, WARM_K, soft_vector, task_categories
from agent.wsr.emit import compile_route

CORPUS = Path(__file__).resolve().parents[2] / "tests/day_layer/corpus/winner_days.json"
OUT = Path(__file__).resolve().parents[2] / "agent/artifact/warm_routes.npz"
REPO = Path("/home/amirelite_ai/Chista/kaggriculture-episodes-analyses/data/replays_parquet")
BOARD = 10
ACTIONS = ("WATER", "PLANT", "FEED", "CARE", "COLLECT_FERTILIZER",
           "HARVEST", "DIG", "DROP", "PLACE")


def recorded_ops(e) -> list[tuple[int, int, str, str | None, int, int]]:
    """The winning unit's action ops for this day: (unit, hour, op, arg1, x, y)."""
    con = duckdb.connect()
    d = f"{REPO}/{e['dump']}"
    q = f"""
    WITH w AS (
      SELECT episode_id, CASE WHEN reward0 >= reward1 THEN 0 ELSE 1 END AS wp
      FROM read_parquet('{d}/episodes.parquet') WHERE episode_id = {e['episode']}
    )
    SELECT h.unit, c.hour, h.op, h.arg1, h.x, h.y
    FROM read_parquet('{d}/hands_steps.parquet') h
    JOIN w ON h.episode_id = w.episode_id AND h.player = w.wp
    JOIN read_parquet('{d}/city_steps.parquet') c
      ON c.episode_id = h.episode_id AND c.step = h.step
    WHERE c.day = {e['day']} AND h.op IN {ACTIONS}
    ORDER BY h.unit, c.hour
    """
    return con.execute(q).fetchall()


def match_route(entry, tasks, rows) -> tuple[list[tuple[int, str, int]], int, int]:
    """Recorded ops -> (hour, task_id, worker): match by (op, item) with the
    item from the op's arg1, falling back to the bare op name; the candidate
    nearest to the recorded cell wins. Returns the route, the matched count
    and the unmatched count."""
    by_op: dict[tuple[str, str | None], list[int]] = defaultdict(list)
    for i in range(tasks.n):
        op0 = str(tasks.ops[i][0])
        item = str(tasks.ops[i][1]) if len(tasks.ops[i]) > 1 else None
        by_op[(op0, item)].append(i)
    claimed: set[int] = set()
    route: list[tuple[int, str, int]] = []
    matched = unmatched = 0
    for unit, hour, op, a1, x, y in rows:
        x, y = int(x), int(y)
        item = str(a1) if a1 is not None and str(a1) != "None" else None
        cands = [i for i in by_op.get((op, item), []) if i not in claimed] or \
                [i for i in by_op.get((op, None), []) if i not in claimed]
        if not cands:
            unmatched += 1
            continue
        i = min(cands, key=lambda j: abs(int(tasks.cells[j][0]) - x)
                + abs(int(tasks.cells[j][1]) - y))
        claimed.add(i)
        matched += 1
        route.append((int(hour), str(tasks.ids[i]), int(unit)))
    return sorted(route), matched, unmatched


def incremental_legal_seed(entry, tasks, day, route) -> tuple[list, int]:
    """Walk the recorded order; a placement joins the seed only if the partial
    route still compiles. Returns (seed placements, dropped count)."""
    seed_route: list[tuple[int, str, int]] = []
    dropped = 0
    for hour, tid, worker in sorted(route):
        trial = seed_route + [(hour, tid, worker)]
        trial_r = B.Result(pool=entry["hands"], route=sorted(trial), complete=False)
        try:
            compile_route(day, tasks, trial_r, horizon=24)
            seed_route = trial_r.route
        except (ValueError, IndexError):
            dropped += 1
    return seed_route, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    days = json.loads(CORPUS.read_text())
    if args.limit:
        days = days[:args.limit]

    seeds: list[dict] = []
    t_start = time.perf_counter()
    vocab: list[str] = []
    vidx: dict[str, int] = {}

    for e in days:
        grid = [(tuple(c), tuple(o), en) for c, o, en in e["chains"]]
        n_shed = sum(1 for _c, o, _e in grid if "FEED" in o or "FERTILIZE" in o)
        tasks = T.build(grid, available={g: int(h) for g, h in e["available"].items()},
                        drop_by=[12] * n_shed)
        day = B.Day(chains=tuple(grid),
                    available={g: int(h) for g, h in e["available"].items()},
                    hire_times=tuple(e["hire_times"]) or (1,) * e["hands"])
        rows = recorded_ops(e)
        route, matched, unmatched = match_route(e, tasks, rows)
        seed_route, dropped = incremental_legal_seed(e, tasks, day, route)
        if len(seed_route) < 2:
            print(f"  skip {e['dump']}/{e['episode']}/d{e['day']}: "
                  f"no legal seed ({matched} matched, {dropped} dropped)")
            continue
        pool = e["hands"]
        from agent.wsr.beam import _start_hours
        first_hours = {w: int(h) for w, h in enumerate(_start_hours(day, pool))}
        doors = [list(d) for d in
                 (B._hand_doors(day, tasks,
                                B.Result(pool=pool, route=seed_route,
                                         complete=False), pool)
                  if pool else [])]
        for _h, tid, _w in seed_route:
            if tid not in vidx:
                vidx[tid] = len(vocab)
                vocab.append(tid)
        cats = task_categories(tasks)
        id_idx = {str(t): i for i, t in enumerate(tasks.ids)}
        seeds.append({
            "pool": pool,
            "hire_hours": [int(h) for h in day.hire_times],
            "soft": soft_vector(day, tasks).tolist(),
            "route": seed_route,
            "categories": [cats[id_idx[str(tid)]] for _h, tid, _w in seed_route
                           if str(tid) in id_idx],
            "first_hours": first_hours,
            "doors": doors,
            "source": f"{e['dump']}/{e['episode']}/d{e['day']}",
            "matched": matched, "dropped": dropped,
        })
        print(f"  built {e['dump']}/{e['episode']}/d{e['day']}: "
              f"{len(seed_route)}/{matched} placements "
              f"({dropped} dropped as illegal)")

    # encode to the compact npz
    n = len(seeds)
    max_len = max((len(s["route"]) for s in seeds), default=0)
    max_hire = max((len(s["hire_hours"]) for s in seeds), default=0)
    max_hands = max((len(s["doors"]) for s in seeds), default=0)
    pool_arr = np.array([s["pool"] for s in seeds], dtype=np.int8)
    hire = np.zeros((n, max_hire), dtype=np.int8)
    soft = np.zeros((n, SOFT_WIDTH), dtype=np.int32)
    route_hour = np.full((n, max_len), -1, dtype=np.int16)
    route_task = np.full((n, max_len), -1, dtype=np.int16)
    route_worker = np.full((n, max_len), -1, dtype=np.int8)
    route_cat = np.full((n, max_len), -1, dtype=np.int8)
    route_len = np.array([len(s["route"]) for s in seeds], dtype=np.int16)
    doors = np.zeros((n, max_hands, 2), dtype=np.int8)
    for i, s in enumerate(seeds):
        hire[i, :len(s["hire_hours"])] = s["hire_hours"]
        soft[i] = s["soft"]
        for k, (h, tid, w) in enumerate(s["route"]):
            route_hour[i, k] = h
            route_task[i, k] = vidx[tid]
            route_worker[i, k] = w
        route_cat[i, :len(s["categories"])] = s["categories"]
        for k, d in enumerate(s["doors"]):
            doors[i, k] = d

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, pool=pool_arr, hire_hours=hire, soft=soft,
        route_hour=route_hour, route_task=route_task, route_worker=route_worker,
        route_cat=route_cat, route_len=route_len, doors=doors,
        vocab=np.array(vocab))
    print(f"\nbuilt {n} seeds in {time.perf_counter() - t_start:.0f}s | "
          f"artifact: {OUT} ({OUT.stat().st_size / 1024:.1f} KB) | "
          f"vocab {len(vocab)} task ids | k={WARM_K}")


if __name__ == "__main__":
    main()
