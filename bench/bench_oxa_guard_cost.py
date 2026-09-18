"""Guard cost: OXA's per-call time on the feed shape, best-of-50 per call.

Usage:
    .venv/bin/python -m bench.bench_oxa_guard_cost

This is the instrument behind docs/F057's timing sentence. It exists because
that sentence used to carry a procedure ("best-of-50 per call, validate=False,
two interleaved runs") with no committed script and no named instance, so a
reader could not repeat it -- and the numbers did not reproduce (R005).

The instance is the one the docs/F057 sweep builds for `feed`: `n` major feed
tasks on the 5x5 grid, ten workers with the first entering at hour 0 and the
rest at hour 1, horizon 24 -- the same `build()` as
`tests/day_layer/test_oxa_false_infeasible.py` and `bench/bench_oxa_fuzz.py`'s
sweep. Compare revisions the F057 way, one revision at a time:

    git show <rev>:day/solvers/oxa_solver.py > /tmp/old.py
    cp /tmp/old.py day/solvers/oxa_solver.py
    .venv/bin/python -m bench.bench_oxa_guard_cost
    git checkout day/solvers/oxa_solver.py

`validate=False` is deliberate: the guard's cost is the solver's own, not the
verifier's, and the runtime path (`agent/`) does not verify.
"""
from __future__ import annotations

import time

from agent.wsr.models import Cell, Instance, Item, MajorTask, Worker
from agent.wsr.solvers.oxa_solver import OxaConfig, solve_oxa

STOCK = {Item.WHEAT: 200, Item.FERTILIZER: 200, Item.COW: 50, Item.SHEEP: 50, Item.GOOSE: 50}
TILES = (8, 12, 20, 25)
CALLS = 50
RUNS = 2


def build(n: int, workers: int = 10) -> Instance:
    return Instance.compile(
        workers=[Worker(index=k, earliest_start=0 if k == 0 else 1) for k in range(workers)],
        major_tasks=[MajorTask(id=f"m{k}", type="feed", cell=Cell(x=k % 5, y=(k // 5) % 5))
                     for k in range(n)],
        warehouse_stock=STOCK,
        horizon=24,
    )


def main() -> int:
    config = OxaConfig(min_workers=1, validate=False)
    print(f"feed shape, best-of-{CALLS} per call, validate=False, "
          f"{RUNS} interleaved runs (milliseconds)")
    print(" tiles   cost   run0     run1")
    for n in TILES:
        instance = build(n)
        status = solve_oxa(instance, OxaConfig(min_workers=1))
        cost = status.solution.reported_cost if status.solution else status.status
        bests = []
        for _ in range(RUNS):
            best = float("inf")
            for _call in range(CALLS):
                start = time.perf_counter()
                solve_oxa(instance, config)
                best = min(best, time.perf_counter() - start)
            bests.append(best * 1e3)
        print(f" {n:5d}  {str(cost):>4}   "
              f"{bests[0]:.3f}   {bests[1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
