# F053 — The monolithic default is not an oracle

**Summary (<=50 words):** `solve_cpsat`'s default builds no objective, so its
cost is an arbitrary feasible schedule — 18 × 0 and 2 × 2 in 20 runs with 8
parallel workers — while `OPTIMAL` describes feasibility. The prefix search's 0
is the verified optimum. The oracle comparison now runs optimizing and
single-worker.

## What was measured

The instance is the aggregating one from
`tests/wrs/test_cpsat_solver.py` (4 × `feed`, 6 workers offered, 50 wheat,
24 hours), against `secretary/solvers/cpsat_solver.py` as ported in F052.

| configuration | 20 runs |
|---|---|
| `solve_cpsat`, default (`feasibility_only=True`, `num_search_workers=8`) | 18 × `(OPTIMAL, cost 0)`, **2 × `(OPTIMAL, cost 2)`** |
| `solve_cpsat`, `feasibility_only=False`, `num_search_workers=1` | **20 × `(OPTIMAL, cost 0)`** |
| `cpsat_binarySearch` (6 runs) | 6 × `(OPTIMAL, cost 0, pool_capped_at=1)`, `verify_solution` valid |

The rate depends on how busy the process is: the same flake is rarer as a fresh
process per run (0 failures in 20 single-test `pytest` invocations on this box)
and dependable in-process (2 of 20), because eight parallel search workers
divide the work differently depending on what else they are doing. The R007
re-introduction therefore reproduces in-process, and the probe is what settles
it — a single fresh-process run can pass by luck.

`verify_solution` accepts the cost-0 solution with a single active worker, and
0 < 2 — so the prefix search's answer is the optimum and the monolithic answer of
2 was not.

## Mechanism (not the time budget)

`cpsat_solver.py` adds `Minimize(Σ fib·u)` **only** when
`config.feasibility_only` is false — its own comment at that site reads
"feasibility_only=True (default): NO Minimize — CP-SAT only checks feasibility".
So in the default mode the model has no objective to be optimal *about*: the
reported cost is whatever feasible schedule the 8-worker parallel search
returns, and `OPTIMAL` is a statement about the feasibility model. With eight
workers the search allocation varies between runs, which is exactly the 10 %
flip seen here (and the 1-in-5 the review saw under a busier machine).

The hypothesis that `feasibility_probe`'s time slicing caused it is refuted by
the clock: the monolithic solve takes 60–77 ms against a 30 s ceiling, and the
prefix search 168–188 ms — nothing is being cut off.

## What changed

- The comparison is now against the **optimizing** solve with a single search
  worker (`feasibility_only=False, num_search_workers=1`), which is
  deterministic, and the `xfail` marker is gone.
- **Which solver was wrong:** `solve_cpsat` as the test used it. It is the
  *default configuration* that is wrong to call an oracle, not the prefix
  search; `cpsat_direct` remains the optimizing entry point for cross-checks.
- The unstable assertion was the only thing that could have caught this: it did,
  and the fix is to make both sides of it reproducible (R005).

## Why this matters beyond the test

An oracle whose answer varies run to run cannot referee anything — and the
drop-loop in the secretary (#14 §6) leans on INFEASIBLE verdicts. This finding
is the reason the oracle is now pinned to an objective-bearing, single-worker
configuration before any solver comparison is trusted.
