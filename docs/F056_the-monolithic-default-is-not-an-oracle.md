# F056 — The oracle is the binary search, and its pool is minimal by construction

**Summary (<=50 words):** The min-worker objective makes `cpsat_binarySearch` the
oracle: feasibility under a shrinking pool cap means the smallest feasible cap is
the optimum. The monolithic solve's default mode builds no objective —
18 × cost 0 and 2 × cost 2 in 20 runs — so its cost can never be the reference.

## What was measured

The instance is the aggregating one from `tests/wrs/test_cpsat_solver.py`
(4 × `feed`, 6 workers offered, 50 wheat, 24 hours), against
`secretary/solvers/cpsat_solver.py` as ported in F052.

| configuration | result |
|---|---|
| `solve_cpsat`, default (`feasibility_only=True`, `num_search_workers=8`) | 18 × `(OPTIMAL, cost 0)`, **2 × `(OPTIMAL, cost 2)`** in 20 runs |
| `cpsat_binarySearch` | 6/6 `(OPTIMAL, cost 0, pool 1)`, `verify_solution` valid |
| one worker fewer, cap = 0 | INFEASIBLE — so the returned pool is minimal |

## The verdict

**The binary search's answer is the optimum.** Its objective is the minimum
number of workers, and it reaches it by probing *feasibility* at shrinking pool
caps: feasible at `k`, infeasible at `k − 1` **is** the optimality proof for that
objective. Nothing else has to certify it — no cost comparison, and no second
solver.

That is why the defect was in the *assertion*, not in a solver. The old
assertion compared the binary search's cost against `solve_cpsat`'s default
answer and called the latter "the monolithic optimum". But the default mode
builds **no objective at all** (see the comment at the `Minimize` site:
"feasibility_only=True (default): NO Minimize — CP-SAT only checks feasibility"),
so its reported cost is an arbitrary feasible schedule — which the 8-worker
parallel search then varies between runs (2 of 20 here). `OPTIMAL` in that mode
is a statement about the feasibility model, not about cost.

**My first fix was wrong and is retracted:** it kept the monolithic solve as the
reference and merely changed its configuration. That keeps the assertion
parasitic on the one solver whose cost is meaningless for this objective.

## What the test asserts now

`tests/wrs/test_cpsat_solver.py::test_binary_search_pool_is_the_minimum_feasible_pool`:

1. the binary search returns a schedule, `verify_solution` accepts it;
2. its pool is **minimal**: re-solving the same instance with the pool capped one
   lower must be infeasible.

Step 2 is the check that can fail, and it is the objective itself rather than a
proxy for it (R007: the failing direction is `worker_pool_cap = active` instead
of `active - 1`, which admits a schedule and reddens the assertion).

## Why it matters

The `xfail` is gone and the suite is deterministic (5/5 runs of the settled test,
80 passed + 1 skipped for `tests/wrs`). The secretary's drop loop (#14 §6) leans
on INFEASIBLE verdicts, so which verdicts are trustworthy had to be settled
before any solver comparison — that is what this finding records.
