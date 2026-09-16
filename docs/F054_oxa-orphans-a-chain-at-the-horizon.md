# F054 — OXA commits a chain's last turn and orphans the rest of the day

**Summary (<=50 words):** OXA's greedy could commit a task on the horizon's last
turn, so its successors — pinned by the committed `exec_time` as a hard lower
bound — could never run, and `solve_oxa` answered INFEASIBLE for 38 days the
oracle schedules. A chain-tail lookahead in the candidate test fixes every one.

## What was measured

The issue-#14 sweep: ten workers (the first entering at hour 0, the rest at hour
1), horizon 24, `n` major tasks of a single kind laid out on the 5x5 grid,
`n = 6..25` (140 instances), shed stock of 200 wheat / 200 fertilizer / 50 of
each animal. Every INFEASIBLE answer was put to the oracle `cpsat_binarySearch`
— the oracle by F053, since feasibility under a shrinking pool cap *is* the
min-worker objective — with `CpSatConfig(time_limit_seconds=10)`.

| kind | `solve_oxa` (pre-fix) INFEASIBLE at n = | the oracle's answer there |
|---|---|---|
| `plnt` | 15..25 (11) | `FEASIBLE` (11 of 11) |
| `wet_harvst` | 15, 16, 17, 18, 19, 22 (6) | `FEASIBLE` (6 of 6) |
| `wet_harvst_plnt` | 8..25 (18) | `FEASIBLE` at 8..17 (10 of 10); no answer inside its 10 s slice at 18..25 |
| `frtz_water` | 7, 8, 9, 18..25 (11) | `FEASIBLE`/`OPTIMAL` (11 of 11) |
| `feed`, `frtz`, `place_animal` | never | — |

**38 verdicts were falsified by the oracle** — 11 + 6 + 10 + 11. The
`wet_harvst_plnt` rows at n = 18..25 are reported separately as *unproven*:
CP-SAT neither schedules nor refutes them inside that slice, so they are not
counted as false, and the sweep's oracle call returned for every one of them
(none raised). The issue-#14 comment's list (`plnt` 15..19, `wet_harvst`
13/15..19/22/23, `frtz_water` 7..9/14/20..25) is **not** a subset of this one and
this one is not a subset of it: n = 13, 14 and 23 came back FEASIBLE pre-fix and
n = 18, 19 came back INFEASIBLE, so the two lists are measurements of different
things (the comment's predates 40d9cfa, when the oracle could not complete at
all). These are the numbers this box measured, at ten workers.

Raising the pool to 4, 8, 10, 12 or 16 workers leaves the verdict unchanged
(`plnt` n=15, `wet_harvst_plnt` n=8, `frtz_water` n=7 are `INFEASIBLE` under
every one of those pools pre-fix and `FEASIBLE`, `verify_solution`-valid, under
every one post-fix), so this is not a capacity limit.

Post-fix the same 140 instances all return a schedule `verify_solution`
accepts, the sweep completes in 0.3 s, and no oracle call is needed because no
INFEASIBLE answer is left to check. The guard is measured, not assumed, to be
cheap on F052's feed shape — best-of-50 per call, `validate=False`, two
interleaved runs: pre-fix 0.11-0.13 / 0.20-0.22 / 0.43-0.51 / 0.67-0.72 ms at
n = 8/12/20/25 against 0.12-0.16 / 0.25-0.26 / 0.52-0.56 / 0.77-0.82 ms
post-fix. That is up to +0.1 ms/call, two orders of magnitude inside the 20 ms
runtime budget.

## Mechanism (the exact lines)

`_build_worker_route` accepted a target task whenever `actual_exec <= horizon`
(the `if actual_exec > horizon: continue` test in its candidate loop) and never
asked whether that task's **successors** could still run. The accepted task's
`exec_time` is written into the shared `exec_times` map when the route closes,
and every later worker computes `ready_t = exec_times[p] + 1` — a hard lower
bound, not a hint. A commit at hour 24 therefore pinned its successor to hour
25 inside a 24-hour horizon **for every worker**, and the leftover target made
`solve_oxa` report INFEASIBLE while workers 3..9 sat idle.

Reproduced on `plnt` n=15 (pre-fix module restored, per-worker routes printed):

```
  w0 -> [... ('m0_plant', 21), ('m0_water', 22), ('m5_plant', 24)]
  w2 -> [('m4_plant', 7), ('m4_water', 8)]
  w3 .. w9 -> []
  UNSCHEDULED: ['m5_water']
```

Worker 0's route ends `m0_water` at t=22 and then commits `m5_plant` at t=24;
`m5_water`'s predecessor is now pinned at 24, so `ready_t` is 25 for all seven
remaining workers and each builds an empty route. Same shape on
`wet_harvst_plnt` n=8, where worker 1 commits `m0_harvest` at t=24 and orphans
`m0_plant`/`m0_water2`.

**The issue-#14 suspicion is refuted.** There is no prefix-size search in this
module: no `r` loop exists, `compute_lower_bound` is never called by
`solve_oxa`, and the verdict comes from the plain leftover-targets check at the
end of the dispatch loop. Extra workers cannot help for the pinning reason
above — not because a search gave up.

## The pre-PR review pass (three defects, one of them mine)

The first version of the guard charged every precedence hop `1 + dist`, and that
is wrong for a plain precedence pair, which **two different workers may serve**:

```
p at (0,0) -> s at (9,9), horizon 10, two workers
  valid schedule: w0 runs p at t=9, w1 runs s at t=10   (the oracle: OPTIMAL)
  first guard:    tail(p) = 1 + dist((0,0),(9,9)) = 19 -> refuses p -> INFEASIBLE
```

`bench/bench_oxa_fuzz.py` (400 seeded random instances: single tasks, same-cell
and cross-cell pairs, item chains with paired acquires and groups) measured the
damage — statuses per solver revision, all on the same seeds:

| revision | OPTIMAL | FEASIBLE | INFEASIBLE | invalid | vs pre-fix |
|---|---|---|---|---|---|
| pre-fix (40d9cfa) | 97 | 92 | 208 | 3 | — |
| first guard (committed earlier in this PR) | 97 | 53 | 246 | 4 | **40 regressions**, 3 improvements |
| reviewed (this commit) | 97 | 99 | 201 | 3 | **0 regressions**, 6 improvements |

So the travel term is now charged only where one worker *must* serve both ends —
an edge inside a `single_worker_group` — and plain edges cost the one turn
precedence forces. The same table caught the second defect: an answer with an
idle hand in the middle (fuzz seed 9) under-reported its `reported_cost`, and
`verify_solution` answered INVALID_SOLUTION for an otherwise legal schedule.
The engine's rule is append-only hiring (`verify.py`, COST ACCOUNTING):
`fib(0..max_active)`, idle middle hands included. Fixed here.

Third, **still open** (pre-existing, not introduced by this PR): the dispatch
loop assigns warehouse entry cells over *all* candidate workers while
`verify_solution` recomputes the placement rule over the workers that are
actually routed, so a worker can be sent out from a cell it does not hold —
`worker 2: first task 'c0_a' reachable too early from entry`. Measured on fuzz
seeds 93, 182 and 205; seeds 182/205 already failed this way before the PR, and
the failure total still improved (208 + 3 → 201 + 3). It needs its own change
(the assignment has to be re-derived from the active set, which is not known
before the dispatch runs) and its own sweep, so it is recorded here rather than
patched blind.

## What changed

- `_successor_gaps` + `_unfinished_tail_turns` (`oxa_solver.py`, stdlib only):
  the precedence DAG inverted over the target tasks, each edge carrying the
  minimum turns that must separate its ends — `1`, plus the travel when the two
  tasks share a `single_worker_group`. The candidate loop skips a task when
  `actual_exec + tail > horizon`; the figure is a lower bound on any completion
  of that chain, so the refusal can only convert a doomed commit into leaving
  the chain to a worker that still has room. Built once per solve.
- `reported_cost` now follows the engine's append-only payroll
  (`fib(0..max_active)`), not the sum over the routes that carry tasks.
- `tests/wrs/test_oxa_false_infeasible.py`: the sweep as a guard (every
  INFEASIBLE answer put to the oracle), the eight measured instances pinned by
  name, the worker-pool invariance check, the cross-cell pair, the gap payroll
  and a positive control that runs the oracle-confirmation branch on green runs.
- `bench/bench_oxa_fuzz.py`: the seeded audit that produced the table above
  (`--oracle` confirms INFEASIBLE verdicts too, `--dump` writes per-seed
  statuses for a differential run against another revision).
- R007 failing direction, all re-checked by re-introducing the bug: with the
  pre-fix module in place the sweep test reddens on `plnt` n=15
  (`1 failed, 2 passed`); with the *first* guard in place the cross-cell pair
  and the gap-payroll tests redden (`2 failed, 18 deselected`); both go green
  again with this commit's solver (`20 passed`).

## Why it matters

OXA is the solver the secretary calls every turn (#14 §6). An INFEASIBLE verdict
drops every chain, the agent passes the day and `planted_tiles` stays 0 — and
`wet_harvst_plnt` is exactly F049's harvest-then-replant cycle, the measured
tile optimum. Nothing in the logs separated "the day did not fit" from "the
solver is broken"; this guard is that difference.
