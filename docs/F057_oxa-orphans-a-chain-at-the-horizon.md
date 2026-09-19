# F057 — OXA commits a chain's last turn and orphans the rest of the day

**Summary (<=50 words):** OXA's greedy could commit a task on the horizon's last
turn, so its successors — pinned by the committed `exec_time` as a hard lower
bound — could never run, and `solve_oxa` answered INFEASIBLE for 31 days of the
issue-#14 sweep that the oracle schedules. A chain-tail lookahead in the
candidate test fixes every one.

## What was measured

The issue-#14 sweep: ten workers (the first entering at hour 0, the rest at hour
1), horizon 24, `n` major tasks of a single kind laid out on the 5x5 grid,
`n = 6..25` (140 instances), shed stock of 200 wheat / 200 fertilizer / 50 of
each animal. Every INFEASIBLE answer was put to the oracle `cpsat_binarySearch`
— the oracle by F056, since feasibility under a shrinking pool cap *is* the
min-worker objective — with `CpSatConfig(time_limit_seconds=10)`. The
instrument is the sweep in `tests/day_layer/test_oxa_false_infeasible.py`, run against
the pre-fix blob with **`PYTHONHASHSEED=0`**:

| kind | `solve_oxa` (pre-fix) INFEASIBLE at n = | the oracle's answer there |
|---|---|---|
| `plnt` | 15..19, 23..25 (8) | `FEASIBLE` (8 of 8) |
| `wet_harvst` | 13, 15..19, 24 (7) | `FEASIBLE` (7 of 7) |
| `wet_harvst_plnt` | 8..25 (18) | `FEASIBLE` at 8..17 (10 of 10); no answer inside its 10 s slice at 18..25 |
| `frtz_water` | 7, 8, 9, 20..25 (9) | `FEASIBLE`/`OPTIMAL` at 7, 8, 9, 20, 21, 23 (6); no answer at 22, 24, 25 |
| `feed`, `frtz`, `place_animal` | never | — |

**31 of those 42 INFEASIBLE verdicts were falsified by the oracle** — 8 + 7 + 10
+ 6. The eleven rows CP-SAT could not decide (`wet_harvst_plnt` 18..25,
`frtz_water` 22, 24, 25) are reported separately as *unproven*: it neither
schedules nor refutes them inside that slice, so they are not counted as false,
and the oracle call returned for every one of them (none raised).

An earlier version of this table said **38** (11 + 6 + 10 + 11) over the sets
`plnt 15..25`, `wet_harvst 15..19 + 22`, `frtz_water 7..9 + 18..25`. That
measurement was taken on a solver whose answer depended on the interpreter's
hash order, and it reproduces under **no** pinned seed I tried (40 of them): see
*Reproducibility*. The seed is part of the measurement, and the numbers above
name theirs.

The issue-#14 comment's list (`plnt` 15..19, `wet_harvst` 13/15..19/22/23,
`frtz_water` 7..9/14/20..25) is **not** a subset of this one and this one is not
a subset of it: under this measurement `wet_harvst` 22 and 23 and `frtz_water`
14 came back FEASIBLE pre-fix where the comment calls them false-INFEASIBLE,
while `plnt` 23..25 and `wet_harvst` 24 are false here and absent from the
comment. The comment's numbers predate the oracle question being settled — F052
found the `cpsat_binarySearch`/`solve_cpsat` comparison unstable, the issue
comment itself says to treat `cpsat_binarySearch` as unverified, and
ac9a833/40d9cfa/F056 settled that the prefix search *is* the oracle — so the two
lists are measurements of different things.

Raising the pool to 4, 8, 10, 12 or 16 workers leaves the verdict unchanged
(`plnt` n=15, `wet_harvst_plnt` n=8, `frtz_water` n=7 are `INFEASIBLE` under
every one of those pools pre-fix and `FEASIBLE`, `verify_solution`-valid, under
every one post-fix), so this is not a capacity limit. Reproduced by
`tests/day_layer/test_oxa_false_infeasible.py::test_a_bigger_worker_pool_does_not_change_the_verdict`
(4/8/12/16) and by hand for 10.

Post-fix the same 140 instances all return a schedule `verify_solution` accepts
(20 `OPTIMAL`, 120 `FEASIBLE`, 0 rejected), and no oracle call is needed because
no INFEASIBLE answer is left to check. The sweep takes 0.33-0.35 s for all 140
instances on this box (`tests/day_layer/test_oxa_false_infeasible.py`'s sweep loop,
three runs).

The guard is measured, not assumed, to be cheap on the sweep's feed shape —
best-of-50 per call, `validate=False`, two interleaved runs, the instrument
`bench/bench_oxa_guard_cost.py`:

| tiles (`feed`) | pre-fix (40d9cfa) | this commit | delta |
|---|---|---|---|
| 8 | 0.076 ms | 0.092 ms | +0.016 |
| 12 | 0.139 ms | 0.169 ms | +0.030 |
| 20 | 0.294 ms | 0.355 ms | +0.061 |
| 25 | 0.452 ms | 0.532 ms | +0.080 |

That is up to about +0.1 ms/call, and the worst case is 0.53-0.55 ms against
F046's 20 ms runtime budget — under 3% of it. Absolute times move with machine
load (an earlier run of the same instrument on a loaded box read 1.5x these,
and repeats on an idle box moved the delta between +0.08 and +0.10 ms/call);
the headroom is what travels, and the instrument is committed so a reader can
re-measure on their own silicon.

## Mechanism (the exact lines)

`_build_worker_route` accepted a target task whenever `actual_exec <= horizon`
(the `if actual_exec > horizon: continue` test in its candidate loop) and never
asked whether that task's **successors** could still run. The accepted task's
`exec_time` is written into the shared `exec_times` map when the route closes,
and every later worker computes `ready_t = exec_times[p] + 1` — a hard lower
bound, not a hint. A commit at hour 24 therefore pinned its successor to hour
25 inside a 24-hour horizon **for every worker**, and the leftover target made
`solve_oxa` report INFEASIBLE while workers 3..9 sat idle.

Reproduced on `plnt` n=15 (pre-fix module in place, `PYTHONHASHSEED=0`,
per-worker routes printed):

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
`solve_oxa` (only `tests/day_layer/test_oxa_solver.py` imports it), and the verdict
comes from the plain leftover-targets check at the end of the dispatch loop.
Extra workers cannot help for the pinning reason above — not because a search
gave up.

## Reproducibility (why this document was rewritten)

The first version of F057 — and the two commits before this one — quoted numbers
a reader could not repeat. `_build_worker_route` built its candidate list by
iterating the `remaining_targets` **set**, and the greedy's winner was the first
candidate to reach the minimum of `(cost, tie_breaker)`. Ties were therefore
broken by the interpreter's hash order, so `solve_oxa` answered differently
under different `PYTHONHASHSEED` values on the *same* instance. Measured:

| revision | statuses over the 400 fuzz seeds, across hash seeds |
|---|---|
| pre-fix (40d9cfa) | `OPTIMAL 97`, `FEASIBLE` 91-94, `INFEASIBLE` 207-209, invalid 2-3 (16 seeds). The document's `97/92/208/3` is the seed-0/1/8 run only |
| reviewed (this commit) before the ordering fix | `97/99/201/3` at seeds 0 and 7, `97/98/202/3` at seeds 1, 2, 3, 6, 8, 9, 11, 14, 15 |
| the sweep's per-kind sets | `plnt` INFEASIBLE at 8 values at seed 0, 6 at seed 2, 7 at seed 3 — the document's 11-value set `15..25` at none of 40 seeds |

The fix is that the candidate choice is now a **total order** —
`key = (cost, tie_breaker, -tail, tid)` — so the winner cannot depend on the
order the candidates were visited in, and the verdict is a function of the
instance alone. `-tail` (serve the more constrained task first) is not only
canonicalisation: it also fixed fuzz seed 307, where the tie between a
standalone tile and a chain head was resolved the other way and the greedy
stranded the chain (`tests/day_layer/test_oxa_reproducibility.py::
test_a_side_task_cannot_strand_a_chain`).

Guards for all of it are in `tests/day_layer/test_oxa_reproducibility.py`: the audit's
400 statuses are pinned as the numbers this document reports, and two runs of
the audit under different hash seeds must produce byte-identical dumps. What
stays seed-dependent is history: the pre-fix and first-guard blobs in the tables
below are `PYTHONHASHSEED=0` measurements, and this document names that seed
wherever it quotes one of them.

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
damage — statuses per solver revision, all on the same seeds, the old blobs
pinned to `PYTHONHASHSEED=0`, the diffs counted by
`bench/bench_oxa_diff.py` (regression = a usable day, `OPTIMAL`/`FEASIBLE`,
became unusable):

| revision | OPTIMAL | FEASIBLE | INFEASIBLE | invalid | vs pre-fix |
|---|---|---|---|---|---|
| pre-fix (40d9cfa) | 97 | 92 | 208 | 3 | — |
| first guard (committed earlier in this PR) | 97 | 53 | 246 | 4 | **43 regressions**, 4 improvements |
| reviewed (this commit) | 97 | 99 | 201 | 3 | **0 regressions**, 7 improvements |

The first version of this table said **40 regressions, 3 improvements** for the
first guard and **6 improvements** for the reviewed one. With the comparison
definition written down in `bench/bench_oxa_diff.py`, the measured answers are
43/4 and 0/7 against the seed-0 pre-fix run (0/46 against the first guard), and
the whole table is reproduced by
`tests/day_layer/test_oxa_reproducibility.py::test_the_audit_reproduces_the_statuses_docs_F057_reports`.
One seed moves in neither direction by this definition: 93 goes
`INFEASIBLE -> INVALID_SOLUTION`, both unusable, counted separately by the tool.

The INFEASIBLE column is a verdict count, not a claim that those days are
impossible: `bench/bench_oxa_fuzz.py --oracle` reports **15 INFEASIBLE verdicts
the oracle schedules** (seeds 18, 45, 52, 58, 62, 100, 179, 191, 208, 233, 235,
251, 262, 321, 337) — pre-existing at all three revisions, and the fourth open
class below.

So the travel term is now charged only where one worker *must* serve both ends —
an edge inside a `single_worker_group` — and plain edges cost the one turn
precedence forces. The same table caught the second defect: an answer with an
idle hand in the middle (fuzz seed 9) under-reported its `reported_cost`, and
`verify_solution` answered INVALID_SOLUTION for an otherwise legal schedule.
The engine's rule is append-only hiring (`verify.py`, COST ACCOUNTING):
`fib(0..max_active)`, idle middle hands included. Fixed here.

Third, **fixed in this branch** (pre-existing at 40d9cfa, 756356a and 6784719):
the dispatch loop assigned warehouse entry cells over *all* candidate workers
while `verify_solution` recomputes the placement rule over the workers that are
actually routed, so a worker could be sent out from a cell it does not hold —
`worker 2: first task 'c0_a' reachable too early from entry`. Measured on fuzz
seeds 93, 182 and 205; seeds 182/205 already failed this way before the PR. The
assignment is now re-derived from the routed set and the dispatch re-run until
it stops moving (`_dispatch` plus the fixed point in `solve_oxa`, capped at four
attempts — the audit converges in one or two: 380 of the 400 seeds need a single
dispatch, 20 need two). The audited row moves from `97 / 99 / 201 / 3 invalid`
to `97 / 101 / 202 / 0`: 0 regressions and 2 improvements by
`bench_oxa_diff.py`, with seed 93 going `INVALID_SOLUTION -> INFEASIBLE` (that
day really is impossible — the oracle answers INFEASIBLE for it too) and seeds
182/205 becoming valid `FEASIBLE` days. The 140-instance sweep is unchanged: all
140 verify, none INFEASIBLE.

Fourth, **partly fixed in this branch** (pre-existing at 40d9cfa, 756356a and
6784719): the dispatch gives each worker one pass, and a task is only
*available* once its predecessors are committed. A day whose only legal
arrangement serves a successor from a lower-indexed worker while a later worker
serves its predecessor is therefore called INFEASIBLE, even though precedence
only orders the two *times*. The minimal case is two plain tasks far apart on
the grid:

```
t0 at (9,0) -> t1 at (0,0), horizon 10, two workers both starting at hour 0
  solve_oxa:          INFEASIBLE        (pre-fix, first guard and this commit)
  cpsat_binarySearch: OPTIMAL           w0 -> t1 at 10 from NW, w1 -> t0 at 9 from NE
                      verify_solution accepts that day (total_cost 1)
```

Worker 0 can reach `t1` (distance 8) but not `t0` (distance 9 → hour 10, and the
guard refuses it: `10 + 1 > 10`); worker 1 can reach `t0` at hour 9, and by then
worker 0's single pass is over, so `t1` is left unserved. In the 400-seed audit
this is not rare: `bench/bench_oxa_fuzz.py --oracle` reports 15 INFEASIBLE
verdicts the oracle schedules — seeds 18, 45, 52, 58, 62, 100, 179, 191, 208,
233, 235, 251, 262, 321, 337 — and all 15 are INFEASIBLE at the pre-fix blob,
the first guard and 6784719 (verified one seed at a time against
`cpsat_binarySearch` at 5 s, `verify_solution`-valid).

**The pass now repeats over the hands that have no route yet**, which is exactly
the minimal case's fix: the greedy finds the oracle's day (`w0 -> t1 at 10`,
`w1 -> t0 at 9`), and fuzz seed 179 goes `INFEASIBLE -> FEASIBLE`. With the
entry-cell fixed point of the third class alongside it, the audited row is
`97 / 102 / 201 / 0` against `97 / 99 / 201 / 3` — 0 regressions, 3 improvements
and 1 reason-changed by `bench_oxa_diff.py` (seed 179 from the repeated pass,
182/205 from the placement, 93's reason changing). **14 of the 15 seeds stay
open**, and their trace says why: they need a *routed* hand's route extended
after another hand commits, not another pass over the idle ones. Seed 18 routes
both hands in the first pass (`w0: c0_a 8, c0_b 14`; `w1: c1_acq 1, c1_cons 11`)
and leaves `c1_tail` with no idle hand left to take it; seed 45 leaves `extra`
behind with two idle hands that cannot reach it inside its 10-hour horizon at
all. Rebuilding a route after the fact means re-running its emission phase (the
batched acquires and their `resolved_qty`) and re-checking the placement rule
per route, so it is its own change; the 14 seeds are listed here so that change
starts from a measurement rather than a guess. This is also why the class is
recorded as *partly* fixed rather than closed.

Fifth, **fixed in this branch** (was open at 40d9cfa, 756356a and 6784719): a
declared precedence edge *into* an aggregatable pickup was dropped on both sides
— such a pickup is not a target task, so neither `target_preds` nor
`_successor_gaps` carried the edge — and the preload phase emitted the pickup at
its own hour. The day then failed validation as a precedence violation:

```
T at (9,9) -> S = PICKUP WHEAT (cell-less) -> C = FEED at (0,0)
group [S, C], horizon 24, two workers starting at hour 0

  solve_oxa           INVALID_SOLUTION  "precedence violated: 'T' must strictly precede 'S'"
                      (40d9cfa, 756356a and this commit; verify_solution confirms)
  cpsat_binarySearch  OPTIMAL           w0 -> S at 11, C at 24; w1 -> T at 10
```

No recipe in `models.py` emits an edge into a pickup — the option chains all go
pickup -> consume — so this input is outside the formulation's own shapes, but
`Instance.compile` accepted it without complaint and the answer was a schedule
the solver's own verifier rejects: an unusable day rather than a named input
error. `solve_oxa` now refuses the shape up front, next to its other input
validation:

```
  raise InfeasibleInputError(
      "precedence 'T' -> 'S' puts a task before a preloaded pickup: 'S' is "
      "emitted as the setup turn at the head of its route, so it cannot "
      "follow another task, and this solver answers INVALID_SOLUTION when "
      "asked to")
```

(`tests/day_layer/test_oxa_solver.py::
test_an_edge_into_a_preloaded_pickup_is_rejected_not_mis_scheduled`, whose
R007 direction is the removal of that one call: the test then stops raising and
gets `INVALID_SOLUTION` back — measured.) The check lives in `solve_oxa` rather
than `Instance.compile` because `cpsat_solver` *can* honour the edge (it
returned the OPTIMAL day above), and taking the input away from the model would
take that away too. Honouring it in the preload instead is the other possible
fix and a deliberate follow-up: it changes when an acquire may be emitted, which
this solver's whole emission phase is built around. The 400-seed audit is blind
to the shape either way — its generator only emits edges *out of* a pickup — so
a hunt that counts *false INFEASIBLE* on it reports zero while it still returned
an unusable answer: the rejection landed in the invalid column, not the
INFEASIBLE one.

## What changed

- `_build_worker_route`'s candidate choice is a total order
  (`key = (cost, tie_breaker, -tail, tid)`), so the verdict no longer depends on
  the interpreter's hash order (see *Reproducibility*).
- `_successor_gaps` + `_unfinished_tail_turns` (`oxa_solver.py`, stdlib only):
  the precedence DAG inverted over the target tasks, each edge carrying the
  minimum turns that must separate its ends — `1`, plus the travel when the two
  tasks share a `single_worker_group`. The candidate loop skips a task when
  `actual_exec + tail > horizon`; the figure is a lower bound on any completion
  of that chain, so the refusal can only convert a doomed commit into leaving
  the chain to a worker that still has room. It is built once per worker route
  (three calls for this sweep's ten-worker pool, which routes three hands —
  counted, not assumed).
- `_dispatch` + `_dispatch_rounds` + the placement fixed point in `solve_oxa`:
  the dispatch became a helper over shared state, repeated for the hands that
  have no route yet (a successor stranded by a later hand's commit is picked up
  again — part of the fourth class, the rest stays open), and the entry-cell
  assignment is re-derived from the workers that are actually routed and the
  dispatch re-run until it stops moving, because `verify_solution` recomputes
  the placement over the routed set (constraint 10) and not over every
  candidate. The pool cap is applied to the lowest-indexed hands, matching
  `cpsat_solver` and the engine's append-only payroll.
- `reported_cost` now follows the engine's append-only payroll
  (`fib(0..max_active)`), not the sum over the routes that carry tasks.
- `tests/day_layer/test_oxa_false_infeasible.py`: the sweep as a guard (every
  INFEASIBLE answer put to the oracle), the eight measured instances pinned by
  name at the seed the table names, the worker-pool invariance check, the
  cross-cell pair, the gap payroll and a positive control that runs the
  oracle-confirmation branch on green runs. The guard's strength is the
  oracle's: when a 10 s slice ends in `UNKNOWN` the branch abstains rather than
  confirming (pre-fix, that is 8 of the 18 `wet_harvst_plnt` rows), which is
  why the positive control pins a day the oracle *decides* — it returns
  `INFEASIBLE` there, not `UNKNOWN`.
- `tests/day_layer/test_oxa_reproducibility.py`: the audit's statuses pinned as the
  numbers above, hash-seed independence, and the stranded-chain instance (fuzz
  seed 307).
- `bench/bench_oxa_fuzz.py`: the seeded audit that produced the table above
  (`--oracle` confirms INFEASIBLE verdicts too, `--dump` writes per-seed
  statuses for a differential run against another revision).
- `bench/bench_oxa_diff.py`: the differential, and the definition of
  "regression" this document quotes — the number's source, committed next to it.
- `bench/bench_oxa_guard_cost.py`: the guard-cost instrument behind the timing
  table.
- R007 failing direction, all re-checked by re-introducing the bug, with the
  tests as they stand now and `PYTHONHASHSEED=0`: with the pre-fix module in
  place the sweep test reddens on `plnt` n=15 (`1 failed, 2 passed`) with
  *"plnt n=15: OXA said INFEASIBLE but the oracle admits a schedule (FEASIBLE) —
  the greedy orphaned a chain instead of leaving it to another worker"*, and the
  eight pinned pairs redden (`8 failed, 13 deselected`); with the *first* guard
  in place the cross-cell pair and the gap-payroll tests redden
  (`2 failed, 19 passed`); with the candidate key reverted to
  `(cost, tie_breaker)` the two reproducibility guards redden
  (`1 failed, 2 passed`, *"the same instances answered differently under another
  hash order: ['307']"*). All green again with this commit's solver:
  `tests/day_layer` 104 passed, 1 skipped.

## Why it matters

OXA is the solver #14 §6 wires in as the day's per-turn dispatcher (the
wiring is not in this tree yet: nothing outside `tests/` and `bench/` calls
`solve_oxa`). Once it is, an INFEASIBLE verdict drops every chain, the agent
passes the day and `planted_tiles` stays 0 — and `wet_harvst_plnt` is exactly
F049's harvest-then-replant cycle, the measured tile optimum. Nothing in the
logs would separate "the day did not fit" from "the solver is broken"; this
guard is that difference.
