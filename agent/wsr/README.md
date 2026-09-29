# wsr/ (Worker-Space-Routing)

The day layer: the planner's high-level chains become the exact, legal tile operations a worker-day is made of. A chain per tile arrives from the planner (`agent/planner/day.py`), is compiled into an indexed `TaskArray`, a deterministic beam search plans which worker does what and when, and the compiler emits the verified op vectors the engine reads.

| file | what it holds |
|---|---|
| `models.py` | `expand_chain`: a chain as its constituent tasks, precedence pairs (`ORDER_MATTERS`), and worker ties |
| `tasks.py` | the day as vectorized arrays: `TaskArray`, `chain_weight`, `effective_latest`, `DISTANCE`, `build()` |
| `beam.py` | the search engine: `Day`, `Result`, `search()`, deterministic multi-key selection, spatial locality, and vectorized deduplication |
| `emit.py` | route compiler into engine ops: `DayOps`, `compile_route()`, `check_route()`, `to_plan()` |
| `routing.py` | spatial geometry: `walk()`, `nearest_shed()`, Manhattan distance utilities |

Nothing here prices anything and nothing here touches the market. What a day costs is the planner's business; what it can pay for is the market's.

This layer is on the runtime critical path: `agent/manager/core.py` -> `agent/planner/day.py` -> here.

---

## 1. The Contract

```python
from agent.wsr import beam as B, tasks as T
from agent.wsr.emit import check_route, compile_route, to_plan

# 1. Build vectorized task arrays from planner chains
tasks  = T.build(chains, available=available, drop_by=drop_by, harvests=harvests) # -> TaskArray

# 2. Package day timetable and labour constraints
day    = B.Day(chains=tuple(chains), available=available, hire_times=hire_times)

# 3. Deterministic structural search
result = B.search(day, tasks, beam=None, hands=None, max_hands=None, warm=warm)  # -> Result

# 4. Compile into turn-by-turn unit ops and market arrivals
ops    = compile_route(day, tasks, result)                                        # -> DayOps
plan   = to_plan(ops, market=rows)                                                # -> {"units", "market"}
```

### Signatures

```python
T.build(chains, *, available: dict[str, int] | None = None, horizon: int = 24,
        drop_by=None, harvests=None) -> TaskArray

B.Day(chains, available: dict[str, int], horizon: int = TURNS_PER_DAY,
      hire_times: tuple[int, ...] = ())

B.search(day: Day, tasks: TaskArray, *, beam: int | None = None,
         hands: int | None = None, max_hands: int | None = None,
         budget_s: float | None = None, warm: Result | None = None) -> Result

compile_route(day: Day, tasks: TaskArray, result: Result, *,
              horizon: int | None = None) -> DayOps

check_route(day: Day, tasks: TaskArray, result: Result) -> list[str]

to_plan(day_ops: DayOps, market=None) -> dict
```

---

## 2. Guidelines for Callers (`agent/planner/day.py`, `manager/core.py`)

1. **`warm=` is a SEED, not a speed-up, and it measured negative in the shape it was tried**:
   `Result` does carry a vectorized `state` and `search(..., warm=route)` starts a beam row from a route
   it did not build. Seeding EVERY day from an earlier route measured net negative on the 57-day corpus
   (placed 5454 against 5563, wins 3 / losses 20): the warm row occupies a beam row and its partial
   packing is a basin the search climbs out of. Do not pass `warm` unless the route is one this same
   day's search produced.

2. **Pass `harvests=` in `T.build` when the caller knows what each tile takes**: a HARVEST's yield comes
   from the board, not from the chain, and `harvests=[(good, units), ...]` aligned with `chains` is how
   the layer learns it. It is also what decides whether a chain's consumers can eat out of the harvester's
   own bag: the one-worker form is built only when the timetable does NOT stock the good at hour 0
   (`tasks.py`, the merge rule). Forcing that form on a stocked day SPENDS turns on the days it was
   measured on (the eight short winner days place four tasks fewer in total), so it is a property of the
   input and not a switch worth flipping.

3. **Give the pool range as `(hands=start, max_hands=ceiling)`, and let the pass find the minimum**:
   the range is searched SILENTLY in one vectorized multipool pass, and larger pools switch off the
   moment a smaller one carries the day, so the smallest carrying pool is exact rather than probed.
   (The O(log2 N) halving this section used to describe, `_smallest_pool`, is not in the tree.)

4. **`budget_s` is real and worth passing on a big day**: the search is bounded by the day's steps and
   the beam's width, but a day with a hundred and eighty tasks costs seconds, so a wall-clock ceiling is
   the only promise that survives one - the search keeps the best route it has found and reports
   `out_of_time=True` rather than running long. `None` means no deadline. Identical inputs still give
   identical routes for a fixed budget: the ceiling decides how much of the search runs, not how ties
   break.

---

## 3. Core Architectural Mechanisms

### Deterministic Multi-Key Shortlist
Candidate selection in `_select` uses an exact lexicographic sort to eliminate random memory-layout tie-breaks:
`order = np.lexsort((worker_of, self_serve_bonus, hop_of, importance, primary))`

- **`primary` (Adaptive Critical-Path Balancing)**:
  `primary = flat_hour * 48 - chain_weight * max(0, 24 - flat_hour) * alpha`
  where $\alpha = 1$ for normal days and $\alpha = 2$ for heavy days ($tasks.n > 130$). Smoothly pulls deep chain tasks earlier in the morning while decaying to pure finish-hour priority by evening so trailing leaf tasks are never starved.
- **`importance` (DAG Downstream Closure)**:
  `tasks.chain_weight` pre-computed in `TaskArray.__post_init__` via boolean reachability matrix squaring. Measures how many downstream tasks are killed if this task is omitted.
- **`hop_of` (Spatial Locality)**:
  Manhattan distance to target tile (`np.abs(hx - cx) + np.abs(hy - cy)`). Breaks finish-hour ties by shortest travel distance, forcing dense geographic clustering and eliminating cross-board wandering.
- **`self_serve_bonus`**:
  Prioritizes workers who already hold the required good in-bag from prior on-field harvests, eliminating trips to the shed.
- **`worker_of`**:
  Deterministic worker index tie-breaker.

### Vectorized State Deduplication
In `_dedupe`, state signatures are formed by packing `done` bits, `free` bytes, and `where` coordinates into contiguous memory blocks viewed as `np.void` structured types. Unique child states are extracted at C-level using `np.unique(..., return_index=True)` in **<0.1 ms**, completely eliminating slow Python loops and byte hashing.

### Effective Chain-Bounded Deadlines
In `TaskArray.__post_init__`, every task's deadline is constrained by its downstream length:
`effective_latest = max(0, latest - chain_weight)`
In `_expand`, `start <= effective_latest` guarantees that a prerequisite (e.g. `FERTILIZE`) is never scheduled at hour 23 when its successor (`WATER`) requires hour 24.

### The Repair Pass (`_repair_unplaced`)

The search appends each task to the END of a worker's day, so on a day the horizon has closed on, the
SHAPE of that day decides whether the last tasks fit. The pass is what re-shapes it, in four families,
each candidate re-timed with the writer's own rules and accepted only when `compile_route` and
`check_route` agree with it:

1. **re-shape** - the unplaced task is offered every place it could take in each worker's day, and the
   day is re-ordered exactly where it is small enough to solve exactly (`_path_order`, Held-Karp over
   the subsets, `PATH_LIMIT`).
2. **hand one task away** - when no day can take it, the nearest day gives one of its own tasks to a
   worker that can still reach that task.
3. **hand several away** - when one is not enough, the same worker hands its work away one task at a
   time, each hand-over shortening ITS day by at least a turn, which bounds the chain by the turns
   actually missing. A chain that does not end with the task placed is rolled back.
4. **the doors belong to the route** - the engine gives each hand the least-occupied shed-access tile at
   its OWN hire turn, so a route the pass changes has NEW doors. They are re-derived here and the day is
   accepted only if it still compiles and checks under them: carried over, the hands walked from tiles
   the engine never gave them and their ops were refused in silence (F047), measured at 28 of 127 ops on
   one winner day against 1 for the game's own route.

Measured on the 36-day bench, the pass is worth about ten tasks and costs ~0.3 s on the heaviest day
(with and without it on the same day, never across days). It closes the twenty-two-tile day of #209
that the search alone leaves one tile short, and its arithmetic has its own guards in
`tests/day_layer/test_path_cover_day.py`.

---

## 4. Shared Engine Invariants

Both search and compiler strictly enforce engine facts:
- **Hours are 0..23** (`TURNS_PER_DAY`). The farmer acts from turn 0; hands act from their first acting hour (`hire turn + 1`, F040).
- **Spawn Doors**: Hands appear on the least-occupied shed door at their hire moment (`_hand_doors`, `Result.doors`). A unit walking off its door in turn 0 moves where subsequent hands land.
- **Door Loads & DROPs**: `DROP` empties the entire worker bag (`kaggriculture.py:343-356`). Any good taken before a drop cannot be consumed after it without refetching (enforced by the R2 ban in `_expand`).
- **Bulk Pickups**: `compile_route` aggregates all door preloads into a single multi-unit `("PICKUP", good, n)` op per good, spending 1 turn instead of $n$ turns.

---

## 5. How a Change is Judged

Four measurements, all of them comparatives - a number without the arm it beat means nothing:

| gauge | what it is | where |
|---|---|---|
| the gauge pair | the twenty-two-tile #209 day (`hands=1` and the mixed day), the planner's own compile of the day the search priced | `tests/day_layer/test_pickup_hour_and_planner_compile_day.py`, `tests/day_layer/test_mixed_*.py` |
| the corpus | 21 winner days from games the player won, with the game's own operations as the reference | `tests/day_layer/test_winner_days.py` |
| the bench | 36 real days, wall-clock budget per day, the broad score | `offline_lab/bench/sweep.py --budget 20` |
| the world | the compiled plan stepped through `FastSim` from the entry's own snapshot, every op scored against the engine's own effect | the winner corpus's engine witness (see the notes beside the corpus) |

The pass's own cost is measured by running the same day twice with the pass replaced by the identity -
never by comparing wall clocks across different days, and never by trusting one day for a constant that
the whole corpus can answer (the shortlist factor was one day's number until the benchmark was asked).

What the layer currently reaches: the bench 3298 of 3332 placed, the winner corpus 47 passed / 8
xfailed - the eight are days the ENGINE provably carried and this layer does not, so they are a search
shortfall and not a model that forbids the game's play. What they do NOT respond to, each measured:
a single task moved between workers, a bounded chain of moves, the one-worker bag form, a global
assignment solved with `scipy`, and the charge fixed point's cost. Their capacity is not the constraint
(our answers have more labour on every one of the eight), which leaves the assignment inside the search
- a design change, not a repair.
