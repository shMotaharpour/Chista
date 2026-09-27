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
         warm: Result | None = None) -> Result

compile_route(day: Day, tasks: TaskArray, result: Result, *,
              horizon: int | None = None) -> DayOps

check_route(day: Day, tasks: TaskArray, result: Result) -> list[str]

to_plan(day_ops: DayOps, market=None) -> dict
```

---

## 2. Guidelines for Callers (`agent/planner/day.py`, `manager/core.py`)

To achieve maximum performance (<500ms solve time) and optimal routing density:

1. **Always Pass `warm=previous_result` During Hourly Re-solves**:
   `Result` carries a native vectorized `state: tuple[np.ndarray, ...]` containing `(done, when, who, free, where, travel)`. When re-solving the same day from turn to turn, passing `warm=last_result` bypasses all string parsing and leg reconstruction, copying the previous solution into the beam in **1 microsecond via NumPy slice assignment**.

2. **Model Tile Harvests with `harvests=` in `T.build`**:
   Pass `harvests=[(crop, units), ...]` aligned with `chains`. This activates **In-Field Self-Serve Consumption**: when a worker harvests wheat or collects fertilizer, subsequent `FEED` and `FERTILIZE` tasks on that worker draw directly from the worker's bag with **zero shed door pickups** (`bag.get(good, 0) == 0`), saving multiple travel turns.

3. **Pass Range Bounds `(hands=start, max_hands=ceiling)` for Pool Optimization**:
   When the optimal pool size is unknown, pass both `hands` and `max_hands`. WSR executes an $O(\log_2 N)$ **Binary Halving Search** (`_smallest_pool`) instead of a linear loop, and dynamically contracts the upper bound `hi` to the actual number of workers utilized (`used_hands`). This cuts search iterations by over 70%.

4. **Do Not Pass Wall-Clock Deadlines (`budget_s`)**:
   Wall-clock timeouts have been eliminated from WSR. The search is structurally bounded by `tasks.n` steps and beam width, guaranteeing that identical inputs produce 100% bit-identical routes without hardware clock jitter.

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

### Targeted Leaf Repair (`_repair_unplaced`)
A fast post-search repair pass slides trailing unplaced leaf tasks (e.g. `WATER`, `DIG`, `CARE`) into remaining idle windows of passing workers, validated against `check_route` and `compile_route`.

---

## 4. Shared Engine Invariants

Both search and compiler strictly enforce engine facts:
- **Hours are 0..23** (`TURNS_PER_DAY`). The farmer acts from turn 0; hands act from their first acting hour (`hire turn + 1`, F040).
- **Spawn Doors**: Hands appear on the least-occupied shed door at their hire moment (`_hand_doors`, `Result.doors`). A unit walking off its door in turn 0 moves where subsequent hands land.
- **Door Loads & DROPs**: `DROP` empties the entire worker bag (`kaggriculture.py:343-356`). Any good taken before a drop cannot be consumed after it without refetching (enforced by the R2 ban in `_expand`).
- **Bulk Pickups**: `compile_route` aggregates all door preloads into a single multi-unit `("PICKUP", good, n)` op per good, spending 1 turn instead of $n$ turns.
