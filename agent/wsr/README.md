# wsr/

The day layer: the planner's chains become the ops a worker-day is made of. A chain per tile
arrives from the planner (`agent/planner/day.py:day_chains`), becomes a task array, a beam search
decides which worker does what and when, and the compiler writes the ops the engine reads.

| file | what it holds |
|---|---|
| `models.py` | `expand_chain`: a chain as the tasks it is made of, the order between them, and the worker ties |
| `tasks.py` | the day as arrays: `DISTANCE`, `TaskArray`, `build()` |
| `beam.py` | the search: `Day`, `Result`, `search()`, and the rules the search and the compiler share |
| `emit.py` | a route into the ops the engine reads: `DayOps`, `compile_route()`, `check_route()`, `to_plan()` |
| `routing.py` | the walk: `walk()` and `nearest_shed()`, the one place a path and a door are spelled |

Nothing here prices anything and nothing here touches the market. What a day costs is the
planner's business; what it can pay for is the market's.

This layer is on the runtime path: `agent/manager/core.py` -> `agent/planner/day.py` -> here.

---

## 1. The contract

Four calls, two objects in, two out:

```python
from agent.wsr import beam as B, tasks as T
from agent.wsr.emit import check_route, compile_route, to_plan

tasks  = T.build(chains, available=available, drop_by=drop_by)       # -> TaskArray
day    = B.Day(chains=tuple(chains), available=available, hands=hands)
result = B.search(day, tasks, beam=None, hands=None, max_hands=None,
                  budget_s=None, warm=None)                          # -> Result
ops    = compile_route(day, tasks, result)                           # -> DayOps
plan   = to_plan(ops, market=rows)                                   # -> {"units", "market"}
```

### Signatures

```python
T.build(chains, *, available: dict[str, int] | None = None, horizon: int = 24,
        drop_by=None) -> TaskArray

B.Day(chains, available: dict[str, int], horizon: int = TURNS_PER_DAY,
      hire_times: tuple[int, ...] = (), hands: int | None = None)

B.search(day: Day, tasks: TaskArray, *, beam: int | None = None,
         hands: int | None = None, max_hands: int | None = None,
         budget_s: float | None = None, warm: Result | None = None) -> Result

compile_route(day: Day, tasks: TaskArray, result: Result, *,
              horizon: int | None = None) -> DayOps

check_route(day: Day, tasks: TaskArray, result: Result) -> list[str]

to_plan(day_ops: DayOps, market=None) -> dict
```

### The input

| name | what it is |
|---|---|
| `chains` | one `(cell, chain_ops, entity)` per tile, in the caller's order, as `day_chains` builds them. `chain_ops` is the expanded chain; this layer never reads the DP's registry. |
| `available` | the hour each good (and seed) is in the shed. A buy in turn 0 is in the shed at hour 1, and a task that consumes a good cannot run before it. |
| `drop_by` | one entry per chain: the latest hour that chain's harvest must be banked, or `None` to leave it for the night. |
| `horizon` | the day's turns; `TURNS_PER_DAY` unless a test asks for fewer. |
| `hands` | the hands the planner offers today, besides the farmer who is always on the field. Zero is an offer: the farmer walks alone. |
| `hire_times` | optional: the hour each offered hand begins. Left out, each starts at the engine's earliest hour, `rules.hire_hour(k)` (a hand hired in turn `t` acts from `t + 1`, F040; ten orders a turn, F031). A recorded day passes its own hours, and then `hands` is their count. |
| `beam` | the width. `None` asks for `beam_for(tasks, workers)`: a step costs `beam x workers x tasks`, so the width follows the day's size. |
| `hands` | the pool to start at. `None` starts at the arithmetic floor (`lower_bound`) and halving-searches the smallest pool that carries the day; a number scans upward from it. |
| `max_hands` | the largest pool allowed, capped by `ceiling_for` (the hands the day offered). Equal to `hands`, it asks one yes-or-no question. |
| `budget_s` | the wall clock. The best route so far comes back with `out_of_time=True`. |
| `warm` | a `Result` from an earlier call on almost this instance. It seeds one extra row on top of the beam, and only at the pool it was searched with. |

### The output

| name | what it is |
|---|---|
| `Result.pool` | the hands the answer was searched with |
| `Result.route` | `[(turn, task_id, worker), ...]` - the turn a task occupies, its id, its worker (0 is the farmer). A drop whose bag is already empty is placed at turn `-1`: done, with no op. |
| `Result.complete` | that pool carried every task |
| `Result.spare` | the worker-turns the route leaves unspent (`spare_turns`), the room a caller may lay more work into |
| `Result.out_of_time` | a deadline stopped the search before it ran out of work to place |
| `Result.infeasible` | the arithmetic floor is above the ceiling, so no allowed pool can carry the day; the route is empty |
| `Result.doors` | the door each hand lands on at its own hire moment - the one statement of where the hands start |
| `Result.can_improve` | incomplete and cut short: more budget may place more work |
| `DayOps.units` | the ops per worker, indexed by turn, PASS-padded |
| `DayOps.arrivals` | `(hour, item, units)` per drop: what the day's market may actually sell today |

`check_route` returns the rules a route breaks (window, precedence, a good before it is in the shed,
a split worker tie) as sentences; empty means none. `compile_route` raises `ValueError` when a walk
does not fit the turns the route left, rather than padding it.

Public helpers a caller may want: `lower_bound(day, tasks)` (the fewest workers the day can need),
`ceiling_for(day, tasks)`, `beam_for(tasks, workers)`, `remaining_turns(day, tasks, result)` (per
worker) and `spare_turns(day, tasks, result)`.

## 2. The call sites

`agent/planner/day.py` is the only caller.

- `fit(chains, *, hands, available=None, budget_s=None, beam=None, hours_committed=0.0, warm=None)
  -> DayFit` asks whether the master's first day walks: `build`, then `search` with the pool the
  master priced (`hands=min(floor, hands)`, `max_hands=hands`), then `check_route` on a complete
  route. `DayFit.reason` is `""` when it fits, else `"hours"`, `"budget"` or `"unstable"`.
  `hands` is passed as it is, so a plan priced with no hands is searched with the farmer alone.
- `compile(day_plan, obs, *, hands=None, ...) -> dict` writes the day the dispatcher slices:
  `search` held to exactly the fitted pool, `compile_route`, then the market rows (hiring that
  pool) from the same chains and `to_plan`. A day that did not fit compiles to nobody doing anything.

The planner fixes the pool itself rather than asking `search(hands=None)` for the smallest one: the
hands are already priced by the master it is answering.

## 3. The rules the search and the compiler share

Each rule is written once and read by both sides, so the day that was priced is the day that is
written. A rule spelled twice drifts, and the engine refuses the difference in silence (F047).

- **Where a task is done** - `leg_target`: a task at its own tile; a DROP at the shed door nearest
  the worker when it drops, because any of the four shed-access tiles takes a DROP.
- **The walk between tasks** - `legs` and `leg_moves`: the path from where the worker stands, and
  after a DROP a detour through the nearest door to pick up again.
- **The door load** - `_bag`: the goods a worker uses before its first DROP, one PICKUP per good.
  After a DROP the bag is empty (the engine's DROP moves every item), so a good used later is
  fetched again on the way, not loaded at the door.
- **When the worker loads** - `loads_before`: before its first task, unless that task is door work
  (`door_work`: on a shed-access tile, needing no good) at a turn before the day's first good
  lands. A worker standing at the shed can pick up whenever it needs to.
- **When a walk is written** - `walk_start_turn`: a walk ends at its task's turn, so a worker with
  slack waits where it stands. This is what puts the hands on the doors the search priced (F040).
- **Where the hands start** - `Result.doors`, from the spawn rule at each hand's own hire moment.

## 4. The drop

A drop is a deadline on a good taken off a tile, not a chain op. `build(..., drop_by=)` derives one
DROP per `HARVEST` or `COLLECT_FERTILIZER` of a chain with a deadline: its `latest` is the deadline,
`banks` names the task it serves, and it is tied to the same worker, because the bag is the worker's.

A DROP empties the whole bag, so one drop banks every harvest since the previous one, and a drop
with nothing new in the bag is free. `DayOps.arrivals` is read off the route for whoever prices the
sell side.

## 5. The engine facts that bite in silence

The engine refuses a bad op without a word (F047), so a wrong day reports success and leaves the
board empty.

- **Hours are 0..23** (F048, `TURNS_PER_DAY`). The farmer has every turn; a hand has the turns
  after its hire (F040, F060).
- **A hand lands on the least-occupied shed door at its hire moment**, and a unit that walks off
  its door moves where every later hand lands (F040).
- **PICKUP and DROP work on any of the four shed-access tiles**; a PICKUP before its good is in
  the shed does nothing.
- **At most ten market orders per turn** (F031); unit ops resolve before market ops (F030).
- **The shed holds a fixed total across all items** and destroys the overflow (F043).
- **A LOCKED tile is not ours** (F042) and spends a unit's turns for nothing.

## 6. How to prove it

Only the board can tell a silent refusal from a plan that worked, so a day is played on the engine
(`offline_lab.fast_sim.FastSim`, which wraps the real interpreter) and asserted on the engine's own
counters. The days live in `tests/day_layer/`, over the recorded corpora in
`tests/day_layer/corpus/`. Re-introduce the bug and watch the guard go red before trusting it
(R007).

## 7. What is deliberately not here

- Prices. The contractor and the master own them.
- The market. `agent/planner/market.py` builds the queue from what it is handed.
- `BUY_LAND`. Quadrants unlock in a fixed order at fixed prices, so it is a planner decision.
