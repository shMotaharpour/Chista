# wsr/

The day layer: the planner's chains become the ops a worker-day is made of. A chain per tile
arrives from the planner (`day_chains`), becomes a task array, a beam search decides which worker
does what and when, and the compiler writes the ops the engine reads.

| file | what it holds |
|---|---|
| `models.py` | a chain expanded into the tasks it is made of, the order between them, and the worker ties |
| `tasks.py` | the day as arrays: `DISTANCE`, `TaskArray`, `ready()`, `window()`, `build()` |
| `beam.py` | the search: `Day`, `search()`, the layered objective, and where the units start |
| `emit.py` | a route into the ops the engine reads: `compile_route()`, `check_route()`, `to_plan()` |
| `routing.py` | the walk: the one place a path between two tiles is spelled |

Nothing here prices anything and nothing here touches the market. What a day costs is the
planner's business; what it can pay for is the market's.

**This layer is on the runtime path**: `agent/manager/core.py` -> `agent/planner/day.py` -> here.
§1 is the call site; the rest of the file is what a caller has to hand it.

---

## 1. The call site

`agent/manager/core.py` asks the planner, and the planner asks this layer
(`agent/planner/day.py`):

```python
# fit(): does the master's first day actually walk?
tasks  = T.build(chains, available=available)
day    = B.Day(chains=tuple(chains), available=available, hire_times=(1,) * max(1, hands))
result = B.search(day, tasks, beam=beam, hands=..., max_hands=...,
                  budget_s=budget_s, warm=warm)
if result.complete and check_route(day, tasks, result, result.settled):
    return DayFit(...)          # the day fits, and what it spent

# plan(): the day the caller dispatches
ops = compile_route(day, tasks, result, horizon=TURNS_PER_DAY, settled=result.settled)
return to_plan(ops, market=market.rows)
```

`fit()` asks a question the master cannot answer itself: its labour row charges each worked day the
ops it runs plus the walk to reach the tile, which is a lower bound by construction, so the master's
answer is an upper bound on what the farm can do. `DayFit.overhead` is the correction — what the
route spent over what the row charged (`agent/planner/day.py`).

`agent/replan.py` is the rung this replaced, and it cannot import: it asks
`agent.wsr.routing.plan_day` for the day, and `routing.py` holds the walk only. Nothing reaches it —
`tests/test_planner_reachable.py` keeps it in its `forbidden` tuple and `tests/test_agent_runtime.py`
keeps it out of the spine — so the file is a record of the old path, not a path.

## 2. What the day layer has to provide

Four pieces. One is this layer; the other three live in `agent/planner/`.

**(a) The needs** — what the day's chains must buy, and the last turn each may land.
`agent/planner/market.py`: `needs` (`:34`) counts what the chains consume, `availability`
(`:73`) says the hour each good is in the shed, and `buy_orders` (`:101`) turns the two into
the engine's own orders.

**(b) The market queue** — one row per turn, in the engine's own settle order:
`agent/planner/market.py:build` (`:221`), over `merge` (`:200`), `hire_orders` (`:137`) and
`sell_rows` (`:151`).

**(c) The day itself** — this layer. The whole contract is two
calls and two objects:

```python
chains = [(cell, chain_ops, entity), ...]        # the DP's winner per tile, in the caller's order
tasks  = T.build(chains, available=available, drop_by=drop_by)
day    = B.Day(chains=chains, available=available, hire_times=hire_times)
result = B.search(day, tasks, beam=None, hands=None, max_hands=MAX_HANDS,
                  budget_s=None, warm=None)
ops    = compile_route(day, tasks, result)       # -> DayOps
```

The input:

| name | what it is |
|---|---|
| `chains` | one `(cell, chain_ops, entity)` per priced tile, as `agent/planner/day.py:day_chains` (`:58`) builds them from the master's choices. `chain_ops` is the expanded chain; this layer never reads the DP's registry. |
| `available` | the hour each good is in the shed. A buy at hour 0 is in the shed at hour 1, and a task that consumes a good cannot run before it. |
| `hire_times` | the hour each offered hand may begin. A hand hired in turn 0 acts from hour 1 (F040). |
| `drop_by` | one entry per chain: the latest hour that chain's harvest must be banked, or `None` to leave it for the night. |
| `beam` | the width. `None` asks for the width the day's size implies, `beam_for(tasks, workers)` — a step costs `beam × workers × tasks`, so a fixed width is a fixed cost only for a fixed day. |
| `hands` | the pool to start at; `None` starts at the arithmetic floor. |
| `max_hands` | the largest pool allowed. Equal to `hands` it asks one yes-or-no question. |
| `budget_s` | the wall clock. The best route so far comes back with `out_of_time=True`. |
| `warm` | a `Result` from an earlier call on almost this instance; the search starts from its state, in a row on top of the beam. |

The output:

| name | what it is |
|---|---|
| `Result.pool` | the hands the answer was searched with |
| `Result.route` | `[(turn, task_id, worker), ...]` — the turn each task occupies, its id, its worker |
| `Result.complete` | whether that pool carried the whole day |
| `Result.out_of_time` | a deadline stopped it before it ran out of work to place |
| `Result.infeasible` | the arithmetic floor is above the ceiling, so no allowed pool can carry it |
| `Result.can_improve` | whether more budget would plausibly place more WORK — the deadline flag under the name of the decision |
| `DayOps.units` | the ops per worker, indexed by turn, PASS-padded |
| `DayOps.arrivals` | `(hour, item, units)` per drop — what the day's market may actually sell today |

`lower_bound(day, tasks)` is public if the caller wants the arithmetic floor itself. When no
allowed pool carries the day the best partial route comes back with `complete=False`, so the
caller keeps the part of the day that works rather than getting nothing.

**(d) The assembly** — `agent/planner/day.py:DayPlan` (`:174`) holds the master's solve, its
assignment and the `DayFit`; `agent/wsr/emit.py:to_plan` (`:136`) writes it as
`{"units": [...], "market": [...]}`, which is what `agent/dispatch.py` slices.

## 2b. The drop, and the pool

**A drop is a deadline on a harvest, not a chain op.** `HARVEST` says the crop left the tile;
whether it has to be in the shed by some hour is the sell side's decision, so it arrives as
`build(chains, drop_by=[...])` - one entry per chain, the latest hour that chain's harvest must be
banked, or `None` to leave it for the night. A DROP task is derived from it: its cell is the door
it hands the bag over at, its `latest` is the deadline, and `banks` names the harvest it serves.
A DROP empties the worker's WHOLE bag, so one drop banks every harvest since the previous one -
which is why a drop with an empty bag is free, the mirror of the fetch. `compile_route` reads the
drops off the route, and `DayOps.arrivals` is `(hour, item, units)` per drop, for whoever prices
the sell side.

**The budget, the warm start, and the answer.** `search(..., budget_s=)` stops at the deadline and
returns the best route it has, `out_of_time=True`. `search(..., warm=<a previous Result>)` starts the
beam from that route, so a caller that re-asks after a small change repairs instead of restarting -
it applies at the pool the route was searched with. `Result.can_improve` is that deadline flag under
the name of the decision, and it is about WORK: a carried day has nothing left to place, and a day
too big for any pool has no better route however long the search runs, so only an incomplete route
that was cut short has more to find. A caller that wants to polish a carried day's makespan is
asking a different question, and the flag says so rather than pretending to answer it.

**The day's input is the planner's, and only the planner's.** `Day(chains, available, hire_times)` -
no units: the engine resets every day to the farmer on the shed's corner door with no hands, so where
the units stand is not a decision the planner has. The hands' own positions are the search's per
route, from the spawn rule and where the units before them walked.

**The pool.** `search(hands=None)` halving-searches the smallest pool that carries the day,
between `lower_bound` and `max_hands`; `hands=` asks for the scan instead. The planner does not use
the halving search: it fixes the pool itself, because the halving costs several times a turn and the
hands are already priced by the master it is answering (`agent/planner/day.py`). A day whose
arithmetic floor is above the ceiling comes back `infeasible=True` with an empty route rather than
raising - a hundred tiles on a five-op chain needs more workers than the ceiling allows, and that is
an answer.

---

## 3. The seam worth knowing: the plan's market rows

`agent/planner/market.py:build` writes the plan's rows - the day's buys, its hires and its sells - in
the engine's own settle order, capped per turn (F031). `agent/market_layer.py` is the rung that used
to take those rows over: it drops the plan's `SELL` rows and appends its own
(`others = [o for o in given if not (o and o[0] == "SELL")]`). It is off the spine now
(`tests/test_agent_runtime.py` keeps it out), so the buys and the sells both come from the planner -
but if it is ever attached again, a plan whose sells are silently discarded looks exactly like a plan
with none.

## 4. The engine facts that bite in silence

These are the ones a day can violate while looking correct. The engine refuses a bad op
without a word (F047), so a wrong day reports success and leaves the board empty.

- **Hours are 0..23** (F048, `TURNS_PER_DAY = 24`). The farmer has 24 turns; a hand hired in
  turn 0 has 23, and the eleventh hired has 22 (F040, F060).
- **A hand lands on a shed door that is free when it is hired**, and a unit that walks off
  its door in turn 0 moves every hand after it (F040). The search prices this and the
  compiler walks the same cells; a disagreement of one cell plants on the neighbour's tile.
- **≤ 10 market orders per turn** (F031, `MAX_ORDERS_PER_TURN = 10`). The cap is per turn,
  not per day.
- **Unit ops resolve before market ops** inside a turn (F030).
- **The shed holds 100 across all items** and destroys the overflow (F043, `SHED_CAPACITY`).
- **A LOCKED tile is not ours** (F042) and spends a unit's hours for nothing.
- **The turn budget** is one second with a bank (F046).

## 5. How to prove it, and what to measure first

Only the board can tell a silent refusal apart from a plan that worked, so a day is replayed
against the real harness (`offline_lab.kaggle_env`) and asserted on what the engine did. The days
are in `tests/day_layer/`, over the recorded corpus (`tests/day_layer/corpus/real_days.json`, built
by `corpus/build_real_days.py`): the mixed day with its pool and second-day variants, the animal-drop
day, the drop and in-bag days, the land-image day, and the bounds, budget and spare-turn days.

They assert on the board's own counters, not on the compiler's intent. Re-introduce the bug and watch
the guard go red before trusting it (R007). `tests/day_layer/test_solve_time_day.py` is where the
search's own cost is measured; it is not asserted in milliseconds, because that is a property of the
machine and not of the plan.

## 6. What is deliberately not here

- Prices. The contractor and the master own them; the day takes them as given.
- The market. `agent/planner/market.py` builds the queue from what it is handed.
- `BUY_LAND`. Quadrants unlock in a fixed order at fixed prices (`LAND_PRICES`), so it is a
  day-level decision the planner owns, not a per-tile one.
