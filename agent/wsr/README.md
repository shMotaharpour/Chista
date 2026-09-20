# wsr/

The day layer: the planner's chains become the ops a worker-day is made of. A chain per tile
arrives from the DP, becomes a task array, a beam search decides which worker does what and
when, and the compiler writes the ops the engine reads.

| file | what it holds |
|---|---|
| `models.py` | a chain expanded into the tasks it is made of, the order between them, and the worker ties |
| `tasks.py` | the day as arrays: `DISTANCE`, `TaskArray`, `ready()`, `window()`, `build()` |
| `beam.py` | the search: `Day`, `search()`, the layered objective, and where the units start |
| `emit.py` | a route into the ops the engine reads: `compile_route()`, `check_route()`, `to_plan()` |
| `routing.py` | the walk: the one place a path between two tiles is spelled |

Nothing here prices anything and nothing here touches the market. What a day costs is the
planner's business; what it can pay for is the market's.

**This layer is not connected to the runtime.** The rest of this file is what an agent needs
to connect it.

---

## 1. The call site, and why nothing calls it

`agent/replan.py:278` builds the day and asks for a plan:

```python
plan = plan_day(tiles, unit_positions(view), new_hands=hires,
                bags=view.private.inventories, shed=view.private.shed,
                money=float(view.me.money), hires_today=view.me.hires_today,
                sells=sells, yields=yields,
                values=[float(v) for v in board.tile_values],
                prices=view.market_prices)
runtime._day_plan = plan
return plan.as_plan()
```

`plan_day` is not in this layer. It is on `origin/main` (`agent/wsr/routing.py:387`), and the
branch that rewrote `routing.py` (`7a91345`) dropped it. So `agent/replan.py` and
`agent/planner/master.py` cannot even import, and the plan rung never fires:

```
runtime.py:134   CHISTA_REPLAN    default off   -> self.replanner is None
runtime.py:139   CHISTA_MARKET    default off   -> the market layer is off
runtime.py:213   _rung_greedy     <- what answers today
```

The default entry point (`agent/main.py`) is the greedy rung. Everything merged above it —
the master's prices, the contractor's chains, this layer, the market layer — is off the path.

## 2. What the day layer has to provide

Four pieces. Three are missing here and exist on `origin/main`; the middle one is this layer.

**(a) The needs** — what the day's chains must buy, and the last turn each may land.
One `Need` per buy: `(order, latest_hour)` where the order is `("BUY_SEED", crop, 1)` or
`("BUY_ANIMAL", species, 1)` or a product. On main: `Need` (`routing.py:72`), `buy_order`
(`:114`), `needs_cost` (`:314`).

**(b) The market queue** — `merge_market(sells, needs, shed, capacity, hires)` returns one
row per turn, in the engine's own settle order. On main: `merge_market` (`:333`),
`hire_cost` (`:328`).

**(c) The day itself** — this layer, and the only part that is new. The whole contract is two
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
| `chains` | one `(cell, chain_ops, entity)` per priced tile — the same `tiles` `agent/replan.py:262` builds. `chain_ops` is the expanded chain; this layer never reads the DP's registry. |
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
| `Result.can_improve` | whether more budget would plausibly find more — the deadline flag under the name of the decision |
| `DayOps.units` | the ops per worker, indexed by turn, PASS-padded |
| `DayOps.arrivals` | `(hour, item, units)` per drop — what the day's market may actually sell today |

`lower_bound(day, tasks)` is public if the caller wants the arithmetic floor itself. When no
allowed pool carries the day the best partial route comes back with `complete=False`, so the
caller keeps the part of the day that works rather than getting nothing.

**(d) The assembly** — `DayPlan(units, market, needs, hires, ...)` with `as_plan()` returning
`{"units": [...], "market": [...]}`, which is what `agent/dispatch.py` slices. On main:
`DayPlan` (`:296`).

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
the name of the decision: False means no budget would find more, so spend the turns elsewhere.

**The day's input is the planner's, and only the planner's.** `Day(chains, available, hire_times)` -
no units: the engine resets every day to the farmer on the shed's corner door with no hands, so where
the units stand is not a decision the planner has. The hands' own positions are the search's per
route, from the spawn rule and where the units before them walked.

**The pool.** `search(hands=None)` halving-searches the smallest pool that carries the day,
between `lower_bound` and `max_hands`; `hands=` asks for the scan instead. A day whose arithmetic
floor is above the ceiling comes back `infeasible=True` with an empty route rather than raising -
a hundred tiles on a five-op chain needs 21 workers against a ceiling of 16, and that is an answer.

---

## 3. The seam worth knowing before you wire it

The market queue has two builders, and they compose rather than duplicate:

- `merge_market` puts the day's buys **and** the sells into the plan's rows.
- `agent/market_layer.py:155` then drops the plan's `SELL` rows and appends its own:
  `others = [o for o in given if not (o and o[0] == "SELL")]`.

So buys come from the day plan and sells come from the market layer. A plan whose sells are
silently discarded looks exactly like a plan with none.

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
against the real harness (`offline_lab.kaggle_env`) and asserted on what the engine did. Two
days are already written that way:

- `tests/day_layer/test_quadrant_day.py` — 25 tiles, 5 hands, 50 plantings.
- `tests/day_layer/test_corners_day.py` — two pastures at the board's ends, a goose beside
  each, an empty barn each, one hand hired and the second quadrant bought.

Both assert on the board's own counters, not on the compiler's intent. Re-introduce the bug
and watch the guard go red before trusting it (R007).

**Measure before turning it on.** The search is priced on a quadrant (24–53 ms) and on a
94-task day (370 ms). It has never been run on a 100-tile board, and the plan rung gets one
turn's budget, once a day.

## 6. What is deliberately not here

- Prices. The contractor and the master own them; the day takes them as given.
- The market. `merge_market` builds the queue from what it is handed.
- `BUY_LAND`. Quadrants unlock in a fixed order at fixed prices (`LAND_PRICES`), so it is a
  day-level decision the planner owns, not a per-tile one.
