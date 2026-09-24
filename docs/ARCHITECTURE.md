# Chista — the system's shape, its one vocabulary, and its one belief

**Revision 2 (2026-09-17).** The single place that defines what the parts are, what
they may read, and what the words mean. Where this document and a module disagree,
the engine and this document win; the module is the bug.

It exists because three vocabularies grew for one world — the engine's action
strings, the tile DP's chain ops, and the WSR day's enums — and every
consumer paid: the DP's market ops looked incomplete, `PLACE` existed twice, the
day's `PRODUCT_ITEMS` lacked every crop, and one chain was expanded twice
(`tile_dp/graph.py::_exec_chain` at build time, `day/models.py::expand_major_task`
at runtime).

---

## 1. The canonical vocabulary — `world/model.py`

Engine-derived (R002), imported by every layer:

Every row below is the name as the module spells it TODAY, with the count read off
the module itself (`len(<name>)`), and the file it lives in named — because the
first version of this table named six symbols that no longer exist and put a
seventh in the wrong module.

| name | what it is | count | module |
|---|---|---|---|
| `PRODUCTS` | what the market trades and the farm sells | 9 | `world/model.py` |
| `CROPS` / `ANIMALS` | the five crops, the three species | 5 / 3 | `world/model.py` |
| `INVENTORY_ITEMS` | what the farm buys or consumes and what it can hold | 12 | `world/model.py` |
| `RESOURCE_NAMES` / `RESOURCE_ID` | the resources by name, and the same as ids | 18 | `world/model.py` |
| `DUAL` | wheat and fertiliser — a resource **and** a product | 2 | `world/model.py` |
| `COLUMNS` | the DP graph's columns = resources + products, in the stored order | 18 | `world/model.py` |
| `UNIT_ACTIONS` | what a unit may be told to do | 18 | `world/model.py` |
| `MOVES` | the four movement ops | 4 | `world/model.py` |
| `WORKER_OPS` | what a unit spends a turn on | 18 | `world/action.py` |
| `MARKET_ORDERS` | the market's own vocabulary: `SELL`, `BUY_SEED/PRODUCT/ANIMAL`, `HIRE`, `BUY_LAND` | 6 | `world/model.py` |
| `STRUCTURES` / `TILE_KINDS` | what a tile can hold, and what a tile can be | 2 / 7 | `world/model.py` |

Gone since the first version, and not renamed: `GOODS`, `RESOURCES`, `VECTOR`,
`ACTIONS`, `MARKET_ACTIONS` and `CHAIN_OPS`. The market/farm split they described is
now carried by `MARKET_ORDERS` versus `WORKER_OPS`, and `WORKER_OPS` lives beside the
engine's handlers in `world/action.py` rather than in `model.py`.

Rules:

1. **Names are the engine's.** Crops and animal products are *products*, never
   resources; seeds, animals, labour, fertiliser and wheat are resources.
2. **The market is not the farm's business.** `WORKER_OPS ∩ MARKET_ACTIONS = ∅`. A
   chain may name a `BUY_*` because it prices that input; the contractor and the
   WSR never touch the market. The only place that supplies market inputs for a
   tile is the test harness that exercises it.
3. **One expansion.** `world/model.py::compile_chain` turns a chain into engine
   actions (`BUILD` → the structure the entity needs, `PLACE` → the animal,
   `NO_ACT` → `PASS`). Nothing else expands a chain.
4. **An op outside the vocabulary is never emitted** — the engine ignores it in
   silence (F047).

The `day/` tree this first version called legacy is gone — `agent/wsr/day/` no longer
exists, and `PLACE_ANIMAL` is not mentioned anywhere in `agent/`. The day layer's
files are the four this document's §2 names. §5's migration list was written against
that tree and needs the same pass.

---

## 2. The DP and the WSR: one chain per tile, one day per unit

A DP chain is the day of **one tile**: its worker ops in order. A WSR major task is
the day of **one unit**. They are not the same object, and the conversion is not
mechanical — it depends on what grows on the tile:

- **One-shot crops**: the engine's window rules force the order. Fertilise, then
  water, then harvest; a harvest before `first_yield_day` is refused, and a
  fertiliser dose only counts inside its three-day window. The chain's op order is
  therefore a *constraint*, not a preference.
- **Ongoing crops and animals**: no such order. Water and feed are daily acts, and
  the tile accumulates yield; the chain's ops commute.
- **Collective work the tile cannot see.** A fertiliser or feed op needs the good in
  the unit's bag, which means a shed trip, a `PICKUP`, and a `BUY_*` one turn
  earlier; a harvest needs a `DROP` to reach the shed before it can be sold. None of
  that is in the chain.

So the conversion is: **one chain → one tile-day, and a search over the day's tiles
→ one day for the whole farm**. `wsr/` holds it: `models.py` expands a chain into
the tasks it is made of, `tasks.py` turns the day's chains into the arrays the
search reads, `beam.py` decides which worker does what and when, and `emit.py`
writes the ops the engine reads. The search owns travel, the shed trips and the
hour assignment; the DP owns which chain runs on which tile.

---

## 3. One belief for the whole system

`belief/` is the only reader of the market and the rival. Everything else asks it.

```python
MarketState = {
  step, day, hour,
  inventory: (9,) int,          # the market's stock, from the observation
  prices:    (9,) int,          # K.market_price at that inventory (parity-tested)
  drain:     {mean: (9,), sd: (9,), horizon: int},   # closed form, no sampling
  rival:     {sales: (9,) per turn, shed_estimate: (9,), slot_order: inferred},
  shed:      {room: int, guard_margin: int},
}
```

- **Only `belief/` reads the market.** The `day/market.py` and `day/opponent.py` this
  first version pointed at do not exist any more; the market and the rival are read in
  `belief/market.py` and `belief/opponent.py`, which is where this rule is enforced
  today.
- **Consumers**: the master prices revenue at `prices` and internal scarcity at its
  own duals; the compiler schedules sells at `drain`; the runtime publishes timing.
- **Not owned**: movement, hours, land, animal chains, the order book's optimiser.
- **R004/R005**: every field names its source — the observation, an engine
  function, or a measurement in `bench/`.

---

## 4. Ownership

| layer | owns | must not |
|---|---|---|
| `world/` | the vocabulary (`model.py`), the town and the rules (`rules.py`: `SHOPS`, `TOWN_CENTER_PRODUCTS`), the engine binding, config | hold policy |
| `belief/` | the one belief: the market, the rival, the demand, the shed projection and the sell plan, and the schemas | decide actions, read the board |
| `tile_dp/` | the chain registry, the per-tile DP | know units, travel, or the market queue |
| `planner/` | the season: what to grow, when to sell, hire, buy, expand | emit engine actions |
| `wsr/` | the day: a chain per tile expanded into the tasks it is made of, a beam search that decides which worker does what and when, and the ops the engine reads | re-price what the master or the belief priced; touch the market |
| `agent/` | the spine: decode, dispatch, deadline, fallback | plan |

`secretary/` is gone: its market and shed halves are `belief/`, its routing and
scheduling halves are `wsr/` (the `day/` package this line used to name no longer
exists - the same tree §1 and §3 still referred to).

---

## 5. Migration — one branch, then one PR

All world changes land on **one branch** (`world/definition`); when the world is
right, it goes to `main` in one PR. No piecemeal merges. Steps:

1. `world/model.py` — names, the resource/product split, the market/worker split,
   the compile table, and the named views. *(Done.)* The guards this step named
   (`tests/test_model.py`) were written against that shape of the module; the module
   was later rebuilt around the enums (`Product`, `Crop`, `Animal`, `Structure`,
   `TileKind`, `UnitAction`, `MarketOrder`, `Move`, `Column`), every guard in that
   file stopped importing, and the file was deleted rather than left red - the
   vocabulary it measured no longer exists under those names.
2. The compiler imports `compile_chain`; the duplicated expansions are deleted.
   *(Done: `agent/replan.py` lost `chain_turns`/`project_day`.)*
3. The packages move: market and shed to `belief/`, routing and scheduling to
   `wsr/`, `secretary/` deleted. *(Done.)*
4. The WSR's enums are deleted; the scheduling layer uses the named views, and its
   task types are **keys into a chain table**, expanded by the model's own
   `CARRIES`/`PLACING_OPS`. *(Done.)* One exception, named rather than hidden:
   `wet_harvst_plnt` is the registry's rotation
   `WATER-HARVEST-DIG-PLANT-WATER`, and the four-op form it keeps is deliberate —
   re-basing it is a re-measure, not a rename.
5. `belief/market.py::forecast` is seeded from `MarketState` (one reader), and the
   market tests are the acceptance. *(Done.)*
6. The DP's op sets are computed views of `world/model.py`, and `PLACE_ANIMAL` is
   retired with the graph rebuild. *(Done.)*

Each step ends with the full suite green. The name ratchet that counted what is left
to move lived in `tests/test_model.py` and went with it: nothing measures hand-spelled
good names now, and re-basing that count on the current names is a fresh measurement.

## 6. The day layer — the hours, the doors, the spare

`agent/wsr/` turns one day into one op list per worker. Four decisions there are not
obvious from the code, and each was paid for:

**The hours are the hourly layer's input.** `Day.hire_times` is the hour each hand is
AVAILABLE to act — the hour its HIRE settles plus one (F040) — and `Day.hands` is
`len(hire_times)`. There is no separate count to disagree with it, and zero hires is
an empty tuple (the farmer is always on the field and is not in the list). Nothing
inside `Day` derives an hour: a default computed there hides where the hands came
from, and the engine's own earliest (`rules.earliest_hire_times`) is an optimistic
BOUND that belongs at the call site. A bad tuple is `Day`'s to refuse.

**Where a hand lands is wsr's own answer.** A hand appears on the door the engine
gives it at ITS OWN hire moment (`_hand_doors`, `_start_positions`), and a unit that
walks off its door in the first turn moves every hand hired after it. So the doors are
the search's fixed point (`_settle`): search, derive the doors from the route, search
again until they agree — two passes or not at all. `Result.doors` is the statement of
where the hands start, and `_consistent` checks the route against it.

**`settled` is not passed to the compiler, deliberately.** `compile_route` takes the
doors and nothing else. `Result` still carries a `settled` field, but no caller hands
it over: an explicit position was the path that wrote the hands from the farmer's start
cell, ignored the doors the search priced, and made a day whose farmer walks in turn 0
uncompilable (issue #162). The doors are the one statement of where the hands start; a
unit's own start is not a choice.

**The spare is one definition.** `remaining_turns(day, tasks, result)` gives each
worker the room its route leaves — its own day (horizon less its own start hour), one
turn per task, one per tile walked FROM ITS OWN START, and the pickups themselves: the
WAIT for goods is room, because the compiler writes PASS for those turns. `spare_turns`
is `sum(remaining_turns(...))` and nothing else. `ceiling_for` bounds the pool by the
day's OWN hands (`len(hire_times)`) rather than by the task count — a pool beyond the
offer is a pool nobody is paying for.

**A trip through the door is a property of the route.** `legs` gives each worker's day
as `(turn, row, fetch)`, `leg_target` names the tile or the DROP's door (the nearest
one when it drops), and `leg_moves` writes the walk, or the walk through the door and
the PICKUP. The search prices this trip and the compiler writes it from the same
place, so the two cannot disagree — which is why every reader of where a unit stands
goes through `_stand_after`. Pricing the trip on the consumer instead and shifting the
whole walk by a count leaves turn 0 idle for a pickup the worker never makes.
