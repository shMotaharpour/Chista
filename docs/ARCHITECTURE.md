# Chista — the system's shape, its one vocabulary, and its one belief

**Revision 2 (2026-09-17).** The single place that defines what the parts are, what
they may read, and what the words mean. Where this document and a module disagree,
the engine and this document win; the module is the bug.

It exists because three vocabularies grew for one world — the engine's action
strings, the tile DP's chain ops, and the WSR secretary's enums — and every
consumer paid: the DP's market ops looked incomplete, `PLACE` existed twice, the
secretary's `PRODUCT_ITEMS` lacked every crop, and one chain was expanded twice
(`tile_dp/graph.py::_exec_chain` at build time, `secretary/models.py::expand_major_task`
at runtime).

---

## 1. The canonical vocabulary — `world/model.py`

Engine-derived (R002), imported by every layer:

| name | what it is | count |
|---|---|---|
| `GOODS` / `PRODUCTS` | what the market trades and the farm sells | 9 |
| `CROPS` / `ANIMALS` | the five crops, the three species | 5 / 3 |
| `RESOURCES` | what the farm buys or consumes: labour, fertiliser, wheat, the five seeds, the three animals | 11 |
| `DUAL` | wheat and fertiliser — a resource **and** a product | 2 |
| `VECTOR` | the DP graph's columns = resources + products, in the stored order | 18 |
| `ACTIONS` | every action string the engine's handlers accept, **extracted from their source** | 24 |
| `WORKER_OPS` | what a unit spends a turn on | 10 |
| `MARKET_ACTIONS` | the market's own vocabulary: `SELL`, `BUY_SEED/PRODUCT/ANIMAL`, `HIRE`, `BUY_LAND` | 6 |
| `CHAIN_OPS` | what a DP chain may name: `WORKER_OPS` + the three `BUY_*` it needs + `NO_ACT` | 14 |

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

Legacy, to be deleted (see §5): `secretary/models.py`'s `Item`, `MinorActionType`,
`CellType`, `expand_major_task`; the DP's `PLACE_ANIMAL` alias.

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

So the conversion is: **one chain → one tile-day, wrapped by the compiler into one
unit-day**. The compiler (`secretary/routing.py`) owns travel, shed trips, pickups,
drops, hour assignment and the market queue; the DP owns which chain runs on which
tile; the WSR's value types are the scheduling half of the same day, and their
expansion is `compile_chain`.

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

- **Only `belief/` reads the market.** `secretary/market.py::forecast` and
  `secretary/opponent.py` become views of `MarketState`, or are deleted where they
  duplicate it.
- **Consumers**: the master prices revenue at `prices` and internal scarcity at its
  own duals; the compiler schedules sells at `drain`; the runtime publishes timing.
- **Not owned**: movement, hours, land, animal chains, the order book's optimiser.
- **R004/R005**: every field names its source — the observation, an engine
  function, or a measurement in `bench/`.

---

## 4. Ownership

| layer | owns | must not |
|---|---|---|
| `world/` | the vocabulary (`model.py`), the town (`vocabulary.py`), the engine binding, config | hold policy |
| `belief/` | the market and rival beliefs, the schemas | decide actions, read the board |
| `tile_dp/` | the chain registry, the per-tile DP | know units, travel, or the market queue |
| `planner/` | the season: what to grow, when to sell, hire, buy, expand | emit engine actions |
| `secretary/` | the day: routing, carries, drops, the market queue, the shed guard; the WSR schedulers (`oxa_solver` — **our** algorithm, copied into this repo and running inside the turn) | re-price what the master or the belief priced; touch the market |
| `agent/` | the spine: decode, dispatch, deadline, fallback | plan |

---

## 5. Migration — one branch, then one PR

All world changes land on **one branch** (`world/definition`); when the world is
right, it goes to `main` in one PR. No piecemeal merges. Steps:

1. `world/model.py` — names, the resource/product split, the market/worker split,
   the compile table. Guards: `tests/test_model.py`.
2. The compiler imports `compile_chain`; the duplicated expansions are deleted.
   *(Done on this branch: `agent/replan.py` lost `chain_turns`/`project_day`.)*
3. The WSR's enums and `expand_major_task` are deleted; `MajorTask` is written over
   a chain tuple.
4. `secretary/market.py::forecast` and `secretary/opponent.py` become views of
   `MarketState`, or are deleted where `belief/` already answers them. The ported
   market tests are the acceptance.
5. The DP's op sets become computed views of `world/model.py`; `PLACE_ANIMAL` is
   retired with the graph rebuild.

Each step ends with the full suite green, and the name ratchet in `tests/test_model.py`
counts what is left to move.
