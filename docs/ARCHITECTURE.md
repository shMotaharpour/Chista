# Chista — the system's shape, its one vocabulary, and its one belief

**Status: revision 1 (2026-09-17).** This is the single place that defines what the
parts of this agent are, what they may read, and what the words mean. Where it and
a module disagree, this document and the engine (`kaggriculture.py`) are the
authority and the module is the bug.

It exists because three vocabularies grew for one world — the engine's action
strings, the tile DP's chain ops (`tile_dp/chains.py`), and the WSR secretary's
enums (`secretary/models.py`) — and every consumer paid for the divergence: the
DP's `MARKET_OPS` was incomplete, `PLACE` existed twice, the secretary's
`PRODUCT_ITEMS` lacked every crop, and its `expand_major_task` and the graph's
`_exec_chain` realised the same chain in two different ways. They are reconciled
here, by making the engine the vocabulary and the **day compiler the only
expansion**.

---

## 1. The canonical vocabulary — `world/model.py`

One module, engine-derived, imported by every layer (R002: never transcribed):

| name | what it is | source |
|---|---|---|
| `GOODS` | the 9 market goods | `K.PRODUCTS` |
| `CROPS` / `ANIMALS` | the 5 crops, the 3 species | `K.CROPS`, `K.ANIMALS` |
| `RESOURCES` | the DP's 18 columns: `LABOR_HOURS`, `FERTILIZER`, `WHEAT`, `SEED_<crop>`, the 4 crop products, `EGG/MILK/WOOL`, `ANIMAL_<species>` | derived from the three above |
| `ACTIONS` | **the engine's action strings** — `PASS`, `NORTH/SOUTH/EAST/WEST`, `PICKUP`, `DROP`, `PLACE`, `PLANT`, `WATER`, `HARVEST`, `FERTILIZE`, `DIG`, `FEED`, `CARE`, `COLLECT_FERTILIZER`, `BUILD_COOP`, `BUILD_PASTURE`, `SELL`, `BUY_SEED`, `BUY_PRODUCT`, `BUY_ANIMAL`, `HIRE`, `BUY_LAND` | `K.FARMER_MOVES` and the handler branches in `kaggriculture.py` |
| `CHAIN_OPS` | the DP's **abstract** chain vocabulary — `PLANT`, `WATER`, `FERTILIZE`, `HARVEST`, `DIG`, `BUILD`, `PLACE`, `FEED`, `CARE`, `COLLECT_FERTILIZER`, `NO_ACT`, and the market ops `BUY_SEED/BUY_PRODUCT/BUY_ANIMAL` | `tile_dp/chains.py` (the registry), asserted to be a subset of the union below |
| `COMPILE` | the **one** abstract→engine table: what each chain op means as engine actions, with its prerequisites | `secretary/routing.py::op_turns` + `carried_item` (they become `world/model.py::compile_chain`) |

Three rules follow, and they are the whole point of the module:

1. **Names are the engine's.** A layer that needs a good, a crop, a species or an
   action string imports it; nothing types its own.
2. **The DP speaks abstract ops, the engine speaks actions, and one function
   translates.** `BUILD` → `BUILD_COOP`/`BUILD_PASTURE` (from the entity),
   `PLACE`/`PLACE_ANIMAL` → `PLACE <animal>`, `NO_ACT` → `PASS`. That translation
   is `compile_chain`; there is no second one.
3. **A market op is never a worker op.** Purchases and sales are the market's
   actions (`plan["market"]`); the worker's day is worker actions only. The chain
   vocabulary mentions them because the DP prices them, not because a unit runs
   them.

### The legacy vocabularies and their fate

| today | becomes |
|---|---|
| `secretary/models.py::Item` (lower-case enum: `wheat`, `cow`, …) | **deleted**; `GOODS`/`CROPS`/`ANIMALS` + `RESOURCES` are the names, and the WSR's `Item` was a third spelling of the same objects |
| `secretary/models.py::MinorActionType`, `MOVEMENT_ACTIONS`, `PRODUCE_ACTIONS`, `CONSUME_ACTIONS` | **deleted**; `ACTIONS`, and the sets are recomputed from it (`movement = K.FARMER_MOVES`, `produce = {PICKUP, HARVEST, COLLECT_FERTILIZER}`, `consume = {PLACE, FEED, FERTILIZE}` — all named in the engine) |
| `secretary/models.py::PRODUCT_ITEMS = {milk, wool, egg}` | **wrong**: the sellable set is `GOODS` (9, including every crop and fertiliser, `:596`). A confidence bug, not a naming one |
| `tile_dp/chains.py::WORKER_OPS` (has `PLACE_ANIMAL`, lacks nothing else) | kept as a **view** of `CHAIN_OPS`, computed from `COMPILE` rather than written by hand |
| `tile_dp/chains.py::MARKET_OPS` | kept, and asserted equal to `{BUY_SEED, BUY_PRODUCT, BUY_ANIMAL}` from `COMPILE` — its "incomplete" reading came from `BUY_*` being absent from `WORKER_OPS`, which is correct and now documented in one place |

---

## 2. The DP and the WSR are the same object, seen twice

The tile DP prices **one day of one tile** as a chain of abstract ops. The WSR
secretary schedules **one day of one unit** as a sequence of minor tasks. Those are
the same thing, and the divergence was that each side expanded it itself:

- `tile_dp/graph.py::_exec_chain` (build time): buys, pickups and ops realised on a
  scratch sim where the worker already stands on the tile — so it can price a chain.
- `secretary/models.py::expand_major_task` (WSR): its own expansion into
  `MinorTask`s, with its own `_SINGLE_WORKER_TYPES = {"feed", "frz", ...}` stringly
  types.

**Resolution: a WSR major task *is* a DP chain, one-to-one, and there is exactly
one expansion.**

| concept | single definition |
|---|---|
| a day's work on a tile | a chain: `tuple` of `CHAIN_OPS` + the entity it names, interned by `tile_dp/chains.py` (`chain_id_of`) |
| a WSR major task | the same tuple — `MajorTask(chain=…, entity=…)`, no second registry, no `MajorTaskType` enum (the chain id IS the type) |
| the expansion into engine actions | `world/model.py::compile_chain(chain, entity, unit_state) -> day` — the compiler's worker-op turn list plus the prerequisites it must satisfy (a seed buy before a `PLANT`, a shed trip before a `FERTILIZE`/`FEED`/`PLACE`, a drop after a `HARVEST`) |
| what the secretary still owns | **routing**: where each unit must stand, when it walks, when it picks up and drops, which hour each market order lands on, and the realised hours it reports back (contract 3: travel is not in the graph) |
| what the secretary loses | its own `Item`, `MinorActionType`, `CellType`, `expand_major_task` and the minor-task model; `MajorTask` becomes a thin value type over a chain |

The 6-vs-7 correspondence between the old major-task types and the chain registry
stops being a thing to maintain: the registry is the type space (54 chains today),
the compiler is the expansion, and the guard is a test that **every chain in the
registry compiles to engine actions the engine accepts**, and that every chain the
DP can choose has a route the compiler can build inside a day.

---

## 3. One belief for the whole system

`belief/` is the system's only reader of the market and the rival. Every other
module asks it; none of them re-derives.

```python
MarketState = {
  step, day, hour,
  inventory: (9,) int,          # the market's own stock, read from the observation
  prices:    (9,) int,          # K.market_price at that inventory (parity-tested)
  drain:     {mean: (9,), sd: (9,), horizon: int},   # closed form, no sampling
  rival:     {sales: (9,) per turn, shed_estimate: (9,), slot_order: inferred},
  shed:      {room: int, guard_margin: int},
}
```

- **Who may read the market directly**: `belief/` only. `secretary/market.py`'s
  `forecast()` and `secretary/opponent.py` become *views* of `MarketState` (the
  forecast is `prices` + `drain` over the horizon; the rival model is
  `rival.sales`), or they are deleted where they duplicate it.
- **Who consumes it**: the master (`planner/master.py`) prices revenue at
  `prices` and internal scarcity at its own duals; the compiler schedules sells at
  `drain`; the runtime publishes timing.
- **What the belief does not own**: movement, hours, land, animal chains, the
  order book's optimiser. It answers "what is true about the market and the rival",
  and nothing else.
- **Provenance rule (R005)**: every field of `MarketState` names its source — the
  observation, the engine's own function, or a measurement in `bench/`. A field
  with no source is a bug, not a heuristic.

---

## 4. Ownership, in one table

| layer | owns | must not |
|---|---|---|
| `world/` (`model.py`, `fast_sim.py`, `replay_agent.py`) | the vocabulary, the engine binding, the configuration object | hold policy |
| `belief/` | the market and rival beliefs, their schemas | decide actions, read the board |
| `tile_dp/` | the chain registry and the per-tile DP | know about units, travel or the market's queue |
| `planner/` | the season: what to grow, when to sell, hire, buy, and expand | emit engine actions |
| `secretary/routing.py` | the day: assignment, routing, carries, drops, the market queue | re-price anything the master or the belief already priced |
| `agent/` | the spine: decode, dispatch, deadline, fallback ladder | plan |

The message types between layers are `belief/schemas.py` (`MarketState`,
`SellIntent`, `PurchaseIntent`, `TileRequirement`, `CarryRequirement`,
`DropRequirement`, `DaySchedule`, `OrderBook`); a message says what must be true
and by when, and the receiver owns how.

---

## 5. Migration, in order, each step guarded

1. **`world/model.py`** — the canonical names and the `COMPILE` table; a test that
   every name equals its engine source and that every chain op compiles to engine
   actions the engine accepts. *(This lands first; nothing else has to change.)*
2. **`secretary/routing.py::op_turns` / `carried_item` move into `COMPILE`** — one
   expansion, imported by the compiler (and by the build-time executor's docstring
   as the reference).
3. **`secretary/models.py` loses `Item`, `MinorActionType`, `CellType`,
   `MOVEMENT_ACTIONS`, `PRODUCE_ACTIONS`, `CONSUME_ACTIONS`** and keeps only the
   scheduling value types; `MajorTask` is written over a chain tuple. `Item` is
   deleted with a one-line shim only if a solver still needs the old spelling.
4. **`secretary/market.py::forecast` becomes a view of `MarketState`** (or is
   deleted where `belief/` already answers it) and `secretary/opponent.py` likewise.
   Guard: the ported market tests must pass against the belief-backed forecast with
   the same numbers (the forecast's own error table is the acceptance).
5. **`tile_dp/chains.py`'s `WORKER_OPS`/`MARKET_OPS` become computed views** of
   `CHAIN_OPS`/`COMPILE`, and `PLACE_ANIMAL` is retired to `PLACE` in the chain
   vocabulary with its legacy alias kept only for the shipped graph's on-disk
   chains (the `.npz` stores chain ids, so the registry keeps its order).

After each step: the full suite, and no new name may be introduced in any module.

---

## 6. Open questions for the owner

1. **Where does `secretary/` end up?** Today it holds the market layer, the shed
   guard, the solvers and the WSR models. Under this document: routing and the
   market queue are `secretary/routing.py`, the market views and the shed guard go
   to `belief/` consumers, and the CP-SAT/OXA solvers are offline helpers. Confirm
   the split — or keep `secretary/` as the name for "everything the day plan needs"
   and put the market views there.
2. **Human-readable enums**: the canonical vocabulary is engine strings (UPPER,
   `SEED_WHEAT`, `ANIMAL_COW`). Do you want `world/model.py` to expose `Enum`
   classes (`Good`, `Crop`, `Species`, `Action`) whose `.value` is the engine
   string, so call sites read `Action.PLANT` — with the guarantee that the enum is
   built *from* the engine list, never typed?
