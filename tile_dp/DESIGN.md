# tile_dp — daily state-action graphs 

## Model

- **State** = the tile at **day start** (hour 0): `NONE | WEED | PLANT | ANIMAL | EMPTY_STRUCTURE`
  with `(crop, age, consec, fert_left, yield)` for plants and
  `(animal, age, unfed, care_bank, yield)` for animals
- **Edge** = one daily action chain in canonical order
  `PLANT → FERTILIZE → WATER → HARVEST` (animals:
  `FEED → CARE → HARVEST → COLLECT_FERTILIZER`), executed by the workers, then
  idle to the next day start. The record is
  `from, to, chain, entity, cost[], produce[]` with `cost` / `produce` as two
  **separate** 18-wide int vectors: `cost` = inputs consumed + `LABOR_HOURS`,
  `produce` = harvested units of the tile's standing product + collected
  fertilizer. A resource may appear in both and they are **never netted**. No
  money and no day in the artifact (both are DP inputs / DP dimensions);
  `entity` = the entity that parameterises the chain's constructive op
  (`PLANT` / `BUILD` / `PLACE`), else the state's own, 0 = none.
- **DIG layering**: a chain may **start** with `DIG` or put `DIG` right
  after `HARVEST`, and may then run one full `NONE` chain, so a single day can
  convert the tile to another kind. One `DIG` per chain; `DIG` never touches an
  occupied animal tile (the engine returns early there); the follow-up must not
  rebuild the kind the tile had before `DIG` (COOP ↔ PASTURE is the legal
  exception, the graph drops the identical-structure case).
- **Cost model** (`chain_requirements` is the single source): `LABOR_HOURS` =
  number of ops in the `WORKER_OPS` allow-list (market buys and `PICKUP` cost 0
  hours), inputs per op: `PLANT` 1 seed of the entity's crop, `FERTILIZE` 1
  fertilizer, `FEED` 1 wheat, `PLACE` / `PLACE_ANIMAL` 1 animal of the entity's
  species. **`LABOR_HOURS` is a floor, not the whole day**: the executor
  actually spends `edge_steps` engine steps per edge (shipped beside
  `edge_cost` — PLANT 2, FERTILIZE / FEED / PLACE / PLACE_ANIMAL 3, the rest 1,
  from `OP_STEPS`), because the secretary layer does not exist yet and the
  purchases are realised inline. In isolation `LABOR_HOURS` under-charges a
  chain exactly as `edge_steps` over-charges it (a PICKUP can carry several
  units; one PASS carries up to ten market orders — F031); the truth depends
  on how the secretary batches (#14). The master's labour row (#12) gets the
  bracket, not a guess.
- **Resource vocabulary**: 18 names, exactly one id per physical item —
  `LABOR_HOURS, FERTILIZER, WHEAT, SEED_WHEAT, SEED_CARROT, SEED_TOMATO,
  SEED_STRAWBERRY, SEED_MELON, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK,
  WOOL, ANIMAL_GOOSE, ANIMAL_COW, ANIMAL_SHEEP`. `WHEAT_FOOD` no longer exists
  (`WHEAT` is the engine's own product id) and the generic `ANIMAL` is gone:
  every op names its species. `NO_ACT` is a whole-chain op only.
- **Applicability filters** (`chains_for`; the CARE rule added 2026-09-14): `CARE`
  requires `FEED` in the same chain (no-op otherwise, and not in the registry),
  `FERTILIZE` dropped for `age < -2`
  (the fertilize effect covers the day plus two), `HARVEST` dropped for
  `age < 0` or `yield_units == 0`.
- **Animals age origin** = first yield day (Hossein's convention): negative age
  `1 - first_yield_day .. -1` while growing up (the placement day is intra-day and
  never a state); from the first yield day on the age is the **cycle phase**
  `0 .. interval-1` and it wraps (goose `{0}`, cow `{0,1}`, sheep `{0,1,2}`) —
  an animal has no lifespan, so the calendar age is not a decision variable.
- **Crops age origin** (Hossein's convention): one-shot = first golden window day
  `SGW = (max_yield_day + 1) // 2`; ongoing = `max_yield_day`. `age -1` is the day
  before the origin. The planned range ends on the last day that still carries a
  decision: one-shot `1 - SGW .. max_yield_day - SGW`, ongoing
  `1 - max_yield_day .. (max_yield - 1) * interval`. The next day (the engine's
  `max_lifespan_step` day — the day the plant *starts* turning into a weed) decodes
  as **WEED**.

## Engine-derived rules baked in

- A plant **not watered on its planting day is weed by the next day**
  (F002) → the planting chain always waters the planting day; the
  first plant day-start state is age - SGW with consec=0.
- `consec` and `fert_left` are **independent** (probed): `(consec=1, fert_left=2)`
  is reachable, so there is no cap of fert_left by consec.
- `care_bank` is **capped at `max_held`** by contract (Hossein): the engine can
  bank more than `max_held` before the first yield day, the model caps the state.
  Probe (2026-09-14): a COW fed+cared nightly banks 7 before its day-8
  production, but the first production consumes
  `min(max_held, yield + 1 + bank)`, so **any bank ≥ `max_held` − 1 is
  decision-equivalent**; the bank also burns on an unfed production night
  (engine zeroes it, kaggriculture.py:826-828). The cap can therefore be
  tightened to `max_held − 1` (one state folded per animal/age/unfed/yield
  combination, a straight DP saving) without losing a decision - on record
  here, not applied mid-PR to keep the state-space contract stable.
- **Dominance is componentwise (contract 2026-09-14)**: two edges to the same
  state prune each other only when one needs no more of EVERY resource and
  produces no less of EVERY resource. The vectors are never netted and no scalar
  sum may decide: 1 wheat against 1 melon, a carrot seed against a wheat seed,
  and one collected fertilizer against nothing all keep both edges. This is what
  keeps the 1286 `COLLECT_FERTILIZER` edges in the model.
- `FERTILIZE→HARVEST` (no water) is **dominated** by bare HARVEST:
  same production, wasted fertilizer (engine-probed) → pruned by the
  generic dominance filter.
- Nightly auto-drop returns unit inventory to the shed → a chain that
  fertilizes re-PICKUPs the fertilizer first (the secretary layer's job).
- Market purchases land one turn before the op that needs them (F030).

## Numbers (engine-verified, build_graph)

- The shipped model is ONE **merged** tile graph (`build_graph()`, engine
  identity tag `tile-dp/reg=55fb92f019f166a9+eng=a2278746+tpd=24+pb=30`):
  685 states / 18217 edges, 42 KB, tracked as
  `tile_dp/models/graph_tile_lifecycle.npz` with
  `tile_dp/models/build_report.json`; rebuild both with
  `.venv/bin/python -m tile_dp.build`. Only day-start states the engine
  actually produces
  become nodes (interned from real execution): PLANT 402, ANIMAL 279,
  EMPTY_COOP 1, EMPTY_PASTURE 1, NONE 1, WEED 1.
- `build_graph(entity=...)` builds the same search restricted to one entity's
  chains: the per-entity views the tests use, not artifacts.
- Each node is expanded with the sim that produced it (**sim inheritance**);
  only the root is a fresh `FastSim` (30-day episode, seed 4242,
  `weedSpawnChance = 0.0`). No replays exist any more.
- Every edge is asserted: it must end in one day (`ChainSpansDays`) and land on
  the state it promises (`StateMismatch`).
- No zero-cost self-loops; dominance-pruned componentwise (FERTILIZE→HARVEST
  absent, while the 1286 fertilizer-collecting edges survive).
- The chain registry: 54 chains (DIG-layered; 30 carry DIG), 18 resource names.

## Files

| file | role |
|---|---|
| `tile_state.py` | TileState, day-start decode from the engine tile |
| `chains.py` | chain registry (per kind, DIG-layered), resource vocabulary, cost/produce vectors, applicability |
| `graph.py` | `build_graph()` — sim-inherited engine edges, assertions, pruning; `TileGraph` / `Edge` |
| `build.py` | `python -m tile_dp.build` — writes the model + `build_report.json` into `models/` |
| `models/` | `graph_tile_lifecycle.npz` (the merged tile graph) + `build_report.json` (counts) |
| `contractor.py` | the DP over this graph: one backward sweep prices the whole board, plans are recovered forward — `price_board()` (issue #11) |

## The DP over this graph (issue #11, `contractor.py`)

The graph is **day-invariant and position-invariant**, so one backward sweep
prices every tile on the board at once — there is no per-tile solve and no
per-day rebuild, and a loop over tiles in the sweep means the property the
architecture rests on has been lost:

```
V_30(s) ≡ 0                                     # F029: no liquidation
V_d(s)  = max over e out of s of  produce(e)·p_d − cost(e)·w_d + V_{d+1}(next(e))
```

Three array ops per day over the shipped CSR (`matvec`, `add`, segmented max),
`V` kept as `(31, 685) float32` ≈ 80 KB, no argmax table: plans are recovered
forward for the tiles actually owned, per-day coefficients included (labour,
inputs, produce) so the master (#12) can couple on them.

Two contracts travel with the graph and are the DP's, not the caller's:

- **R006 — non-negativity.** Dominance pruning is optimality-preserving only
  while `p ≥ 0` and `w ≥ 0` componentwise; a negative component makes a pruned
  edge the true optimum, so the contractor asserts on entry, every call, and the
  master clamps its duals. See `docs/R006_non-negative-prices-and-wages.md`.
- **No empty edge slices.** `np.maximum.reduceat` returns the element *at* the
  index for an empty group — a plausible wrong value rather than an error — so
  the shipped graph asserts every state owns at least one out-edge.

Measured on the dev box (issue #11 acceptance: sweep ≤ 15 ms, 100 recoveries
≤ 5 ms): **sweep 5.8-6.2 ms, 100 tile recoveries 2.1-3.1 ms**, and in
`bench_turn_budget` **solo 5.63 ms / contended 9.33 ms (ratio 1.66)**. Two
measurements that changed the code rather than decorating it:

- The per-tile recovery loop the issue describes costs **20.8 ms for 100 tiles**
  against the 5 ms ceiling — every ~25-element slice pays full numpy dispatch
  cost — so the tiles walk in lockstep instead.
- Folding produce and cost into one `[EP | −EC] @ [p ; w]` matvec looks like
  fewer passes and measured **85-126 ms a sweep** against 8.5-10.4 ms for the
  issue's two-matvec form: the wider `gemv` loses to BLAS thread dispatch.

`agent/replan.py` wires it into the runtime as the hour-0 rung (the pricing
oracle in place): duals and routing are stood in for by the engine's own quotes
(#12) and by "each unit works the tile it stands on" (#14), so the rung is
opt-in behind `CHISTA_REPLAN=1` until the secretary can carry its inputs.
