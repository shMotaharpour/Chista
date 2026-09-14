# tile_dp — daily state-action graphs 

## Model

- **State** = the tile at **day start** (hour 0): `NONE | WEED | PLANT | ANIMAL | EMPTY_STRUCTURE`
  with `(crop, age, consec, fert_left, yield)` for plants and
  `(animal, age, unfed, care_bank, yield)` for animals
- **Edge** = one daily action chain in canonical order
  `PLANT → FERTILIZE → WATER → HARVEST` (animals:
  `FEED → CARE → HARVEST → COLLECT_FERTILIZER`), executed by the workers, then
  idle to the next day start.
- **DIG layering (v14)**: a chain may **start** with `DIG` or put `DIG` right
  after `HARVEST`, and may then run one full `NONE` chain, so a single day can
  convert the tile to another kind. One `DIG` per chain; `DIG` never touches an
  occupied animal tile (the engine returns early there); the follow-up must not
  rebuild the kind the tile had before `DIG` (COOP ↔ PASTURE is the legal
  exception, the graph drops the identical-structure case).
- **Cost model** (`chain_requirements` is the single source): `LABOR_HOURS` =
  number of ops in the `WORKER_OPS` allow-list (market buys and `PICKUP` cost 0
  hours), inputs per op: `PLANT` 1 seed of the entity's crop, `FERTILIZE` 1
  fertilizer, `FEED` 1 wheat, `PLACE` / `PLACE_ANIMAL` 1 animal of the entity's
  species.
- **Applicability filters** (`chains_for`): `FERTILIZE` dropped for `age < -2`
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
  (F002) → the canonical replay always waters the planting day; the
  first plant day-start state is age - SGW with consec=0.
- `consec` and `fert_left` are **independent** (probed): `(consec=1, fert_left=2)`
  is reachable, so there is no cap of fert_left by consec.
- `care_bank` is **capped at `max_held`** by contract (Hossein): the engine can
  bank more than `max_held` before the first yield day, the model caps the state.
- `FERTILIZE→HARVEST` (no water) is **dominated** by bare HARVEST:
  same production, wasted fertilizer (engine-probed) → pruned by the
  generic dominance filter.
- Nightly auto-drop returns unit inventory to the shed → the fert day
  re-PICKUPs before FERTILIZE in the canonical replay.
- Market purchases land one turn before the op that needs them (F030).

## Numbers (engine-verified, build_graph)

- reachable day-start states are whatever the engine actually produces; the
  authoritative counts are `build_graph()`'s output (see the status table in
  `README.md`).
- The graph builder interns states from real execution: only states the engine
  actually produces become nodes (the v13 build: 8 graphs, 456 nodes / 2267
  edges; authoritative counts are the table in `README.md`).
- No zero-cost self-loops; dominance-pruned (FERTILIZE→HARVEST absent).
- The chain registry is now **v14** (54 chains, DIG-layered): the stored graphs
  are still the v13 build and must be rebuilt in the graph's turn.

## Files

| file | role |
|---|---|
| `tile_state.py` | TileState, day-start decode from the engine tile |
| `chains.py` | chain registry (per kind, DIG-layered), cost model, applicability |
| `graph.py` | `build_graph()` — engine-driven edges + pruning; `TileGraph` |
| — | (next step: the DP over this graph, not written yet) |
