# tile_dp — daily state-action graphs 

## Model

- **State** = the tile at **day start** (hour 0): `NONE | WEED | PLANT | ANIMAL | EMPTY_STRUCTURE`
  with `(crop, age, consec, fert_left, yield)` for plants and
  `(animal, age, unfed, care_bank, yield)` for animals
- **Edge** = one daily action chain in canonical order
  `PLANT → FERTILIZE → WATER → HARVEST` (+ DIG for clearing), executed
  by the workers, then idle 24 turns to the next day start.
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
- The graph builder interns states from real execution: only states the
  engine actually produces become nodes (≈45 nodes, 100+ edges —
  authoritative count is `build_graph()`'s output).
- No zero-cost self-loops; dominance-pruned (FERTILIZE→HARVEST absent).

## Files

| file | role |
|---|---|
| `tile_state.py` | TileState, day-start decode from the engine tile |
| `chains.py` | chain registry, canonical order, per-state applicability |
| `graph.py` | `build_graph()` — engine-driven edges + pruning; `TileGraph` |
| — | (next step: the DP over this graph, not written yet) |
