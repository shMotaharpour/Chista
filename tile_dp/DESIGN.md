# tile_dp — daily state-action graphs 

## Model

- **State** = the tile at **day start** (hour 0): `NONE | WEED | PLANT | COOP | PASTURE | AMIMAL`
  with `(age, consec, fert_left, yield)` for plants and (age, comsec, held, yield) for animals
- **Edge** = one daily action chain in canonical order
  `PLANT → FERTILIZE → WATER → HARVEST` (+ DIG for clearing), executed
  by the workers, then idle 24 turns to the next day start.
- **Animals age origin** = first harvest day (Hossein's convention): age -1 =
  the day before the first harvest;
- **Corps age origin** = first golden windows day (Hossein's convention): age -1 =
  the day before the SGW (Start of Golden Windows);

## Engine-derived rules baked in

- A plant **not watered on its planting day is weed by the next day**
  (F002) → the canonical replay always waters the planting day; the
  first plant day-start state is age - SGW with consec=0.
- `consec=1` (yesterday dry) **caps fert_left at 1**: the fertilizer
  day must be at age > -2.
- `FERTILIZE→HARVEST` (no water) is **dominated** by bare HARVEST:
  same production, wasted fertilizer (engine-probed) → pruned by the
  generic dominance filter.
- Nightly auto-drop returns unit inventory to the shed → the fert day
  re-PICKUPs before FERTILIZE in the canonical replay.
- Market purchases land one turn before the op that needs them (F030).

## Numbers (engine-verified, build_graph)

- 57 reachable day-start plant states (+ NONE.
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
| `contractor.py` | (next step: the DP over this graph) |
