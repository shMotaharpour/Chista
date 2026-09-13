# tile_dp — carrot daily state-action graph

Branch: `single_tile_daily_state_action` — step 1 per Hossein's spec.

## Model

- **State** = the tile at **day start** (hour 0): `NONE | WEED | PLANT`
  with `(age, consec, fert_left, yield)` for plants.
- **Edge** = one daily action chain in canonical order
  `PLANT → FERTILIZE → WATER → HARVEST` (+ DIG for clearing), executed
  by the workers, then idle 24 turns to the next day start.
- **Age origin** = first harvest day (Hossein's convention): age -1 =
  the day before the first harvest; age 2 = the last living day for
  carrot (weed from age 3 — never a day-start state).

## Engine-derived rules baked in

- A plant **not watered on its planting day is weed by the next day**
  (F002) → the canonical replay always waters the planting day; the
  first plant day-start state is age -1 with consec=0.
- `consec=1` (yesterday dry) **caps fert_left at 1**: the fertilizer
  day must be a watered day, and yesterday was dry (engine-probed).
- `FERTILIZE→HARVEST` (no water) is **dominated** by bare HARVEST:
  same production, wasted fertilizer (engine-probed) → pruned by the
  generic dominance filter.
- Nightly auto-drop returns unit inventory to the shed → the fert day
  re-PICKUPs before FERTILIZE in the canonical replay.
- Market purchases land one turn before the op that needs them (F030).

## Numbers (engine-verified, build_graph)

- 57 reachable day-start plant states (+ NONE; WEED is unreachable as a
  day-start state in v1 — dry-night deaths decode to NONE, weed-spawn
  RNG ignored).
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
| `CARROT_GRAPH.md` | full SC/AC listing + Mermaid diagram |
