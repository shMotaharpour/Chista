# tile_dp — Daily State-Action Graphs (single_tile_daily_state_action)

Per-tile **daily** state-action graphs. State = the tile at day start
(hour 0); edge = one daily action chain executed by the workers, then
idle to the next day start.

## Status

| entity | kind | planned days (age range) | states | edges | graph npz |
|---|---|---|---|---|---|
| CARROT | one-shot crop | 3 (−1..1) | 13 | 54 | `graph_CARROT_lifecycle.npz` |
| WHEAT | one-shot crop | 4 (−1..2) | 23 | 106 | `graph_WHEAT_lifecycle.npz` |
| TOMATO | ongoing crop | 11 (−7..3) | 89 | 482 | `graph_TOMATO_lifecycle.npz` |
| STRAWBERRY | ongoing crop | 16 (−9..6) | 149 | 914 | `graph_STRAWBERRY_lifecycle.npz` |
| MELON | one-shot crop | 12 (−5..6) | 83 | 474 | `graph_MELON_lifecycle.npz` |
| GOOSE | animal | 4 ages (−3..0), cycle 1 | 14 | 60 | `graph_GOOSE_lifecycle.npz` |
| COW | animal | 9 ages (−7..1), cycle 2 | 31 | 125 | `graph_COW_lifecycle.npz` |
| SHEEP | animal | 8 ages (−5..2), cycle 3 | 32 | 149 | `graph_SHEEP_lifecycle.npz` |

Counts are `build_graph()` output with `ENGINE_TAG = tile-dp-v12`
(`artifacts/tile_dp/build_report.json`).

## Known issue — canonical replay vs node label

`build_graph()` computes a node's outgoing edges by replaying that node with a
canonical chain (water/feed/care every day, fert on one derived day). That
replay does **not** always reproduce the node itself, so such a node's edges are
in fact computed from a different day-start state. Measured with
`scripts/replay_mismatch.py` on v12 (mismatch / states):

CARROT 6/13, WHEAT 13/23, TOMATO 58/89, STRAWBERRY 103/149, MELON 49/83,
GOOSE 10/14, COW 22/31, SHEEP 24/32.

## State definition

- **Crops** — `(crop, age, consec, fert_left, yield)`:
  - `age` origin (Hossein's convention): **one-shot = START OF GOLDEN WINDOW
    (SGW) = (max_yield_day + 1) // 2**, **ongoing = max_yield_day**.
    `age 0` = that day's day-start; `age -1` = the day before it (for a
    one-shot crop the day after planting — the first day-start it exists; for
    an ongoing crop the last day before its first production).
  - planned range (the days that still carry a decision):
    one-shot `1 - SGW .. max_yield_day - SGW` → WHEAT −1..2, CARROT −1..1,
    MELON −5..6; ongoing `1 - max_yield_day .. (max_yield - 1) * interval` →
    TOMATO −7..3, STRAWBERRY −9..6.
  - the day the plant **starts** turning into a weed (the engine's
    `max_lifespan_step` day: `max_yield_day + 1` after planting for one-shot,
    the day after the last production for ongoing) decodes as **WEED** — the
    states end the day before it, because the project never plans on it.
  - `consec`: 0|1 — yesterday watered/dry. Two dry nights = weed (F002),
    so a consec=1 day MUST be watered or the plant dies. `consec` and
    `fert_left` are independent (no cap): `(consec=1, fert_left=2)` is
    reachable (engine-measured).
  - `fert_left`: 0..2 remaining fertilizer-covered days (F004).
  - `yield`: units on the tile (0..cap).
- **Animals** — `(animal, age, unfed, care_bank, yield)`:
  - `age` = position in the production cycle (Hossein's convention): negative
    while growing up, `1 - first_yield_day .. -1` (the placement day itself is
    intra-day, so it is not a day-start state); from the first production on the
    positive age is the phase `0 .. interval - 1` and it **wraps** — goose
    `{0}`, cow `{0,1}`, sheep `{0,1,2}` (interval 1/2/3). The calendar age is
    not a decision variable: the tile has no lifespan.
  - `unfed`: 0|1 (2 = escaped overnight, F017) — the structure REMAINS,
    so a new animal can be placed without DIG (F024: DIG fails on an
    occupied structure) or DIG to make it NONE.
  - `care_bank`: 0..max_held (goose 4, cow 6, sheep 6) — banked CARE
    nights (F019); CARE banks only on fed days. **Cap is a contract**
    (Hossein): the engine can bank more than max_held before the first
    production; the model caps the state at max_held.
  - `yield`: units on the animal (0..max_held).

## Action chains (canonical order)

Within one day, ops run in canonical order (Hossein's rule):
- crops: `PLANT → FERTILIZE → WATER → HARVEST` (+ DIG last for clearing)
- animals: `FEED → CARE → HARVEST → COLLECT_FERTILIZER`

Chains = canonical subsets of the state's ops. Engine-verified notes:
- WATER strictly before HARVEST for one-shot crops (the watered unit
  lands immediately — F026).
- Order of FERTILIZE vs WATER within the day does not change the
  outcome for ongoing corps ( fert covers the whole day) — canonical keeps it fixed.
- Fertilizer must be in the UNIT's bag for FERTILIZE (F004) and the
  nightly auto-drop returns it to the shed → each fert day re-PICKUPs.
- Market purchases land one turn BEFORE the unit op that needs them
  (F030) — the BUY turn itself costs no labor.

## Pruning (engine-driven, no hand tables)

1. **No-op sweep**: a chain that changed nothing and consumed nothing
   is dropped (F047 silent no-ops).
2. **Dominance**: for edges with the SAME next state, e1 dominates e2
   iff prod(e1) ≥ prod(e2) AND every resource use(e1) ≤ use(e2) (with
   at least one strict). Dominated edges are removed — the DP never
   needs them.

## v1 simplifications (documented)

- Weed-spawn RNG ignored (0.005/tile/day, single tile).
- preconditions explicit on edges
  but assumed satisfied.
- Selling = harvesting at the secretary's day-price; no warehousing.
