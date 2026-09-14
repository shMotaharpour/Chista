# tile_dp — Daily State-Action Graphs (single_tile_daily_state_action)

Per-tile **daily** state-action graphs. State = the tile at day start
(hour 0); edge = one daily action chain executed by the workers, then
idle to the next day start.

## Status

| entity | kind | lifecycle | states | edges | graph npz |
|---|---|---|---|---|---|
| CARROT | one-shot crop | 6 days | 22 | 90 | `graph_CARROT_lifecycle.npz` |
| WHEAT | one-shot crop | 7 days | 34 | 158 | `graph_WHEAT_lifecycle.npz` |
| TOMATO | ongoing crop | 14 days | 99 | 567 | `graph_TOMATO_lifecycle.npz` |
| STRAWBERRY | ongoing crop | 19 days | 159 | 999 | `graph_STRAWBERRY_lifecycle.npz` |
| MELON | one-shot crop | 15 days | 89 | 368 | `graph_MELON_lifecycle.npz` |
| GOOSE / COW / SHEEP | animals | — | deferred | — | next step |

## State definition

- **Crops** — `(crop, age, consec, fert_left, yield)`:
  - `age`: origin = FIRST HARVEST DAY (age -1 = the day before it);
    age 0..max_yield-first = golden window (water/fert give +2, F005);
    after max_yield the plant decays 1 unit per 2 turns (F008) until weed.
  - `consec`: 0|1 — yesterday watered/dry. Two dry nights = weed (F002),
    so a consec=1 day MUST be watered or the plant dies.
  - `fert_left`: 0..2 remaining fertilizer-covered days (F004).
  - `yield`: units on the tile (0..cap).
- **Animals** (deferred; Hossein's convention noted for the next step):
  - `age` wraps inside the positive production range:
    goose `{0}`, cow `{0,1}`, sheep `{0,1,2}` (interval = 1/2/3);
    pre-yield ages are negative (placement day itself is intra-day).
  - `unfed`: 0|1 (2 = escaped overnight, F017) — the structure REMAINS,
    so a new animal can be placed without DIG (F024: DIG fails on an
    occupied structure).
  - `care_bank`: 0..max_held (goose 4, cow 6, sheep 6) — banked CARE
    nights (F019); CARE banks only on fed days.
  - `yield`: units on the animal (0..cap).

## Action chains (canonical order)

Within one day, ops run in canonical order (Hossein's rule):
- crops: `PLANT → FERTILIZE → WATER → HARVEST` (+ DIG last for clearing)
- animals: `FEED → CARE → HARVEST → COLLECT_FERTILIZER`

Chains = canonical subsets of the state's ops. Engine-verified notes:
- WATER strictly before HARVEST for one-shot crops (the watered unit
  lands immediately — F026).
- Order of FERTILIZE vs WATER within the day does not change the
  outcome ( fert covers the whole day) — canonical keeps it fixed.
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
- No animals yet; wheat+carrot only; preconditions explicit on edges
  but assumed satisfied.
- Selling = harvesting at the secretary's day-price; no warehousing.
