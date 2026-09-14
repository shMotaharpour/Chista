# tile_dp — Daily State-Action Graphs (single_tile_daily_state_action)

Per-tile **daily** state-action graphs. State = the tile at day start
(hour 0); edge = one daily action chain executed by the workers, then
idle to the next day start.

## Status

| entity | kind | planned days (age range) | states | edges | graph npz |
|---|---|---|---|---|---|
| CARROT | one-shot crop | 3 (−1..1) | 13 | 50 | `graph_CARROT_lifecycle.npz` |
| WHEAT | one-shot crop | 4 (−1..2) | 26 | 109 | `graph_WHEAT_lifecycle.npz` |
| TOMATO | ongoing crop | 11 (−7..3) | 91 | 441 | `graph_TOMATO_lifecycle.npz` |
| STRAWBERRY | ongoing crop | 16 (−9..6) | 152 | 827 | `graph_STRAWBERRY_lifecycle.npz` |
| MELON | one-shot crop | 12 (−5..6) | 94 | 503 | `graph_MELON_lifecycle.npz` |
| GOOSE | animal | 4 ages (−3..0), cycle 1 | 15 | 61 | `graph_GOOSE_lifecycle.npz` |
| COW | animal | 9 ages (−7..1), cycle 2 | 32 | 126 | `graph_COW_lifecycle.npz` |
| SHEEP | animal | 8 ages (−5..2), cycle 3 | 33 | 150 | `graph_SHEEP_lifecycle.npz` |

Counts are `build_graph()` output with `ENGINE_TAG = tile-dp-v13`
(`artifacts/tile_dp/build_report.json`). Animal graphs now carry one
`EMPTY_STRUCTURE` node each (the `BUILD` chain, v13).
The chain registry has since moved to **v14** (DIG layering), so these numbers
are still the v13 build: the graphs must be rebuilt (edge counts will grow) in
the graph's turn.

## Known issue — canonical replay vs node label

`build_graph()` computes a node's outgoing edges by replaying that node with a
canonical chain. That replay does not always reproduce the node itself, so the
edges are computed from a different day-start state (a node that cannot be
reproduced is unreachable and should be dropped).

v13 progress: the crop replay now skips WATER on the node's own dry streak
(`consec`), so dry crop nodes replay correctly. Animal nodes are still open
(the replay feeds + cares every day, so `unfed > 0` and `bank` labels are not
reproduced). Measured with `scripts/replay_mismatch.py`:

v12 baseline (mismatch / states): CARROT 6/13, WHEAT 13/23, TOMATO 58/89,
STRAWBERRY 103/149, MELON 49/83, GOOSE 10/14, COW 22/31, SHEEP 24/32.

v13 (after the dry-streak fix): TOTAL 186/456 — WHEAT 8/26, CARROT 1/13,
TOMATO 31/91, STRAWBERRY 62/152, MELON 28/94, GOOSE 10/15, COW 22/32,
SHEEP 24/33. Remaining classes: (a) `WEED` nodes replay as `NONE` (no WEED
replay exists), (b) crop `y` is off by one on some fert histories, (c) animal
`unfed`/`bank` labels (the replay feeds and cares every night).

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

## Action chains (v14)

Within one day, ops run in canonical order (Hossein's rule):
- crops: `PLANT → FERTILIZE → WATER → HARVEST` (+ DIG last for clearing)
- animals: `FEED → CARE → HARVEST → COLLECT_FERTILIZER`

A chain is the set of WORKER ops applied on one tile in one day. `NO_ACT` = the
worker does nothing on this tile (0 hours); the day still passes and the state
advances. The engine's `PASS` (1 hour) is never used in chain definitions.

Per-kind chains (explicit lists in `chains.py`, before pruning). `NONE` is ONE
state for crops and animals; the two selections are what the graphs pick from:

| kind | chains | what |
|---|---|---|
| `NONE` crop | 2 | `NO_ACT`, `PLANT+WATER` (F002) |
| `NONE` animal | 5 | `NO_ACT`, `BUILD`, `BUILD+PLACE`, `BUILD+PLACE+FEED`, `BUILD+PLACE+FEED+CARE` |
| `WEED` | 7 | `NO_ACT`, `DIG` + the DIG layering |
| `PLANT` mature | 33 | subsets of {FERTILIZE, WATER, HARVEST} + the layering |
| `PLANT` young | 9 | the same without HARVEST (F026) |
| `ANIMAL` | 16 | subsets of {FEED, CARE, HARVEST, COLLECT_FERTILIZER}, no DIG |
| `EMPTY_STRUCTURE` | 8 | `NO_ACT`, `PLACE_ANIMAL` + the layering |

`BUILD` is a single worker action that turns `NONE` into a structure, so
`EMPTY_STRUCTURE` is a real state again (`NONE → EMPTY_STRUCTURE → ANIMAL`, and
an escape lands back in `EMPTY_STRUCTURE` because the structure remains).
Registry: **54** chains (`CHAIN_NAMES`), 26 of them carry `DIG`. Chain names are
stable (`chain_name` / `ops_of_name`), ids are internal, and the registry is one
flat list — a layered chain may cross domains (a `DIG` follow-up can `BUILD`),
so the graph selects the subset of its own domain.

**DIG layering (2026-09-14, Hossein's rule):** a chain may START with `DIG` or
put `DIG` right after `HARVEST`, and may then run one full `NONE` chain, so a
single day can convert a tile from one kind to another. Rules (`chains.py`:
`_layer` / `_dig_tail`): one DIG per chain; DIG only where the tile is diggable
(never on an occupied animal tile — the engine's `DIG` returns early there); the
follow-up must not rebuild the kind the tile had before DIG, except an empty
structure (COOP ↔ PASTURE is legal; the graph drops the identical-structure
case). The longest chain is 8 hours
(`FERTILIZE+WATER+HARVEST+DIG+BUILD+PLACE+FEED+CARE`) — nothing tops the day.

Cost model (contract, 2026-09-14) — `chain_requirements` is the single source:
- `LABOR_HOURS` = number of ops in the `WORKER_OPS` allow-list. Market buys
  (`BUY_*`) are the market's action and a `PICKUP` is a carry of the secretary
  layer, so both cost 0 hours: `(BUILD, PLACE, FEED)` = 3, `(PICKUP,)` = 0.
- Inputs per op, summed by the chain: `PLANT` 1 seed of the entity's crop,
  `FERTILIZE` 1 fertilizer, `FEED` 1 wheat, and `PLACE` / `PLACE_ANIMAL` 1
  animal of the entity's species (`ANIMAL_GOOSE` / `ANIMAL_COW` /
  `ANIMAL_SHEEP`).
- Money is NOT in the state (out of scope by design); the shed is not modelled.
- **One chain = exactly one day**: after the ops the sim is stepped until the
  day rolls over. A chain that would span more than one day raises
  `ChainSpansDays`, and an op the engine silently refuses raises
  `ChainNotRealised` — both are loud, never dropped.

Applicability filters in `chains_for` (2026-09-14): `FERTILIZE` is dropped for
`age < -2` (the engine's fertilize effect covers the day itself plus two, so an
earlier dose cannot reach the window) and `HARVEST` for `age < 0` or
`yield_units == 0`. These filters need `age` (for ANIMAL too) and `yield_units`
from the caller, which `graph.py` does not pass yet — pending, graph's turn.

**Secretary gap (open):** the inputs (seed / fertilizer / wheat / animal) have
to be bought and carried by a secretary layer that does NOT exist yet; until
then the graph executor realises the purchase inline as a stand-in.

Engine-verified notes:
- WATER strictly before HARVEST for one-shot crops (the watered unit
  lands immediately — F026).
- Order of FERTILIZE vs WATER within the day does not change the
  outcome for ongoing corps ( fert covers the whole day) — canonical keeps it fixed.
- Fertilizer must be in the UNIT's bag for FERTILIZE (F004) and the
  nightly auto-drop returns it to the shed → each fert day re-PICKUPs.
- Market purchases land one turn BEFORE the unit op that needs them
  (F030) — the BUY turn itself costs no labor.

## Pruning

1. **No-op sweep**: a chain that changed nothing and consumed nothing
   is dropped (F047 silent no-ops).
2. **Dominance**: for edges with the SAME next state, e1 dominates e2
   iff prod(e1) ≥ prod(e2) AND every resource use(e1) ≤ use(e2) (with
   at least one strict). Dominated edges are removed — the DP never
   needs them.

## v1 simplifications (documented)

- Weed-spawn RNG ignored (0.005/tile/day, single tile).
- No precondition checks on edges: an op that the engine refuses is only caught
  by `ChainNotRealised` for the plant / place / build chains.
- Selling = harvesting at the secretary's day-price; no warehousing.
