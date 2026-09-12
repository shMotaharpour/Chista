# tile_dp — TileContractor design (v1: wheat + carrot, plants only)

Scope: the single-tile 30-day optimizer serving the FarmSecretary
(Dantzig-Wolfe master, later). Calibrated against the shipped engine —
per R002/R003 no game rule is transcribed into tables; the graph builder
asks the engine.

## Decisions locked with Hossein (2026-09)

1. **Package:** `tile_dp/`. Tests at repo-level `tests/test_tile_dp.py`
   (beside the package, like every other suite in this repo).
2. **Selling is NOT a DP decision.** The contractor only produces; the
   secretary prices production. Harvesting a unit of `crop` on day `d` is
   valued at `prices[crop][d]` (her per-day shadow price). HARVEST *timing*
   stays a DP decision; warehousing is the secretary's business.
3. **`wage` is a per-day VECTOR of per-resource costs** (not a scalar):
   `wage[resource][day]` — labor-hour, seed, fertilizer, (later animals) —
   each independently priced per day.
4. **`production`** = the per-day, per-crop harvested-units vector implied by
   the plan. **`schedule`** = per-day action-chain IDs.
5. **All integers.** Every quantity in this domain is discrete; int32
   arrays are half the memory of float64 and faster in arithmetic. The
   secretary's prices and wages arrive as ints (money units).
6. **numpy int arrays are the core representation** — measured 7.3x faster
   than a dict-of-tuples DP at v1 scale (880 us vs 6.4 ms per solve) and
   ~10x smaller. Numeric IDs everywhere in the hot path; string names only
   at the module boundary (decode/encode once, zero dict lookups inside
   the DP loops).
7. **RAM-first.** Per-crop graphs are ~1 MB (see sizing); all five plant
   crops together ~4 MB. solve() touches only RAM. The disk cache
   (`artifacts/tile_dp/graph_<CROP>_d30.npz` + sidecar JSON metadata) is a
   cold-start shortcut with an engine-version check — never read inside
   solve().
8. **Typed variables everywhere** — frozen dataclasses / type aliases below
   are the contract.

## Numeric ID registries (the only string <-> int boundary)

```
crop_id:     0=WHEAT, 1=CARROT            (v1; engine order for future crops)
resource_id: 0=LABOR_HOURS, 1=SEED_WHEAT, 2=SEED_CARROT, 3=FERTILIZER
chain_id:    per-species registry (chains.py); stable, documented
state_id:    assigned at graph build; mapping state<->id stored in the
             graph (a TileState is 5 small ints, packed into one int64 key)
```

## Typed contract

```python
CropId = np.int8
ResourceId = np.int8
ChainId = np.int16
StateId = np.int32
DayIndex = int  # 0..29 (python int at the boundary)

# --- inputs from the secretary (all int) ------------------------------------
PriceVector = Mapping[CropId, np.ndarray]   # prices[crop_id] : int32[30]
WageVector = Mapping[ResourceId, np.ndarray]  # wage[res_id] : int32[30]

# --- core arrays (one graph per crop) ---------------------------------------
@dataclass(frozen=True)
class TileGraph:
    crop_id: CropId
    season_days: int                     # 30
    n_states: int
    # edges grouped by (day, from_state); CSR-style offsets into the arrays
    edge_offsets: np.ndarray             # int64[(days+1, states+1)] flat
    edge_next: np.ndarray                # int32[E]  to-state id
    edge_chain: np.ndarray               # int16[E]  chain id
    edge_prod: np.ndarray                # int32[crops, E]  harvested units
    edge_use: np.ndarray                 # int32[resources, E] consumed units
    # state decoding table (packed int64 key <-> TileState fields)
    state_keys: np.ndarray               # int64[n_states]
    # chain registry: chain_id -> tuple of op names (boundary decode only)
    chain_ops: tuple[tuple[str, ...], ...]

    def solve(self, prices: PriceVector, wage: WageVector,
              start_state: StateId, start_day: DayIndex) -> "ContractorSolution"
```

Objective per edge e on day d:

    value(e, d) = Σ_c edge_prod[c, e] * prices[c][d]
                − Σ_r edge_use[r, e] * wage[r][d]
                + V[edge_next[e], d+1]

Backward pass over d = 29..start_day; forward pass recovers the argmax
schedule. Everything is int arithmetic on numpy arrays; V is int64[n_states].

## Output (boundary types, decoded once)

```python
@dataclass(frozen=True)
class DailyPlan:
    day: DayIndex
    chain_id: ChainId
    chain_ops: tuple[str, ...]           # decoded for readability

@dataclass(frozen=True)
class ContractorSolution:
    schedule: tuple[DailyPlan, ...]
    production: np.ndarray               # int32[days, crops] harvested units
    resource_use: np.ndarray             # int32[days, resources] consumed
    total_profit: int
```

`production`/`resource_use` are the DW column (quantity part) the
FarmSecretary consumes; `schedule` is the per-day chain IDs she can replay.

## RAM sizing (measured)

| crop       | states | edges    | graph RAM |
|------------|--------|----------|-----------|
| WHEAT      | ~114   | ~27,360  | ~0.7 MB   |
| CARROT     | ~66    | ~15,840  | ~0.5 MB   |
| TOMATO     | ~360   | ~86,400  | ~2 MB     |
| STRAWBERRY | ~420   | ~100,800 | ~2.5 MB   |
| MELON      | ~672   | ~161,280 | ~4 MB     |

All five resident: ~10 MB. Graph building (engine-driven, slow) happens once
per crop; solve() is ~1 ms.

## v1 simplifications (documented)

- Weed-spawn RNG ignored (0.005/tile/day, single tile).
- No animals; wheat+carrot only; preconditions assumed satisfied but
  explicit on edges.
- Selling = harvesting at the secretary's day-price; no warehousing.
