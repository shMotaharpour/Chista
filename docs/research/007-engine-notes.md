# Agent v1 Design Notes — engine-order and batching rules (user-verified)

## 1. Turn processing order (confirmed from source + user)

The engine's `interpreter` runs in this order:
1. Farmer + hand actions
2. Market orders (SELL/BUY/HIRE processed after actions)
3. Town consumption
4. Decay / end-of-day refresh

**Consequence — HIRE timing:** a HIRE issued on turn t spawns the hand at the END of turn t
(market phase). The hand first acts on turn t+1, and disappears at the end of that day.
So a hire issued in the FIRST turn of a day gets 23 working turns; issued later, fewer.
→ **Rule: issue all HIREs on hour 0 of each day.**

## 2. Action ordering constraints (within the action phase)

Farmer/hand actions run in list order: farmer first, then hands by index. Same-tile
state effects cascade: if the farmer WATERs tile X, a hand standing on X sees
watered_today=True only NEXT turn (each unit's action is applied sequentially against
the live state). Design consequence:
- A unit's planned action must be validated against the state BEFORE its own action
  (pre-state), not after other units' actions.
- Two units cannot both benefit from the same tile state change in one turn.
- PLANT is atomic-validated: if total PLANT requests for a crop exceed seeds, ALL are
  dropped → dispatcher must sum plant requests across farmer+hands and never over-commit.

## 3. Batching (amortized) actions — cost model

One unit performing `PICKUP WHEAT 5` = 1 turn for 5 items.
Five units each `PICKUP WHEAT 1` = 5 unit-turns for the same 5 items.

→ **Batching rule:** when a task involves n identical units at the same spot (pickup,
drop, sell-from-shed), assign it to ONE unit with the count argument, never spread
across units. Conversely, tile actions (WATER/HARVEST/FEED) are per-tile anyway —
parallelize those across hands.

Shed batching also applies to SELL: `SELL WHEAT 50` is one market order regardless of
size — market orders are counted per order (max 10/turn), not per unit.

## 4. Selling in tranches (per user spec)

Premium products must be split across a sell window (multi-day drip), not dumped:
- MILK: spread sales over 2 days (safe pace ~8/day → ~4/day if smoothing to 2 days)
- WOOL: 3-day window (~6/day)
- STRAWBERRY: 2-3 day window (safe pace 7/day)
- MELON: up to ~10/day lossless; bigger stock → 2-day window
- WHEAT/EGG: same-day dump is fine (66/day+ absorbs)
- FERTILIZER: consume first; sell surplus ≤ 53/day
- Days 29-30: forced full sale of everything regardless of pace.

Implementation: maintain a per-product sell schedule derived from the safe-pace table;
each day sell `min(shed, safe_pace)` until shed is empty; prioritize selling the
most-perishable-value item first (largest revenue-at-risk from price crash).

## Engine-order summary (per source, line ~936-947)

1. Farmer + hands actions (in listed order)
2. Market orders (lockstep with opponent)
3. Town consumption
4. Plant decay
5. End-of-day refresh (if last turn of day): plants/animals refresh, inventory drop to shed, hands disappear, hires_today reset

## Hand spawn placement — exact definition (source line 534)

`_spawn_hand(farm, board_size)`:

1. Candidates = the FOUR shed-access tiles only:
   `_shed_access_tiles(10) = [(4,4), (5,4), (4,5), (5,5)]` in NWSE order
   (inner corners around the shed, one per quadrant).
2. Count occupants: farmer + all current hands standing on those tiles.
3. Sort by `(occupancy, NWSE index)` and take the minimum — i.e. LEAST occupied
   tile, ties broken NWSE.

Practical consequences:
- A hire always appears within 1 tile of the shed; it can spawn on a LOCKED
  tile (spawn ignores locks; only tile ACTIONS no-op on locked, movement is free).
- Consecutive hires in one turn are computed sequentially: each new hire updates
  occupancy, so hires spread over the four tiles (1 each, then wrap).
- The farmer starts on (4,4). First hire of a day therefore takes (5,4) — which
  is in NE. If NE is not yet bought, that hand stands on locked ground and must
  spend one turn walking back to owned tiles.
- Spawn position is fully predictable from the observation (farmer + hands lists),
  so the agent can pre-plan a fresh hand's route with zero wasted turns.

Assignment insight for the dispatcher: all hands start from the same 4-tile
cluster, so optimal need-assignment = sort needs by distance from the shed and
hand them out in order (nearest to the first hand, next to the second, ...).
Our current greedy per-unit claiming approximates this but a shed-sorted
assignment would save walking distance.
