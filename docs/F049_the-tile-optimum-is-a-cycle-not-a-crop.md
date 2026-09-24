# F049 — The tile optimum is a cycle, not a crop

**Summary (<=50 words):** The honest single-cycle maxima of F009 are not the
season-optimal policy. With a flat wheat price and zero wages the DP harvests 5
units on day 3 and replants the same day, 48 wheat over 30 days; F009's 6-by-day-4
cycle yields 6 per 4 days. Throughput, not per-cycle yield, is what the tile
maximises.

## What was measured

Both numbers come from the shipped graph through the DP (`agent/tile_dp/contractor.py`,
issue #11), not from a hand-built schedule:

- **Horizon that admits one cycle** (`days=5` from the bare tile, price 1 on
  wheat, wages 0): the plan plants on day 0, fertilizes and waters, and harvests
  **6 units on day 4** — F009's honest maximum, reproduced by the DP itself.
- **Full season** (`days=30`, same duals): the plan plants on day 0, fertilizes
  and waters, and harvests **5 units on day 3**, then on the same day runs
  `DIG, PLANT, WATER` and starts the next cycle. Ten harvests (9 × 5 + 3) for
  **48 wheat over the season**.

## What is read off those numbers

The 3-day cycle carries 5 units per 3 tile-days (5/3 ≈ 1.67); the honest
single-cycle calendar carries 6 per 4 tile-days (6/4 = 1.5). The engine's
HARVEST frees the tile (F007) and `DIG`+`PLANT`+`WATER` fits in one day, so a
tile is a *repeating* resource and the per-cycle maximum is not the per-season
optimum.

This is inference, not a second measurement: it explains the two readings above
and is consistent with F006's fertilized wheat sequence (1, 1, 3, 5, 6), where
day 3 already holds 5 units.

## Why it matters

- Any planning layer that treats F009's honest yield as "what a wheat tile is
  worth" understates a tile by about 14 % per cycle, and by more once replanting
  compounds.
- The master (#12) prices *columns*, and the column the contractor returns is
  the rotation, not the crop. A reduced cost computed against a fixed
  "wheat = 6 per cycle" assumption would be wrong in the direction that makes
  good columns look bad.
- F009 stays true as written — it is the honest maximum of one cycle. It must
  not be read as a season budget, which is what this finding exists to prevent.

## Scope of the claim

Measured with a flat wheat price and zero wages, i.e. with the DP free to
replant and nothing charged for labour, seed or fertilizer. With positive input
prices the cycle length is a function of the duals, and the point stands a
fortiori: what the tile is worth is decided by the rotation, not by the crop's
calendar.
