# F045 — Weed RNG couples farms to tomorrow's prices

**Summary (<=50 words):** _spawn_weeds draws rng.random() once per empty tile
on both farms before the same stream picks the town's next shop
(weedSpawnChance = 0.005; LOCKED tiles draw nothing). Planting a tile
therefore changes tomorrow's prices, and no price path can be precomputed
offline.

## Finding

- `weedSpawnChance = 0.005` in the competition configuration.
- `_spawn_weeds` draws `rng.random()` once per empty tile on both farms
  before the same stream picks the town's next shop. So **planting a tile
  changes tomorrow's prices**, and no price path can be precomputed offline.
- A LOCKED tile draws nothing.

*Source: "Kaggriculture — the rules, as we have established them" research
document, section 10 (Weeds and the RNG).*
