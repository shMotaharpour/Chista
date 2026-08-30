# Shop Distributions, Crop Correlations, Land Modes (phase 3)

Data: 6,269 competition episodes.

## 1. P(shop opens at each unlock slot)

Slots are the 8 unlock events at days 3, 6, 9, ..., 24. Every slot is ~uniform
across the 8 shop types (11.5% - 13.3% each). **The unlock draw is effectively
uniform with replacement** — no shop is meaningfully more likely in any slot.
Players therefore cannot plan around WHICH shops appear; only how many and when.

## 2. First unlocked shop (day 3) vs winner's crop mix

For every first-shop type, the winner's crop mix is essentially identical:
WHEAT 73-90% of all plants, strawberry/melon second. The first shop type has
**no measurable influence** on what winners plant. The day-3 shop is too early
to matter (crops were chosen before it appeared) and winners don't pivot.

## 3. Winner vs opponent crop correlation (same episode)

| crop | episodes | winner avg share | loser avg share | corr |
|---|---|---|---|---|
| WHEAT | 6,269 | 67.5% | 68.1% | 0.202 |
| STRAWBERRY | 6,269 | 20.0% | 20.0% | 0.301 |
| MELON | 6,269 | 7.8% | 8.3% | 0.317 |
| CARROT | 6,086 | 4.5% | 3.6% | 0.279 |
| TOMATO | 549 | 3.9% | 1.0% | -0.078 |

Interpretation:
- Winner/loser shares track each other weakly (corr 0.2-0.3) — everyone follows
  roughly the same meta (wheat+strawberry backbone), the winner is not counter-
  picking the opponent.
- TOMATO is the only differentiation: winners plant 4x more tomato (3.9% vs 1.0%
  share, appearing in only 549 episodes) — a niche edge crop, not a backbone.

## 4. Mode of each of the 25 NW tiles at day 25 hour 0 (winner farm)

Spatial layout revealed (x grows east, y grows south; shed center at 4-5):

- **(4,2), (5,2): PASTURE** — both pasture tiles sit in the CENTER ROW
  (6,102 and 5,586 of 6,269 episodes — near-universal!)
- **WHEAT dominates most tiles**: rows y=0..3 majority WHEAT (feed line + sell crop)
- **STRAWBERRY cluster**: tiles (8,0),(9,0),(6,1),(7,1),(8,1),(8,2),(9,2),(8,3)... —
  the FAR EAST columns are strawberry (ongoing crop, long-term yield)
- **CARROT pockets**: (0,0), (1,1) — near-shed early-harvest slots
- Weeds rare at day 25 (only (9,1): WEED in 2,363 eps) — winners keep the farm clean

**Standard winning layout: wheat everywhere, strawberry in east columns,
pasture pair in the center row, carrot near the shed.**

## 5. Average crops planted per 3-day period (winners)

| days | total plants | plants/episode |
|---|---|---|
| 0-2 | 112,241 | 17.9 |
| 3-5 | 28,142 | **4.6** (trough — capital locked in first cycle) |
| 6-8 | 90,789 | 14.5 |
| 9-11 | 113,681 | 18.1 |
| 12-14 | 209,415 | **33.4** (peak — strawberry expansion) |
| 15-17 | 92,613 | 14.8 |
| 18-20 | 105,522 | 16.8 |
| 21-23 | 123,472 | 19.7 |
| 24-26 | 179,487 | **28.6** (second peak — wheat wall for end-season sell) |
| 27-29 | 128,135 | 20.7 (still planting until the end!) |

Planting never stops — even days 27-29 average 20 plants/episode. The
"NO_PLANT_AFTER" idea from our v1 is wrong: winners keep planting cheap
wheat to the last day (it matures in 2 days and sells).
