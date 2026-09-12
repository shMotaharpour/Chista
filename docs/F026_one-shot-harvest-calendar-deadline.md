# F026 — One-shot harvest calendar and deadline

**Summary (<=50 words):** One-shot crops have exactly one harvest, refused while age < first_yield_day, and it destroys the plant. Ready days: WHEAT 2, CARROT 2, MELON 10. Water before harvesting the same day — WATER's unit lands immediately. The deadline is real: a day late costs most of the tile.

## Finding

- Days count from the day the seed went in (day 0); production happens at nightfall, so what a night makes is in hand the next morning.
- HARVEST is refused while `age < first_yield_day`, and the plant is destroyed by the harvest — there is exactly one.
| crop | first legal day | held then, watered daily | last full day | with one fertilizer |
|---|---|---|---|---|
| WHEAT | day 2 | 2 | day 4 (4 units) | day 4, 6 units |
| CARROT | day 2 | 2 | day 3 (3 units) | day 3, 4 units |
| MELON | day 10 | 6 — already the cap | day 12 (6 units) | day 10 is enough |
- Water *before* harvesting on the same day: WATER adds its unit the moment it runs.
- The deadline is real: from the morning of `planted + max_yield_day + 1` the plant loses one unit every two turns and is a weed within hours. Harvesting a day late is not a small loss, it is most of the tile.

*Source: "Crops and animals — every rule, numbered" research document.*
