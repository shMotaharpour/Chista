# F011 — Ongoing production cadence

**Summary (<=50 words):** Ongoing crops produce at nightfall from first_yield_day, repeating every interval days: +1, or +2 if that day was both watered and fertilized. TOMATO: seed 50, first day 8, daily, 4 productions. STRAWBERRY: seed 100, first day 10, every 2 days, 4 productions.

## Finding

- Production happens at nightfall, starting on `first_yield_day` and repeating every `interval` days: +1, or **+2 if that day was both watered and fertilized**.
| crop | seed | first yield | interval | productions | held cap |
|---|---|---|---|---|---|
| TOMATO | 50 | day 8 | 1 | 4 | 4 |
| STRAWBERRY | 100 | day 10 | 2 | 4 | 4 |

*Source: "Crops and animals — every rule, numbered" research document, project [AgriOracle](https://github.com/shMotaharpour/AgriOracle).*
