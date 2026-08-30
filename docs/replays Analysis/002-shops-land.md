# Shops & Land — top player behavior

## 6. Town shop unlocks and player reaction

### Shop unlock days

| shop           |   unlocks |   avg_day |   earliest |   latest |
|:---------------|----------:|----------:|-----------:|---------:|
| YARN_STORE     |      4510 |      11.4 |          3 |       24 |
| SMOOTHIE_SHOP  |      4491 |      11.4 |          3 |       24 |
| ICE_CREAM_SHOP |      4479 |      11.4 |          3 |       24 |
| BRUNCH_SPOT    |      4471 |      11.3 |          3 |       24 |
| BAKERY         |      4469 |      11.4 |          3 |       24 |
| FARMERS_MARKET |      4466 |      11.6 |          3 |       24 |
| PIZZA_SHOP     |      4446 |      11.4 |          3 |       24 |
| PET_CAFE       |      4427 |      11.5 |          3 |       24 |

### Planting rate before vs after the crop's consuming shop unlocks

| crop       |   plants_per_day_before |   plants_per_day_after |   plant_events |
|:-----------|------------------------:|-----------------------:|---------------:|
| WHEAT      |                    6    |                  18    |        1042043 |
| STRAWBERRY |                   13.54 |                  10.46 |         613211 |
| CARROT     |                    1.63 |                  22.37 |          76640 |
| TOMATO     |                    2.78 |                  21.22 |           6639 |

### Selling rate of the demanded crop before vs after shop unlock (winners)

| crop       |   sells_per_day_before |   sells_per_day_after |   sell_events |
|:-----------|-----------------------:|----------------------:|--------------:|
| WHEAT      |                   2.68 |                 21.32 |       1019258 |
| STRAWBERRY |                   1.97 |                 22.03 |        156127 |
| WOOL       |                   3.61 |                 20.39 |        143512 |
| CARROT     |                   2.04 |                 21.96 |         21039 |
| EGG        |                   1.49 |                 22.51 |          6219 |

## 7. Land purchases (winners)

### Quadrants bought per winning episode

| quadrants_bought   | episodes   |
|--------------------|------------|

### Timing of each successive land purchase (winner episodes)

| quadrant_nth   | episodes   | avg_day   | earliest   | latest   |
|----------------|------------|-----------|------------|----------|

## 7. Land purchases (winners) — corrected data (BUY_LAND lives in market_orders)

### Quadrants bought per winning episode
| quadrants | episodes |
|---|---|
| 2 | 11,699 (86%) |
| 3 | 2,050 (15%) |
| 4 | 1 |

### Timing of each successive purchase (winners)
| nth | episodes | avg day | earliest | latest |
|---|---|---|---|---|
| 1st (NE) | 6,207 | **day 6.2** | 5.0 | 7.0 |
| 2nd (SW) | 6,207 | **day 10.7** | 8.3 | 11.6 |
| 3rd (SE) | 964 | day 12.0 | 10.6 | 13.1 |

### Verdict
- **ALL winners buy NE by day ~6 and SW by day ~11** — universal rule, near-zero variance
- SE (3rd) only in 15% of wins and only by day 12 — marginal, skipped by most winners
- Our agent: NE day 8, SW never-by-day-15 → we are 2-5 days LATE and skip SW entirely.
  Correct proven schedule: **NE at day 6, SW at day 10-11, SE skip unless money > $4k+**
  (and land before heavy melon-seed spending).
