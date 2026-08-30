# How Top Players Trade — deep replay analysis

Data: 6,269 competition episodes (Aug 15-24). 'Winner' = higher final bank.

## 1. Price-timed or calendar? Price when selling vs season median

| item | sells | avg price at sell | season median | ratio |
|---|---|---|---|---|
| WHEAT | 779,386 | $44.8 | $41 | 1.09 |
| CARROT | 29,849 | $59.8 | $38 | 1.57 |
| TOMATO | 4,446 | $145.5 | $65 | 2.24 |
| STRAWBERRY | 236,877 | $117.0 | $158 | 0.74 |
| MELON | 64,858 | $145.6 | $131 | 1.11 |
| EGG | 7,787 | $58.1 | $52 | 1.12 |
| MILK | 246,086 | $97.2 | $172 | 0.57 |
| WOOL | 205,625 | $118.1 | $185 | 0.64 |

## 2. Dump vs drip — order sizes (winners)

| item       |   orders |   avg_units_per_order |   max_order |
|:-----------|---------:|----------------------:|------------:|
| FERTILIZER |   903360 |                  11.4 |       1e+06 |
| WHEAT      |   779386 |                   8.3 |       1e+06 |
| MILK       |   246086 |                  14.6 |       1e+06 |
| STRAWBERRY |   236877 |                  15.9 |       1e+06 |
| WOOL       |   205625 |                   5.4 |     999     |
| MELON      |    64858 |                   8.7 |     999     |
| CARROT     |    29849 |                   5.5 |     999     |
| EGG        |     7787 |                   4.9 |     999     |
| TOMATO     |     4446 |                   7.5 |     999     |

Orders-per-turn distribution (SELL, winners):
|   n_orders |   turns |
|-----------:|--------:|
|          1 | 1831460 |
|          2 |  946871 |
|          3 |  364287 |
|          4 |  114209 |
|          5 |   41040 |
|          6 |   12081 |
|          7 |    2958 |
|          8 |    1392 |
|          9 |     162 |
|         10 |    2943 |

## 3. BUY_PRODUCT usage by winners (arbitrage?)

| item       |   buys |         units |   episodes_using |
|:-----------|-------:|--------------:|-----------------:|
| WHEAT      | 489857 |   3.42213e+06 |             6207 |
| WOOL       |    243 | 583           |                6 |
| FERTILIZER |      8 |  95           |                4 |

## 4. Planting portfolio by phase (winners)

| phase   | crop       |   plants |
|:--------|:-----------|---------:|
| d0-5    | WHEAT      |    66290 |
| d0-5    | MELON      |    64683 |
| d0-5    | STRAWBERRY |     7162 |
| d0-5    | CARROT     |      767 |
| d16-25  | WHEAT      |   556417 |
| d16-25  | CARROT     |    47138 |
| d16-25  | TOMATO     |     2705 |
| d16-25  | STRAWBERRY |      362 |
| d16-25  | MELON      |      153 |
| d6-15   | STRAWBERRY |   224289 |
| d6-15   | WHEAT      |   169342 |
| d6-15   | MELON      |    26128 |
| d6-15   | CARROT     |     4778 |
| d6-15   | TOMATO     |     1212 |

## 5. Revenue by period (winners)

| period     |       units |   sells |
|:-----------|------------:|--------:|
| before d25 | 1.39006e+07 | 1740580 |
| d25-27     | 3.35888e+06 |  486710 |
| d28-30     | 8.82046e+06 |  250984 |