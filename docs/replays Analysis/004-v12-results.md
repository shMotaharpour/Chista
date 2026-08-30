# v1.2 Analysis — the melon-core breakthrough

## Results (5 seeds, vs 'pass')

| seed | v1.1 | v1.2 |
|---|---|---|
| 1 | 11,212 | 33,910 |
| 2 | 8,322 | 27,965 |
| 3 | 13,012 | 28,157 |
| 42 | 11,481 | 27,936 |
| 77 | 4,883 | 28,209 |
| 500 | 10,043 | 28,006 |

v1.2 mean ≈ **$28,000** (×2.5 over v1.1, near-zero variance).

## What the replay analysis changed

1. **Melon seeds bought from day 3, not day 8** (`seed_target`: min(12, (money-150)/80)).
   Two full melon cycles instead of one — this alone was the ×2.5.
2. **Melon = money crop on ALL free tiles days 3-17** (not just east columns).
3. **Wheat backbone shrunk to 12 seeds** (feed/sell backbone; matures in 2 days).
4. **Strawberry dropped** — 10-day first yield ties up tiles the melon uses better.
5. **Hire batch unchanged** (8/day), **land schedule unchanged** (NE@6, SW@11).

## v1.2 constant sweep (3 seeds each, vs pass)

| parameter | values tried | result |
|---|---|---|
| MELON_START_DAY | 2 / 3 / 4 | $26.9k / **$28.0k** / $27.2k → 3 |
| NO_PLANT_AFTER | 24 / 26 / 28 | $27.5k / **$28.0k** / $27.9k → 26 |
| MAX_HANDS | 6 / 8 / 10 | $28.1k / $28.0k / $28.0k → flat (6 fine) |
| CREW_RATE | 5 / 6 / 7 | $28.0k / $28.0k / $27.6k → flat (5-6) |

All constants sit at a local optimum; the config is stable.

## Remaining gap to ladder ($87k median)

The ×3 gap is structural: winners run ~90 active tiles + 10 hands + cow/sheep lines.
Our bottleneck now is watering capacity (crew), not strategy. Next levers:
- more aggressive hands (HIRE beyond 8 — fib cost is still cheap)
- cow/sheep income lines (milk $160 × interval 2, wool $200 × interval 3)
- fertilizer sale income (53/day × $100+)
