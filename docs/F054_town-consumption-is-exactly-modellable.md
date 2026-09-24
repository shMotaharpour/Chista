Summary: Town consumption is exactly modellable from the observation —
shops every 4 turns (single-product shops 2×), the centre every 24:
6,462/6,462 item-step deltas matched. Only the next shop unlock is
random, so a forward inventory forecast priced by the engine's own
function errs ≤1.32 % of I0 at 10 days.

Source: `bench/bench_market_forecast.py --probe` (the cadence model
against the engine's own per-turn deltas, seed 7, `weedSpawnChance 0`,
PASS-vs-PASS, 719 turns) and `--error --seeds 20` (forecast vs realised
day-start inventory, seeds 0..19).

Evidence: the model `town_deltas(shops, step)` — the shop set the
observation shows at step `s` is the set that consumes during turn `s`,
shops at `step % townShopSellInterval == 0` (a single-product shop at
multiplier 2, each instance independently), the town centre at
`step % townCenterSellInterval == 0` — reproduced **every** observed
inventory delta over a full episode: 6,462/6,462 item-step comparisons,
0 mismatches. The only unknown in the horizon is the next shop unlock
(`townShopUnlockInterval` days, drawn with replacement, capped at
`MAX_SHOP_INSTANCES`); the unlocked set itself is public.

Error of the forward forecast (worst item, worst seed of 20), inventory
as % of `I0 = 10,000` and price in coins:

| horizon | `none` policy | `mean` policy |
|---|---|---|
| day +3 | 0.000 % / 0 coins | 0.000 % / 0 coins |
| day +6 | 0.360 % / 22 coins | 0.315 % / 12 coins |
| day +10 | 1.320 % / 52 coins | 1.140 % / 30 coins |
| day +20 | 3.720 % / 115 coins | 2.865 % / 78 coins |

The day-3 zero is structural, not luck: the first unlock lands at the
start of day 3 and has not consumed yet at that day's first turn. The
`none` policy is biased one way — missing unlocks under-counts
consumption, so it holds MORE inventory and prices it LOWER.

Consequence: `agent/planner/market.py` forecasts inventory and calls the
engine's own `market_price` (R002 — the curve is never transcribed); the
`mean` policy is the default because it is measured better at every
horizon past day 3. The `d+1` cash contract is unaffected — see F055.
