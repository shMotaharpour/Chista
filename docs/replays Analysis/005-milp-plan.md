# MILP Season Planner v2 — CP-SAT implementation plan (revised)

> Supersedes 005-milp-plan.md model sections. Solver: OR-Tools CP-SAT (all-integer).
> All constants verified: docs/research/006 (mechanics), docs/replays Analysis/000-004.

## Animal model — verified mechanics (user correction + live tests)

| animal | first yield | interval | max_held | production (verified source) |
|---|---|---|---|---|
| GOOSE | day 4 | 1 (DAILY) | 4 eggs on tile | 1 egg/day base; care bank caps at 4 total (max_held) |
| COW | day 8 | 2 (every other day) | 6 milk on tile | 1 milk/production day base; care bank up to 6 total |
| SHEEP | day 6 | 3 (every 3rd day) | 6 wool on tile | 1 wool/production day; care bank up to 6 total |

Feeding effect on production: NONE — base production (1/animal/production day) is
unconditional (source line 828: yield_units += base + bonus, base added regardless).
Feeding gates ONLY the care bank accumulation (+1/day if fed AND cared).
GOOSE full_care: egg daily; with care bank up to max_held=4 total per harvest.
"care دوتاش می‌کنه" = with bank, 2 eggs per production (1 base + 1 bank) capped at 4 on tile.

Key corrections:
- GOOSE first egg at day 4 (4-day wait). During those 4 days CARE banks +1/day
  (fed+care) → first production day yields 1 base + 4 banked = 5 (verified 006 V6-like).
- After day 4: KEEP_ALIVE (feed every 2nd day) → 1 egg per 2 days (0.5/day).
  FULL_CARE (feed+care daily) → 2/day (1 base + 1 banked).
- SHEEP: interval 3 does NOT align with 2-day survival feeding → keep_alive for sheep
  needs feed days exactly on production days (every 3rd day) BUT then 2 consecutive
  unfed days occur between productions (day 10, 11 between productions on 9 and 12)
  → ESCAPE. So SHEEP must be full_care. COW: interval 2 aligns with 2-day feeding —
  keep_alive works (feed on production days only). GOOSE: interval 1 means production
  daily when fed daily; alternating feed (every 2 days) = production every 2 days
  (only on fed days) — matches user: "غاز یک روز در میان تخم می‌دهد".

## CP-SAT model

### Variables
```
plant[c,d]      ∈ [0, 30]     int — crop c planted on day d
land[q]         ∈ {0,1}       — quadrant q purchased
hire[d]         ∈ [0, MAX]    — hands hired day d
crew[d]         ∈ [1, 12]     — total workers (1 + carried + hired)
sell[p,d]       ∈ [0, 200]    int — units of p sold day d
stock[p,d]      ∈ [0, 100]    int — shed units end of day d
harvest[p,d]    ∈ [0, 50]     int — units of p harvested day d
mode[a]         ∈ {0,1}       0=KEEP_ALIVE 1=FULL_CARE (per animal, once bought)
buy_animal[a,d] ∈ [0, 2]      — animals of type a bought day d
alive[a,d]      ∈ [0, 10]     — animals of type a alive day d
fert_collect[d] ∈ [0, 10]     — fertilizer units collected day d
```

### Constraints
1. Cash path (integer dollars):
   money[d+1] = money[d] + Σ_p sell[p,d]·price[p,d] − Σ_c plant[c,d]·seed[c]
                − Σ_q land[q]·price_q·[d == buy_day] − hire_cost(d) − feed_cost(d) − animal_cost(d)
   ≥ 0 ∀d
2. Tiles: Σ_c live_plants(c,d) ≤ 25·(1 + Σ_q land[q]·(d ≥ LAND_DAYS[q]))
3. Watering/work capacity (VRP proxy):
   Σ_c water_actions(c,d) + feed_actions(d) + care_actions(d) ≤ crew[d] × 24
   where water_actions: wheat/carrot = 1/tile, melon/tomato/strawberry = 0.5 (alternating)
   feed_actions: Σ_a (1 if mode FULL_CARE else 0.5 scaled)
   → integer modeling: multiply by 2 (half-actions allowed).
4. Shed: stock[p,d] ≤ 100
5. Harvest: harvest[c,d] = plant[c, d − maxyd] × yield(c)  (one-time crops)
   ongoing: harvest every interval days from first_yield_day, capped at max_yield
6. Market absorption: revenue is piecewise-linear in total daily sell volume.
   Model: sell[p,d] split over K tranches with per-tranche marginal price
   (precomputed from lab/prices.py MARKET_PARAMS).
7. Animals:
   - buy: buy_animal[a,d] ≤ 1, first production at d + first_yield
   - alive[a,d] = alive[a,d-1] + bought − escaped (escape if KEEP_ALIVE misaligned —
     prevented by construction: feed schedule aligned to production days)
   - mode[a]: binary. FULL_CARE → daily feed+care actions, production = 1 + bank
     (bank +1/day while alive under FULL_CARE, consumed each production)
     KEEP_ALIVE → feed on production days only, production = 1/production day, no bank
   - fertilizer: alive[a,d] → 1 fertilizer available per day (all modes)
8. Terminal: Σ_p sell[p,29] + sell[p,30] = stock[p,28]; maximize money[30]
9. Feeding: wheat needed per day = Σ aliveanimals fed that day; sourced from
   buy_wheat[d] + stock[wheat,d] (feed = 1 wheat/animal/fed-day)

### Objective
maximize money[30]

### Feeding-mode economics (verified live):
- GOOSE KEEP_ALIVE: feed every 2nd day, egg every 2nd day (0.5 egg/day), fert 1/day
  → 0.5 action/day + 0.5 feed cost; vs FULL_CARE 1 egg/day + bank.
- COW KEEP_ALIVE: feed production days only (aligned, interval 2) → survives,
  1 milk/2 days, no bank. FULL_CARE: 2 milk/2 days (1+bank).
- SHEEP: interval 3 misaligns → full_care only.
Decision variable in model; choose by prices at solve time.

### Layout (003 §4): strawberry east (x≥8), wheat west, pasture center —
spatial cost modeled as Manhattan distance from shed (4.5, 4.5); VRP proxy =
Σ task distances / crew, bounded by 24 turns/crew/day.

## Implementation
- lab/milp/cpsat_model.py — build model from constants
- lab/milp/solve.py — solve, export plan.json
- lab/milp/execute.py — plan.json → agent ops (reuses v1.2 dispatcher for tile ops)
- validate: run env with plan vs 'pass' on 5 seeds, compare money vs predicted

## Baseline to beat
v1.2: $28k mean (rule-based). Target: > $40k predicted, > $20k executed.
